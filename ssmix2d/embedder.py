"""Training-free audio embedder built on the 2-D (time x frequency) cross-scale scale-space mixer."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .crossscale2d import CrossScaleMixer2D

ArrayLike = Union[np.ndarray, torch.Tensor]

DEFAULT_READOUT = ("bands", "stats", "mixed", "attn", "env")
READOUT_TERMS = ("bands", "stats", "mixed", "attn", "env")


class AudioEmbedder2D(nn.Module):
    """Training-free audio transformer that mixes jointly across time and frequency.

    Pipeline:
        log-mel spectrogram (25 ms window, 10 ms hop)  ->  per-clip normalisation
        depth x [ separable Daubechies tight frame over (time, mel)
                  -> 2-D cross-scale attention (time and frequency siblings, parents, children)
                  -> exact synthesis ]
        readout of every block, pooled over time.

    The model has no learnable parameters and needs no training data.

    Args:
        depth:        number of mixer blocks.
        sample_rate:  rate the model operates at; `embed` resamples inputs to it.
        n_mels:       mel bands.
        order:        Daubechies order along time (support 2N-1).
        freq_order:   Daubechies order along the mel axis.
        levels:       time levels J_t (None: chosen from the clip length).
        freq_levels:  mel-axis levels J_f (None: chosen from n_mels).
        group:        mel bins per attention token (a divisor of 16).
        window:       sibling half-width k on each axis.
        tau:          attention temperature.
        readout:      subset of ("bands", "stats", "mixed", "attn", "env").

    Readout terms (per block unless noted):
        bands  log mean_t |W_{s,r}| before mixing, per time band s, mel band r and mel bin
        stats  mean_t and std_t of the normalised log-mel state (also once after the last block)
        mixed  log mean_t |W'_{s,r}| after mixing
        attn   mean_t attention weight per band, link and mel group
        env    mean log-mel spectrum before normalisation (once, appended last)

    Example:
        >>> emb = AudioEmbedder2D()
        >>> E = emb.embed([wave_a, wave_b], sample_rate=44100, length=5.0)   # (2, D) numpy array
    """

    def __init__(self, depth: int = 2, sample_rate: int = 32000, n_mels: int = 128, order: int = 4,
                 freq_order: int = 2, levels: Optional[int] = None, freq_levels: Optional[int] = None,
                 group: int = 8, window: int = 2, tau: float = 0.1, readout: Sequence[str] = DEFAULT_READOUT):
        super().__init__()
        unknown = set(readout) - set(READOUT_TERMS)
        if unknown:
            raise ValueError(f"unknown readout terms: {sorted(unknown)}")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        win = int(round(0.025 * sample_rate))
        hop = int(round(0.010 * sample_rate))
        self.sample_rate = int(sample_rate)
        self.depth = int(depth)
        self.n_mels = int(n_mels)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=win, win_length=win, hop_length=hop,
                                                        n_mels=n_mels, power=2.0)
        self.mixer = CrossScaleMixer2D(order=order, freq_order=freq_order, levels=levels, freq_levels=freq_levels,
                                       group=group, window=window, tau=tau)
        self.readout = tuple(readout)
        self._dim_cache: dict = {}
        self._config = dict(depth=int(depth), sample_rate=int(sample_rate), n_mels=int(n_mels), order=int(order),
                            freq_order=int(freq_order), levels=None if levels is None else int(levels),
                            freq_levels=None if freq_levels is None else int(freq_levels), group=int(group),
                            window=int(window), tau=float(tau), readout=list(readout))

    # ------------------------------------------------------------- hub I/O
    @property
    def config(self) -> dict:
        """Constructor arguments; `AudioEmbedder2D(**emb.config)` rebuilds the identical model."""
        return dict(self._config, readout=list(self._config["readout"]))

    def save_pretrained(self, directory: str) -> str:
        """Write config.json to `directory` (there are no weights). Returns the file path."""
        import json
        import os
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "config.json")
        with open(path, "w") as f:
            json.dump({"model_type": "ssmix2d-audio-embedder", **self.config}, f, indent=2)
        return path

    @classmethod
    def from_pretrained(cls, name_or_path: str, revision: Optional[str] = None) -> "AudioEmbedder2D":
        """Build from a local directory holding config.json, or from a Hugging Face Hub repo id
        (e.g. "user/repo"; needs `pip install huggingface_hub`)."""
        import json
        import os
        if os.path.isdir(name_or_path):
            path = os.path.join(name_or_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(name_or_path, "config.json", revision=revision)
        with open(path) as f:
            cfg = json.load(f)
        cfg.pop("model_type", None)
        cfg["readout"] = tuple(cfg.get("readout", DEFAULT_READOUT))
        return cls(**cfg)

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def logmel(self, wav: torch.Tensor):
        """(B, N) waveforms -> per-clip normalised log-mel (B, T, n_mels) and the mean log-mel envelope (B, n_mels)."""
        X = torch.log(self.mel(wav).transpose(1, 2) + 1e-6)
        env = X.mean(1)
        return X - env.unsqueeze(1), env

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Embed a batch of equal-length waveforms.

        Args:
            wav: (B, N) float tensor at `sample_rate`, values in [-1, 1].
        Returns:
            (B, D) embedding.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        X, env = self.logmel(wav)
        return self.from_logmel(X, env)

    @torch.no_grad()
    def from_logmel(self, X: torch.Tensor, env: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Readout from a normalised log-mel (B, T, n_mels); `env` is appended when given and requested."""
        r, m, feats = set(self.readout), self.mixer, []
        for _ in range(self.depth):
            W, plan = m.coefficients(X)
            ts, fs = m.valid(plan)
            g0, g1 = fs.start // m.group, -(-fs.stop // m.group)
            if "bands" in r:
                feats.append(torch.log(W[..., ts, fs].abs().mean(-2) + 1e-6).flatten(1))
            if "stats" in r:
                feats += [X.mean(1), X.std(1)]
            m.att_crop = (ts.start, ts.stop, g0, g1)
            try:
                Wm = m.mix(W)
            finally:
                m.att_crop = None
            if "mixed" in r:
                feats.append(torch.log(Wm[..., ts, fs].abs().mean(-2) + 1e-6).flatten(1))
            if "attn" in r:
                feats.append(torch.cat([a.flatten(1) for a in m.last_attention.values()], 1))
            X = m.synthesize(Wm, plan)
        if "stats" in r:
            feats += [X.mean(1), X.std(1)]
        if "env" in r and env is not None:
            feats.append(env)
        return torch.cat(feats, 1)

    # ------------------------------------------------------------ convenience
    def dim(self, num_samples: int) -> int:
        """Embedding dimension for clips of `num_samples` samples at `sample_rate`."""
        if num_samples not in self._dim_cache:
            p = next(self.mel.buffers())
            self._dim_cache[num_samples] = self.forward(torch.zeros(1, num_samples, device=p.device)).shape[1]
        return self._dim_cache[num_samples]

    @torch.no_grad()
    def embed(self, audio: Union[ArrayLike, Iterable[ArrayLike]], sample_rate: Optional[int] = None,
              batch_size: int = 4, length: Optional[float] = None) -> np.ndarray:
        """Embed one clip, a batch, or a list of clips of any lengths.

        Args:
            audio:       1-D waveform, (B, N) array/tensor, or a list of 1-D waveforms (mono, [-1, 1]).
            sample_rate: rate of `audio`; resampled to `self.sample_rate` when different.
            batch_size:  clips per forward pass.
            length:      if given, every clip is zero-padded or cut to this many seconds, so all
                         embeddings share one dimension. Without it, clips of different lengths
                         can have different embedding sizes (the number of time levels follows
                         the clip length).
        Returns:
            (B, D) float32 numpy array (rows in input order).
        """
        if isinstance(audio, (np.ndarray, torch.Tensor)) and np.ndim(audio) == 2:
            clips = [audio[i] for i in range(audio.shape[0])]
        elif isinstance(audio, (np.ndarray, torch.Tensor)):
            clips = [audio]
        else:
            clips = list(audio)
        p = next(self.mel.buffers())
        sr = self.sample_rate if sample_rate is None else int(sample_rate)
        prepared = []
        for c in clips:
            t = torch.as_tensor(np.asarray(c, dtype=np.float32) if not torch.is_tensor(c) else c, dtype=torch.float32)
            if t.dim() != 1:
                raise ValueError("each clip must be a 1-D mono waveform")
            if sr != self.sample_rate:
                t = torchaudio.functional.resample(t, sr, self.sample_rate)
            if length is not None:
                n = int(round(length * self.sample_rate))
                t = torch.nn.functional.pad(t, (0, max(0, n - t.numel())))[:n]
            prepared.append(t)
        order = sorted(range(len(prepared)), key=lambda i: prepared[i].numel())
        out = [None] * len(prepared)
        i = 0
        while i < len(order):
            n = prepared[order[i]].numel()
            group = [order[i]]
            while i + len(group) < len(order) and len(group) < batch_size and prepared[order[i + len(group)]].numel() == n:
                group.append(order[i + len(group)])
            x = torch.stack([prepared[g] for g in group]).to(p.device)
            e = self.forward(x).float().cpu().numpy()
            for g, row in zip(group, e):
                out[g] = row
            i += len(group)
        dims = {r.shape[0] for r in out}
        if len(dims) > 1:
            raise ValueError("clips produced different embedding sizes; pass length=... to fix the duration")
        return np.stack(out)
