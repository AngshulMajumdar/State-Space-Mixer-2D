"""ssmix2d: a training-free time-frequency audio transformer on Daubechies wavelets (Hugging Face Space)."""
import os
import tempfile
import wave

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from ssmix2d import AudioEmbedder2D  # noqa: E402

torch.set_grad_enabled(False)
EMB = AudioEmbedder2D()  # zero parameters: nothing to download
SR = EMB.sample_rate
CLIP_SECONDS = 5.0
MAX_SECONDS = 30.0


# ------------------------------------------------------------------ audio helpers
def to_wave(audio):
    """Gradio (sample_rate, ndarray) -> float32 mono at SR, clipped to MAX_SECONDS."""
    if audio is None:
        raise gr.Error("Please upload or record a clip.")
    sr, x = audio
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0:
        raise gr.Error("The clip is empty.")
    t = torch.from_numpy(x)
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t[: int(MAX_SECONDS * SR)]


def fixed_length(t, seconds=CLIP_SECONDS):
    n = int(seconds * SR)
    return torch.nn.functional.pad(t, (0, max(0, n - t.numel())))[:n]


# ------------------------------------------------------------------ compare tab
def compare(a, b):
    ta, tb = fixed_length(to_wave(a)), fixed_length(to_wave(b))
    E = EMB(torch.stack([ta, tb])).numpy()
    Z = E / np.linalg.norm(E, axis=1, keepdims=True)
    cos = float(Z[0] @ Z[1])
    path = os.path.join(tempfile.mkdtemp(), "ssmix2d_embeddings.npy")
    np.save(path, E.astype(np.float32))
    msg = (f"### Cosine similarity: **{cos:.3f}**\n"
           f"Each clip is padded or cut to {CLIP_SECONDS:.0f} s and mapped to a {E.shape[1]:,}-dimensional "
           f"embedding with zero learned parameters. For classification, use the embeddings with a linear model "
           f"or k-NN; raw cosine is a quick, unsupervised view.")
    return msg, path


# ------------------------------------------------------------------ inspect tab
KINDS = (("time: finer", "#9ecae9"), ("time: sibling", "#4c78a8"), ("time: coarser", "#1f3b63"),
         ("frequency: finer", "#ffbf79"), ("frequency: sibling", "#f58518"), ("frequency: coarser", "#a14d00"))


def inspect(a):
    x = fixed_length(to_wave(a))
    X, _ = EMB.logmel(x.unsqueeze(0))  # (1, T, F) normalised log-mel
    m = EMB.mixer
    W, plan = m.coefficients(X)
    ts, fs = m.valid(plan)
    g0, g1 = fs.start // m.group, -(-fs.stop // m.group)
    m.mix(W)
    S, R = plan["S"], plan["R"]
    T = plan["T"]
    L = m.links(S, R)
    tb = [f"D{j + 1}" for j in range(S - 1)] + [f"A{S - 1}"]
    fb = [f"d{j + 1}" for j in range(R - 1)] + [f"a{R - 1}"]
    t_axis = np.arange(T) * 0.01

    fig, ax = plt.subplots(4, 1, figsize=(9, 13), gridspec_kw={"height_ratios": [1.2, 1.1, 1, 1]})
    ax[0].imshow((X[0] + 0).T.numpy(), origin="lower", aspect="auto", cmap="magma",
                 extent=[0, t_axis[-1], 0, X.shape[2]])
    ax[0].set_title("Log-mel spectrogram (128 bands, 32 kHz), normalised per clip")
    ax[0].set_ylabel("mel band")

    E = W[0, :, :, ts, fs].abs().mean((-1, -2)).numpy()  # (S, R)
    im = ax[1].imshow(np.log10(E + 1e-8).T, origin="lower", aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(S), tb)
    ax[1].set_yticks(range(R), fb)
    ax[1].set_xlabel("time scale (db4): fine → coarse")
    ax[1].set_ylabel("mel scale (db2)")
    ax[1].set_title("2-D Daubechies scale space: log10 mean |coefficient| of every time × frequency band")
    fig.colorbar(im, ax=ax[1], fraction=0.02, pad=0.01)

    away = np.stack([np.mean([1.0 - m.last_attention[(s, r)][0, 0, ts, g0:g1].mean(-1).numpy() for r in range(R)], 0)
                     for s in range(S)])  # (S, T)
    im = ax[2].imshow(away, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=max(away.max(), 1e-3),
                      extent=[0, t_axis[-1], -0.5, S - 0.5])
    ax[2].set_yticks(range(S), tb)
    ax[2].set_xlabel("time (s)")
    ax[2].set_title("Cross-scale attention: weight each coefficient takes from its neighbours (1 − self weight)")
    fig.colorbar(im, ax=ax[2], fraction=0.02, pad=0.01)

    acc = np.zeros((S, len(KINDS)))
    for (s, r), ks in L.items():
        w = m.last_attention[(s, r)][0][:, ts, g0:g1].mean((-1, -2)).numpy()
        for (s2, r2, dt, df), v in zip(ks, w):
            if (s2, r2, dt, df) == (s, r, 0, 0):
                continue
            if r2 == r and df == 0:
                k = 1 if s2 == s else (0 if s2 < s else 2)
            else:
                k = 4 if r2 == r else (3 if r2 < r else 5)
            acc[s, k] += float(v) / R
    left = np.zeros(S)
    for k, (name, c) in enumerate(KINDS):
        ax[3].barh(range(S), acc[:, k], left=left, color=c, label=name)
        left += acc[:, k]
    ax[3].set_yticks(range(S), tb)
    ax[3].set_xlabel("mean attention weight (averaged over mel scales)")
    ax[3].set_title("Where each time band's attention goes (excluding itself)")
    ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ examples (synthetic, no licences)
def _write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return path


def make_examples():
    d = tempfile.mkdtemp()
    t = np.arange(int(4 * 16000)) / 16000
    rng = np.random.default_rng(0)
    tone = sum(np.sin(2 * np.pi * k * 220 * t) / k for k in range(1, 6)) * 0.15
    tone2 = sum(np.sin(2 * np.pi * k * 330 * t) / k for k in range(1, 6)) * 0.15
    chirp = 0.3 * np.sin(2 * np.pi * (100 * t + 400 * t ** 2))
    bursts = np.zeros_like(t)
    for c in rng.uniform(0, 3.8, size=8):
        i = int(c * 16000)
        bursts[i:i + 1200] = rng.standard_normal(len(bursts[i:i + 1200])) * 0.4
    return {n: _write(os.path.join(d, f"{n}.wav"), s) for n, s in
            (("tone_220Hz", tone), ("tone_330Hz", tone2), ("chirp", chirp), ("noise_bursts", bursts))}


EX = make_examples()

ABOUT = """
# ssmix2d · training-free time–frequency audio transformer
A 2-D Daubechies wavelet frame over time × mel + cross-scale attention, with **zero learned parameters** and
**no training data**. Each coefficient attends to its siblings, parents and children along time **and** frequency.
The embedding is read out from the 2-D scale space and from the attention maps themselves.
[Code](https://github.com/AngshulMajumdar/State-Space-Mixer-2D) ·
[Article](https://angshulmajumdar.github.io/State-Space-Mixer-2D/article.html) ·
[Docs](https://angshulmajumdar.github.io/State-Space-Mixer-2D/)
"""

with gr.Blocks(title="ssmix2d: training-free time-frequency audio transformer") as demo:
    gr.Markdown(ABOUT)
    with gr.Tab("Compare two clips"):
        with gr.Row():
            a = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip A")
            b = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip B")
        go = gr.Button("Embed and compare", variant="primary")
        out = gr.Markdown()
        f = gr.File(label="Embeddings (.npy, shape 2 × D)")
        go.click(compare, [a, b], [out, f])
        gr.Examples([[EX["tone_220Hz"], EX["tone_330Hz"]], [EX["tone_220Hz"], EX["noise_bursts"]],
                     [EX["chirp"], EX["noise_bursts"]]], inputs=[a, b])
    with gr.Tab("Inspect the scale space"):
        c = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip")
        go2 = gr.Button("Analyse", variant="primary")
        plot = gr.Plot()
        go2.click(inspect, c, plot)
        gr.Examples([[EX["chirp"]], [EX["noise_bursts"]], [EX["tone_220Hz"]]], inputs=[c])

if __name__ == "__main__":
    demo.launch()
