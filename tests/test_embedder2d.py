import json
import os

import numpy as np
import pytest
import torch

from ssmix2d import AudioEmbedder2D


def clip(sr, sec, f0=220.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * sec)) / sr
    return (0.2 * np.sin(2 * np.pi * f0 * t) + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def emb():
    return AudioEmbedder2D()


def test_zero_parameters(emb):
    assert sum(p.numel() for p in emb.parameters()) == 0


def test_shape_dim_and_finite(emb):
    E = emb.embed([clip(32000, 2.0), clip(32000, 2.0, 330)])
    assert E.shape == (2, emb.dim(64000)) and E.dtype == np.float32 and np.isfinite(E).all()


def test_dimension_for_five_seconds(emb):
    assert emb.dim(5 * 32000) == 59392


def test_deterministic_and_batch_invariant(emb):
    clips = [clip(32000, 1.5, f, s) for s, f in enumerate((200, 300, 400))]
    a = emb.embed(clips, batch_size=1)
    b = emb.embed(clips, batch_size=3)
    assert np.allclose(a, b, atol=1e-4)
    assert np.array_equal(emb.embed(clips), emb.embed(clips))


def test_resampling_and_length(emb):
    E = emb.embed([clip(16000, 1.0), clip(44100, 3.0)], sample_rate=16000, length=2.0)
    assert E.shape[0] == 2 and E.shape[1] == emb.dim(64000)


def test_input_forms(emb):
    x = clip(32000, 1.0)
    a = emb.embed(x)
    b = emb.embed(np.stack([x, x]))
    c = emb.embed(torch.from_numpy(x))
    assert a.shape == (1, b.shape[1]) and np.allclose(a[0], b[1], atol=1e-5) and np.allclose(a, c, atol=1e-6)


def test_mixed_lengths_need_length(emb):
    with pytest.raises(ValueError):
        emb.embed([clip(32000, 0.5), clip(32000, 3.0)])


def test_rejects_multichannel(emb):
    with pytest.raises(ValueError):
        emb.embed([np.zeros((2, 32000), np.float32)])


def test_distinguishes_sounds(emb):
    tones = [clip(32000, 1.0, f, s) for s, f in enumerate((210, 220, 230))]
    t = np.arange(32000) / 32000
    noise = [(np.random.default_rng(s).standard_normal(32000) * 0.2 * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
             for s in range(3)]
    E = emb.embed(tones + noise)
    Z = (E - E.mean(0)) / (E.std(0) + 1e-8)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T
    within = (S[:3, :3].sum() - 3 + S[3:, 3:].sum() - 3) / 12
    across = S[:3, 3:].mean()
    assert within > across


def test_readout_subsets_and_validation():
    full = AudioEmbedder2D().dim(32000)
    parts = sum(AudioEmbedder2D(readout=(t,)).dim(32000) for t in ("bands", "stats", "mixed", "attn", "env"))
    assert parts == full
    with pytest.raises(ValueError):
        AudioEmbedder2D(readout=("nope",))
    with pytest.raises(ValueError):
        AudioEmbedder2D(depth=0)


def test_config_round_trip(tmp_path):
    e = AudioEmbedder2D(depth=1, order=6, freq_order=3, window=1, tau=0.2, n_mels=64, group=4)
    path = e.save_pretrained(str(tmp_path))
    cfg = json.load(open(path))
    assert cfg["model_type"] == "ssmix2d-audio-embedder"
    e2 = AudioEmbedder2D.from_pretrained(str(tmp_path))
    assert e2.config == e.config
    x = clip(32000, 1.0)
    assert np.array_equal(e.embed(x), e2.embed(x))
    assert AudioEmbedder2D(**e.config).config == e.config


def test_from_logmel_matches_forward(emb):
    x = torch.from_numpy(np.stack([clip(32000, 1.0), clip(32000, 1.0, 330)]))
    X, env = emb.logmel(x)
    assert torch.allclose(emb.from_logmel(X, env), emb(x))
