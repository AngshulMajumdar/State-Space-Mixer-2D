"""ssmix2d quickstart: training-free time-frequency audio embeddings and the 2-D mixer as a PyTorch layer.

    python examples/quickstart.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmix2d import AudioEmbedder2D, CrossScaleMixer2D  # noqa: E402

rng = np.random.default_rng(0)
sr = 16000
t = np.arange(2 * sr) / sr


def tone():  # harmonic tone with random pitch
    f0 = rng.uniform(200, 400)
    return sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6)) * 0.1


def bursts():  # trains of short noise bursts
    x = np.zeros_like(t)
    for c in rng.uniform(0, 1.9, size=6):
        i = int(c * sr)
        x[i:i + 800] = rng.standard_normal(800) * 0.3
    return x


# 1) Embeddings: no training, any sample rate, any clip length (fixed with `length`)
clips = [tone() for _ in range(6)] + [bursts() for _ in range(6)]
labels = np.array([0] * 6 + [1] * 6)
emb = AudioEmbedder2D()
E = emb.embed(clips, sample_rate=sr, length=2.0)
print("embeddings:", E.shape)

# leave-one-out 1-nearest-neighbour on standardised embeddings
Z = (E - E.mean(0)) / (E.std(0) + 1e-8)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
S = Z @ Z.T
np.fill_diagonal(S, -np.inf)
print("1-NN leave-one-out accuracy:", (labels[S.argmax(1)] == labels).mean())

# 2) The 2-D mixer is an ordinary nn.Module on (batch, time, frequency)
X = torch.randn(2, 500, 128)
mixer = CrossScaleMixer2D(order=4, freq_order=2, window=2, tau=0.1)
W, plan = mixer.coefficients(X)                    # (2, S, R, nt, nf) time x frequency scale space
print("2-D scale-space coefficients:", tuple(W.shape))
print("tight-frame reconstruction error:", (mixer.synthesize(W, plan) - X).abs().max().item())
Y = mixer(X)                                       # cross-scale attention over time and frequency
print("cross-scale mixer output:", tuple(Y.shape), "parameters:", sum(p.numel() for p in mixer.parameters()))
