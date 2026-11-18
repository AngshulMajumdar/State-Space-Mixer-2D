"""Frozen ssmix2d embeddings on a prepared dataset (run prep.py first).

    python benchmarks/embed.py <esc50|fsdd>

Writes benchmarks/cache/emb_<dataset>_ssmix2d.npy. The embedding matrix is written row by row to a
memory-mapped file, so the whole dataset never has to sit in RAM.
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ssmix2d import AudioEmbedder2D  # noqa: E402

C = os.path.join(ROOT, "benchmarks", "cache")


def main(ds, batch_size=4):
    z = np.load(f"{C}/{ds}_wav.npz")
    W = z["w32"]
    emb = AudioEmbedder2D()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb.to(dev)
    D = emb.dim(W.shape[1])
    E = np.lib.format.open_memmap(f"{C}/emb_{ds}_ssmix2d.npy", mode="w+", dtype=np.float32, shape=(len(W), D))
    t0 = time.time()
    for i in range(0, len(W), batch_size):
        E[i:i + batch_size] = emb(torch.from_numpy(W[i:i + batch_size]).to(dev)).cpu().numpy()
    E.flush()
    print(f"{ds}: {E.shape} in {time.time() - t0:.0f}s on {dev}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
