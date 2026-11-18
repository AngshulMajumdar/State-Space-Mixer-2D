"""Throughput of the 2-D frame and cross-scale mixer vs. parameter-free softmax self-attention over the same
tokens (all time x mel-group positions, 8 bins each).

    python benchmarks/speed.py [--device cuda]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmix2d import CrossScaleMixer2D  # noqa: E402


def timeit(fn, x, reps=3):
    """Best-of-reps wall time in ms (the first call warms caches and FFT plans)."""
    fn(x)
    best = float("inf")
    for _ in range(reps):
        if x.is_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def round_trip(m):
    def f(x):
        W, plan = m.coefficients(x)
        return m.synthesize(W, plan)
    return f


def softmax_attention(x, chunk=4096):
    """Q = K = V = tokens of 8 mel bins, no parameters, O(N^2) in the number of tokens (queries chunked)."""
    B, T, F = x.shape
    z = x.reshape(B, T * F // 8, 8)
    out = torch.empty_like(z)
    for i in range(0, z.shape[1], chunk):
        out[:, i:i + chunk] = torch.softmax(z[:, i:i + chunk] @ z.transpose(1, 2) / 8 ** 0.5, -1) @ z
    return out.reshape(B, T, F)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    print(f"device={dev}  threads={torch.get_num_threads()}  B=1  128 mel bins  fp32\n")
    print(f"{'frames':>7} {'2-D frame':>10} {'2-D mixer':>10} {'softmax attn':>13}   (ms)")
    for T in (1024, 2048, 4096):
        x = torch.randn(1, T, 128, device=dev)
        m = CrossScaleMixer2D(levels=6)
        print(f"{T:>7d} {timeit(round_trip(m), x):>10.1f} {timeit(m, x, 2):>10.1f} {timeit(softmax_attention, x, 1):>13.1f}")
    x = torch.randn(1, 1024, 128, device=dev)
    print("\n1024 frames: frame cost vs Daubechies order along time")
    for N in (1, 4, 16):
        print(f"  db{N:<3d} support={2 * N - 1:<4d} {timeit(round_trip(CrossScaleMixer2D(order=N, levels=6)), x):7.1f} ms")


if __name__ == "__main__":
    main()
