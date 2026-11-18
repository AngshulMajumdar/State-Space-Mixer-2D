"""Load ESC-50 and FSDD once; store float32 waveforms at 16 kHz and 32 kHz.

ESC-50: 2000 x 5 s, official folds 1-5.
FSDD:   3000 clips; every clip zero-padded (appended) to 2.3 s so all models see
        identical fixed-length input; official split: take index 0-4 -> test.
"""
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# holds ESC-50/ and fsdd/ (git clones of karolpiczak/ESC-50, Jakobovski/free-spoken-digit-dataset)
D = os.environ.get("SSMIX_DATA", os.path.join(ROOT, "data"))
OUT = os.path.join(ROOT, "benchmarks", "cache")
os.makedirs(OUT, exist_ok=True)


def load(path, srs=(16000, 32000), fixed_sec=None):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    out = []
    for s in srs:
        y = AF.resample(x, sr, s) if sr != s else x
        if fixed_sec is not None:
            n = int(round(fixed_sec * s))
            y = torch.nn.functional.pad(y, (0, max(0, n - len(y))))[:n]
        out.append(y.numpy())
    return out


def esc50():
    meta = pd.read_csv(f"{D}/ESC-50/meta/esc50.csv")
    a16, a32 = zip(*[load(f"{D}/ESC-50/audio/{f}", fixed_sec=5.0) for f in meta.filename])
    np.savez(f"{OUT}/esc50_wav.npz", w16=np.stack(a16), w32=np.stack(a32),
             y=meta.target.values, fold=meta.fold.values)


def fsdd():
    files = sorted(glob.glob(f"{D}/fsdd/recordings/*.wav"))
    y, spk, idx = [], [], []
    a16, a32 = [], []
    for f in files:
        d, s, i = os.path.basename(f)[:-4].split("_")
        y.append(int(d)); spk.append(s); idx.append(int(i))
        w16, w32 = load(f, fixed_sec=2.3)
        a16.append(w16); a32.append(w32)
    np.savez(f"{OUT}/fsdd_wav.npz", w16=np.stack(a16), w32=np.stack(a32), y=np.array(y),
             speaker=np.array(spk), take=np.array(idx))


if __name__ == "__main__":
    esc50()
    fsdd()
    for n in ("esc50", "fsdd"):
        z = np.load(f"{OUT}/{n}_wav.npz")
        print(n, z["w16"].shape, z["w32"].shape, np.bincount(z["y"]).min(), np.bincount(z["y"]).max())
