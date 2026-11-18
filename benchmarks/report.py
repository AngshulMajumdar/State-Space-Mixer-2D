"""Probe the ssmix2d embeddings and merge them into benchmarks/results.json (read by scripts/build_docs.py and
scripts/build_space.py).

    python benchmarks/report.py

Baseline rows (the 1-D ssmix embedder, hand-crafted sets, VGGish, wav2vec 2.0, PaSST, EfficientAT, PANNs, ...)
come from the State-Space-Mixer benchmark suite, which uses the same caches (prep.py) and the same probe.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe import esc50, fsdd  # noqa: E402

OUT = os.path.join(HERE, "results.json")


def r1(x):
    return round(float(x), 1)


if __name__ == "__main__":
    with open(OUT) as f:
        R = json.load(f)
    a = esc50()
    R["esc50"]["ssmix2d"] = {"folds": [r1(v) for v in a], "mean": r1(np.mean(a)), "std": r1(np.std(a))}
    o, s = fsdd()
    R["fsdd"]["ssmix2d"] = r1(o)
    R["fsdd_speakers"]["ssmix2d"] = {"per_speaker": {k: r1(v) for k, v in s.items()}, "mean": r1(np.mean(list(s.values())))}
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({"esc50": R["esc50"]["ssmix2d"], "fsdd": R["fsdd"]["ssmix2d"],
                      "fsdd_speakers": R["fsdd_speakers"]["ssmix2d"]["mean"]}, indent=1))
