"""Identical frozen-embedding linear probe (the same protocol as State-Space-Mixer).

StandardScaler + multinomial logistic regression; C chosen by 3-fold CV on the training portion only,
from {1e-3, 1e-2, 1e-1, 1}. Runs single-process in float32 so 60k-dimensional embeddings fit in memory.
ESC-50: official 5-fold cross-validation.
FSDD:   official split (take 0-4 is the test set) and leave-one-speaker-out over the 6 speakers.

    python benchmarks/probe.py <esc50|fsdd> [model]       (model defaults to ssmix2d)
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "cache")


def fit_score(Xtr, ytr, Xte, yte):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    gs = GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1]},
                      cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=1)
    gs.fit(Xtr, ytr)
    return gs.score(Xte, yte)


def load(ds, name):
    return np.load(f"{C}/{ds}_wav.npz"), np.load(f"{C}/emb_{ds}_{name}.npy").astype(np.float32)


def esc50(name="ssmix2d"):
    z, X = load("esc50", name)
    return [100 * fit_score(X[z["fold"] != f], z["y"][z["fold"] != f], X[z["fold"] == f], z["y"][z["fold"] == f])
            for f in range(1, 6)]


def fsdd(name="ssmix2d"):
    z, X = load("fsdd", name)
    te = z["take"] < 5
    official = 100 * fit_score(X[~te], z["y"][~te], X[te], z["y"][te])
    speakers = {str(s): 100 * fit_score(X[z["speaker"] != s], z["y"][z["speaker"] != s],
                                        X[z["speaker"] == s], z["y"][z["speaker"] == s])
                for s in np.unique(z["speaker"])}
    return official, speakers


if __name__ == "__main__":
    ds, name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "ssmix2d")
    if ds == "esc50":
        a = esc50(name)
        print(f"esc50 {name}: {np.mean(a):.1f} +/- {np.std(a):.1f}  folds {[round(v, 1) for v in a]}")
    else:
        o, s = fsdd(name)
        print(f"fsdd {name}: official {o:.1f}  speaker-held-out {np.mean(list(s.values())):.1f}  {s}")
