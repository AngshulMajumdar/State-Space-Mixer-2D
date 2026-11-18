"""Assemble the Hugging Face Space (Gradio) for the ssmix2d audio embedder.

    python scripts/build_space.py <output_dir>

Writes README.md (Space metadata + description, numbers from benchmarks/results.json), app.py,
requirements.txt and a copy of the ssmix2d package, so the Space runs without installing from GitHub.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ssmix2d import AudioEmbedder2D, __version__  # noqa: E402

REPO = "https://github.com/AngshulMajumdar/State-Space-Mixer-2D"
DOCS = "https://angshulmajumdar.github.io/State-Space-Mixer-2D/"
GRADIO = "6.32.0"


def main(out):
    with open(os.path.join(ROOT, "benchmarks", "results.json")) as f:
        R = json.load(f)
    E, Fd, Sp = R["esc50"], R["fsdd"], R["fsdd_speakers"]
    dim = AudioEmbedder2D().dim(5 * 32000)

    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "space", "app.py"), out)
    shutil.copy(os.path.join(ROOT, "space", "requirements.txt"), out)
    pkg = os.path.join(out, "ssmix2d")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    shutil.copytree(os.path.join(ROOT, "ssmix2d"), pkg, ignore=shutil.ignore_patterns("__pycache__", ".*"))

    readme = f"""---
title: ssmix2d Audio Embedder
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {GRADIO}
app_file: app.py
pinned: false
short_description: Training-free time-frequency audio transformer on wavelets
tags:
- audio
- audio-embeddings
- wavelets
- training-free
---

# ssmix2d audio embedder

A **training-free audio transformer that mixes across time and frequency**. Its token mixer is cross-scale
attention on an exact 2-D Daubechies wavelet frame over (time, mel). It has **zero learned parameters** and needs
**no training data**.

- **Compare two clips**: upload or record two clips and get the cosine similarity of their {dim:,}-dimensional
  ssmix2d embeddings, plus the embeddings as a `.npy` file.
- **Inspect the scale space**: see a clip's log-mel spectrogram, the energy of every time × frequency wavelet band,
  how strongly each coefficient attends to its neighbours, and where the attention goes (finer, sibling or coarser
  scales, along time or along frequency).

Code and API: [{REPO.replace("https://", "")}]({REPO}) · Article: [{DOCS}article.html]({DOCS}article.html)

## How it works

1. A 128-band log-mel spectrogram (25 ms windows, 10 ms hop), normalised per clip.
2. A separable **2-D Daubechies tight frame**: db4 along time, db2 along mel, every time scale × every frequency
   scale. It is Parseval-tight, so reconstruction is exact.
3. **2-D cross-scale attention**. Every coefficient attends to its dyadic siblings, parents and children along
   time and along frequency, using parameter-free cosine weights. There are two blocks.
4. Readout: band magnitudes before and after mixing, the attention maps, the state statistics and the spectral
   envelope.

## Results (frozen embeddings, identical linear probe)

**ESC-50**, environmental sound, 5-fold: ssmix2d **{E['ssmix2d']['mean']}%**, ahead of the 1-D ssmix ({E['ssmix']['mean']}%),
all eight hand-crafted feature sets combined ({E['handcrafted_all']['mean']}%), openSMILE ComParE ({E['compare']['mean']}%),
VGGish ({E['vggish']['mean']}%) and wav2vec 2.0 ({E['wav2vec2']['mean']}%).

**FSDD**, spoken digits, official split: ssmix2d **{Fd['ssmix2d']}%**, ahead of the 1-D ssmix ({Fd['ssmix']}%),
EfficientAT ({Fd['efficientat_mn10']}%), PaSST ({Fd['passt']}%), wav2vec 2.0 ({Fd['wav2vec2']}%) and PANNs ({Fd['panns']}%).

**Unseen speakers** (FSDD, leave one speaker out): **{Sp['ssmix2d']['mean']}%**, against {Sp['ssmix']['mean']}% for the 1-D ssmix.

## Use it in Python

```bash
pip install git+{REPO}
```

```python
from ssmix2d import AudioEmbedder2D
E = AudioEmbedder2D().embed(waves, sample_rate=44100, length=5.0)   # (N, {dim}) numpy
```

ssmix2d {__version__}.
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf8") as f:
        f.write(readme)
    print("space assembled in", out)


if __name__ == "__main__":
    main(sys.argv[1])
