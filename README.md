# ssmix2d

**Training-free time–frequency transformers on Daubechies wavelets.**

ssmix2d mixes audio jointly across **time and frequency** with exact multiscale operators:

- A separable 2-D Daubechies tight frame over (time, mel): every time scale × every frequency scale, computed with
  batched FFTs and reconstructed exactly.
- A 2-D cross-scale attention mixer: each coefficient attends to its siblings, parents and children along **both**
  axes.
- A pooled scale-space readout that includes the attention maps.

Nothing is trained, and there are no weights to download.

- Documentation and article: **https://angshulmajumdar.github.io/State-Space-Mixer-2D/**
- Live demo (Hugging Face Space): **https://huggingface.co/spaces/Angshul/ssmix2d-audio-embedder**
- 1-D predecessor: [State-Space-Mixer](https://github.com/AngshulMajumdar/State-Space-Mixer)

| | |
|---|---|
| ESC-50, linear probe on frozen embeddings | **76.0 % with no training**: ahead of the 1-D ssmix (74.6), all eight hand-crafted sets combined (73.9), openSMILE ComParE (71.2), VGGish (71.0) and wav2vec 2.0 (68.4) |
| FSDD spoken digits, official split | **99.7 %**: ahead of the 1-D ssmix (99.0), EfficientAT (98.7), PaSST (98.0), wav2vec 2.0 (97.0) and PANNs (85.7) |
| FSDD, unseen speakers (leave one speaker out) | **71.9 %**, +3.8 over the 1-D ssmix (68.1) |
| Reconstruction error of the 2-D frame | below 10⁻¹⁴ (float64) |
| Learnable parameters | 0 |

## Install

```bash
pip install git+https://github.com/AngshulMajumdar/State-Space-Mixer-2D
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, torchaudio, NumPy and mpmath.

## Audio embeddings

```python
from ssmix2d import AudioEmbedder2D

emb = AudioEmbedder2D()                                        # parameter-free; the benchmarked configuration
E = emb.embed(waves, sample_rate=44100, length=5.0)            # list of 1-D arrays -> (N, D) numpy
```

Inputs are resampled to 32 kHz. With `length`, every clip is padded or cut to the same duration, so all
embeddings share one dimension (59,392 for 5 s). Use them with any linear model, k-NN or clustering.
`AudioEmbedder2D.from_pretrained` / `save_pretrained` store the configuration (there are no weights).

## The 2-D mixer as a PyTorch layer

```python
import torch
from ssmix2d import CrossScaleMixer2D

X = torch.randn(8, 500, 128)                                   # (batch, time, frequency)
mixer = CrossScaleMixer2D(order=4, freq_order=2, window=2, tau=0.1)
Y = mixer(X)                                                   # same shape, cross-scale attention on both axes

W, plan = mixer.coefficients(X)                                # (8, S, R, nt, nf) time x frequency scale space
X2 = mixer.synthesize(W, plan)                                 # exact inverse
```

The time support is set by `order` and the frequency support by `freq_order` (N vanishing moments, support 2N−1).
Any 2-D field works as input; the first axis is treated as time and the second as frequency.

## Public API

| Object | Role |
|---|---|
| `AudioEmbedder2D` | Training-free audio transformer: log-mel → 2-D cross-scale mixer blocks → pooled scale-space readout; `embed`, `from_pretrained`, `save_pretrained` |
| `CrossScaleMixer2D` | Cross-scale attention on the separable 2-D Daubechies tight frame; `coefficients`, `mix`, `synthesize`, `links` |
| `ScaleSpaceMixer` | FFT-batched 1-D Daubechies tight frame for any length (the building block of each axis) |
| `daubechies`, `daubechies_lowpass`, `qmf_highpass` | Daubechies filters of any order (spectral factorisation, mpmath roots) |

## Repository layout

```
ssmix2d/       package
tests/         pytest suite (exactness, Parseval, brute-force equality, equivariance, API)
benchmarks/    prep.py, embed.py, probe.py, report.py, speed.py, results.json
examples/      quickstart.py
space/         Gradio app for the Hugging Face Space (app.py, requirements.txt)
scripts/       build_docs.py (regenerates docs/), build_space.py (assembles the Space)
docs/          GitHub Pages site with the article (Settings -> Pages -> Deploy from branch, folder /docs)
```

## Tests

```bash
pip install pytest PyWavelets
pytest tests
```

## Reproducing the benchmarks

```bash
git clone https://github.com/karolpiczak/ESC-50 data/ESC-50
git clone https://github.com/Jakobovski/free-spoken-digit-dataset data/fsdd
pip install scikit-learn pandas soundfile
python benchmarks/prep.py
python benchmarks/embed.py esc50
python benchmarks/embed.py fsdd
python benchmarks/report.py
```

Baseline rows come from the [State-Space-Mixer](https://github.com/AngshulMajumdar/State-Space-Mixer) benchmark
suite, which uses the same data preparation and the same probe.
