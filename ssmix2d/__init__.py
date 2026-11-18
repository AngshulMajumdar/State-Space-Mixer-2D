"""ssmix2d: training-free 2-D (time x frequency) scale-space transformers on Daubechies wavelets."""
from .crossscale2d import CrossScaleMixer2D
from .daubechies import daubechies, daubechies_lowpass, qmf_highpass
from .embedder import AudioEmbedder2D
from .frame import ScaleSpaceMixer

__version__ = "0.1.0"

__all__ = [
    "AudioEmbedder2D",
    "CrossScaleMixer2D",
    "ScaleSpaceMixer",
    "daubechies",
    "daubechies_lowpass",
    "qmf_highpass",
]
