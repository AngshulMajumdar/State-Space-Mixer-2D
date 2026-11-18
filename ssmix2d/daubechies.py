"""Daubechies extremal-phase filters of arbitrary order, built from scratch.

order N  ->  N vanishing moments, 2N taps, support [0, 2N-1].

Construction (Daubechies 1988): |H(w)|^2 = 2 cos^{2N}(w/2) P(sin^2(w/2)),
P(y) = sum_{k<N} C(N-1+k, k) y^k.  Each root y_k of P maps to a pair
z, 1/z with z + 1/z = 2 - 4 y_k; keeping |z| < 1 gives the minimum-phase
(extremal-phase) factor, identical to PyWavelets' `dbN.rec_lo`.
Roots are found with mpmath at high precision, so large N stays accurate.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np

try:
    import mpmath as _mp
except ImportError:  # pragma: no cover
    _mp = None


@lru_cache(maxsize=None)
def _lowpass_tuple(order: int) -> tuple:
    N = int(order)
    if N < 1:
        raise ValueError("order must be >= 1")
    if N == 1:
        return (2 ** -0.5, 2 ** -0.5)
    coeffs_desc = [comb(N - 1 + k, k) for k in range(N)][::-1]
    if _mp is not None:
        with _mp.workdps(max(50, 4 * N)):
            ys = _mp.polyroots(coeffs_desc, maxsteps=500, extraprec=8 * N)
            zs = []
            for y in ys:
                b = 2 - 4 * y
                d = _mp.sqrt(b * b - 4)
                z = (b + d) / 2
                if abs(z) >= 1:
                    z = (b - d) / 2
                zs.append(z)
            # h(x) = (1 + x)^N * prod(1 - z_k x),  x = z^{-1}
            poly = [_mp.mpc(1)]
            for _ in range(N):
                poly = [a + b for a, b in zip(poly + [0], [0] + poly)]
            for z in zs:
                poly = [a - z * b for a, b in zip(poly + [0], [0] + poly)]
            s = sum(poly)
            h = [float(_mp.re(c * _mp.sqrt(2) / s)) for c in poly]
        return tuple(h)
    # numpy fallback (accurate to ~N=20)
    ys = np.roots(np.array(coeffs_desc, dtype=np.float64))
    b = 2 - 4 * ys.astype(np.complex128)
    d = np.sqrt(b * b - 4)
    z = np.where(np.abs((b + d) / 2) < 1, (b + d) / 2, (b - d) / 2)
    poly = np.array([1.0 + 0j])
    for _ in range(N):
        poly = np.convolve(poly, [1.0, 1.0])
    for zk in z:
        poly = np.convolve(poly, [1.0, -zk])
    h = np.real(poly * np.sqrt(2) / poly.sum())
    return tuple(h.tolist())


def daubechies_lowpass(order: int) -> np.ndarray:
    """Orthonormal lowpass h (sum h = sqrt 2, sum h^2 = 1), length 2*order."""
    return np.array(_lowpass_tuple(order), dtype=np.float64)


def qmf_highpass(h: np.ndarray) -> np.ndarray:
    """g[n] = (-1)^n h[L-1-n]  (PyWavelets rec_hi convention)."""
    L = len(h)
    n = np.arange(L)
    return ((-1.0) ** n) * h[::-1]


def daubechies(order: int):
    """(h, g): orthonormal Daubechies lowpass and highpass filters of the given order (2*order taps).
    Identical to PyWavelets dbN rec_lo / rec_hi."""
    h = daubechies_lowpass(order)
    return h, qmf_highpass(h)
