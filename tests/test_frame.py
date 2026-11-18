import math

import numpy as np
import pytest
import torch

from ssmix2d import ScaleSpaceMixer, daubechies
from ssmix2d.frame import atrous_reference

DEV = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ------------------------------------------------------------------ filters
@pytest.mark.parametrize("N", list(range(1, 39)))
def test_filters_match_pywavelets(N):
    pywt = pytest.importorskip("pywt")
    h, g = daubechies(N)
    w = pywt.Wavelet(f"db{N}")
    assert np.abs(h - w.rec_lo).max() < 1e-12
    assert np.abs(g - w.rec_hi).max() < 1e-12


@pytest.mark.parametrize("N", [1, 2, 5, 12, 25, 45, 60])
def test_filters_orthonormal_and_vanishing_moments(N):
    h, g = daubechies(N)
    L = len(h)
    assert L == 2 * N
    for k in range(N):
        assert abs(np.dot(h[: L - 2 * k], h[2 * k:]) - (k == 0)) < 1e-13
        assert abs(np.dot(h[: L - 2 * k], g[2 * k:])) < 1e-13
    n = np.arange(L) / (L - 1)
    for p in range(N):
        assert abs(np.sum(g * n ** p)) < 1e-12


def test_support_argument():
    assert ScaleSpaceMixer(support=7).order == 4
    with pytest.raises(ValueError):
        ScaleSpaceMixer(support=6)


# ------------------------------------------------------------- frame / PR
@pytest.mark.parametrize("N", [1, 3, 8])
@pytest.mark.parametrize("J", [1, 4, 7])
def test_tight_frame(N, J):
    m = ScaleSpaceMixer(order=N, levels=J, pad="circular")
    plan = m._plan(1000, torch.device("cpu"), torch.float64)
    s = (plan["psi"].abs() ** 2).sum(0)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-12)


@pytest.mark.parametrize("dev", DEV)
@pytest.mark.parametrize("pad", ["reflect", "circular"])
@pytest.mark.parametrize("T", [17, 97, 128, 1000, 4099])
@pytest.mark.parametrize("N", [1, 2, 6, 20])
def test_perfect_reconstruction(dev, pad, T, N):
    x = torch.randn(2, T, 5, dtype=torch.float64, device=dev)
    m = ScaleSpaceMixer(order=N, rule="identity", pad=pad)
    assert (m(x) - x).abs().max() < 1e-12


@pytest.mark.parametrize("N", [1, 2, 4, 9])
@pytest.mark.parametrize("T", [64, 101, 777])
def test_fft_equals_atrous_reference_and_parseval(N, T):
    J = 5
    x = torch.randn(3, T, 4, dtype=torch.float64)
    m = ScaleSpaceMixer(order=N, levels=J, rule="identity", pad="circular")
    W = m.analysis(x)
    R = atrous_reference(x, N, J, m._plan(T, x.device, x.dtype)["delays"])
    assert (W - R).abs().max() < 1e-12
    assert abs(W.pow(2).sum() / x.pow(2).sum() - 1) < 1e-12
    assert (m.synthesis(W) - x).abs().max() < 1e-12


def test_float32_reconstruction():
    x = torch.randn(2, 3000, 16)
    assert (ScaleSpaceMixer(order=8, rule="identity")(x) - x).abs().max() < 1e-5


# ---------------------------------------------------------- attention rule
def test_attention_is_parameter_free_and_homogeneous():
    m = ScaleSpaceMixer(order=4, heads=2)
    assert sum(p.numel() for p in m.parameters()) == 0
    x = torch.randn(2, 2048, 8, dtype=torch.float64)
    assert torch.allclose(m(3.7 * x), 3.7 * m(x), atol=1e-10)


def test_attention_tends_to_identity_with_temperature():
    torch.manual_seed(0)
    x = torch.randn(1, 4000, 4, dtype=torch.float64)
    errs = [((ScaleSpaceMixer(order=4, tau=t)(x) - x).norm() / x.norm()).item() for t in (1, 10, 100, 1e4)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-3


def test_attention_gates_preserve_local_energy():
    x = torch.randn(2, 3000, 6, dtype=torch.float64)
    m = ScaleSpaceMixer(order=3, heads=3, pad="circular")
    plan = m._plan(3000, x.device, x.dtype)
    Wp = m._analysis_padded(x.transpose(1, 2), plan)
    g = m._gate_attention(Wp, plan, 3000)
    B, S, C, n = Wp.shape
    E = Wp.view(B, S, 3, 2, n).pow(2).mean(3)
    E = torch.fft.irfft(torch.fft.rfft(E, dim=-1) * plan["gauss"][None, :, None, :], n=n, dim=-1).clamp_min(m.eps)
    gh = g.view(B, S, 3, 2, n)[:, :, :, 0]
    assert torch.allclose((gh * gh * E).sum(1), E.sum(1), rtol=1e-10)


def test_attention_selects_dominant_scale():
    """A pure tone in one octave should receive the largest gate in that band."""
    T = 8192
    t = torch.arange(T, dtype=torch.float64)
    m = ScaleSpaceMixer(order=8, levels=8, tau=0.5)
    for j in range(1, 8):  # centre of detail band j: 3*pi / 2^(j+1)
        w = 3 * math.pi / 2 ** (j + 1)
        x = (torch.sin(w * t) + 0.01 * torch.randn(T, dtype=torch.float64))[None, :, None]
        g = m.gates(x)[0, :, 1000:-1000, 0].mean(-1)
        assert int(g.argmax()) == j - 1


# ------------------------------------------------------------- shrink rule
def test_shrink_denoises():
    torch.manual_seed(0)
    T = 8192
    t = torch.arange(T, dtype=torch.float64)
    clean = torch.sin(2 * math.pi * t / 400) + ((t > 4000) & (t < 4100)) * torch.sin(2 * math.pi * t / 9)
    x = (clean + 0.5 * torch.randn(T, dtype=torch.float64))[None, :, None]
    snr = lambda y: 10 * math.log10(clean.pow(2).sum() / (y.flatten() - clean).pow(2).sum())
    assert snr(ScaleSpaceMixer(order=8, rule="shrink")(x)) > snr(x) + 6


# ----------------------------------------------------------------- misc
def test_gradients_flow():
    x = torch.randn(2, 512, 8, requires_grad=True)
    ScaleSpaceMixer(order=3, heads=4)(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_order_does_not_change_shapes_or_cost_path():
    x = torch.randn(2, 1024, 8)
    for N in (1, 4, 16, 40):
        assert ScaleSpaceMixer(order=N, levels=6)(x).shape == x.shape
