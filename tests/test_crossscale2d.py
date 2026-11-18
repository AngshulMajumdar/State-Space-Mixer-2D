import math

import numpy as np
import pytest
import torch

from ssmix2d import CrossScaleMixer2D

dev = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def brute_force(W, m, tokens):
    """Direct evaluation of the 2-D cross-scale attention for a few tokens."""
    B, S, R, nt, nf = W.shape
    L = m.links(S, R)
    Wn = W.cpu().numpy()
    out = []
    for b, s, r, t, g in tokens:
        q = Wn[b, s, r, t, m.group * g:m.group * (g + 1)]
        q = q / max(np.linalg.norm(q), 1e-12)
        vs, sc = [], []
        for s2, r2, dt, df in L[(s, r)]:
            v = Wn[b, s2, r2, (t + dt) % nt, (np.arange(m.group * g, m.group * (g + 1)) + df) % nf]
            vs.append(v)
            sc.append(q @ v / max(np.linalg.norm(v), 1e-12) / m.tau)
        a = np.exp(np.array(sc) - max(sc))
        a /= a.sum()
        out.append((a[:, None] * np.array(vs)).sum(0))
    return np.array(out)


@pytest.mark.parametrize("device", dev)
@pytest.mark.parametrize("T,F", [(231, 128), (97, 64), (500, 40)])
@pytest.mark.parametrize("order,freq_order", [(4, 2), (1, 1), (8, 3)])
def test_perfect_reconstruction(device, T, F, order, freq_order):
    X = torch.randn(2, T, F, dtype=torch.float64, device=device)
    m = CrossScaleMixer2D(order=order, freq_order=freq_order)
    W, plan = m.coefficients(X)
    assert (m.synthesize(W, plan) - X).abs().max().item() < 1e-10


def test_parseval_on_padded_grid():
    X = torch.randn(2, 231, 128, dtype=torch.float64)
    m = CrossScaleMixer2D()
    W, plan = m.coefficients(X)
    xp = m.ft._pad(X.transpose(1, 2), plan["pt"])
    xpp = m.ff._pad(xp.transpose(1, 2), plan["pf"])
    assert abs((W ** 2).sum().item() / (xpp ** 2).sum().item() - 1) < 1e-10


def test_band_count_and_links():
    m = CrossScaleMixer2D(levels=3, freq_levels=2, window=2)
    W, plan = m.coefficients(torch.randn(1, 200, 64))
    assert W.shape[1:3] == (4, 3)
    L = m.links(4, 3)
    assert all(ks[0][:2] == key and ks[0][2:] == (0, 0) for key, ks in L.items())
    assert len(L[(1, 1)]) == 1 + 14 + 14  # self, 3 time scales x 5 offsets - 1, 3 freq scales x 5 offsets - 1
    assert all(len(set(ks)) == len(ks) for ks in L.values())


def test_matches_brute_force():
    X = torch.randn(2, 231, 128, dtype=torch.float64)
    m = CrossScaleMixer2D()
    W, _ = m.coefficients(X)
    Wm = m.mix(W)
    rng = np.random.default_rng(0)
    B, S, R, nt, nf = W.shape
    tokens = [(rng.integers(B), rng.integers(S), rng.integers(R), rng.integers(nt), rng.integers(nf // m.group))
              for _ in range(200)]
    ref = brute_force(W, m, tokens)
    got = np.array([Wm[b, s, r, t, m.group * g:m.group * (g + 1)].numpy() for b, s, r, t, g in tokens])
    assert np.abs(ref - got).max() < 1e-9


def test_zero_temperature_is_identity():
    X = torch.randn(1, 200, 64, dtype=torch.float64)
    errs = [(CrossScaleMixer2D(tau=t)(X) - X).abs().max().item() for t in (1e-2, 1e-4, 1e-6)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3


def test_parameter_free_and_homogeneous():
    m = CrossScaleMixer2D()
    assert sum(p.numel() for p in m.parameters()) == 0
    X = torch.randn(2, 150, 64, dtype=torch.float64)
    assert torch.allclose(m(3.0 * X), 3.0 * m(X), atol=1e-10)


def test_shift_equivariance_circular():
    m = CrossScaleMixer2D(pad="circular")
    X = torch.randn(1, 256, 128, dtype=torch.float64)
    assert (torch.roll(m(X), 17, 1) - m(torch.roll(X, 17, 1))).abs().max() < 1e-10
    assert (torch.roll(m(X), 16, 2) - m(torch.roll(X, 16, 2))).abs().max() < 1e-10


def test_attention_is_a_distribution():
    m = CrossScaleMixer2D()
    W, _ = m.coefficients(torch.randn(2, 120, 64))
    m.mix(W)
    for a in m.last_attention.values():
        assert torch.all(a >= 0)
        assert torch.allclose(a.sum(1), torch.ones_like(a.sum(1)), atol=1e-5)


def test_att_crop_matches_full_maps():
    m = CrossScaleMixer2D()
    W, plan = m.coefficients(torch.randn(1, 120, 64))
    m.mix(W)
    full = {k: v.clone() for k, v in m.last_attention.items()}
    m.att_crop = (10, 50, 2, 6)
    m.mix(W)
    m.att_crop = None
    for k, v in m.last_attention.items():
        assert torch.allclose(v, full[k][:, :, 10:50, 2:6].mean(2), atol=1e-6)


def test_gradients_flow():
    X = torch.randn(1, 100, 64, requires_grad=True)
    CrossScaleMixer2D()(X).square().sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all() and X.grad.abs().sum() > 0


def test_float32_and_shape():
    X = torch.randn(3, 333, 96)
    Y = CrossScaleMixer2D()(X)
    assert Y.shape == X.shape and Y.dtype == torch.float32 and torch.isfinite(Y).all()


@pytest.mark.parametrize("bad", [dict(group=3), dict(window=-1), dict(tau=0.0)])
def test_invalid_arguments(bad):
    with pytest.raises(ValueError):
        CrossScaleMixer2D(**bad)


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        CrossScaleMixer2D()(torch.randn(10, 64))
