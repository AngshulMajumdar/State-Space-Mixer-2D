"""GPU-efficient, training-free scale-space mixer on Daubechies wavelets.

Pipeline on a token sequence X in R^{B x T x C}:

  1. Analysis   W_s = F^{-1}[ Psi_s . F X ],   s = 0..J   (J details + 1 approx)
  2. Gate       g_s = rule(W)                  (parameter-free, data-dependent)
  3. Synthesis  Y   = sum_s F^{-1}[ conj(Psi_s) . F(g_s W_s) ]

Psi_s are the transfer functions of the undecimated (a trous / stationary)
Daubechies transform, evaluated exactly on the DFT grid:

  D_j(w) = G~(2^{j-1} w) prod_{k<j-1} H~(2^k w),   A_J(w) = prod_{k<J} H~(2^k w),
  H~ = H / sqrt 2,  G~ = G / sqrt 2,  |H~|^2 + |G~|^2 = 1
  =>  sum_s |Psi_s(w)|^2 = 1   (Parseval tight frame, exact PR for any T).

All J+1 bands come out of one forward rFFT and one batched inverse, so cost
is O(B C T log T (J+1)) regardless of the filter order: support is free.
Each band gets an integer delay compensation (a pure phase), so band
coefficients at index t describe the same instant across scales; tightness
and PR are unaffected.

Gate rules
----------
identity   g = 1                              -> Y == X exactly.
attention  scale attention: per head h and time t, the scale tokens are
           scored by white-noise-normalised local energy
               score_s = log(E_s(t) / n_s),  E_s = Gauss_{sigma_s} * mean_head(W_s^2),
           n_s = |Psi_s|^2 averaged over frequency (energy of band s for white
           input), sigma_s ~ kappa * 2^{level}.  Then
               a = softmax_s(score / tau),  g = (J+1) a,
           rescaled so sum_s g_s^2 E_s = sum_s E_s (local energy preserved).
           White noise -> g ~ 1 (identity); tau -> inf -> identity;
           tau -> 0 -> winner-take-all scale selection (Lindeberg-style).
           Gates are scale-invariant, so the mixer is 1-homogeneous.
shrink     translation-invariant soft shrinkage on detail bands
           (== cycle-spinning VisuShrink): g = (1 - lam sigma_s / |W|)_+,
           sigma_1 = MAD(finest band)/0.6745 per channel, sigma_s = sigma_1
           sqrt(n_s / n_1), lam = sqrt(2 log T).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .daubechies import daubechies


def _fast_len(n: int) -> int:
    """Smallest 5-smooth integer >= n that is a multiple of 16 (for n > 16):
    cuFFT / pocketfft real transforms are fastest on such sizes."""
    if n <= 16:
        return n
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = p35 * 16
            while m < n:
                m *= 2
            best = min(best, m)
            p35 *= 3
        p5 *= 5
    return best


def _freq_response(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = np.arange(len(f))
    return np.exp(-1j * np.outer(w, n)) @ f


class ScaleSpaceMixer(nn.Module):
    """Undecimated Daubechies tight frame (FFT-batched, any T) with parameter-free gate rules.

    Args:
        order:     Daubechies order N (N vanishing moments, support 2N-1).
        support:   alternative to `order`; must be odd (= 2N-1).
        levels:    number of detail levels J (None -> as many as fit T, max 10).
        heads:     channel groups sharing one scale-attention pattern.
        rule:      'attention' | 'shrink' | 'identity'.
        tau:       attention temperature.
        kappa:     energy-smoothing width, in units of each band's dyadic scale.
        pad:       'reflect' (no wrap-around artefacts) or 'circular'.
        delay_compensation: align bands in time (pure phase; keeps PR).
        fast_fft:  extend padded length to a 5-smooth FFT size.
        pad_tol:   reflect padding covers all but this fraction of every band's
                   impulse-response energy (PR is exact for any value).
    Input/Output: (B, T, C).
    """

    def __init__(
        self,
        order: Optional[int] = 4,
        support: Optional[int] = None,
        levels: Optional[int] = None,
        heads: int = 1,
        rule: str = "attention",
        tau: float = 1.0,
        kappa: float = 1.0,
        pad: str = "reflect",
        delay_compensation: bool = True,
        fast_fft: bool = True,
        pad_tol: float = 1e-8,
        eps: float = 1e-12,
    ):
        super().__init__()
        if support is not None:
            if support < 1 or support % 2 == 0:
                raise ValueError("Daubechies support is 2N-1: must be odd and >= 1")
            order = (support + 1) // 2
        if rule not in ("attention", "shrink", "identity"):
            raise ValueError(f"unknown rule {rule!r}")
        if pad not in ("reflect", "circular"):
            raise ValueError("pad must be 'reflect' or 'circular'")
        self.order = int(order)
        self.support = 2 * self.order - 1
        self.levels = levels
        self.heads = int(heads)
        self.rule = rule
        self.tau = float(tau)
        self.kappa = float(kappa)
        self.pad = pad
        self.delay_compensation = bool(delay_compensation)
        self.fast_fft = bool(fast_fft)
        self.pad_tol = float(pad_tol)
        self.eps = float(eps)
        self.h, self.g = daubechies(self.order)
        self._cache: dict = {}

    # ------------------------------------------------------------------ setup
    def num_levels(self, T: int) -> int:
        """Detail levels J used for length T (the `levels` argument, or as many as fit, at most 10)."""
        if self.levels is not None:
            return int(self.levels)
        L = len(self.h)
        J = int(math.floor(math.log2(max(T, 2) / max(L - 1, 1)))) if L > 2 else int(math.log2(max(T, 2)))
        return int(min(max(J, 1), 10))

    def _delays_and_reach(self, J: int):
        """Integer centre-of-energy delay of every band's equivalent filter, and
        the one-sided radius (around that centre) holding 1 - pad_tol of the
        energy of every band: the padding that boundary handling actually needs."""
        L = len(self.h)
        M = 1 << int(math.ceil(math.log2(4 * (2 ** J) * L + 8)))
        psi = self._bands_np(np.arange(M // 2 + 1) * 2 * np.pi / M, J, np.zeros(J + 1))
        e = np.fft.irfft(psi, n=M, axis=-1) ** 2
        t = np.arange(M)
        centre = np.round((e * t).sum(-1) / e.sum(-1))
        reach = 0
        for s in range(J + 1):
            d = np.abs(t - centre[s])
            order_ = np.argsort(d, kind="stable")
            cum = np.cumsum(e[s][order_]) / e[s].sum()
            k = int(np.searchsorted(cum, 1.0 - self.pad_tol))
            reach = max(reach, int(d[order_[min(k, M - 1)]]))
        delays = centre if self.delay_compensation else np.zeros(J + 1)
        if not self.delay_compensation:
            reach = (2 ** J - 1) * (L - 1) + 1
        return delays, reach

    def _bands_np(self, w: np.ndarray, J: int, delays: np.ndarray) -> np.ndarray:
        r2 = math.sqrt(2.0)
        bands = []
        acc = np.ones_like(w, dtype=np.complex128)
        for j in range(1, J + 1):
            s = 2.0 ** (j - 1)
            bands.append(acc * _freq_response(self.g, s * w) / r2)
            acc = acc * _freq_response(self.h, s * w) / r2
        bands.append(acc)
        psi = np.stack(bands)  # (J+1, F)
        return psi * np.exp(1j * np.outer(delays, w))

    def _plan(self, T: int, device, dtype):
        J = self.num_levels(T)
        key = (T, J, str(device), dtype)
        if key in self._cache:
            return self._cache[key]
        delays, reach = self._delays_and_reach(J)
        if self.pad == "reflect":
            P = min(T - 1, reach // 2 + 1)
            n0 = T + 2 * P
        else:
            P, n0 = 0, T
        n = _fast_len(n0) if (self.fast_fft and self.pad == "reflect") else n0
        w = np.arange(n // 2 + 1) * 2 * np.pi / n
        psi = self._bands_np(w, J, delays)
        energy_ref = (np.abs(psi) ** 2).mean(-1)  # white-noise band energy n_s
        # Gaussian smoothing per band, sigma_s = kappa * 2^{level}
        lev = np.array(list(range(1, J + 1)) + [J], dtype=np.float64)
        sig = self.kappa * 2.0 ** lev
        gauss = np.exp(-0.5 * (sig[:, None] * w[None, :]) ** 2)
        cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
        plan = dict(
            J=J, P=P, n=n, n0=n0,
            psi=torch.as_tensor(psi, dtype=cdt, device=device),
            psi_c=torch.as_tensor(np.conj(psi), dtype=cdt, device=device),
            gauss=torch.as_tensor(gauss, dtype=dtype, device=device),
            log_ref=torch.as_tensor(np.log(energy_ref), dtype=dtype, device=device),
            delays=delays,
        )
        self._cache[key] = plan
        return plan

    # -------------------------------------------------------------- transforms
    def _pad(self, x, plan):
        """x: (B, C, T) -> (B, C, n)."""
        P, n, n0 = plan["P"], plan["n"], plan["n0"]
        if P > 0:
            x = F.pad(x, (P, P), mode="reflect")
        if n > n0:
            x = F.pad(x, (0, n - n0))
        return x

    def _crop(self, y, plan, T):
        P = plan["P"]
        return y[..., P:P + T]

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, J+1, T, C) band coefficients (on the unpadded grid)."""
        B, T, C = x.shape
        plan = self._plan(T, x.device, x.dtype)
        W = self._analysis_padded(self._pad(x.transpose(1, 2), plan), plan)
        return self._crop(W, plan, T).permute(0, 1, 3, 2)

    def _analysis_padded(self, xp, plan):
        Xf = torch.fft.rfft(xp, dim=-1)  # (B, C, F)
        return torch.fft.irfft(Xf.unsqueeze(1) * plan["psi"][None, :, None, :], n=plan["n"], dim=-1)

    def _synthesis_padded(self, Wp, plan):
        Wf = torch.fft.rfft(Wp, dim=-1)  # (B, S, C, F)
        Yf = (Wf * plan["psi_c"][None, :, None, :]).sum(1)
        return torch.fft.irfft(Yf, n=plan["n"], dim=-1)

    def synthesis(self, W: torch.Tensor) -> torch.Tensor:
        """Adjoint of `analysis` in circular mode; (B, J+1, T, C) -> (B, T, C).
        (In reflect mode use forward(); W here lives on the unpadded grid.)"""
        B, S, T, C = W.shape
        plan = self._plan(T, W.device, W.dtype)
        if plan["n"] != T:
            raise RuntimeError("synthesis() of cropped coefficients needs pad='circular'")
        return self._synthesis_padded(W.permute(0, 1, 3, 2), plan).transpose(1, 2)

    # ------------------------------------------------------------------- gates
    def _gate_attention(self, W, plan, T_valid):
        B, S, C, n = W.shape
        H = self.heads
        if C % H:
            raise ValueError("channels must be divisible by heads")
        E = W.view(B, S, H, C // H, n).pow(2).mean(3)  # (B, S, H, n)
        E = torch.fft.irfft(torch.fft.rfft(E, dim=-1) * plan["gauss"][None, :, None, :], n=n, dim=-1)
        E = E.clamp_min(self.eps)
        score = (E.log() - plan["log_ref"][None, :, None, None]) / self.tau
        g = S * torch.softmax(score, dim=1)
        num = E.sum(1, keepdim=True)
        den = (g * g * E).sum(1, keepdim=True).clamp_min(self.eps)
        g = g * torch.sqrt(num / den)
        return g.repeat_interleave(C // H, dim=2)  # (B, S, C, n)

    def _gate_shrink(self, W, plan, T_valid):
        B, S, C, n = W.shape
        P = plan["P"]
        # Donoho: noise level from the finest band, carried to band s by the
        # known white-noise energy n_s of the tight frame.
        sigma1 = W[:, 0, :, P:P + T_valid].abs().median(dim=-1, keepdim=True).values / 0.6745  # (B, C, 1)
        ratio = torch.exp(0.5 * (plan["log_ref"][:-1] - plan["log_ref"][0]))  # (S-1,)
        sigma = sigma1[:, None] * ratio[None, :, None, None]  # (B, S-1, C, 1)
        lam = math.sqrt(2.0 * math.log(max(T_valid, 2))) * sigma
        gd = torch.relu(1.0 - lam / W[:, :-1].abs().clamp_min(self.eps))
        return torch.cat([gd, torch.ones_like(W[:, -1:])], dim=1)

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """Gate field (B, J+1, T, C) the mixer applies to x (for inspection)."""
        B, T, C = x.shape
        plan = self._plan(T, x.device, x.dtype)
        Wp = self._analysis_padded(self._pad(x.transpose(1, 2), plan), plan)
        g = self._gate(Wp, plan, T)
        return self._crop(g, plan, T).permute(0, 1, 3, 2)

    def _gate(self, Wp, plan, T):
        if self.rule == "attention":
            return self._gate_attention(Wp, plan, T)
        if self.rule == "shrink":
            return self._gate_shrink(Wp, plan, T)
        return None

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C): analysis, gate rule, synthesis (rule='identity' returns x)."""
        if x.dim() != 3:
            raise ValueError("expected (B, T, C)")
        B, T, C = x.shape
        plan = self._plan(T, x.device, x.dtype)
        xp = self._pad(x.transpose(1, 2), plan)
        Wp = self._analysis_padded(xp, plan)
        g = self._gate(Wp, plan, T)
        if g is not None:
            Wp = Wp * g
        y = self._synthesis_padded(Wp, plan)
        return self._crop(y, plan, T).transpose(1, 2)

    def extra_repr(self):
        return (f"order={self.order}, support={self.support}, levels={self.levels}, heads={self.heads}, "
                f"rule={self.rule!r}, tau={self.tau}, pad={self.pad!r}")


# ---------------------------------------------------------------------------
# Independent reference: explicit a trous circular convolutions (rolls).
# Used only to validate the FFT path; O(L J T C).
# ---------------------------------------------------------------------------
def atrous_reference(x: torch.Tensor, order: int, J: int, delays=None) -> torch.Tensor:
    """x: (B, T, C), circular. Returns (B, J+1, T, C)."""
    h, g = daubechies(order)
    h = torch.as_tensor(h / math.sqrt(2), dtype=x.dtype)
    g = torch.as_tensor(g / math.sqrt(2), dtype=x.dtype)

    def cconv(a, f, dil):
        out = torch.zeros_like(a)
        for k in range(len(f)):
            out = out + f[k] * torch.roll(a, shifts=k * dil, dims=1)
        return out

    a, outs = x, []
    for j in range(1, J + 1):
        dil = 2 ** (j - 1)
        outs.append(cconv(a, g, dil))
        a = cconv(a, h, dil)
    outs.append(a)
    W = torch.stack(outs, 1)
    if delays is not None:
        W = torch.stack([torch.roll(W[:, s], shifts=-int(d), dims=1) for s, d in enumerate(delays)], 1)
    return W
