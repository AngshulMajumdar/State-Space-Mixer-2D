"""Two-dimensional cross-scale attention on a separable Daubechies tight frame (time x frequency).

Frame. The undecimated, delay-aligned Daubechies frame is applied along both axes of a 2-D field X (B, T, F):
db `order` with J_t levels along the first (time) axis and db `freq_order` with J_f levels along the second
(frequency) axis. Band (s, r) has the transfer function Psi_s(w_t) Psi_r(w_f), and

    sum_{s, r} |Psi_s(w_t)|^2 |Psi_r(w_f)|^2 = (sum_s |Psi_s|^2) (sum_r |Psi_r|^2) = 1,

so the 2-D frame is Parseval-tight and the adjoint is its exact inverse.

Tokens. Coefficient vectors over `group` consecutive frequency bins, at every band (s, r), instant t and
bin group g.

Mixer. Token (s, r, t, g) attends to
    time-scale neighbours  (s + ds, r) at time offsets       o * 2^lev_t(s + ds), |o| <= k, ds in {-1, 0, 1}
    freq-scale neighbours  (s, r + dr) at frequency offsets  o * 2^lev_f(r + dr), |o| <= k, dr in {-1, 0, 1}
(lev = dyadic level of the key band), with per-token cosine softmax at temperature tau. Values are the raw
coefficients, shifted alike; synthesis is the exact adjoint. The operator has no parameters, is
shift-equivariant in time (and in frequency for shifts by multiples of `group`), 1-homogeneous, and tends to
the identity as tau -> 0.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .frame import ScaleSpaceMixer

Link = Tuple[int, int, int, int]  # (time band, frequency band, time shift, frequency shift)


class CrossScaleMixer2D(nn.Module):
    """Parameter-free cross-scale attention on the separable 2-D Daubechies tight frame.

    Args:
        order:        Daubechies order along time (support 2*order - 1).
        freq_order:   Daubechies order along frequency.
        levels:       time levels J_t (None: as many as fit the length).
        freq_levels:  frequency levels J_f (None: as many as fit the number of bins).
        group:        frequency bins per token (a divisor of 16).
        window:       half-width k of the sibling neighbourhood on each axis.
        tau:          attention temperature (tau -> 0 is the identity).
        pad:          boundary handling of both frames, "reflect" or "circular".

    Input and output: (B, T, F) real tensors.
    """

    def __init__(self, order: int = 4, freq_order: int = 2, levels: Optional[int] = None,
                 freq_levels: Optional[int] = None, group: int = 8, window: int = 2, tau: float = 0.1,
                 pad: str = "reflect"):
        super().__init__()
        if group < 1 or 16 % group:
            raise ValueError("group must divide 16 (1, 2, 4, 8 or 16)")
        if window < 0:
            raise ValueError("window must be >= 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.ft = ScaleSpaceMixer(order=order, levels=levels, rule="identity", pad=pad)
        self.ff = ScaleSpaceMixer(order=freq_order, levels=freq_levels, rule="identity", pad=pad)
        self.order, self.freq_order = int(order), int(freq_order)
        self.group, self.window, self.tau = int(group), int(window), float(tau)
        self.last_attention: Optional[Dict[Tuple[int, int], torch.Tensor]] = None
        self.att_crop: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------ frame
    def coefficients(self, X: torch.Tensor):
        """(B, T, F) -> padded coefficients W (B, S, R, nt, nf) and the plan needed for synthesis."""
        if X.dim() != 3:
            raise ValueError("expected a (B, T, F) tensor")
        B, T, Fb = X.shape
        pt = self.ft._plan(T, X.device, X.dtype)
        pf = self.ff._plan(Fb, X.device, X.dtype)
        Wt = self.ft._analysis_padded(self.ft._pad(X.transpose(1, 2), pt), pt)  # (B, S, F, nt)
        S, nt = Wt.shape[1], Wt.shape[-1]
        y = Wt.permute(0, 1, 3, 2).reshape(B * S, nt, Fb)
        Wf = self.ff._analysis_padded(self.ff._pad(y, pf), pf)  # (B*S, R, nt, nf)
        R, nf = Wf.shape[1], Wf.shape[-1]
        if nf % self.group:
            raise ValueError(f"padded frequency length {nf} is not a multiple of group={self.group}")
        plan = dict(pt=pt, pf=pf, T=T, F=Fb, S=S, R=R, nt=nt, nf=nf)
        return Wf.reshape(B, S, R, nt, nf), plan

    def synthesize(self, W: torch.Tensor, plan: dict) -> torch.Tensor:
        """Padded coefficients (B, S, R, nt, nf) -> (B, T, F) by the tight-frame adjoint (exact inverse)."""
        B, S, R, nt, nf = W.shape
        pt, pf = plan["pt"], plan["pf"]
        y = self.ff._synthesis_padded(W.reshape(B * S, R, nt, nf), pf)  # (B*S, nt, nf)
        y = self.ff._crop(y, pf, plan["F"]).reshape(B, S, nt, plan["F"]).permute(0, 1, 3, 2)
        x = self.ft._synthesis_padded(y, pt)  # (B, F, nt)
        return self.ft._crop(x, pt, plan["T"]).transpose(1, 2)

    def valid(self, plan: dict):
        """Slices of the padded grid holding the input's own samples: (time slice, frequency slice)."""
        Pt, Pf = plan["pt"]["P"], plan["pf"]["P"]
        return slice(Pt, Pt + plan["T"]), slice(Pf, Pf + plan["F"])

    # ------------------------------------------------------------------ links
    def links(self, S: int, R: int) -> Dict[Tuple[int, int], List[Link]]:
        """Key list of every band (s, r): [(s', r', time shift, frequency shift)], self first."""
        lt = [s + 1 for s in range(S - 1)] + [S - 1]
        lf = [r + 1 for r in range(R - 1)] + [R - 1]
        k = self.window
        out = {}
        for s in range(S):
            for r in range(R):
                ks = [(s, r, 0, 0)]
                for d in (-1, 0, 1):
                    s2 = s + d
                    if 0 <= s2 < S:
                        ks += [(s2, r, o * 2 ** lt[s2], 0) for o in range(-k, k + 1) if (d, o) != (0, 0)]
                    r2 = r + d
                    if 0 <= r2 < R:
                        ks += [(s, r2, 0, o * 2 ** lf[r2]) for o in range(-k, k + 1) if (d, o) != (0, 0)]
                out[(s, r)] = ks
        return out

    @staticmethod
    def _shift(x: torch.Tensor, dt: int, df: int) -> torch.Tensor:
        """x (B, nt, nf): the value at (t, f) becomes x(t + dt, f + df), circular on the padded grid."""
        if dt == 0 and df == 0:
            return x
        sh, dims = [], []
        if dt:
            sh.append(-dt)
            dims.append(1)
        if df:
            sh.append(-df)
            dims.append(2)
        return torch.roll(x, sh, dims)

    # ------------------------------------------------------------------ mixing
    def mix(self, W: torch.Tensor) -> torch.Tensor:
        """One synchronous cross-scale attention update of padded coefficients (B, S, R, nt, nf).

        Every key is normalised over its own `group` bins after shifting (key norms come from a sliding
        window sum of squares), so frequency shifts that are not multiples of `group` are exact.
        Attention weights are kept in `last_attention[(s, r)]` as (B, links, nt, groups), or, if `att_crop`
        = (t0, t1, g0, g1) is set, as their mean over t0:t1 restricted to groups g0:g1 (B, links, g1 - g0).
        """
        B, S, R, nt, nf = W.shape
        G, D = nf // self.group, self.group
        nrm = W.reshape(B, S, R, nt, G, D).norm(dim=-1).clamp_min(1e-12)
        Qn = W.reshape(B, S, R, nt, G, D) / nrm.unsqueeze(-1)
        sq = torch.cat([W * W, (W * W)[..., :D - 1]], -1).cumsum(-1)
        N2 = sq[..., D - 1:] - torch.cat([torch.zeros_like(sq[..., :1]), sq[..., :-D]], -1)
        del sq
        out = torch.empty_like(W)
        att = {}
        for (s, r), ks in self.links(S, R).items():
            q = Qn[:, s, r]
            sc = torch.stack([(q * self._shift(W[:, s2, r2], dt, df).reshape(B, nt, G, D)).sum(-1)
                              / self._shift(N2[:, s2, r2], dt, df)[..., ::D].clamp_min(1e-24).sqrt()
                              for s2, r2, dt, df in ks], 1) / self.tau  # (B, M, nt, G)
            a = torch.softmax(sc, dim=1)
            acc = torch.zeros(B, nt, G, D, dtype=W.dtype, device=W.device)
            for m, (s2, r2, dt, df) in enumerate(ks):
                acc.addcmul_(a[:, m].unsqueeze(-1), self._shift(W[:, s2, r2], dt, df).reshape(B, nt, G, D))
            out[:, s, r] = acc.reshape(B, nt, nf)
            if self.att_crop is None:
                att[(s, r)] = a
            else:
                t0, t1, g0, g1 = self.att_crop
                att[(s, r)] = a[:, :, t0:t1, g0:g1].mean(2)
        self.last_attention = att
        return out

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, T, F): 2-D analysis, one cross-scale attention update, exact synthesis."""
        W, plan = self.coefficients(X)
        return self.synthesize(self.mix(W), plan)

    def extra_repr(self) -> str:
        return (f"order={self.order}, freq_order={self.freq_order}, group={self.group}, "
                f"window={self.window}, tau={self.tau}")
