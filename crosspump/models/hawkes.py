"""Cross-platform marked Hawkes process (the excitation model of CrossPump).

P = 3 platforms (social, wallet, orderflow). For an episode with events (t_l, p_l, m_l):
    lambda_k(t) = mu_k + sum_j alpha_jk * sum_{l: p_l = j, t_l < t} g_j(m_l) * exp(-beta_jk (t - t_l)),     g_j(m) = exp(gamma_j * m~)
with m~ = standardised magnitude mark (mark[:, 0]; NaN -> 0). Parameters are shared across episodes (mu, alpha, beta, gamma);
alpha_jk / beta_jk = branching ratio from platform j to platform k = the cross-platform excitation matrix. Fitted by exact
maximum likelihood (exponential kernels -> O(E) recursion per episode, batched over episodes in torch).

Two uses:
    * `row_state(streams, T, N)`: for every decision row t and episode i the PIT intensity lambda_k(t) and the attribution
      matrix A_jk(t) = contribution of platform j to lambda_k(t), using only events with avail <= t (events enter the recursion
      at ceil(avail), consistent with EventStreams.before / rasterise); plus the coordination score (models/coordination.py).
    * `MarkedHawkes.fit(streams, assets)`: fit on the given episodes; `log_lik_rows` gives the running log-likelihood of the
      observed events under this model, so two fits (pump episodes vs controls) give a sequential log-likelihood ratio.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from ..data.streams import EventStreams, P


def _softplus_inv(x: float) -> float:
    return float(np.log(np.expm1(x))) if x > 1e-6 else -12.0


class MarkedHawkes:
    def __init__(self, n_platforms: int = P, device: Optional[torch.device] = None, mark_clip: float = 3.0, l2: float = 1e-3):
        self.P = int(n_platforms)
        self.device = device or torch.device("cpu")
        self.mark_clip = float(mark_clip)
        self.l2 = float(l2)
        Pn = self.P
        self._mu = torch.nn.Parameter(torch.full((Pn,), _softplus_inv(0.05), device=self.device))
        self._alpha = torch.nn.Parameter(torch.full((Pn, Pn), _softplus_inv(0.1), device=self.device))
        self._beta = torch.nn.Parameter(torch.full((Pn, Pn), _softplus_inv(0.5), device=self.device))
        self._gamma = torch.nn.Parameter(torch.zeros(Pn, device=self.device))
        self.mark_loc = np.zeros(Pn); self.mark_scale = np.ones(Pn)
        self.fitted = False
        self.history: List[float] = []

    # ------------------------------------------------------------------ parameters
    @property
    def mu(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._mu) + 1e-6

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._alpha)

    @property
    def beta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._beta) + 1e-3

    @property
    def gamma(self) -> torch.Tensor:
        return torch.clamp(self._gamma, -2.0, 2.0)

    def parameters(self) -> List[torch.nn.Parameter]:
        return [self._mu, self._alpha, self._beta, self._gamma]

    def branching_matrix(self) -> np.ndarray:
        """(P, P) n_jk = alpha_jk / beta_jk: expected number of platform-k events triggered by one unit-mark platform-j event."""
        with torch.no_grad():
            return (self.alpha / self.beta).cpu().numpy()

    def summary(self) -> Dict[str, Any]:
        with torch.no_grad():
            return {"mu": self.mu.cpu().numpy().round(5).tolist(), "alpha": self.alpha.cpu().numpy().round(4).tolist(), "beta": self.beta.cpu().numpy().round(4).tolist(),
                    "gamma": self.gamma.cpu().numpy().round(4).tolist(), "branching": self.branching_matrix().round(4).tolist(), "fitted": self.fitted, "n_iter": len(self.history)}

    # ------------------------------------------------------------------ marks
    def _std_marks(self, streams: EventStreams, fit: bool = False, assets: Optional[Sequence[int]] = None) -> np.ndarray:
        m = np.nan_to_num(streams.mark[:, 0].astype(np.float64), nan=0.0)
        if fit:
            in_fit = np.isin(streams.asset, np.asarray(list(assets), dtype=np.int64)) if assets is not None else np.ones(len(streams), bool)
            for p in range(self.P):
                sel = (streams.platform == p) & in_fit                      # statistics from the fitting episodes only (PIT)
                if sel.sum() > 5:
                    self.mark_loc[p] = float(np.median(m[sel])); self.mark_scale[p] = float(np.std(m[sel]) + 1e-6)
        z = (m - self.mark_loc[streams.platform]) / self.mark_scale[streams.platform]
        return np.clip(z, -self.mark_clip, self.mark_clip)

    # ------------------------------------------------------------------ likelihood (batched over episodes)
    def _pack(self, streams: EventStreams, assets: Sequence[int], T: int, fit_marks: bool = False):
        assets = [int(a) for a in assets]
        z_all = self._std_marks(streams, fit=fit_marks, assets=assets)
        seqs = []
        for a in assets:
            lo, hi = streams._bounds(a)
            seqs.append((streams.time[lo:hi], streams.platform[lo:hi].astype(np.int64), z_all[lo:hi]))
        E = max([len(s[0]) for s in seqs] + [1])
        B = len(seqs)
        t = np.full((B, E), float(T)); p = np.zeros((B, E), np.int64); z = np.zeros((B, E)); msk = np.zeros((B, E), bool)
        for b, (tt, pp, zz) in enumerate(seqs):
            n = len(tt); t[b, :n] = tt; p[b, :n] = pp; z[b, :n] = zz; msk[b, :n] = True
        dev = self.device
        return (torch.as_tensor(t, dtype=torch.float64, device=dev), torch.as_tensor(p, device=dev), torch.as_tensor(z, dtype=torch.float64, device=dev),
                torch.as_tensor(msk, device=dev), B, E)

    def neg_log_lik(self, packed, T: int) -> torch.Tensor:
        t, p, z, msk, B, E = packed
        mu, alpha, beta, gamma = self.mu.double(), self.alpha.double(), self.beta.double(), self.gamma.double()
        Pn = self.P
        R = torch.zeros(B, Pn, Pn, dtype=torch.float64, device=self.device)            # R[b, j, k]
        t_prev = torch.zeros(B, dtype=torch.float64, device=self.device)
        ll = torch.zeros(B, dtype=torch.float64, device=self.device)
        eye = torch.eye(Pn, dtype=torch.float64, device=self.device)
        for e in range(E):
            te, pe, ze, me = t[:, e], p[:, e], z[:, e], msk[:, e]
            dt = torch.clamp(te - t_prev, min=0.0)
            R = R * torch.exp(-beta[None] * dt[:, None, None])
            lam = mu[None, :] + (alpha[None] * R).sum(1)                                  # (B, P)
            lam_e = lam.gather(1, pe[:, None]).squeeze(1)
            ll = ll + torch.where(me, torch.log(lam_e + 1e-12), torch.zeros_like(ll))
            g = torch.exp(gamma[pe] * ze) * me.double()                                     # mark gain of the event (0 if padded)
            R = R + eye[pe][:, :, None] * g[:, None, None]                                  # add g to row j = pe (for all k)
            t_prev = torch.where(me, te, t_prev)
        # compensator: mu_k T + sum_l alpha_{p_l k} g_l (1 - exp(-beta_{p_l k}(T - t_l))) / beta_{p_l k}
        g_all = torch.exp(gamma[p] * z) * msk.double()                                      # (B, E)
        rem = torch.clamp(float(T) - t, min=0.0)                                            # (B, E)
        a_rows = alpha[p]; b_rows = beta[p]                                                 # (B, E, P)
        comp = (a_rows * g_all[..., None] * (1.0 - torch.exp(-b_rows * rem[..., None])) / b_rows).sum((1, 2)) + mu.sum() * float(T)
        nll = -(ll - comp).sum() / max(B, 1)
        return nll + self.l2 * (self.alpha ** 2).sum()

    def fit(self, streams: EventStreams, assets: Sequence[int], T: int, iters: int = 200, lr: float = 0.05, verbose: bool = False) -> "MarkedHawkes":
        packed = self._pack(streams, assets, T, fit_marks=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for it in range(iters):
            opt.zero_grad()
            loss = self.neg_log_lik(packed, T)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            opt.step()
            self.history.append(float(loss.detach()))
            if verbose and it % 50 == 0:
                print(f"iter {it} nll/episode {float(loss):.3f}")
        self.fitted = True
        return self

    # ------------------------------------------------------------------ PIT row state
    @torch.no_grad()
    def row_state(self, streams: EventStreams, T: int, N: int, assets: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        """For rows t = 0..T-1 and episodes i: lam (T, N, P), attribution A (T, N, P, P) [j -> k], using events with avail <= t
        (an event enters the state at row ceil(avail) with kernel exp(-beta (t - time))). Also the running event log-likelihood
        increments ll_inc (T, N): sum over events becoming known at row t of log lam_{p}(time) minus the compensator increment,
        scored against the intensity implied by events usable BEFORE row t (strictly PIT; a bar-level approximation of the exact likelihood)."""
        assets = list(range(N)) if assets is None else [int(a) for a in assets]
        mu, alpha, beta, gamma = self.mu.double().cpu().numpy(), self.alpha.double().cpu().numpy(), self.beta.double().cpu().numpy(), self.gamma.double().cpu().numpy()
        Pn = self.P
        lam = np.zeros((T, N, Pn)); A = np.zeros((T, N, Pn, Pn)); ll_inc = np.zeros((T, N))
        z_all = self._std_marks(streams, fit=False)
        decay = np.exp(-beta)                                                               # per-bar decay (P, P)
        for i in assets:
            lo, hi = streams._bounds(i)
            tm, pl, av, z = streams.time[lo:hi], streams.platform[lo:hi].astype(int), streams.avail[lo:hi], z_all[lo:hi]
            row_in = np.clip(np.ceil(av), 0, T).astype(int)                                 # row at which the event becomes usable
            g = np.exp(gamma[pl] * z)
            order = np.argsort(row_in, kind="stable")
            tm, pl, row_in, g = tm[order], pl[order], row_in[order], g[order]
            R = np.zeros((Pn, Pn)); ptr = 0
            for t in range(T):
                R = R * decay
                lam_prev = mu + (alpha * R).sum(0)                                          # intensity from events usable before this row
                while ptr < len(row_in) and row_in[ptr] <= t:
                    ll_inc[t, i] += np.log(lam_prev[pl[ptr]] + 1e-12)                       # PIT: scored against the state known before row t
                    R[pl[ptr], :] += g[ptr] * np.exp(-beta[pl[ptr], :] * max(t - tm[ptr], 0.0))
                    ptr += 1
                contrib = alpha * R                                                         # (j, k)
                A[t, i] = contrib; lam[t, i] = mu + contrib.sum(0)
                # compensator over (t, t+1] from the usable state: mu_k + sum_j alpha_jk R_jk (1 - e^{-beta_jk}) / beta_jk
                ll_inc[t, i] -= mu.sum() + (contrib * (1.0 - decay) / beta).sum()
        return {"lam": lam, "attr": A, "ll_inc": ll_inc}
