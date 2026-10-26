"""CrossPump: linked cross-platform evidence -> coordination score -> alarm gate -> avoid/exit signal.

Pipeline for a decision row (t, i), using only events with avail <= t and bar features up to t:
    1. Cross-platform Hawkes state (models/hawkes.py): two shared marked Hawkes models fitted on the train episodes, H_pump on
       pump episodes and H_norm on control episodes; PIT intensities lam_k(t), attribution matrix A_jk(t) (contribution of
       platform j to platform k's intensity), and the sequential log-likelihood ratio LLR(t) = sum_{s <= t} [ll_pump - ll_norm]
       over a trailing window of W bars (the 'linked-evidence' statistic: how much better the pump excitation structure
       explains the recent cross-platform stream than the normal one).
    2. Coordination score C(t) = cross-platform excitation share x synchrony:
           share(t) = sum_{j != k} A_jk / sum_{j,k} A_jk   (0 if no excitation)
           sync(t)  = geometric mean over platforms present in the episode of (lam_k(t) / mu_k)  (relative intensity lift)
           C(t) = share(t) * log1p(sync(t))     (high when activity on different platforms excites each other and all are lifted)
    3. Linked-evidence head: logistic regression (class-weighted, L2) on the causal feature vector
           [log lam_k, log(lam_k / mu_k), vec(A share), C, LLR_W, counts over {3, 12, 48} bars per platform (availability-based),
            prior-involvement maxima of recent social / wallet sources, causal z-scores of taker_imb / log_volume]
       trained on train rows with the episode label (1 inside [anchor - lead, peak]); abstains (NaN) only when nothing is
       known yet (warm-up).  Ablations: head_features in {all, hawkes_only, counts_only}, platforms subset, llr_only.
    4. Alarm gate (models/gate.py) at the false-alarm budget alpha with delayed feedback (tau = hold + latency); each alarm is
       stamped with its row -> lead time; the policy head emits the exit overlay (w = -w_max on alarm) for the shared backtest,
       and eval/follower.py turns alarms into avoid / exit decisions for the follower benchmark.
Registered as baseline "crosspump".
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch

from ..baselines.common import XPBaseline, register
from ..data.streams import PLATFORMS, P, EventStreams
from .hawkes import MarkedHawkes


def _causal_z(x: np.ndarray, warmup: int = 24, halflife: float = 48.0) -> np.ndarray:
    """EWMA z-scores with statistics from strictly earlier rows, vectorised over columns. x: (T, N)."""
    T, N = x.shape
    lam = 0.5 ** (1.0 / halflife)
    m = np.zeros(N); v = np.zeros(N); n = np.zeros(N, dtype=int)
    z = np.full((T, N), np.nan)
    for t in range(T):
        xt = np.nan_to_num(x[t], nan=0.0)
        ok = n >= warmup
        with np.errstate(invalid="ignore", divide="ignore"):
            z[t] = np.where(ok & (v > 1e-12), (xt - m) / np.sqrt(np.maximum(v, 1e-12)), np.nan)
        d = xt - m
        first = n == 0
        m = np.where(first, xt, lam * m + (1 - lam) * xt)
        v = np.where(first, 0.0, lam * v + (1 - lam) * d * d)
        n = n + np.isfinite(x[t]).astype(int)
    return z


class _LogisticHead:
    """Class-weighted L2 logistic regression in numpy (no sklearn dependency at runtime)."""

    def __init__(self, l2: float = 1e-2, iters: int = 300, lr: float = 0.1):
        self.l2, self.iters, self.lr = l2, iters, lr
        self.w: Optional[np.ndarray] = None; self.b = 0.0; self.mu = None; self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticHead":
        X = np.nan_to_num(X.astype(np.float64)); y = y.astype(np.float64)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        pos = max(y.sum(), 1.0); neg = max(n - y.sum(), 1.0)
        wgt = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        w = np.zeros(d); b = 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = (wgt * (p - y))
            gw = Z.T @ g / n + self.l2 * w; gb = g.mean()
            w -= self.lr * gw; b -= self.lr * gb
        self.w, self.b = w, b
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = (np.nan_to_num(X.astype(np.float64)) - self.mu) / self.sd
        return Z @ self.w + self.b


@register("crosspump")
class CrossPump(XPBaseline):
    name = "crosspump"
    paper = "CrossPump (this paper): cross-platform marked Hawkes + coordination score + linked-evidence head + delayed-ACI gate"
    year = 2026; venue = "this paper"; status = "implemented"
    modalities = {"social", "wallet", "orderflow", "graph"}

    def __init__(self, cfg, dims, horizons, device):
        super().__init__(cfg, dims, horizons, device)
        m = dict(cfg.get("model", {}))
        self.iters = int(m.get("hawkes_iters", 150)); self.lr = float(m.get("hawkes_lr", 0.05))
        self.max_train_episodes = int(m.get("max_train_episodes", 200))
        self.llr_window = int(m.get("llr_window", 48))
        self.head_features = str(m.get("head_features", "all"))            # all | hawkes_only | counts_only | llr_only
        self.use_two_models = bool(m.get("two_models", True))
        self.count_windows = tuple(m.get("count_windows", (3, 12, 48)))
        self.warmup = int(m.get("warmup", 12))
        self.hp: Optional[MarkedHawkes] = None; self.hn: Optional[MarkedHawkes] = None
        self.head = _LogisticHead(float(m.get("head_l2", 1e-2)), int(m.get("head_iters", 300)))
        self._F: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------ features
    def _build_features(self) -> np.ndarray:
        T, N = self.T, self.N
        feats: List[np.ndarray] = []; names: List[str] = []
        present = np.zeros((N, P), dtype=bool)
        if len(self.streams):
            present[self.streams.asset, self.streams.platform.astype(int)] = True
        if self.head_features != "counts_only" and self.hp is not None:
            sp = self.hp.row_state(self.streams, T, N)
            lam, A = sp["lam"], sp["attr"]
            mu = np.asarray(self.hp.summary()["mu"])[None, None, :]
            lift = lam / mu
            share = (A.sum((2, 3)) - np.trace(A, axis1=2, axis2=3)) / np.maximum(A.sum((2, 3)), 1e-9)
            n_present = np.maximum(present.sum(1), 1)[None, :]
            log_lift = np.where(present[None, :, :], np.log(np.maximum(lift, 1e-9)), 0.0)
            sync = np.exp(log_lift.sum(2) / n_present)
            coord = share * np.log1p(np.maximum(sync - 1.0, 0.0))
            ll = sp["ll_inc"]
            if self.use_two_models and self.hn is not None:
                ll = ll - self.hn.row_state(self.streams, T, N)["ll_inc"]
            cs = np.concatenate([np.zeros((1, N)), np.cumsum(ll, 0)], 0)
            idx = np.arange(T)
            llr_w = cs[idx + 1] - cs[np.maximum(idx + 1 - self.llr_window, 0)]
            self._coord, self._llr, self._share, self._sync = coord, llr_w, share, sync
            if self.head_features == "llr_only":
                feats.append(llr_w[..., None]); names.append("llr_w")
            else:
                feats += [np.log(np.maximum(lam, 1e-9)), np.log(np.maximum(lift, 1e-9)), A.reshape(T, N, P * P) / np.maximum(lam.sum(2, keepdims=True), 1e-9),
                          coord[..., None], llr_w[..., None], share[..., None], np.log1p(np.maximum(sync, 0))[..., None]]
                names += [f"log_lam_{p}" for p in PLATFORMS] + [f"log_lift_{p}" for p in PLATFORMS] + [f"A_{j}_{k}" for j in PLATFORMS for k in PLATFORMS] + ["coord", "llr_w", "share", "log_sync"]
        if self.head_features in ("all", "counts_only"):
            cs = np.concatenate([np.zeros((1, N, P)), np.cumsum(self.counts, 0)], 0)
            idx = np.arange(T)
            for w in self.count_windows:
                feats.append(np.log1p(cs[idx + 1] - cs[np.maximum(idx + 1 - w, 0)])); names += [f"cnt{w}_{p}" for p in PLATFORMS]
            # prior involvement of recent sources (channel / wallet track record), causal max over the last 48 bars
            pri = self.mark_sums[..., 3]                                                   # (T, N, P) sums of prior-involvement marks per bar
            csp = np.concatenate([np.zeros((1, N, P)), np.cumsum(pri, 0)], 0)
            feats.append(np.log1p(np.maximum(csp[idx + 1] - csp[np.maximum(idx + 1 - 48, 0)], 0))); names += [f"prior48_{p}" for p in PLATFORMS]
            fn = list(self.panel.meta.get("feat_names", []))
            for col in ("taker_imb", "log_volume"):
                if col in fn:
                    feats.append(np.nan_to_num(_causal_z(self.feat[:, :, fn.index(col)].astype(np.float64), self.warmup))[..., None]); names.append(f"z_{col}")
        F = np.concatenate(feats, -1).astype(np.float32)
        self.feature_names = names
        return F

    # ------------------------------------------------------------------ fit / score
    def _fit(self, panel, train_assets, val_assets):
        is_pump = np.asarray(panel.meta["is_pump"])
        tr = np.asarray(train_assets)
        pumps = tr[is_pump[tr]]; ctrls = tr[~is_pump[tr]]
        rng = np.random.default_rng(0)
        if len(pumps) > self.max_train_episodes:
            pumps = rng.choice(pumps, self.max_train_episodes, replace=False)
        if len(ctrls) > self.max_train_episodes:
            ctrls = rng.choice(ctrls, self.max_train_episodes, replace=False)
        if self.head_features != "counts_only" and len(self.streams):
            self.hp = MarkedHawkes(device=torch.device("cpu")).fit(self.streams, pumps if len(pumps) else tr, self.T, self.iters, self.lr)
            if self.use_two_models and len(ctrls):
                self.hn = MarkedHawkes(device=torch.device("cpu")).fit(self.streams, ctrls, self.T, self.iters, self.lr)
            self.diag["hawkes_pump"] = self.hp.summary(); self.diag["hawkes_norm"] = self.hn.summary() if self.hn else None
        self._F = self._build_features()
        t, i = self.rows_of(tr)
        X = self._F[t, i]; y = panel.labels[t, i].astype(float)
        ok = np.isfinite(X).all(1)
        self.head.fit(X[ok], y[ok])
        self.diag.update({"n_features": int(self._F.shape[-1]), "feature_names": self.feature_names, "train_rows": int(ok.sum()), "train_pos_rate": float(y[ok].mean())})
        top = np.argsort(-np.abs(self.head.w))[:8]
        self.diag["top_weights"] = {self.feature_names[k]: round(float(self.head.w[k]), 3) for k in top}

    def _score_rows(self, t, i):
        X = self._F[t, i]
        s = self.head.decision(X)
        s[t < self.warmup] = np.nan
        return s

    def alert_table(self, t: np.ndarray, i: np.ndarray, alarm: np.ndarray) -> Dict[str, np.ndarray]:
        """Each alarm stamped with its row, the coordination score and the LLR at that row (for the lead-time tables)."""
        sel = alarm.astype(bool)
        return {"t": t[sel], "asset": i[sel], "coord": self._coord[t[sel], i[sel]] if hasattr(self, "_coord") else np.full(sel.sum(), np.nan),
                "llr": self._llr[t[sel], i[sel]] if hasattr(self, "_llr") else np.full(sel.sum(), np.nan)}
