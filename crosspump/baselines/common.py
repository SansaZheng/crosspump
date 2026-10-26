"""Scaffold for every CrossPump method and baseline (B1-B10): score rows -> alarm gate at a false-alarm budget ->
exit-overlay trade -> the shared Warn2Trade `out` dict, so the event backtester and the metric matrix are identical
for every method. Subclasses implement only

    _fit(self, panel, train_assets, val_assets) -> None
    _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray     # (B,) higher = more pump-like; NaN = abstain

using, for a row (t, i), only information knowable at bar t (avail <= t): helpers `self.counts` (T, N, P) causal
per-bar event counts (events counted at their availability bar), `self.mark_sums` (T, N, P, DM), `self.streams_before(i, t)`,
`self.feat` = panel.price_feat (raw; a subclass normalises causally if it needs to), `self.labels_train` masked outside the
train episodes. Modalities a baseline reads are declared in `modalities`; the runner can switch streams off (ablation /
coverage study) through cfg['platforms'].

Trade semantics (exit overlay): w = -w_max on alarm, i.e. a follower who is long reduces to flat; its PnL under the shared
engine equals the loss avoided by exiting now rather than `hold` bars later. The paper's headline economic metric
(follower loss avoided vs the Twitter-follower benchmark) is computed separately in crosspump/eval/follower.py.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np
import torch
from torch.utils.data import DataLoader

from warn2trade.baselines.event_base import BaseBaseline, register_event_baseline
from ..data.streams import EventStreams, get_streams, PLATFORMS, P, DM
from ..models.gate import make_gate, FixedQuantileGate

MODALITIES = ("social", "wallet", "orderflow", "text", "graph")


def register(name: str):
    return register_event_baseline(name)


class XPBaseline(BaseBaseline):
    name = "xp_base"
    paper = ""
    year = 0
    venue = ""
    official_repo = ""
    status = "reimplemented"           # official | reimplemented | reimplemented-fallback | wrapper-needed
    modalities: Set[str] = set()
    unspecified: List[str] = []

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        super().__init__(cfg, dims, horizons, device)
        self.panel = cfg.get("panel")
        assert self.panel is not None, "XPBaseline needs cfg['panel'] (the runner passes it)"
        self.L = int(cfg.get("lookback", 48))
        self.latency = int(cfg.get("latency", 1))
        self.hold_index = int(cfg.get("hold_index", 2))
        self.h = int(self.horizons[self.hold_index])
        self.bars_per_year = int(cfg.get("bars_per_year", self.panel.meta.get("bars_per_year", 105120)))
        self.alpha = float(cfg.get("alpha", 0.01))
        self.platforms = tuple(cfg.get("platforms", PLATFORMS))             # streams visible to this run (ablations)
        self.gate: FixedQuantileGate = make_gate(dict(cfg.get("gate", {"type": "delayed_aci"})) | {"alpha": self.alpha}, self.h, self.latency)
        self.exit_sign = float(cfg.get("exit_sign", -1.0))
        self.T, self.N = self.panel.T, self.panel.N
        self.feat = self.panel.price_feat
        es = get_streams(self.panel)
        if es is None:
            es = EventStreams.empty()
        keep = np.isin(es.platform, [PLATFORMS.index(p) for p in self.platforms])
        self.streams: EventStreams = es.subset(keep) if len(es) else es
        self.counts = self.streams.rasterise(self.T, self.N, pit=True)                                   # (T, N, P)
        self.mark_sums = np.stack([self.streams.rasterise(self.T, self.N, pit=True, weights=np.nan_to_num(self.streams.mark[:, k])) for k in range(DM)], -1) if len(self.streams) else np.zeros((self.T, self.N, P, DM))
        self.train_assets: np.ndarray = np.zeros(0, dtype=int)
        self.val_assets: np.ndarray = np.zeros(0, dtype=int)
        self._last_t = -1
        self.diag: Dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers for subclasses
    def streams_before(self, i: int, t: float) -> EventStreams:
        return self.streams.before(int(i), float(t), pit=True)

    def cum_counts(self, t: np.ndarray, i: np.ndarray, back: int) -> np.ndarray:
        """(B, P) event counts in bars (t - back, t] (availability-based)."""
        cs = np.concatenate([np.zeros((1, self.N, P)), np.cumsum(self.counts, 0)], 0)
        lo = np.clip(t - back, 0, self.T); hi = np.clip(t + 1, 0, self.T)
        return cs[hi, i] - cs[lo, i]

    def labels_for(self, assets: np.ndarray) -> np.ndarray:
        """Training labels restricted to the given episodes (others masked to NaN) — never read test labels."""
        y = self.panel.labels.astype(np.float64).copy()
        mask = np.ones(self.N, dtype=bool); mask[assets] = False
        y[:, mask] = np.nan
        return y

    def rows_of(self, assets: np.ndarray, lo: Optional[int] = None, hi: Optional[int] = None):
        lo = self.L if lo is None else lo; hi = self.T - max(self.horizons) - self.latency if hi is None else hi
        tt, ii = np.meshgrid(np.arange(lo, hi), np.asarray(assets), indexing="ij")
        return tt.ravel(), ii.ravel()

    # ------------------------------------------------------------------ hooks
    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        raise NotImplementedError

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------ fit / predict
    @staticmethod
    def _assets_in(loader: Optional[DataLoader]) -> np.ndarray:
        if loader is None:
            return np.zeros(0, dtype=int)
        ds = loader.dataset
        return np.unique(ds.index[:, 1]) if hasattr(ds, "index") else np.zeros(0, dtype=int)

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        self.train_assets = self._assets_in(train)
        self.val_assets = self._assets_in(val) if val is not None else np.zeros(0, dtype=int)
        self._fit(self.panel, self.train_assets, self.val_assets)
        cal_assets = self.val_assets if len(self.val_assets) else self.train_assets
        t, i = self.rows_of(cal_assets)
        s = self._score_rows(t, i) if len(t) else np.zeros(0)
        y = self.panel.labels[t, i] if len(t) else np.zeros(0)
        self.gate.calibrate(s, labels=y)
        self.diag.update({"n_train_episodes": int(len(self.train_assets)), "n_val_episodes": int(len(self.val_assets)), "alpha": self.alpha,
                          "platforms": list(self.platforms), "calib_rows": int(np.isfinite(s).sum()), "gate": self.gate.kind, "modalities": sorted(self.modalities)})
        self._last_t = -1

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        t = batch["t"].cpu().numpy(); i = batch["asset"].cpu().numpy(); y = batch["y"].cpu().numpy()
        s = self._score_rows(t, i)
        B = len(t)
        if int(t.min()) < self._last_t:
            self.gate.reset_stream()
        alarm = np.zeros(B, dtype=bool); a_prob = np.zeros(B)
        for k in np.lexsort((i, t)):
            alarm[k], _ = self.gate.step(float(s[k]), int(t[k]), float(y[k]))
            a_prob[k] = self.gate.prob(float(s[k]))
        self._last_t = int(t.max())
        dev = self.device
        a_t = torch.as_tensor(alarm, dtype=torch.float32, device=dev)
        return {"a_prob": torch.as_tensor(a_prob, dtype=torch.float32, device=dev), "w": a_t * self.exit_sign * self.w_max, "abstain": 1.0 - a_t,
                "hold_p": self._hold_fixed(B, self.hold_index), "kappa_p": self._immediate(B),
                "score": torch.as_tensor(np.nan_to_num(s, nan=-1.0), dtype=torch.float32, device=dev)}


# ---------------------------------------------------------------------- reference rows
@register("xp_counts_rule")
class CountsRule(XPBaseline):
    """Reference row: z-scored total event count over the last `back` bars (any visible platform), no learning."""
    name = "xp_counts_rule"; paper = "reference rule (this paper)"; year = 2026; status = "implemented"; modalities = {"social", "wallet", "orderflow"}

    def _fit(self, panel, train_assets, val_assets):
        self.back = int(self.cfg.get("back", 12))
        t, i = self.rows_of(train_assets)
        c = self.cum_counts(t, i, self.back).sum(-1)
        self.mu, self.sd = float(c.mean()), float(c.std() + 1e-6)

    def _score_rows(self, t, i):
        return (self.cum_counts(t, i, self.back).sum(-1) - self.mu) / self.sd
