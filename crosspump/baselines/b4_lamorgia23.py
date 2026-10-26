"""B4 — La Morgia, Mei, Sassi, Stefa, "The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency
Manipulations", ACM Transactions on Internet Technology 23(1), 2023, doi:10.1145/3561300, arXiv:2105.00733 (conference
version ICCCN 2020). Official code: https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset. Registry key: xp_lamorgia23.

What the official code does (read on c225, data_raw/crosspump/sapienza_pd).
    features.py  trades of the pumped coin are filtered to side == 'buy' and grouped in chunks of 5 / 15 / 25 seconds;
                 rush order = several buy trades with the same millisecond timestamp (L9-12, L22-25); per chunk:
                 std_rush_order (L9-19), avg_rush_order (L22-32), std_trades (L35-41), std_volume (L44-50), avg_volume (L53-59),
                 std_price (L62-67), avg_price (L70-75), avg_price_max (L78-83) — each = pct_change() of a rolling statistic
                 over `rolling_freq` chunks — plus hour_sin / hour_cos / minute_sin / minute_cos (L116-119); the chunk
                 containing the pump start is labelled gt = 1.
    classifier.py RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1) on those 12 features (L13-29),
                 StratifiedKFold(5) cross_val_predict, precision / recall / F1 (L37-40).

Adaptation to bars (status official-features-reimplemented). The panel has bars, not trades, so
    rush bar     taker_imb > 0 (net taker buying) AND trade count above its rolling mean over the previous `rush_base` bars
                 (log_volume replaces the trade count when the panel has none);
    for each window w in {3, 12, 48} bars: relative one-bar change (pct_change, clipped) of
                 std / mean of the rush indicator, std of trades, std / mean of volume, std of log price; one-bar change of the
                 mean log price and of the mean bar-high log price over w bars;
    clock        the official sin / cos formulas on the bar close (UTC).
= 8 x 3 + 4 = 28 features -> RandomForest with the official hyper-parameters, trained on the rows of the train episodes
(chronological split instead of the official stratified 5-fold CV, which mixes past and future pumps). Entry [t, i] uses
bars <= t of episode i only; rows whose shortest-window features are not yet defined abstain (NaN). Sheet: sheets/lamorgia23.md.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from .common import XPBaseline, register
from ._feats_d import rolling, clock_features, feat_cols, bar_column, fit_forest, forest_scores, subsample_rows


def _relchg(a: np.ndarray, clip: float = 20.0, eps: float = 1e-9) -> np.ndarray:
    """pandas pct_change() of a (T, N) array along rows, with a guarded denominator and clipping (the official code leaves inf)."""
    prev = np.concatenate([np.full_like(a[:1], np.nan), a[:-1]], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((a - prev) / (np.abs(prev) + eps), -clip, clip)


def _diff(a: np.ndarray) -> np.ndarray:
    return a - np.concatenate([np.full_like(a[:1], np.nan), a[:-1]], 0)


@register("xp_lamorgia23")
class LaMorgia23Baseline(XPBaseline):
    name = "xp_lamorgia23"
    paper = ("La Morgia, Mei, Sassi, Stefa, 'The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency Manipulations', "
             "ACM Trans. Internet Technology 23(1), 2023, doi:10.1145/3561300, arXiv:2105.00733 (ICCCN 2020)")
    year = 2023
    venue = "ACM TOIT 23(1)"
    official_repo = "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset"
    status = "official-features-reimplemented"
    modalities = {"orderflow"}
    unspecified = [
        "rush orders need trade-level timestamps (same-millisecond buy trades); on bars: taker_imb > 0 and trade count above its rolling mean of the previous `rush_base` = 48 bars",
        "chunk sizes 5 / 15 / 25 s and rolling_freq chunks become rolling windows of {3, 12, 48} bars (model.windows)",
        "the official code keeps only buy-side trades; bars carry total volume / trades, the buy side enters through taker_imb in the rush indicator",
        "pct_change() of a rolling statistic can be inf / undefined when the previous value is 0: denominator |prev| + 1e-9 and clipping to +-20",
        "prices are log prices rebuilt from log_ret relative to the first bar of the episode; avg_price / avg_price_max use one-bar differences of the rolling mean",
        "evaluation protocol: official = stratified 5-fold CV over all pumps; here chronological train episodes only",
        "label: official = the single chunk containing the pump start; here the benchmark label on [anchor - lead, peak]",
    ]

    def _features(self, panel) -> np.ndarray:
        m = self.m
        T, N = self.T, self.N
        cols = feat_cols(panel, self.feat.shape[-1])
        ret = np.nan_to_num(bar_column(self.feat, cols, "log_ret")); logp = np.cumsum(ret, 0)
        hl = np.nan_to_num(bar_column(self.feat, cols, "hl_range"))
        vol = np.exp(np.clip(bar_column(self.feat, cols, "log_volume"), -50, 50))
        ltr = bar_column(self.feat, cols, "log_trades")
        trades = np.expm1(np.clip(ltr, 0, 50)) if np.isfinite(ltr).any() else vol
        self.diag["trade_count_source"] = "log_trades" if np.isfinite(ltr).any() else "log_volume (no trade count in the panel)"
        imb = bar_column(self.feat, cols, "taker_imb")
        base = rolling(trades, int(m.get("rush_base", 48)), "mean", int(m.get("rush_min_periods", 12)))
        base = np.concatenate([np.full((1, N), np.nan), base[:-1]], 0)                     # statistics of strictly earlier bars
        with np.errstate(invalid="ignore"):
            rush = ((imb > 0) & (trades > base)).astype(np.float64)
        rush[~np.isfinite(base) | ~np.isfinite(trades)] = np.nan
        out: List[np.ndarray] = []; names: List[str] = []
        for w in self.windows:
            mp = min(w, max(2, w // 2))
            out += [_relchg(rolling(rush, w, "std", mp)), _relchg(rolling(rush, w, "mean", mp)), _relchg(rolling(trades, w, "std", mp)), _relchg(rolling(vol, w, "std", mp)),
                    _relchg(rolling(vol, w, "mean", mp)), _relchg(rolling(logp, w, "std", mp)), _diff(rolling(logp, w, "mean", mp)), _diff(rolling(logp + 0.5 * hl, w, "mean", mp))]
            names += [f"{k}_{w}" for k in ("std_rush_order", "avg_rush_order", "std_trades", "std_volume", "avg_volume", "std_price", "avg_price", "avg_price_max")]
        clock = clock_features(panel, T, N)
        self.feat_names = names + ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
        self.n_core = 8                                                                    # features of the shortest window decide abstention
        return np.concatenate([np.stack(out, -1), clock], -1)

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        self.m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        m = self.m
        self.windows = sorted(int(w) for w in m.get("windows", [3, 12, 48]))
        self.X = self._features(panel)                                                     # (T, N, F)
        self.ready = np.isfinite(self.X[:, :, : self.n_core]).all(-1)                      # (T, N)
        y = self.labels_for(train_assets)
        t, i = self.rows_of(np.asarray(train_assets, dtype=int)) if len(train_assets) else (np.zeros(0, int), np.zeros(0, int))
        ok = (self.ready[t, i] & np.isfinite(y[t, i])) if len(t) else np.zeros(0, bool)
        t, i = t[ok], i[ok]; yy = y[t, i] if len(t) else np.zeros(0)
        keep = subsample_rows(yy, int(m.get("max_train_rows", 200_000)), int(m.get("seed", 1)))
        self.model, how = fit_forest(self.X[t[keep], i[keep]] if len(keep) else np.zeros((0, self.X.shape[-1])), yy[keep], kind="rf", n_estimators=int(m.get("n_estimators", 200)),
                                     max_depth=m.get("max_depth", 5), class_weight=m.get("class_weight"), seed=int(m.get("seed", 1)))
        if how != "rf":
            self.status = "reimplemented-fallback"
        self.diag.update({"classifier": how, "n_features": int(self.X.shape[-1]), "train_rows": int(len(keep)), "train_pos": int(yy[keep].sum()) if len(keep) else 0})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        s = forest_scores(self.model, self.X[t, i])
        s[~self.ready[t, i]] = np.nan
        return s
