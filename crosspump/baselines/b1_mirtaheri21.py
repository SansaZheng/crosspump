"""B1 — Mirtaheri, Abu-El-Haija, Morstatter, Ver Steeg, Galstyan, "Identifying and Analyzing Cryptocurrency
Manipulations in Social Media", IEEE Transactions on Computational Social Systems 8(3):607-617, 2021,
doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110. Registry key: xp_mirtaheri21.

What the paper does. Social-only, two platforms. (i) A Telegram message classifier separates pump announcements from
other chatter. (ii) For every pump attempt, Twitter activity about the coin in windows before the pump (tweet volume,
number of unique users, sentiment, bot share) plus coin descriptors feeds LR / RF / NN classifiers that predict whether
the attempt succeeds. No market microstructure and no on-chain data are used as inputs.

Adaptation to the stacked-episode panel. One row = (bar t, episode i). From the SOCIAL stream only, with events
counted at the first row at which they are usable (ceil(avail) <= t), over trailing windows {3, 12, 48, 144} bars:
    count         number of messages                               (tweet / message volume)
    unique        number of distinct known source ids               (unique users)
    reach         sum of mark[1] (reach_or_size)                    (audience)
    text_sum      sum of mark[2] (text_score)                       (sentiment / pump-language mass)
    text_mean     text_sum / count                                  (mean sentiment)
    pump_msgs     number of messages with text_score >= pump_thr    (output of the pump-message classifier)
-> sklearn RandomForest (fallback LogisticRegression) trained on the rows of the train episodes with the episode label
(1 on [anchor - lead, peak]). Rows of episodes without a social stream abstain (NaN; loader flag has_social, or — when
the loader gives no flag — no social event usable yet). All features are precomputed per episode in
_fit; entry [t, i] depends only on events of episode i with avail <= t. Convention sheet: sheets/mirtaheri21.md.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from .common import XPBaseline, register
from ._feats_d import CumRaster, unique_source_raster, CausalCoverage, fit_forest, forest_scores, subsample_rows, SOCIAL


@register("xp_mirtaheri21")
class Mirtaheri21Baseline(XPBaseline):
    name = "xp_mirtaheri21"
    paper = ("Mirtaheri, Abu-El-Haija, Morstatter, Ver Steeg, Galstyan, 'Identifying and Analyzing Cryptocurrency Manipulations in Social Media', "
             "IEEE Trans. Computational Social Systems 8(3):607-617, 2021, doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110")
    year = 2021
    venue = "IEEE TCSS 8(3)"
    official_repo = ""                      # no public code located; data release only (sheet)
    status = "reimplemented"
    modalities = {"social", "text"}
    unspecified = [
        "prediction unit: the paper classifies pump ATTEMPTS (success / failure) once per event; here every bar of every episode is scored with the episode label",
        "window lengths: the paper uses hours-to-days windows before the pump; here {3, 12, 48, 144} bars (config model.windows)",
        "sentiment / pump-message classifier: replaced by the loader's text_score mark (mark[2]); pump message = text_score >= 0.5 (model.pump_thr)",
        "bot share and coin descriptors (market cap, exchange listings) are not in the streams and are dropped",
        "classifier hyper-parameters are not reported: RandomForest 200 trees, max_depth 8, class_weight balanced_subsample",
        "Twitter and Telegram are merged into the single social platform of the stream schema",
    ]

    def _features(self) -> np.ndarray:
        m = self.m
        T, N = self.T, self.N
        s = self.streams
        thr = float(m.get("pump_thr", 0.5))
        cnt = CumRaster(self.counts[:, :, SOCIAL]); reach = CumRaster(self.mark_sums[:, :, SOCIAL, 1]); txt = CumRaster(self.mark_sums[:, :, SOCIAL, 2])
        if len(s):
            with np.errstate(invalid="ignore"):
                is_pm = ((s.platform == SOCIAL) & (np.nan_to_num(s.mark[:, 2], nan=-np.inf) >= thr)).astype(np.float64)
            pm = CumRaster(s.rasterise(T, N, pit=True, weights=is_pm)[:, :, SOCIAL])
        else:
            pm = CumRaster(np.zeros((T, N)))
        uniq = unique_source_raster(s, T, N, SOCIAL, self.windows)
        tt, ii = np.meshgrid(np.arange(T), np.arange(N), indexing="ij"); tt, ii = tt.ravel(), ii.ravel()
        cols: List[np.ndarray] = []; names: List[str] = []
        for w in self.windows:
            c = cnt.window(tt, ii, w); ts = txt.window(tt, ii, w)
            cols += [c, uniq[w].ravel(), reach.window(tt, ii, w), ts, ts / np.maximum(c, 1.0), pm.window(tt, ii, w)]
            names += [f"{k}_{w}" for k in ("count", "unique", "reach", "text_sum", "text_mean", "pump_msgs")]
        self.feat_names = names
        return np.stack(cols, -1).reshape(T, N, -1)

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        self.m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        m = self.m
        self.windows = [int(w) for w in m.get("windows", [3, 12, 48, 144])]
        self.cov = CausalCoverage(panel, self.counts, self.platforms).table(SOCIAL)          # (T, N) social known to exist at row t
        self.X = self._features()                                                           # (T, N, F), causal per entry
        y = self.labels_for(train_assets)
        t, i = self.rows_of(np.asarray(train_assets, dtype=int)) if len(train_assets) else (np.zeros(0, int), np.zeros(0, int))
        ok = self.cov[t, i] & np.isfinite(y[t, i]); t, i = t[ok], i[ok]
        yy = y[t, i] if len(t) else np.zeros(0)
        keep = subsample_rows(yy, int(m.get("max_train_rows", 200_000)), int(m.get("seed", 0)))
        self.model, how = fit_forest(self.X[t[keep], i[keep]] if len(keep) else np.zeros((0, self.X.shape[-1])), yy[keep], kind=str(m.get("classifier", "rf")),
                                     n_estimators=int(m.get("n_estimators", 200)), max_depth=m.get("max_depth", 8),
                                     class_weight=m.get("class_weight", "balanced_subsample"), seed=int(m.get("seed", 0)))
        if how != "rf":
            self.status = "reimplemented-fallback" if "lr" in how else self.status
        self.diag.update({"classifier": how, "n_features": int(self.X.shape[-1]), "train_rows": int(len(keep)), "train_pos": int(np.nansum(yy[keep])) if len(keep) else 0,
                          "episodes_with_social": int(self.cov[-1].sum())})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        s = forest_scores(self.model, self.X[t, i])
        s[~self.cov[t, i]] = np.nan
        return s
