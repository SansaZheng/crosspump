"""B6 — Bolz, Bründler, Kane, Patsias, Tessendorf, Gogol, Kim, Tessone, "Machine Learning-Based Detection of
Pump-and-Dump Schemes in Real-Time", arXiv:2412.18848 (Dec 2024); v2 "Real-Time Machine Learning Detection of
Telegram-Based Pump-and-Dump Schemes", ACM 2025, doi:10.1145/3733815.3764042. Registry key: xp_bolz24.

What the paper does. Telegram pump channels are monitored live; high-frequency order-book / trade data of every
candidate coin on the exchange is streamed; a Z-score model on order-flow features (buy volume, trade count, order-book
imbalance against a rolling baseline) flags the target coin seconds before the announced pump, and supervised
classifiers rank the candidate coins (top-5 hit rate 55.8 %, top-10 74.4 %).

Adaptation to the stacked-episode panel (one column = one episode, bars of `bar_minutes`):
    z-part      per episode, causal EWMA z-scores (statistics from bars < t, warm-up abstains) of the raw order-flow
                features taker_imb / log_volume / log_trades; the row's order-flow evidence is the max over the last
                `back` bars of the max over those features (the "recent Z-score peak").
    social part text evidence of the channel feed: sum of mark[2] (text_score) and log1p(count) of the social
                events usable at t (avail <= t) over the last `social_back` bars (availability-based rasters).
    combination logistic regression on [z_peak, log1p(social_count), social_text_sum] fitted on the train episodes
                (the paper's supervised ranking step), score = decision function; `combine: additive` gives the
                documented un-trained alternative z_peak + beta * social_text_sum.
Everything is precomputed per episode in _fit (each entry depends only on bars <= t and events with avail <= t of
its own episode), so _score_rows is a gather. Convention sheet: crosspump/baselines/sheets/bolz24.md.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .common import XPBaseline, register
from ._feats_e import causal_ewma_z, trailing_sum, trailing_max, coverage_mask, feature_names_index, DEFAULT_FEAT_INDEX


@register("xp_bolz24")
class Bolz24Baseline(XPBaseline):
    name = "xp_bolz24"
    paper = ("Bolz, Bründler, Kane, Patsias, Tessendorf, Gogol, Kim, Tessone, 'Machine Learning-Based Detection of Pump-and-Dump Schemes in "
             "Real-Time', arXiv:2412.18848 (2024); v2 'Real-Time Machine Learning Detection of Telegram-Based Pump-and-Dump Schemes', ACM 2025, "
             "doi:10.1145/3733815.3764042")
    year = 2024
    venue = "arXiv 2412.18848 (Dec 2024) / ACM 2025, doi:10.1145/3733815.3764042"
    official_repo = ""                     # no public code found (sheet: bolz24.md)
    status = "reimplemented"
    modalities = {"orderflow", "social", "text"}
    unspecified = [
        "rolling baseline of the Z-score: paper = seconds-level order-book stream; here EWMA with halflife 60 bars and a 30-bar warm-up (abstain)",
        "which order-flow features: paper lists buy volume / trades / book imbalance; here taker_imb, log_volume, log_trades (the panel's raw proxies)",
        "aggregation of several Z-scores: max over features and over the last `back` bars (a peak detector, as in their real-time trigger)",
        "how the Telegram signal enters the score: paper monitors channels for the announcement; here text_score sum + log1p(count) over `social_back` bars",
        "classifier for the combination: paper reports several supervised rankers; here a logistic regression on 3 features (additive fallback)",
    ]

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        warmup = int(m.get("warmup", 30)); halflife = float(m.get("halflife", 60.0))
        back = int(m.get("back", 3)); social_back = int(m.get("social_back", 6))
        feats = list(m.get("features", ["taker_imb", "log_volume", "log_trades"]))
        combine = str(m.get("combine", "logistic")); beta = float(m.get("beta", 1.0))
        cols = feature_names_index(panel, feats, DEFAULT_FEAT_INDEX)
        T, N = self.T, self.N
        # --- order-flow Z-score peak (per episode, causal)
        if cols:
            z = causal_ewma_z(self.feat[:, :, list(cols.values())], warmup=warmup, halflife=halflife)       # (T, N, F)
            with np.errstate(invalid="ignore"):
                zmax = np.fmax.reduce(z, axis=-1) if z.shape[-1] > 1 else z[..., 0]
            z_peak = trailing_max(zmax, back)
        else:
            z_peak = np.full((T, N), np.nan)
        # --- social / text evidence (availability-based rasters: an event counts from ceil(avail) on)
        soc_cnt = trailing_sum(self.counts[:, :, 0], social_back)
        soc_txt = trailing_sum(self.mark_sums[:, :, 0, 2], social_back)
        X = np.stack([z_peak, np.log1p(soc_cnt), soc_txt], -1)                                              # (T, N, 3)
        cov = coverage_mask(self.streams, N)
        self.has_social = cov[:, 0]
        self.has_orderflow = np.isfinite(z_peak).any(0) if cols else np.zeros(N, dtype=bool)
        # --- supervised combination on the train episodes only
        self.coef = np.array([1.0, 0.0, beta]); self.intercept = 0.0; backend = "additive"
        if combine == "logistic":
            y = self.labels_for(np.asarray(train_assets))
            t, i = self.rows_of(np.asarray(train_assets))
            Xtr = X[t, i]; ytr = y[t, i]
            ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
            if ok.sum() > 10 and 0 < ytr[ok].sum() < ok.sum():
                try:
                    from sklearn.linear_model import LogisticRegression
                    mu = Xtr[ok].mean(0); sd = Xtr[ok].std(0) + 1e-6
                    lr = LogisticRegression(C=float(m.get("C", 1.0)), class_weight="balanced", max_iter=500)
                    lr.fit((Xtr[ok] - mu) / sd, ytr[ok].astype(int))
                    self.coef = lr.coef_[0] / sd; self.intercept = float(lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum()); backend = "logistic"
                except Exception as e:                                                                       # pragma: no cover
                    self.diag["combine_fallback"] = f"{type(e).__name__}: {e}"[:200]
        score = self.intercept + X @ self.coef                                                                # NaN where z_peak is NaN (warm-up)
        score[:, ~(self.has_social | self.has_orderflow)] = np.nan                                           # no modality -> abstain
        self.score_grid = score
        self.diag.update({"combine": backend, "coef": [float(c) for c in self.coef], "intercept": float(self.intercept), "feature_cols": cols,
                          "back": back, "social_back": social_back, "warmup": warmup, "halflife": halflife,
                          "episodes_with_social": int(self.has_social.sum()), "episodes_with_orderflow": int(self.has_orderflow.sum())})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.score_grid[np.asarray(t, dtype=int), np.asarray(i, dtype=int)]
