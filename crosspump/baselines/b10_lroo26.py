"""B10 — Shoaei, Pishdar, Bag-Mohammadi, Karami, "LROO Rug Pull Detector: A Leakage-Resistant Framework Based on
On-Chain and OSINT Signals", arXiv:2603.11324 (Mar 2026); companion dataset TM-RugPull, arXiv:2602.21529.
Registry key: xp_lroo26.

What the paper does. One tabular row per token project (1,000 projects): on-chain behavioural metrics (holders,
transfers, concentration, liquidity activity) plus temporally aligned OSINT signals (social / search activity), all
extracted strictly BEFORE any liquidity withdrawal (the "leakage-resistant" part); a TabPFN classifier predicts rug pull.

Adaptation to the stacked-episode panel: one tabular row per (decision bar t, episode i), every feature causal
(events enter at their usable bar ceil(avail); bar statistics from bars < t):
    wallet (on-chain)   per window W in `windows`: #inflow events, #distinct wallet ids, sum of mark[0] (magnitude) and of
                        mark[1] (size vs ADV), share of the window's size carried by the largest wallet (concentration),
                        max prior involvement (mark[3]); NaN until the first wallet event of the episode is usable
                        (missing modality, never "zero activity").
    order flow          per window: #burst events and sum of burst magnitude (mark[0]); plus the causal EWMA z-score of
                        log_volume and taker_imb at t.
    social (OSINT)      per window: #messages, sum of text_score (mark[2]), sum of reach (mark[1]).
    coverage flags      wallet seen / social seen (causal).
Classifier: TabPFN (`from tabpfn import TabPFNClassifier`) on a class-stratified subsample of <= `max_train_rows`
(1,000) train rows; when tabpfn is not importable, cannot load its weights or fails, sklearn HistGradientBoosting on
the train rows (<= `fallback_max_rows`), recorded in self.status ('reimplemented-fallback') and self.diag. Labels
only through labels_for(train_assets). Scores are computed lazily for the requested rows and cached (TabPFN inference
is expensive). Rows with no platform seen yet abstain (NaN). Convention sheet: crosspump/baselines/sheets/lroo26.md.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from .common import XPBaseline, register
from ._feats_e import (causal_ewma_z, trailing_sum, trailing_max, raster_sum, raster_max, trailing_unique, seen_mask, fit_boosted_classifier,
                       predict_proba1, feature_names_index, DEFAULT_FEAT_INDEX)


def _fit_tabpfn(X: np.ndarray, y: np.ndarray, m: Dict[str, Any]) -> Tuple[Any, str]:
    """TabPFN classifier (optional dependency). Raises when the package / weights are unavailable."""
    from tabpfn import TabPFNClassifier                                            # optional dependency
    kw = {"device": str(m.get("device", "cpu")), "random_state": int(m.get("seed", 42))}
    if m.get("tabpfn_estimators") is not None:
        kw["n_estimators"] = int(m["tabpfn_estimators"])
    try:
        model = TabPFNClassifier(ignore_pretraining_limits=True, **kw)
    except TypeError:                                                              # older tabpfn signature
        try:
            model = TabPFNClassifier(**kw)
        except TypeError:
            model = TabPFNClassifier()
    model.fit(X, y)
    model.predict_proba(X[: min(4, len(X))])                                       # fail here (not at scoring time) if inference is broken
    return model, "tabpfn"


@register("xp_lroo26")
class Lroo26Baseline(XPBaseline):
    name = "xp_lroo26"
    paper = ("Shoaei, Pishdar, Bag-Mohammadi, Karami, 'LROO Rug Pull Detector: A Leakage-Resistant Framework Based on On-Chain and OSINT Signals', "
             "arXiv:2603.11324 (2026); companion dataset TM-RugPull arXiv:2602.21529")
    year = 2026
    venue = "arXiv 2603.11324 (Mar 2026)"
    official_repo = "https://github.com/mohamadpishdar/RugPull-Defender (data acquisition code of the companion dataset; no LROO training code found)"
    status = "reimplemented"
    modalities = {"wallet", "orderflow", "social"}
    unspecified = [
        "unit of prediction: paper = one row per project (features up to a pre-withdrawal cut-off); here one row per decision bar of an episode",
        "feature list: paper's on-chain / OSINT columns are project-level aggregates; here stream statistics over trailing windows {3, 12, 48} bars",
        "concentration measure: share of window inflow size carried by the largest wallet id (top-1 share)",
        "TabPFN context: <= 1,000 train rows, class-stratified (all positives up to half of the budget, negatives sampled uniformly, seeded)",
        "TabPFN version / ensemble size: package defaults (tabpfn_estimators overrides); CPU",
        "fallback when tabpfn is unavailable: sklearn HistGradientBoosting (100 iterations, depth 6, lr 0.1, balanced class weights)",
        "missing wallet platform: wallet features NaN (both backends accept NaN) rather than zeros",
    ]

    # ------------------------------------------------------------------ features
    def _features(self, panel, m: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        windows = [int(w) for w in m.get("windows", [3, 12, 48])]
        T, N = self.T, self.N
        es = self.streams
        E = len(es)
        plat = es.platform if E else np.zeros(0, dtype=np.int8)
        soc, wal = plat == 0, plat == 1
        seen = seen_mask(self.counts)                                              # (T, N, P) causal coverage
        self.seen = seen
        feats: List[np.ndarray] = []; names: List[str] = []
        size_adv = np.nan_to_num(es.mark[:, 1].astype(np.float64)) if E else np.zeros(0)
        prior_r = raster_max(es, T, N, es.mark[:, 3] if E else np.zeros(0), wal)
        wal_size_r = raster_sum(es, T, N, wal, size_adv)
        for W in windows:
            # --- wallet / on-chain behaviour
            uniq, conc, total = trailing_unique(es, T, N, W, mask=wal, size=size_adv)
            w_feats = [total, uniq, trailing_sum(self.mark_sums[:, :, 1, 0], W), trailing_sum(wal_size_r, W), conc, np.nan_to_num(trailing_max(prior_r, W))]
            for f, nm in zip(w_feats, ["n", "uniq", "mag_sum", "size_sum", "top1_share", "prior_max"]):
                f = f.astype(np.float64).copy(); f[~seen[:, :, 1]] = np.nan
                feats.append(f); names.append(f"wallet_{nm}_{W}")
            # --- order-flow bursts
            feats += [trailing_sum(self.counts[:, :, 2], W), trailing_sum(self.mark_sums[:, :, 2, 0], W)]
            names += [f"of_n_{W}", f"of_mag_sum_{W}"]
            # --- social / OSINT
            feats += [trailing_sum(self.counts[:, :, 0], W), trailing_sum(self.mark_sums[:, :, 0, 2], W), trailing_sum(self.mark_sums[:, :, 0, 1], W)]
            names += [f"social_n_{W}", f"social_text_sum_{W}", f"social_reach_sum_{W}"]
        cols = feature_names_index(panel, ["log_volume", "taker_imb"], DEFAULT_FEAT_INDEX) if "orderflow" in self.platforms else {}
        if cols:
            z = causal_ewma_z(self.feat[:, :, list(cols.values())], warmup=int(m.get("warmup", 30)), halflife=float(m.get("halflife", 60.0)))
            for k, nm in enumerate(cols):
                feats.append(z[:, :, k]); names.append(f"z_{nm}")
        feats += [seen[:, :, 1].astype(np.float64), seen[:, :, 0].astype(np.float64)]
        names += ["wallet_seen", "social_seen"]
        self.has_bar_z = bool(cols)
        return np.stack(feats, -1), names

    # ------------------------------------------------------------------ fit / score
    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        seed = int(m.get("seed", 42))
        self.X, self.feat_names = self._features(panel, m)                          # (T, N, F), per-episode causal
        self.any_seen = self.seen.any(-1)                                           # (T, N) some platform delivered an event by t
        self.chunk = int(m.get("predict_chunk", 2000))
        self.score_grid = np.full((self.T, self.N), np.nan); self.done = np.zeros((self.T, self.N), dtype=bool)
        self.model = None; self.backend = "none"
        train_assets = np.asarray(train_assets, dtype=int)
        y = self.labels_for(train_assets)
        t, i = self.rows_of(train_assets)
        Xtr, ytr = self.X[t, i], y[t, i]
        ok = np.isfinite(ytr) & self.any_seen[t, i]
        Xtr, ytr = Xtr[ok], ytr[ok].astype(int)
        self.diag.update({"n_features": int(self.X.shape[-1]), "windows": [int(w) for w in m.get("windows", [3, 12, 48])], "train_rows_available": int(len(ytr)),
                          "train_pos_available": int(ytr.sum())})
        if len(ytr) <= 10 or ytr.sum() == 0 or ytr.sum() == len(ytr):
            self.diag.update({"backend": "none", "fallback_reason": "no two-class training rows"})
            return
        rng = np.random.default_rng(seed)
        prefer = str(m.get("backend", "tabpfn"))
        reason = None
        if prefer == "tabpfn":
            cap = int(m.get("max_train_rows", 1000))
            pos = np.nonzero(ytr == 1)[0]; neg = np.nonzero(ytr == 0)[0]
            if len(ytr) > cap:
                pos = rng.choice(pos, size=min(len(pos), cap // 2), replace=False)
                neg = rng.choice(neg, size=min(len(neg), cap - len(pos)), replace=False)
            sub = np.sort(np.concatenate([pos, neg]))
            try:
                self.model, self.backend = _fit_tabpfn(Xtr[sub], ytr[sub], m)
                self.diag.update({"train_rows": int(len(sub)), "train_pos": int(ytr[sub].sum())})
            except Exception as e:                                                   # ImportError, missing weights, runtime failure
                reason = f"tabpfn unavailable ({type(e).__name__}: {e})"[:240]
                self.model = None
        if self.model is None:
            cap = int(m.get("fallback_max_rows", 200000))
            sub = np.arange(len(ytr))
            if len(sub) > cap:
                pos = np.nonzero(ytr == 1)[0]; neg = np.nonzero(ytr == 0)[0]
                neg = rng.choice(neg, size=max(cap - len(pos), 1), replace=False)
                sub = np.sort(np.concatenate([pos, neg]))
            self.model, self.backend, r2 = fit_boosted_classifier(Xtr[sub], ytr[sub], n_estimators=int(m.get("n_estimators", 100)), learning_rate=float(m.get("learning_rate", 0.1)),
                                                                  max_depth=int(m.get("max_depth", 6)), num_leaves=int(m.get("num_leaves", 31)), seed=seed, prefer="sklearn")
            self.status = "reimplemented-fallback"
            self.diag.update({"train_rows": int(len(sub)), "train_pos": int(ytr[sub].sum())})
            if prefer != "tabpfn":
                reason = f"backend '{prefer}' requested in the config"
        self.diag.update({"backend": self.backend, "fallback_reason": reason})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=int); i = np.asarray(i, dtype=int)
        if self.model is None:
            return np.full(len(t), np.nan)
        need = ~self.done[t, i]
        if need.any():
            flat = np.unique(t[need].astype(np.int64) * self.N + i[need])           # row-wise model: each row scored from its own features only
            tt, ii = flat // self.N, flat % self.N
            for s in range(0, len(flat), self.chunk):
                sl = slice(s, s + self.chunk)
                self.score_grid[tt[sl], ii[sl]] = predict_proba1(self.model, self.X[tt[sl], ii[sl]])
            self.done[tt, ii] = True
        s = self.score_grid[t, i].copy()
        s[~self.any_seen[t, i]] = np.nan                                            # nothing observed on any platform yet -> abstain
        return s
