"""B8 — Mahrous & Di Pietro, "PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on
Telegram", IEEE ICBC 2026, arXiv:2605.09431; data + code Zenodo doi:10.5281/zenodo.19616802. Registry key: xp_pumpsense26.

What the paper does. 283,017 Telegram messages from 39 groups (43 group folders in the Zenodo archive, 2,246 labelled
pump announcements, column `Pump` of chat_data/<group>/result11.csv). Messages are turned into message windows
(reproduce_paper_metrics.py::build_windows: window_size 7/11/15 messages, centered or causal) labelled
`nearby_pump` = 1 when a labelled announcement falls inside the window; LightGBM on TF-IDF (F1 0.79) and a fine-tuned
BGE-M3 (F1 0.83) classify windows; an LLM / seq2seq step extracts coin and exchange from positive windows.

Adaptation to the panel. The social stream is the message feed of the episode (one message = one social event with
mark = [1, reach, text_score, prior_involvement]); the "window" is the last `window_bars` bars of usable messages
(avail <= t). Window features (all availability-based, per episode, causal):
    counts         log1p(#messages) over window_bars and over long_bars, excess of the short window over the long rate
    text_score     sum / mean / max of mark[2] over the window (the hash-tfidf or classifier relevance score the loader gives)
    reach          mean mark[1] over the window; max prior involvement (mark[3]) over the window
    gaps           bars since the last message (capped), mean inter-message gap = window_bars / #messages
    embeddings     mean of panel.text document embeddings usable at t over the window (first `text_dims` dims) when the
                   panel carries text (hash_embed, CONVENTIONS.md §1), otherwise omitted
Classifier: LightGBM with the paper's hyper-parameters (lightgbm_classification.py L157-167) if importable, else sklearn
HistGradientBoosting with the same depth / leaves / rate (backend recorded in self.status / self.diag).
score = P(window contains a pump announcement); NaN (abstain) until the episode's first message is usable.
Convention sheet: crosspump/baselines/sheets/pumpsense26.md.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .common import XPBaseline, register
from ._feats_e import trailing_sum, trailing_max, raster_max, last_event_row, seen_mask, text_window_mean, fit_boosted_classifier, predict_proba1


@register("xp_pumpsense26")
class PumpSense26Baseline(XPBaseline):
    name = "xp_pumpsense26"
    paper = ("Mahrous, Di Pietro, 'PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on Telegram', IEEE ICBC 2026, "
             "arXiv:2605.09431; data/code doi:10.5281/zenodo.19616802")
    year = 2026
    venue = "IEEE ICBC 2026 (arXiv 2605.09431)"
    official_repo = "https://doi.org/10.5281/zenodo.19616802 (AhmedMahrous00/icbc_2026_reproducibility v1.1)"
    status = "reimplemented"
    modalities = {"social", "text"}
    unspecified = [
        "window definition: paper = 7/11/15 consecutive messages of one group (centered or causal); here the last `window_bars` bars of usable messages",
        "features: paper = TF-IDF(1-2 grams, 15k) of the concatenated window text; here window statistics of the social marks (+ hashed embeddings if the panel has text)",
        "positive label: paper = an annotated announcement inside the window; here the panel label (rows [anchor - lead, peak] of pump episodes)",
        "class balance: paper uses class_weight='balanced' (kept); subsampling none",
        "backend: LightGBM if importable, else sklearn HistGradientBoosting with matching depth / leaves / learning rate (recorded)",
    ]

    def _features(self) -> np.ndarray:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        W = int(m.get("window_bars", 12)); WL = int(m.get("long_bars", 48)); text_dims = int(m.get("text_dims", 32))
        T, N = self.T, self.N
        es = self.streams
        soc = (es.platform == 0) if len(es) else np.zeros(0, dtype=bool)
        cnt = self.counts[:, :, 0]
        n_w = trailing_sum(cnt, W); n_l = trailing_sum(cnt, WL)
        txt_sum = trailing_sum(self.mark_sums[:, :, 0, 2], W)
        reach_sum = trailing_sum(self.mark_sums[:, :, 0, 1], W)
        txt_max = trailing_max(raster_max(es, T, N, es.mark[:, 2] if len(es) else np.zeros(0), soc), W)
        prior_max = trailing_max(raster_max(es, T, N, es.mark[:, 3] if len(es) else np.zeros(0), soc), W)
        last = last_event_row(cnt)
        since = np.where(last >= 0, np.arange(T)[:, None] - last, WL).astype(np.float64)
        feats = [np.log1p(n_w), np.log1p(n_l), n_w - n_l * (W / float(WL)), txt_sum, txt_sum / np.maximum(n_w, 1.0), np.nan_to_num(txt_max),
                 reach_sum / np.maximum(n_w, 1.0), np.nan_to_num(prior_max), np.minimum(since, WL), W / np.maximum(n_w, 1.0)]
        X = np.stack(feats, -1)
        emb = text_window_mean(self.panel, T, N, W, dims=text_dims)
        if emb is not None:
            X = np.concatenate([X, emb], -1)
        self.diag.update({"window_bars": W, "long_bars": WL, "n_features": int(X.shape[-1]), "text_used": emb is not None})
        return X

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        self.X = self._features()                                                   # (T, N, F), per-episode causal
        self.seen_social = seen_mask(self.counts)[:, :, 0]                          # (T, N) causal coverage (a message was usable by t)
        train_assets = np.asarray(train_assets, dtype=int)
        y = self.labels_for(train_assets)
        t, i = self.rows_of(train_assets)
        Xtr, ytr = self.X[t, i], y[t, i]
        ok = np.isfinite(ytr) & self.seen_social[t, i]
        self.model = None
        if ok.sum() > 10 and 0 < ytr[ok].sum() < ok.sum():
            self.model, backend, reason = fit_boosted_classifier(Xtr[ok], ytr[ok], n_estimators=int(m.get("n_estimators", 100)), learning_rate=float(m.get("learning_rate", 0.1)),
                                                                 max_depth=int(m.get("max_depth", 6)), num_leaves=int(m.get("num_leaves", 31)), seed=int(m.get("seed", 42)),
                                                                 prefer=str(m.get("backend", "lightgbm")))
            self.diag.update({"backend": backend, "fallback_reason": reason, "train_rows": int(ok.sum()), "train_pos": int(ytr[ok].sum())})
            if backend != "lightgbm":
                self.status = "reimplemented-fallback"
        else:
            self.diag.update({"backend": "none", "fallback_reason": "no two-class training rows"})
        grid = np.full((self.T, self.N), np.nan)
        if self.model is not None:
            flat = self.X.reshape(-1, self.X.shape[-1])
            grid = predict_proba1(self.model, flat).reshape(self.T, self.N)
        grid[~self.seen_social] = np.nan                                             # no message usable yet / missing modality -> abstain
        self.score_grid = grid

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.score_grid[np.asarray(t, dtype=int), np.asarray(i, dtype=int)]
