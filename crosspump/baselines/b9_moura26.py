"""B9 — Moura, Paoletti, Ferreira, Almeida, "Don't You Know, Pump it Up!", ICWSM 2027 (accepted), arXiv:2609.01176;
code https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027. Registry key: xp_moura26.

What the paper does. A RoBERTa classifier keeps the crypto-related Telegram messages of the TGDataset (01_classify.ipynb);
for every coin the DAILY count of crypto-related mentions is scanned with a 1-D Cell-Averaging CFAR detector
(02_anomaly_detection.ipynb); the bursty days are then validated against market data with RDD / DiD (04_causality.ipynb).

CA-CFAR as published (02_anomaly_detection.ipynb, read on c225):
    cell 5   NUM_TRAINING_CELLS_PER_SIDE = 5, NUM_GUARD_CELLS_PER_SIDE = 1, P_FA = 0.05, USE_STL = False
    cell 17  ca_cfar_1d: alpha = N * (Pfa ** (-1 / N) - 1) with N = num_training_cells (per side) -> 4.1028 for N = 5;
             noise = mean of the left training window and of the right training window, averaged; "edge cells with only
             one valid window use that side's mean alone"; no valid training cell -> threshold = inf (never detected);
             detection iff signal[i] > alpha * noise.

Adaptation to the stacked-episode panel (no training, no labels):
    series      per episode, the availability-based per-bar count of social events (self.counts[:, :, 0]; a message is
                counted at the first bar at which it is usable). The loader's text_score (mark[2]) is NOT used as a
                filter: the paper's RoBERTa relevance filter is a preprocessing step of the loader.
    cell        `cell_bars` bars (default 3 = 15 minutes at 5-minute bars); the cell under test is the trailing cell
                ending at the decision bar t, evaluated at every bar (sliding).
    causal      the two-sided window uses future days. Here ONLY the left (past) window is used — exactly the notebook's
                own edge rule for a cell whose right window does not exist — with their guard (1 cell) and training
                (5 cells) sizes and their alpha. Training cells that would start before the window are dropped (partial
                left window, as in the notebook); no training cell yet -> abstain (NaN; notebook: threshold = inf).
    score       log((count + eps) / (alpha * noise + eps)), eps = `eps` (default 1.0). For any eps > 0 the score is
                > 0 iff count > alpha * noise, i.e. iff the notebook declares a detection; eps only keeps the ratio
                finite when the noise floor is zero (sparse 5-minute series).
    abstain     NaN until the first social event of the episode is usable (causal coverage: a missing social platform
                is a missing modality, never "zero mentions").
Convention sheet: crosspump/baselines/sheets/moura26.md.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .common import XPBaseline, register
from ._feats_e import ca_cfar_causal, cfar_alpha, seen_mask


@register("xp_moura26")
class Moura26Baseline(XPBaseline):
    name = "xp_moura26"
    paper = ("Moura, Paoletti, Ferreira, Almeida, 'Don't You Know, Pump it Up!', ICWSM 2027 (accepted), arXiv:2609.01176; "
             "code https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027")
    year = 2026
    venue = "ICWSM 2027 (accepted; arXiv 2609.01176)"
    official_repo = "https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027"
    status = "reimplemented"               # algorithm and parameters from 02_anomaly_detection.ipynb (cells 5 and 17)
    modalities = {"social"}
    unspecified = [
        "cell size on intraday bars: the paper uses DAILY mention counts; here a cell = `cell_bars` bars (default 3), sliding, ending at the decision bar",
        "causality: the notebook averages a left and a right training window; only the left (past) window is used (its own edge rule for a missing right window)",
        "score for ranking: the notebook outputs a 0/1 detection; here log((count + eps) / (alpha * noise + eps)), positive iff the notebook would detect",
        "zero noise floor on sparse series: additive eps = 1.0 inside the log ratio (does not move the detection boundary)",
        "relevance filter: the RoBERTa crypto-relatedness classifier is a loader-side preprocessing step, not applied here (every social event counts)",
        "STL detrending: off, as in the notebook default (USE_STL = False)",
    ]

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        cell = int(m.get("cell_bars", 3)); n_train = int(m.get("n_train_cells", 5)); n_guard = int(m.get("n_guard_cells", 1))
        p_fa = float(m.get("p_fa", 0.05)); eps = float(m.get("eps", 1.0))
        cnt = self.counts[:, :, 0]                                                   # (T, N) social events at their usable bar
        cut, thr, n_valid = ca_cfar_causal(cnt, cell=cell, n_train=n_train, n_guard=n_guard, p_fa=p_fa)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.log((cut + eps) / (thr + eps))                               # NaN while no training cell exists
        seen = seen_mask(self.counts)[:, :, 0]
        score[~seen] = np.nan                                                        # no social event usable yet -> abstain
        self.score_grid = score
        self.detect_grid = (cut > thr) & seen                                        # the notebook's 0/1 decision (diagnostics)
        self.diag.update({"cell_bars": cell, "n_train_cells": n_train, "n_guard_cells": n_guard, "p_fa": p_fa, "alpha_cfar": cfar_alpha(n_train, p_fa), "eps": eps,
                          "trained": False, "detection_rate": float(self.detect_grid.mean()) if self.detect_grid.size else 0.0,
                          "episodes_with_social": int(seen[-1].sum()) if seen.size else 0})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.score_grid[np.asarray(t, dtype=int), np.asarray(i, dtype=int)]
