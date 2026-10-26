"""Fold runner for the stacked-episode layout: chronological episode folds -> fit / predict every method through the shared
Warn2Trade event backtester (exit overlay) -> the shared trading / detection matrix + CrossPump metrics (lead time, precision
at fixed false-alarm rate, event recall, follower loss avoided, HR@k) -> results/crosspump/<dataset>/<method>/seed<k>/fold<j>.json.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch

from warn2trade.data.base import build_loader
from warn2trade.data.schema import Panel
from warn2trade.data.splits import Fold
from warn2trade.backtest.costs import CostModel
from warn2trade.backtest.engine import EventBacktester
from warn2trade.backtest import metrics as WM
from warn2trade.train.trainer import evaluate_outputs, to_device
from warn2trade.utils.registry import BASELINES
from warn2trade.utils.seed import set_seed
from .splits import EpisodeSplitter, EpisodeDataset
from .metrics import lead_time, precision_at_far, event_metrics, hit_rate_at_k
from .follower import follower_overlay


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set,)):
        return sorted(o)
    return str(o)


class XPRunner:
    def __init__(self, panel: Panel, horizons: Sequence[int], bars_per_year: int, data_cfg: Dict[str, Any], split_cfg: Dict[str, Any], cost_cfg: Dict[str, Any],
                 eval_cfg: Dict[str, Any], device: torch.device, out_dir: str, seed: int = 42, dataset_name: str = "dataset"):
        self.panel, self.horizons, self.bars_per_year = panel, list(horizons), int(bars_per_year)
        self.d, self.sp, self.c, self.e = dict(data_cfg), dict(split_cfg), dict(cost_cfg), dict(eval_cfg)
        self.device, self.out_dir, self.seed, self.dataset_name = device, out_dir, int(seed), dataset_name
        self.latency = max(int(self.d.get("latency", 1)), 1)
        self.exec_max = int(self.d.get("exec_max", 4))
        self.lookback = int(self.d.get("lookback", 48))
        self.bar_minutes = float(panel.meta.get("bar_minutes", 1))

    # ------------------------------------------------------------------ folds
    def folds(self, max_folds: Optional[int] = None) -> List[Fold]:
        purge_ns = int(self.sp.get("purge_hours", 0) * 3600 * 1e9)
        purge_ns = max(purge_ns, int(self.panel.T * self.bar_minutes * 60 * 1e9))          # at least one window length
        sp = EpisodeSplitter(self.panel.meta["anchor_ns"], float(self.sp.get("train_frac", 0.6)), float(self.sp.get("val_frac", 0.15)), purge_ns,
                             int(self.sp.get("n_folds", 1)), int(self.sp.get("min_train", 8)), max_folds)
        return list(sp)

    def cost_model(self, bps: float) -> CostModel:
        return CostModel(bps, self.c.get("use_half_spread", True), self.c.get("impact_eta", 0.1), self.c.get("delay_zeta", 0.05), self.c.get("notional", 0.05))

    def _loaders(self, fold: Fold, batch_size: int = 1024):
        rng = np.random.default_rng(self.seed)
        mk = lambda assets, neg: EpisodeDataset(self.panel, assets, self.lookback, self.horizons, self.latency, self.exec_max, int(self.d.get("vol_window", 60)), neg_subsample=neg, rng=rng)
        ds = {"train": mk(fold.train, self.d.get("neg_subsample")), "val": mk(fold.val, None), "test": mk(fold.test, None)}
        return {k: build_loader(v, batch_size, shuffle=False, seed=self.seed) for k, v in ds.items()}

    # ------------------------------------------------------------------ one fold
    def run_fold(self, name: str, base_cfg: Dict[str, Any], fold: Fold, save: bool = True, tag: Optional[str] = None) -> Dict[str, Any]:
        set_seed(self.seed + fold.k)
        cls = BASELINES.get(name)
        cfg = dict(base_cfg); cfg.update({"panel": self.panel, "lookback": self.lookback, "latency": self.latency, "bars_per_year": self.bars_per_year})
        loaders = self._loaders(fold)
        bl = cls(cfg, self.panel.dims(), self.horizons, self.device)
        t0 = time.time(); bl.fit(loaders["train"], loaders["val"]); fit_sec = time.time() - t0
        t0 = time.time()
        batches = [to_device(b, self.device) for b in loaders["test"]]
        outs = [bl.predict(b) for b in batches]
        pred_sec = time.time() - t0
        t = np.concatenate([b["t"].cpu().numpy() for b in batches]); i = np.concatenate([b["asset"].cpu().numpy() for b in batches])
        y = np.concatenate([b["y"].cpu().numpy() for b in batches])
        s = np.concatenate([o["score"].cpu().numpy() for o in outs]) if outs and "score" in outs[0] else np.concatenate([o["a_prob"].cpu().numpy() for o in outs])
        s = np.where(s <= -1.0, np.nan, s)
        alarm = np.concatenate([(1.0 - o["abstain"].cpu().numpy()) > 0.5 for o in outs]) if outs and "abstain" in outs[0] else np.concatenate([(o["w"].cpu().numpy() != 0) for o in outs])
        rows = pd.DataFrame({"t": t, "asset": i, "y": y})
        res: Dict[str, Any] = {"dataset": self.dataset_name, "baseline": name, "seed": self.seed, "fold": fold.k, "fit_sec": fit_sec, "pred_sec": pred_sec,
                               "n_train": int(len(fold.train)), "n_val": int(len(fold.val)), "n_test": int(len(fold.test)), "test_rows": int(len(t)),
                               "modalities": sorted(getattr(bl, "modalities", [])), "status": getattr(bl, "status", ""), "eval": {}}
        # ---- shared matrix at every cost level (exit overlay)
        grid = sorted(set([float(b) for b in self.e.get("cost_grid_bps", (1, 5, 10, 15, 30))] + [float(self.c.get("commission_bps", 5.0))]))
        bench = np.concatenate([[0.0], np.nan_to_num(np.nanmean(np.log(self.panel.mid[1:, fold.test]) - np.log(self.panel.mid[:-1, fold.test]), axis=1))])   # (T,) equal-weight test-episode return
        trades_main = None
        for bps in grid:
            bt = EventBacktester(self.horizons, self.latency, self.cost_model(bps))
            trades = pd.concat([bt.collect(o, b) for o, b in zip(outs, batches)], ignore_index=True)
            m = evaluate_outputs(trades, self.bars_per_year, self.panel.T, 0, bench, len(fold.test))
            m.update(follower_overlay(rows, alarm, self.panel, int(self.e.get("hold_late_bars", 12)), self.latency, bps))
            res["eval"][f"{bps:g}"] = m
            if bps == float(self.c.get("commission_bps", 5.0)):
                trades_main = trades
        # ---- CrossPump metrics (cost-free)
        xp: Dict[str, Any] = {}
        xp.update(lead_time(rows, alarm, self.panel.meta, int(self.e.get("early_window_bars", 0))))
        ok = np.isfinite(s)
        tc, ic = bl.rows_of(fold.val if len(fold.val) else fold.train)
        sc = bl._score_rows(tc, ic) if len(tc) else np.zeros(0); yc = self.panel.labels[tc, ic] if len(tc) else np.zeros(0)
        xp.update(precision_at_far(y[ok], s[ok], tuple(self.e.get("far_grid", (0.001, 0.005, 0.01, 0.05))), sc, yc))
        xp.update(event_metrics(rows, alarm, self.panel.meta, getattr(bl, "alpha", None)))
        xp.update(hit_rate_at_k(rows, s, self.panel.meta))
        xp["AUC_PR_rows"] = WM.auc_pr(y[ok], s[ok]) if ok.any() and y[ok].sum() > 0 else float("nan")
        xp["AUC_ROC_rows"] = WM.auc_roc(y[ok], s[ok]) if ok.any() and y[ok].sum() > 0 else float("nan")
        xp["abstain_rate"] = float((~ok).mean())
        res["xp"] = xp
        res["diag"] = {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)) for k, v in getattr(bl, "diag", {}).items()}
        if save:
            d = os.path.join(self.out_dir, self.dataset_name, name, tag or "main", f"seed{self.seed}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"fold{fold.k}.json"), "w") as fh:
                json.dump(res, fh, indent=2, default=_json_default)
            if trades_main is not None:
                trades_main.to_csv(os.path.join(d, f"trades_fold{fold.k}.csv"), index=False)
            pd.DataFrame({"t": t[alarm], "asset": i[alarm], "y": y[alarm], "score": s[alarm]}).to_csv(os.path.join(d, f"alerts_fold{fold.k}.csv"), index=False)
        res["_baseline"] = bl
        return res

    def run(self, name: str, base_cfg: Dict[str, Any], max_folds: Optional[int] = None, tag: Optional[str] = None) -> Dict[str, Any]:
        folds = self.folds(max_folds)
        if not folds:
            raise RuntimeError("no fold: too few episodes for the split configuration")
        results = [self.run_fold(name, base_cfg, f, tag=tag) for f in folds]
        return {"folds": results, "agg": aggregate(results, f"{float(self.c.get('commission_bps', 5.0)):g}")}


def aggregate(results: List[Dict[str, Any]], bps_key: str) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    keys = set()
    for r in results:
        keys |= set(r["eval"].get(bps_key, {}).keys()) | set(r.get("xp", {}).keys())
    for k in sorted(keys):
        vals = []
        for r in results:
            v = r["eval"].get(bps_key, {}).get(k, r.get("xp", {}).get(k))
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["n_folds"] = len(results)
    return agg
