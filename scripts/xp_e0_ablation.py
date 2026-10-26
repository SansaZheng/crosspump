#!/usr/bin/env python3
"""E0/E5 on the synthetic testbed: head-feature and platform ablations of CrossPump over seeds.
    python3 scripts/xp_e0_ablation.py --seeds 1 2 3 4 5 --out results/crosspump/e0_ablation.csv
Reports row AUC-PR, early-episode AUC-PR (max pre-label score per episode, pump vs control), P@FAR 1 %, share warned early, lead."""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from warn2trade.utils import DATASETS
from warn2trade.backtest.metrics import auc_pr
import crosspump.data.synthetic  # noqa: F401
from crosspump.models.crosspump import CrossPump
from crosspump.eval.splits import EpisodeSplitter, EpisodeDataset
from crosspump.eval.metrics import lead_time, precision_at_far
from crosspump.eval.follower import follower_overlay

VARIANTS = {"full": {}, "counts_only": {"model": {"head_features": "counts_only"}}, "hawkes_only": {"model": {"head_features": "hawkes_only"}},
            "llr_only": {"model": {"head_features": "llr_only"}}, "single_model": {"model": {"two_models": False}},
            "no_wallet": {"platforms": ["social", "orderflow"]}, "no_social": {"platforms": ["wallet", "orderflow"]}, "no_orderflow": {"platforms": ["social", "wallet"]},
            "social_only": {"platforms": ["social"]}, "orderflow_only": {"platforms": ["orderflow"]}}

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5]); ap.add_argument("--n-pumps", type=int, default=60)
ap.add_argument("--n-controls", type=int, default=120); ap.add_argument("--iters", type=int, default=150); ap.add_argument("--out", default="results/crosspump/e0_ablation.csv")
a = ap.parse_args()
rows = []
for seed in a.seeds:
    p = DATASETS.get("xp_synth")({"seed": seed, "n_pumps": a.n_pumps, "n_controls": a.n_controls}).build_panel()
    f = list(EpisodeSplitter(p.meta["anchor_ns"], 0.6, 0.15, purge_ns=int(p.T * 300e9), n_folds=1))[0]
    ds = EpisodeDataset(p, f.test, 24, p.meta["horizons"]); t, i = ds.index[:, 0], ds.index[:, 1]; y = p.labels[t, i]
    early = t < p.meta["pre_bars"] - p.meta["lead"]
    for name, over in VARIANTS.items():
        cfg = {"panel": p, "lookback": 24, "alpha": 0.01, "model": {"hawkes_iters": a.iters}}
        cfg["model"].update(over.get("model", {})); cfg.update({k: v for k, v in over.items() if k != "model"})
        m = CrossPump(cfg, p.dims(), p.meta["horizons"], torch.device("cpu")); m._fit(p, f.train, f.val)
        s = m._score_rows(t, i); ok = np.isfinite(s)
        tv, iv = m.rows_of(f.val); sv = m._score_rows(tv, iv); yv = p.labels[tv, iv]
        alarm = ok & (s > np.quantile(sv[(yv == 0) & np.isfinite(sv)], 0.99))
        df = pd.DataFrame({"t": t, "asset": i, "y": y})
        ep = {int(x): np.nanmax(np.where(ok, s, -np.inf)[(i == x) & early]) for x in np.unique(i)}
        ye = np.array([p.meta["is_pump"][x] for x in ep]); se = np.array(list(ep.values()))
        lt = lead_time(df, alarm, p.meta); fo = follower_overlay(df, alarm, p, 12)
        r = {"seed": seed, "variant": name, "row_AP": auc_pr(y[ok], s[ok]), "early_episode_AP": auc_pr(ye, se), "P@FAR1%": precision_at_far(y[ok], s[ok], s_cal=sv, y_cal=yv)["P@FAR0.01"],
             "event_recall": lt.get("event_recall"), "share_early": lt.get("share_early_vs_anchor"), "lead_min": lt.get("lead_anchor_min_mean"), "loss_avoided_bps": fo.get("loss_avoided_bps"), "pump_share": float(ye.mean())}
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out), exist_ok=True); df.to_csv(a.out, index=False)
print(df.groupby("variant").agg(["mean", "std"]).drop(columns=["seed"], level=0).round(3).to_string())
