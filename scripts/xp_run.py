#!/usr/bin/env python3
"""Run one (dataset, baseline, seed) through the CrossPump fold runner.

    python3 scripts/xp_run.py --dataset synth --baseline crosspump --seed 42 --device cpu --smoke
    python3 scripts/xp_run.py --dataset hu_pd --baseline xp_lamorgia23 --seed 2024

Writes results/crosspump/<dataset>/<baseline>/<tag|main>/seed<k>/{fold<j>.json, trades_fold<j>.csv, alerts_fold<j>.csv, summary.json}.
"""
from __future__ import annotations
import os
import time
from xp_common import base_parser, load_cfgs, build_panel, make_runner, import_all, save_json, log, BASELINES


def main():
    args = base_parser("CrossPump single run").parse_args()
    status = import_all(verbose=False)
    cfg, ds_cfg, bcfg = load_cfgs(args)
    if args.baseline not in BASELINES.keys():
        raise SystemExit(f"baseline {args.baseline!r} not registered; available: {[k for k in BASELINES.keys() if k.startswith('xp_') or k == 'crosspump']}\nmodule status: {status}")
    builder, panel = build_panel(ds_cfg)
    log.info(f"panel {args.dataset}: T={panel.T} N={panel.N} pumps={panel.meta.get('n_pumps')} coverage={panel.meta.get('coverage', {}).get('episodes_with_platform')}")
    runner = make_runner(cfg, ds_cfg, builder, panel, args)
    t0 = time.time()
    res = runner.run(args.baseline, bcfg, cfg["split"].get("max_folds"), tag=args.tag)
    a = res["agg"]
    f = lambda k, d=3: f"{a.get(k, float('nan')):.{d}f}"
    log.info(f"[{args.dataset} | {args.baseline} | seed {args.seed}] folds={a['n_folds']} AUC_PR={f('AUC_PR_rows')} P@FAR1%={f('P@FAR0.01')} "
             f"recall={f('event_recall')} early={f('share_early_vs_anchor', 2)} lead={f('lead_anchor_min_mean', 0)}min alarm={f('alarm_rate_normal_rows')} "
             f"loss_avoided={f('loss_avoided_bps', 0)}bps SR={f('SR', 2)} ({time.time() - t0:.0f}s)")
    d = os.path.join(args.out, args.dataset, args.baseline, args.tag or "main", f"seed{args.seed}")
    save_json({"agg": a, "config": cfg, "dataset_config": ds_cfg, "baseline_config": {k: v for k, v in bcfg.items() if k != "panel"}, "module_status": status,
               "panel_meta": {k: v for k, v in panel.meta.items() if k in ("dataset", "n_pumps", "n_controls", "coverage", "skipped", "bar_minutes", "pre_bars", "post_bars", "assumptions", "label_rule", "source", "licence")}},
              os.path.join(d, "summary.json"))


if __name__ == "__main__":
    main()
