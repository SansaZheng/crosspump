"""Shared plumbing for scripts/xp_*.py (CrossPump track). Imports every crosspump loader / baseline module tolerantly
(a missing optional dependency or an unfinished module disables one entry, never the script)."""
from __future__ import annotations
import argparse
import glob
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from warn2trade.utils import load_config, deep_merge, apply_overrides, DATASETS, BASELINES, get_logger
from warn2trade.train import resolve_device
from crosspump.eval.runner import XPRunner

log = get_logger("xp")
CFG_DIR = os.path.join(ROOT, "configs", "crosspump")
RESULTS = os.path.join(ROOT, "results", "crosspump")
CORE_MODULES = ("crosspump.data.synthetic", "crosspump.baselines.common", "crosspump.models.crosspump")


def _discover(pattern: str, pkg: str) -> List[str]:
    return [f"{pkg}.{os.path.basename(f)[:-3]}" for f in sorted(glob.glob(os.path.join(ROOT, *pkg.split("."), pattern))) if not os.path.basename(f).startswith("_")]


def import_all(verbose: bool = False) -> Dict[str, str]:
    status = {}
    mods = list(CORE_MODULES) + _discover("d*_*.py", "crosspump.data.loaders") + _discover("b*_*.py", "crosspump.baselines")
    for m in mods:
        try:
            importlib.import_module(m); status[m] = "ok"
        except Exception as e:  # noqa: BLE001
            status[m] = f"unavailable: {type(e).__name__}: {e}"
            if verbose:
                log.warning(f"{m}: {status[m]}")
    return status


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dataset", default="synth", help="configs/crosspump/datasets/<name>.yaml")
    p.add_argument("--baseline", default="crosspump", help="registry key (crosspump, xp_counts_rule, xp_hu23, ...)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true", help="merge the baseline yaml's smoke: block and the dataset's smoke sizes")
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides a.b=c; prefix baseline. for the baseline config")
    return p


def load_cfgs(args) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = load_config(os.path.join(CFG_DIR, "default.yaml"))
    ds_path = os.path.join(CFG_DIR, "datasets", f"{args.dataset}.yaml")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"no dataset config {ds_path}")
    ds_cfg = load_config(ds_path)
    for k in ("lookback", "latency", "exec_max", "vol_window", "neg_subsample"):
        if k in ds_cfg:
            cfg["data"][k] = ds_cfg[k]
    if "split" in ds_cfg:
        cfg["split"] = deep_merge(cfg["split"], ds_cfg["split"])
    if "horizons" in ds_cfg:
        cfg["horizons"] = list(ds_cfg["horizons"])
    if args.smoke and "smoke" in ds_cfg:
        ds_cfg = deep_merge(ds_cfg, ds_cfg["smoke"])
    key = args.baseline[3:] if args.baseline.startswith("xp_") else args.baseline
    bl_path = os.path.join(CFG_DIR, "baselines", f"{key}.yaml")
    bcfg = load_config(bl_path) if os.path.exists(bl_path) else {}
    if args.smoke and "smoke" in bcfg:
        bcfg = deep_merge(bcfg, bcfg["smoke"])
    bcfg.pop("smoke", None)
    bcfg = deep_merge({"n_exec": cfg["policy"]["n_exec"], "w_max": cfg["policy"]["w_max"], "hold_index": cfg["policy"]["hold_index"]}, bcfg)
    cfg = apply_overrides(cfg, [o for o in args.override if not o.startswith("baseline.")])
    bcfg = apply_overrides(bcfg, [o[len("baseline."):] for o in args.override if o.startswith("baseline.")])
    if args.max_folds is not None:
        cfg["split"]["max_folds"] = args.max_folds
    return cfg, ds_cfg, bcfg


def build_panel(ds_cfg: Dict[str, Any], cache: bool = True):
    """Build (or load from results/crosspump/panel_cache/<name>-<cfg hash>.pkl) the dataset panel."""
    import hashlib, pickle
    builder = DATASETS.get(ds_cfg["name"])(ds_cfg)
    key = hashlib.sha256(json.dumps(ds_cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = os.path.join(RESULTS, "panel_cache", f"{ds_cfg['name']}-{key}.pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as fh:
            return builder, pickle.load(fh)
    panel = builder.build_panel()
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(panel, fh, protocol=4)
        os.replace(tmp, path)
    return builder, panel


def make_runner(cfg, ds_cfg, builder, panel, args) -> XPRunner:
    horizons = cfg.get("horizons") or list(panel.meta.get("horizons", builder.horizons))
    bpy = int(ds_cfg.get("bars_per_year", panel.meta.get("bars_per_year", builder.bars_per_year)))
    return XPRunner(panel, horizons, bpy, cfg["data"], cfg["split"], cfg["cost"], cfg["eval"], resolve_device(args.device), args.out, args.seed, args.dataset)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
