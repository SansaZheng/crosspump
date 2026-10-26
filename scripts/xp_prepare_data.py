#!/usr/bin/env python3
"""Download / prepare the raw files of the CrossPump datasets (public sources only; every loader module documents its
own RECIPE and implements prepare(root, args)). Nothing is downloaded unless you run this script.

    python3 scripts/xp_prepare_data.py --list
    python3 scripts/xp_prepare_data.py --dataset hu_pd [--root data_raw] [--klines] [--aggtrades] [--onchain] [--embed]
    python3 scripts/xp_prepare_data.py --dataset all --klines

Keys: --klines fetch Binance 1-minute klines for the pumped symbols (data.binance.vision); --aggtrades daily aggTrades around
each pump; --onchain ERC-20 transfers via the Etherscan V2 API (needs env ETHERSCAN_API_KEY); --embed hashed text embeddings.
"""
from __future__ import annotations
import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xp_common import _discover, log

NAMES = {"hu_pd": "d1_hu_pd", "sapienza_pd": "d2_sapienza_pd", "xu_pumpolymp": "d3_xu_pumpolymp", "pumpsense": "d4_pumpsense", "tgdataset_pd": "d5_tgdataset_pd",
         "ardia_pd": "d6_ardia_pd", "mirtaheri": "d7_mirtaheri", "dex_wash": "d8_dex_wash", "rugpull1k": "d9_rugpull1k", "memechain": "d10_memechain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None); ap.add_argument("--root", default="data_raw"); ap.add_argument("--list", action="store_true")
    for k in ("klines", "aggtrades", "onchain", "embed", "tgdataset", "controls"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-pumps", type=int, default=None)
    a = ap.parse_args()
    if a.list or a.dataset is None:
        for k, m in NAMES.items():
            try:
                mod = importlib.import_module(f"crosspump.data.loaders.{m}")
                print(f"{k:14s} {getattr(mod, 'SOURCE_URL', '?')}\n{' ' * 15}licence: {getattr(mod, 'LICENCE', '?')}\n{' ' * 15}required: {getattr(mod, 'REQUIRED', '?')}")
            except Exception as e:  # noqa: BLE001
                print(f"{k:14s} loader unavailable: {type(e).__name__}: {e}")
        return
    names = list(NAMES) if a.dataset == "all" else a.dataset.split(",")
    for k in names:
        mod = importlib.import_module(f"crosspump.data.loaders.{NAMES[k]}")
        if not hasattr(mod, "prepare"):
            log.warning(f"{k}: loader has no prepare(); recipe:\n{getattr(mod, 'RECIPE', '')}"); continue
        log.info(f"{k}: prepare(root={a.root}) ...")
        mod.prepare(a.root, {kk: getattr(a, kk) for kk in ("klines", "aggtrades", "onchain", "embed", "tgdataset", "controls", "workers", "max_pumps")})


if __name__ == "__main__":
    main()
