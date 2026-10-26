"""D2 -- La Morgia, Mei, Sassi, Stefa, "The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency
Manipulations", ACM TOIT 23(1), 2023, doi:10.1145/3561300, arXiv:2105.00733 (ICCCN 2020 version: "Pump and Dumps in the
Bitcoin Era"). Registry key: xp_sapienza_pd.

Raw files (git clone of https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset, MIT, at <root>/crosspump/sapienza_pd),
structure as observed on the clone:
    pump_telegram.csv                       REQUIRED. header symbol,group,date,hour,exchange (date YYYY-MM-DD, hour HH:MM);
                                            1,110 rows: binance 520, yobit 276, cryptopia 210, bittrex 102, kucoin 2;
                                            20 group codes; 2017-07-05 .. 2021-01-18. Binance: 85 coins, 338 distinct
                                            (symbol, date, hour), 2018-01-03 .. 2021-01-18.
    groups.csv                              optional. header group_name,group_code,last_time_checked, telegram_link (13 rows)
    labeled_features/features_{5,15,25}S.csv.gz   optional. header date, pump_index, std_rush_order, avg_rush_order,
                                            std_trades, std_volume, avg_volume, std_price, avg_price, avg_price_max, hour_sin,
                                            hour_cos, minute_sin, minute_cos, symbol, gt. One block of chunks per pump
                                            (pump_index 0..326, about +-1 day around the pump); gt == 1 on exactly ONE chunk
                                            for 317 pumps = the manually corrected pump start (e.g. BRD 2018-12-30 17:00:00,
                                            POLY 2018-10-17 19:00:50, BRD 2019-10-03 14:59:35). The 25 s file is read
                                            (refine_file), only columns date / symbol / gt.
    features/, data/                        empty in the clone (the authors' downloader.py fills data/ from the Binance API).
Market data: Binance 1-minute klines (crosspump/binance_klines, see binance_fetch.py). The precomputed rush-order features
are NOT used as model inputs (their normalisation spans the whole +-1 day block, i.e. it is not point-in-time).

Episodes. One pump per (symbol, exchange) cluster of rows within merge_hours. Announcement time = date + hour, UTC.
Anchor = the corrected pump start (gt chunk) of the same symbol when one lies within refine_tolerance_minutes (60) of the
announcement, else the announcement time (episode['anchor_source'] = 'labeled_features_gt' | 'table'; the table time
stays in episode['announce_ns'] and is the time of the social event). Peak = argmax bar high within peak_window_bars.

Assumptions (also in meta['assumptions']).
  A1 date / hour columns and the chunk timestamps of labeled_features are UTC (the authors state UTC for the table; the
     labelled chunks agree with the table hours for the pumps inspected).
  A2 the table has no pair column: every Binance row is taken as a <SYMBOL>BTC pump (the paper studies BTC pairs); rows
     whose <SYMBOL>BTC klines do not exist are dropped and counted (meta['pumps_without_klines']).
  A3 the corrected start is used for the ANCHOR / label window only; it never enters a feature or a stream.
  A4 no message text in this dataset: social stream = announcement events only (source = stable id of the group code).
  A5 exchange filter 'binance' ('all' keeps other venues with Binance klines of the same symbol: cross-venue view).
  A6 wallet platform: optional <root>/crosspump/onchain/sapienza_pd_transfers.csv; absent = missing modality.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from . import common
from . import binance_fetch as bf

SOURCE_URL = "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset"
LICENCE = "MIT (LICENSE in the repository, Copyright (c) 2020 SystemsLab-Sapienza)"
PAPER = "La Morgia et al., ACM TOIT 23(1), 2023, doi:10.1145/3561300, arXiv:2105.00733"
SUBDIR = "sapienza_pd"
TABLE = "pump_telegram.csv"
REQUIRED = [f"crosspump/{SUBDIR}/{TABLE}"]
RECIPE = f"""git clone --depth 1 {SOURCE_URL} <root>/crosspump/{SUBDIR}
  (needs {TABLE}; optional groups.csv and labeled_features/features_25S.csv.gz for the corrected pump start)
python3 scripts/xp_prepare_data.py --dataset sapienza_pd --klines      # Binance 1m klines -> <root>/crosspump/binance_klines/<SYMBOL>BTC-1m-YYYY-MM.csv
python3 scripts/xp_prepare_data.py --dataset sapienza_pd --aggtrades   # optional: daily aggTrades for pump and control windows
ETHERSCAN_API_KEY=... python3 scripts/xp_prepare_data.py --dataset sapienza_pd --onchain   # optional wallet platform (ERC-20 / BEP-20 only)"""


def read_table(root: str, subdir: str = SUBDIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    common.require(root, [os.path.join("crosspump", subdir, TABLE)], RECIPE)
    df = pd.read_csv(os.path.join(root, "crosspump", subdir, TABLE), dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["symbol", "group", "date", "hour", "exchange"]
    if any(c not in df for c in need):
        raise ValueError(f"{TABLE}: expected columns {need}, got {list(df.columns)}")
    ts = pd.to_datetime(df["date"].str.strip() + " " + df["hour"].str.strip(), utc=True, errors="coerce")
    ok = ts.notna() & df["symbol"].notna()
    out = pd.DataFrame({"symbol": df["symbol"].str.strip().str.upper(), "exchange": df["exchange"].map(bf.norm_exchange), "channel": df["group"].str.strip(),
                        "peak_ns": np.nan})[ok].reset_index(drop=True)
    out["announce_ns"] = ts[ok].values.astype("datetime64[ns]").astype(np.int64)
    out["channel_id"] = [bf.stable_id(c) for c in out["channel"]]
    return out, {"rows": int(len(df)), "rows_bad_timestamp": int((~ok).sum()), "exchanges": out["exchange"].value_counts().to_dict(), "groups": int(out["channel"].nunique())}


def read_groups(root: str, subdir: str) -> Dict[str, Dict[str, str]]:
    p = os.path.join(root, "crosspump", subdir, "groups.csv")
    if not os.path.exists(p):
        return {}
    g = pd.read_csv(p, dtype=str)
    g.columns = [c.strip().lower() for c in g.columns]
    if "group_code" not in g:
        return {}
    return {str(r["group_code"]).strip(): {"name": str(r.get("group_name", "")).strip(), "link": str(r.get("telegram_link", "")).strip()} for _, r in g.iterrows()}


def read_corrected_starts(root: str, subdir: str, fname: str) -> Dict[str, np.ndarray]:
    """labeled_features/<fname>: rows with gt == 1 -> {symbol: sorted start times (ns, UTC)}. Empty dict when absent."""
    p = os.path.join(root, "crosspump", subdir, "labeled_features", fname)
    if not os.path.exists(p):
        return {}
    out: Dict[str, List[int]] = {}
    for ch in pd.read_csv(p, usecols=["date", "symbol", "gt"], chunksize=200000):
        g = ch[pd.to_numeric(ch["gt"], errors="coerce") == 1]
        if not len(g):
            continue
        ts = pd.to_datetime(g["date"], utc=True, errors="coerce")
        for s, t in zip(g["symbol"].astype(str).str.strip().str.upper(), ts):
            if pd.notna(t):
                out.setdefault(s, []).append(int(t.value))
    return {s: np.sort(np.array(v, dtype=np.int64)) for s, v in out.items()}


def load_pumps(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    c = {**bf.DEFAULTS, **cfg}
    root = str(c.get("root", "data_raw")); subdir = str(c.get("subdir", SUBDIR)); quote = str(c["quote"])
    rows, info = read_table(root, subdir)
    ex = bf.norm_exchange(c.get("exchange", "binance"))
    sel = rows[(rows["exchange"] == ex) | (ex == "all")]
    pumps = bf.merge_announcements(sel, float(c["merge_hours"]), quote, "sapienza_pd")
    starts = read_corrected_starts(root, subdir, str(c.get("refine_file", "features_25S.csv.gz"))) if c.get("refine_anchor", True) else {}
    tol = float(c.get("refine_tolerance_minutes", 60.0)) * 60 * 10 ** 9
    shifts = []
    for p in pumps:
        p["anchor_source"] = "table"; p["anchor_shift_s"] = 0.0
        st = starts.get(p["symbol"])
        if st is None or not len(st) or p["exchange"] != "binance":
            continue
        d = st - int(p["announce_ns"]); j = int(np.argmin(np.abs(d)))
        if abs(int(d[j])) <= tol:
            p["anchor_ns"] = int(st[j]); p["anchor_source"] = "labeled_features_gt"; p["anchor_shift_s"] = float(d[j]) / 1e9
            p["key"] = bf.episode_key(p["symbol"], quote, p["anchor_ns"]); shifts.append(float(d[j]) / 1e9)
    info.update({"rows_selected": int(len(sel)), "pumps_merged": len(pumps), "corrected_starts_available": int(sum(len(v) for v in starts.values())),
                 "anchors_refined": len(shifts), "anchor_shift_s_median": float(np.median(shifts)) if shifts else None,
                 "anchor_shift_s_abs_max": float(np.max(np.abs(shifts))) if shifts else None})
    pumps.sort(key=lambda d: d["anchor_ns"])
    return bf.cap_pumps(pumps, c.get("max_pumps")), rows, info


def prepare(root: str, args) -> None:
    bf.prepare_cex(root, args, dataset="sapienza_pd", subdir=SUBDIR, repo_url=SOURCE_URL, required=REQUIRED, pumps_fn=lambda c: load_pumps(c)[0], builder=SapienzaPD)


@common.register_dataset("xp_sapienza_pd")
class SapienzaPD(BaseDatasetBuilder):
    name = "xp_sapienza_pd"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self):
        cfg = dict(self.cfg); cfg.setdefault("horizons", list(self.horizons))
        root = str(cfg.get("root", "data_raw")); subdir = str(cfg.get("subdir", SUBDIR))
        pumps, rows, info = load_pumps(cfg)
        groups = read_groups(root, subdir)
        entities = {"channels": {int(bf.stable_id(g)): {"code": str(g), **groups.get(str(g), {})} for g in sorted(rows["channel"].unique())}}
        assumptions = [
            "A1 date / hour of pump_telegram.csv and the chunk timestamps of labeled_features are UTC.",
            "A2 no pair column: Binance rows are taken as <SYMBOL>BTC pumps; rows without such klines are dropped and counted.",
            "A3 anchor = manually corrected pump start (labeled_features gt chunk of the same symbol within refine_tolerance_minutes of the table time) when present, "
            "else the table time; the correction only moves the anchor / label window, the social event stays at the table time.",
            "A4 no message text: the social stream holds announcement events only (source = stable id of the group code).",
            "A5 exchange filter '%s' (with 'all', prices are still Binance klines of the same symbol)." % cfg.get("exchange", "binance"),
            "A6 the authors' precomputed rush-order features are not used (block-normalised, not point-in-time)."]
        return bf.build_cex_panel(cfg, pumps, name=self.name, dataset="sapienza_pd", source=f"{SOURCE_URL} ({PAPER})", licence=LICENCE, recipe=RECIPE,
                                  all_pump_times=bf.pump_times_by_symbol(rows), entities=entities, assumptions=assumptions, extra_meta={"table": info, "paper": PAPER})
