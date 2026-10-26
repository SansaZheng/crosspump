"""D6 -- Ardia & Bluteau, "Twitter and cryptocurrency pump-and-dumps", International Review of Financial Analysis 95
(2024) 103479, doi:10.1016/j.irfa.2024.103479. Registry key: xp_ardia_pd.

Raw files (git clone of https://github.com/ArdiaD/PumpDump, branch master, at <root>/crosspump/ardia; data DOI
10.5281/zenodo.12019080), structure as observed on the clone:
    data/list_pd_events.csv   REQUIRED. header Date,Channel,Currency,Exchange,success,pump_date; 1,160 rows, all Binance,
                              95 channels (Telegram channel titles), 221 coins, 2019-01-02 .. 2022-02-02.
                              Date = ISO UTC announcement ('2019-01-07T14:27:13Z'); success in {0 (838), 1 (322)};
                              pump_date = minute of maximum volume ('2019-01-07T14:25:00Z'), empty when success == 0.
                              pump_date - Date: median -0.2 min, quartiles -1.0 / -0.1 min, min -60 min, max +2 min, i.e.
                              the volume peak usually falls in the minute of the announcement and is often stamped
                              BEFORE the recorded message time.
Market data: Binance 1-minute klines (crosspump/binance_klines, see binance_fetch.py).

Episodes. One pump per (coin, exchange) cluster of rows within merge_hours (several channels relay the same pump).
Curated peak = end of the pump_date minute. Anchor = min(earliest announcement, pump_date): when the curated volume peak
precedes the recorded message the label window must not start after its own peak (episode['anchor_source'] =
'pump_date_before_announcement', else 'table'); the social event stays at the recorded message time (so on those episodes
the announcement becomes usable only after the peak, which is what the data say). Unsuccessful events (success == 0) stay
pump episodes with episode['success'] = False and a derived peak; success_only: true drops them.

Assumptions (also in meta['assumptions']).
  A1 Date and pump_date are UTC (ISO 'Z').
  A2 the table has no pair column: events are taken as <SYMBOL>BTC pumps (the paper's sample is Binance BTC markets);
     events without such klines are dropped and counted.
  A3 success of a merged episode = any relayed row has success == 1; curated peak = first non-empty pump_date of the cluster.
  A4 no message text / follower counts in the table: social stream = announcement events only (source = stable id of the
     channel title), reach NaN.
  A5 wallet platform: optional <root>/crosspump/onchain/ardia_pd_transfers.csv; absent = missing modality.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from . import common
from . import binance_fetch as bf

SOURCE_URL = "https://github.com/ArdiaD/PumpDump"
LICENCE = ("no licence file; README terms: cite Ardia & Bluteau (2024, doi:10.1016/j.irfa.2024.103479) and reference the data DOI 10.5281/zenodo.12019080 "
           "in a footnote -- redistribution unverified")
PAPER = "Ardia & Bluteau, IRFA 95 (2024) 103479, doi:10.1016/j.irfa.2024.103479"
SUBDIR = "ardia"
TABLE = os.path.join("data", "list_pd_events.csv")
REQUIRED = [f"crosspump/{SUBDIR}/data/list_pd_events.csv"]
RECIPE = f"""git clone --depth 1 --branch master {SOURCE_URL} <root>/crosspump/{SUBDIR}
  (needs data/list_pd_events.csv)
python3 scripts/xp_prepare_data.py --dataset ardia_pd --klines      # Binance 1m klines -> <root>/crosspump/binance_klines/<SYMBOL>BTC-1m-YYYY-MM.csv
python3 scripts/xp_prepare_data.py --dataset ardia_pd --aggtrades   # optional: daily aggTrades for pump and control windows
ETHERSCAN_API_KEY=... python3 scripts/xp_prepare_data.py --dataset ardia_pd --onchain   # optional wallet platform (ERC-20 / BEP-20 only)"""


def read_table(root: str, subdir: str = SUBDIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    common.require(root, [os.path.join("crosspump", subdir, TABLE)], RECIPE)
    df = pd.read_csv(os.path.join(root, "crosspump", subdir, TABLE), dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["date", "channel", "currency", "exchange", "success", "pump_date"]
    if any(c not in df for c in need):
        raise ValueError(f"{TABLE}: expected columns {need}, got {list(df.columns)}")
    ts = pd.to_datetime(df["date"].str.strip(), utc=True, errors="coerce")
    pk = pd.to_datetime(df["pump_date"].astype(str).str.strip().replace({"": None, "nan": None, "NA": None}), utc=True, errors="coerce")
    ok = ts.notna() & df["currency"].notna()
    out = pd.DataFrame({"symbol": df["currency"].str.strip().str.upper(), "exchange": df["exchange"].map(bf.norm_exchange), "channel": df["channel"].astype(str).str.strip(),
                        "success": pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)})[ok].reset_index(drop=True)
    out["announce_ns"] = ts[ok].values.astype("datetime64[ns]").astype(np.int64)
    pkv = pk[ok]
    out["pump_minute_ns"] = np.where(pkv.notna().to_numpy(), pkv.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64), np.nan)
    out["peak_ns"] = out["pump_minute_ns"] + 60 * 10 ** 9 - 1            # end of the minute of maximum volume
    out["channel_id"] = [bf.stable_id(c) for c in out["channel"]]
    return out, {"rows": int(len(df)), "rows_bad_timestamp": int((~ok).sum()), "exchanges": out["exchange"].value_counts().to_dict(), "channels": int(out["channel"].nunique()),
                 "rows_success": int(out["success"].sum()), "rows_with_pump_date": int(out["pump_minute_ns"].notna().sum())}


def load_pumps(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    c = {**bf.DEFAULTS, **cfg}
    root = str(c.get("root", "data_raw")); quote = str(c["quote"])
    rows, info = read_table(root, str(c.get("subdir", SUBDIR)))
    ex = bf.norm_exchange(c.get("exchange", "binance"))
    sel = rows[(rows["exchange"] == ex) | (ex == "all")]
    pumps = bf.merge_announcements(sel, float(c["merge_hours"]), quote, "ardia_pd", ann_cols=("success", "pump_minute_ns"))
    n_early = 0
    for p in pumps:
        p["success"] = bool(any(int(a.get("success", 0)) == 1 for a in p["announcements"]))
        p["anchor_source"] = "table"
        mins = [float(a["pump_minute_ns"]) for a in p["announcements"] if a.get("pump_minute_ns") is not None and np.isfinite(a["pump_minute_ns"])]
        if mins and int(min(mins)) < int(p["anchor_ns"]):
            p["anchor_ns"] = int(min(mins)); p["anchor_source"] = "pump_date_before_announcement"; n_early += 1
            p["key"] = bf.episode_key(p["symbol"], quote, p["anchor_ns"])
        if p.get("peak_ns") is not None and int(p["peak_ns"]) < int(p["anchor_ns"]):
            p["peak_ns"] = int(p["anchor_ns"]) + 60 * 10 ** 9 - 1
    if c.get("success_only", False):
        pumps = [p for p in pumps if p["success"]]
    pumps.sort(key=lambda d: d["anchor_ns"])
    info.update({"rows_selected": int(len(sel)), "pumps_merged": len(pumps), "pumps_success": int(sum(p["success"] for p in pumps)),
                 "anchors_moved_to_pump_date": n_early, "success_only": bool(c.get("success_only", False))})
    return bf.cap_pumps(pumps, c.get("max_pumps")), rows, info


def prepare(root: str, args) -> None:
    bf.prepare_cex(root, args, dataset="ardia_pd", subdir=SUBDIR, repo_url=SOURCE_URL, required=REQUIRED, pumps_fn=lambda c: load_pumps(c)[0], builder=ArdiaPD, branch="master")


@common.register_dataset("xp_ardia_pd")
class ArdiaPD(BaseDatasetBuilder):
    name = "xp_ardia_pd"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self):
        cfg = dict(self.cfg); cfg.setdefault("horizons", list(self.horizons))
        pumps, rows, info = load_pumps(cfg)
        entities = {"channels": {int(bf.stable_id(ch)): str(ch) for ch in sorted(rows["channel"].unique())}}
        assumptions = [
            "A1 Date and pump_date are UTC.",
            "A2 no pair column: events are taken as <SYMBOL>BTC pumps; events without such klines are dropped and counted.",
            "A3 curated peak = end of the pump_date minute (minute of maximum volume); anchor = min(earliest announcement, pump_date) "
            "(episode['anchor_source']); the social event stays at the recorded message time.",
            "A4 success of a merged episode = any relayed row has success == 1; unsuccessful events stay pump episodes (episode['success'] False, derived peak) "
            "unless success_only is set.",
            "A5 no message text / follower counts: the social stream holds announcement events only, reach NaN."]
        panel = bf.build_cex_panel(cfg, pumps, name=self.name, dataset="ardia_pd", source=f"{SOURCE_URL} ({PAPER})", licence=LICENCE, recipe=RECIPE,
                                   all_pump_times=bf.pump_times_by_symbol(rows), entities=entities, assumptions=assumptions, extra_meta={"table": info, "paper": PAPER})
        panel.meta["success"] = np.array([bool(e.get("success", False)) for e in panel.meta["episodes"]])
        return panel
