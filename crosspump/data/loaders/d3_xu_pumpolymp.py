"""D3 -- Xu & Livshits, "The Anatomy of a Cryptocurrency Pump-and-Dump Scheme", USENIX Security 2019, arXiv:1811.10109.
Registry key: xp_xu_pumpolymp.

Raw files (git clone of https://github.com/xujiahuayz/pumpdump at <root>/crosspump/xu_pumpdump), structure as observed:
    code and data/rawpumpolymp.json   REQUIRED (25 MB). A JSON list of 435 PumpOlymp records (the paper analyses 412 after
                                      its own filtering) with keys currency, exchange (Cryptopia 216, Yobit 121, Binance 71,
                                      Bittrex 27), channelTitle, channelLink (92 channels), signalTime (ISO UTC, 'Z',
                                      optional fractional seconds; one corrupt record dated 1999), duration ('HH:MM'), max,
                                      priceBeforePump, volume, ourBuyPrice/Time, ourProfit, theoreticalBuyPrice/Time,
                                      theoreticalProfit, id (uuid), marketHistoryEntries (list of {rate, amount, timeStamp}
                                      trades from about 2 h before to about 15 min after the signal; no trade side).
                                      Binance records: 71 pumps, 38 coins, 2018-09-10 .. 2019-02-26.
    code and data/casestudy.csv       not used (tick trades of one Cryptopia case study, BVB).
    code and data/*.RData, tables.xlsx, outpost(s).csv   not used (R workspaces / regression tables / Telegram post counts).
Market data: Binance 1-minute klines (crosspump/binance_klines, see binance_fetch.py). Cryptopia / Yobit / Bittrex have no
free historical klines, so only Binance records are kept by default.

Episodes. Anchor = signalTime (the coin-release message as timestamped by PumpOlymp). Peak = time of the highest trade in
marketHistoryEntries at or after signalTime - 60 s (episode['peak_source'] = 'curated'), never earlier than the anchor;
when a record has no such trade the peak is derived (argmax bar high within peak_window_bars). Records of the same
(coin, exchange) within merge_hours are one pump relayed by several channels.

Assumptions (also in meta['assumptions']).
  A1 the records carry no pair field: Binance records are taken as <SYMBOL>BTC pumps (PumpOlymp tracked BTC markets; the
     recorded rates are BTC-denominated). meta['price_check'] reports how many episodes have a kline close just before
     the anchor within 25 % of priceBeforePump.
  A2 signalTime and the trade timestamps are UTC.
  A3 marketHistoryEntries is used for the PEAK (label side) only; it is not turned into an order-flow stream because it
     exists for pump records only (controls would lack it) and has no trade side.
  A4 records with signalTime before 2017 are corrupt and dropped (counted in meta['table']).
  A5 no message text: the social stream holds announcement events only (source = stable id of channelLink).
  A6 exchange filter 'binance' ('all' keeps other venues with Binance klines of the same symbol: cross-venue view).
  A7 wallet platform: optional <root>/crosspump/onchain/xu_pumpolymp_transfers.csv; absent = missing modality.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from . import common
from . import binance_fetch as bf

SOURCE_URL = "https://github.com/xujiahuayz/pumpdump"
LICENCE = "no licence file in the repository (research use; cite Xu & Livshits, USENIX Security 2019) -- redistribution unverified"
PAPER = "Xu & Livshits, USENIX Security 2019, arXiv:1811.10109"
SUBDIR = "xu_pumpdump"
TABLE = os.path.join("code and data", "rawpumpolymp.json")
REQUIRED = [f"crosspump/{SUBDIR}/code and data/rawpumpolymp.json"]
RECIPE = f"""git clone --depth 1 {SOURCE_URL} <root>/crosspump/{SUBDIR}
  (needs 'code and data/rawpumpolymp.json')
python3 scripts/xp_prepare_data.py --dataset xu_pumpolymp --klines      # Binance 1m klines -> <root>/crosspump/binance_klines/<SYMBOL>BTC-1m-YYYY-MM.csv
python3 scripts/xp_prepare_data.py --dataset xu_pumpolymp --aggtrades   # optional: daily aggTrades for pump and control windows
ETHERSCAN_API_KEY=... python3 scripts/xp_prepare_data.py --dataset xu_pumpolymp --onchain   # optional wallet platform (ERC-20 / BEP-20 only)"""
MIN_VALID_NS = int(pd.Timestamp("2017-01-01", tz="UTC").value)


def _ts(x: Any) -> float:
    try:
        t = pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return np.nan
    return float(t.value) if pd.notna(t) else np.nan


def _first(d: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] not in (None, ""):
            return low[n.lower()]
    return None


def _peak_from_history(entries: Any, signal_ns: int) -> float:
    """Time of the highest trade at or after signal - 60 s (first occurrence of the max rate); NaN when unavailable."""
    if not isinstance(entries, list) or not entries:
        return np.nan
    best_r, best_t = -np.inf, np.nan
    for m in entries:
        if not isinstance(m, dict):
            continue
        r = _first(m, ("rate", "price")); t = _ts(_first(m, ("timeStamp", "timestamp", "time")))
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(t) or t < signal_ns - 60 * 10 ** 9:
            continue
        if r > best_r or (r == best_r and t < best_t):
            best_r, best_t = r, t
    return max(best_t, float(signal_ns)) if np.isfinite(best_t) else np.nan


def read_table(root: str, subdir: str = SUBDIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """rawpumpolymp.json -> DataFrame[symbol, exchange, channel, channel_id, channel_title, announce_ns, peak_ns, pumpolymp_id,
    price_before, price_max, duration]. Defensive: accepts a list of records or a dict wrapping one."""
    common.require(root, [os.path.join("crosspump", subdir, TABLE)], RECIPE)
    with open(os.path.join(root, "crosspump", subdir, TABLE), "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        obj = lists[0] if lists else list(obj.values())
    rec = []; bad_time = 0; bad_rec = 0
    for x in obj if isinstance(obj, list) else []:
        if not isinstance(x, dict):
            bad_rec += 1; continue
        sym = _first(x, ("currency", "coin", "symbol")); t = _ts(_first(x, ("signalTime", "signal_time", "time")))
        if sym is None:
            bad_rec += 1; continue
        if not np.isfinite(t) or t < MIN_VALID_NS:
            bad_time += 1; continue
        link = _first(x, ("channelLink", "channel_link")); title = _first(x, ("channelTitle", "channel_title", "channel"))
        ch = str(link or title or "unknown").strip()
        num = lambda k: pd.to_numeric(_first(x, (k,)), errors="coerce")
        rec.append({"symbol": str(sym).strip().upper(), "exchange": bf.norm_exchange(_first(x, ("exchange",)) or ""), "channel": ch, "channel_id": bf.stable_id(ch),
                    "channel_title": str(title or ""), "announce_ns": int(t), "peak_ns": _peak_from_history(x.get("marketHistoryEntries"), int(t)),
                    "pumpolymp_id": str(_first(x, ("id",)) or ""), "price_before": float(num("priceBeforePump")), "price_max": float(num("max")),
                    "duration": str(_first(x, ("duration",)) or "")})
    cols = ["symbol", "exchange", "channel", "channel_id", "channel_title", "announce_ns", "peak_ns", "pumpolymp_id", "price_before", "price_max", "duration"]
    out = pd.DataFrame(rec, columns=cols)
    return out, {"records": int(len(obj)) if isinstance(obj, list) else 0, "records_bad_time": bad_time, "records_malformed": bad_rec,
                 "exchanges": out["exchange"].value_counts().to_dict(), "channels": int(out["channel"].nunique()), "records_with_curated_peak": int(out["peak_ns"].notna().sum())}


def load_pumps(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    c = {**bf.DEFAULTS, **cfg}
    root = str(c.get("root", "data_raw")); quote = str(c["quote"])
    rows, info = read_table(root, str(c.get("subdir", SUBDIR)))
    ex = bf.norm_exchange(c.get("exchange", "binance"))
    sel = rows[(rows["exchange"] == ex) | (ex == "all")]
    pumps = bf.merge_announcements(sel, float(c["merge_hours"]), quote, "xu_pumpolymp", extra_cols=("pumpolymp_id", "price_before", "price_max", "duration"))
    info.update({"rows_selected": int(len(sel)), "pumps_merged": len(pumps)})
    return bf.cap_pumps(pumps, c.get("max_pumps")), rows, info


def prepare(root: str, args) -> None:
    bf.prepare_cex(root, args, dataset="xu_pumpolymp", subdir=SUBDIR, repo_url=SOURCE_URL, required=REQUIRED, pumps_fn=lambda c: load_pumps(c)[0], builder=XuPumpOlymp)


@common.register_dataset("xp_xu_pumpolymp")
class XuPumpOlymp(BaseDatasetBuilder):
    name = "xp_xu_pumpolymp"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self):
        cfg = dict(self.cfg); cfg.setdefault("horizons", list(self.horizons))
        pumps, rows, info = load_pumps(cfg)
        ent = rows.drop_duplicates("channel")
        entities = {"channels": {int(r.channel_id): {"link": r.channel, "title": r.channel_title} for r in ent.itertuples(index=False)}}
        assumptions = [
            "A1 no pair field: Binance records are taken as <SYMBOL>BTC pumps; meta['price_check'] compares the kline close before the anchor with priceBeforePump.",
            "A2 signalTime and marketHistoryEntries timestamps are UTC.",
            "A3 curated peak = time of the highest marketHistoryEntries trade at or after signalTime - 60 s, clipped to >= anchor; label side only (not a stream: "
            "controls would lack it and trades carry no side).",
            "A4 records with signalTime before 2017 are corrupt and dropped.",
            "A5 no message text: the social stream holds announcement events only (source = stable id of channelLink).",
            "A6 exchange filter '%s' (with 'all', prices are still Binance klines of the same symbol)." % cfg.get("exchange", "binance")]
        panel = bf.build_cex_panel(cfg, pumps, name=self.name, dataset="xu_pumpolymp", source=f"{SOURCE_URL} ({PAPER})", licence=LICENCE, recipe=RECIPE,
                                   all_pump_times=bf.pump_times_by_symbol(rows), entities=entities, assumptions=assumptions, extra_meta={"table": info, "paper": PAPER})
        pre = int(panel.meta["pre_bars"]); n = ok = 0
        for i, e in enumerate(panel.meta["episodes"]):
            pb = e.get("price_before")
            if not e.get("is_pump") or pb is None or not np.isfinite(pb) or pb <= 0:
                continue
            m = panel.mid[max(0, pre - 12) : pre, i]; m = m[np.isfinite(m)]
            if len(m):
                n += 1; ok += int(abs(m[-1] / pb - 1.0) <= 0.25)
        panel.meta["price_check"] = {"episodes_checked": n, "close_within_25pct_of_priceBeforePump": ok,
                                     "note": "sanity check of assumption A1 (BTC pair); label-side information only, never a feature"}
        return panel
