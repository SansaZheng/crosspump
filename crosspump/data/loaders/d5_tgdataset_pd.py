"""D5 Moura et al. bursts over the TGDataset -- Moura, Paoletti, Ferreira, Almeida, "Don't You Know, Pump it Up!",
ICWSM 2027 (accepted), arXiv:2609.01176; repo https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027
(MIT, commit 36f7091 of 2026-08-26). Raw messages = TGDataset (La Morgia, Mei, Mongardini, KDD 2025; Zenodo record
7640712, CC-BY-4.0, ~71 GB zipped / ~460 GB) -- OPTIONAL. Registry key: xp_tgdataset_pd.

Raw layout (observed on c225, 2026-10-03, clone at <root>/crosspump/moura):
    reference_data/anomaly_time_intervals.json
        {"1INCH": [{"date": "2020-12-25", "first_message_unix": 1608854402, "last_message_unix": 1608939178}], ...}
        681 coins, 810 bursty (coin, day) pairs, dates 2016-08-26 .. 2022-07-31, burst duration 0 .. 24 h (median 14.6 h)
    reference_data/coins_exchanges_and_trades_normalized.json
        {"GOAT": {"ascendex": ["USDT"], "kraken": ["EUR", "USD"], ...}, ...}   4,364 coins; exchange keys are lower case
        ('binance' lists 61 of the 681 burst coins = 78 bursts: 56 with a BTC pair, 22 USDT only)
    The 47 validated P&D events of the paper, the raw messages and the OHLCV are NOT in the repository.
    Stage 02 of the paper (02_anomaly_detection.ipynb): CA-CFAR over each coin's DAILY mention count with 5 training
    cells and 1 guard cell PER SIDE, P_fa 0.05 -- i.e. the detector looks up to 6 days into the FUTURE.

Episodes: one per (coin, burst day) whose coin has cached Binance klines (quote preference cfg quotes = [BTC, USDT];
the first quote with kline files wins, so quotes are mixed across episodes -- meta['quote_of']); anchor =
first_message_unix. PROXY LABEL (the event list is not public), meta['label_rule']:
    is_pump  iff  max close within peak_window_bars after the burst start >= (1 + min_rise) x mean close of the hour
             before the burst start (min_rise default 0.10)  AND  the close falls back by at least half of that rise
             within post_bars.
Other bursts are source controls (is_pump False, source 'burst_no_reaction'); sampled controls (common.sample_controls)
are added on top and stay away from every burst of their coin.

Social stream
  * without the TGDataset (default): a BURST-LEVEL stream -- one event per burst interval of the episode's coin that
    starts inside the window: time = first_message_unix, magnitude = log1p(duration in hours), text_score NaN,
    source 0. Its AVAILABILITY is governed by cfg burst_avail:
        'cfar' (default)   end of the burst day + cfar_future_days (6): the first moment the paper's two-sided CA-CFAR
                           could have flagged the day. With a 72 h + 8 h window this is after the window end for every
                           same-window burst, so the events are DROPPED and the social modality is honestly MISSING.
        'day_end'          end of the burst's UTC day (when its daily count and last_message_unix are complete);
                           still ignores the 6 future days the detector needs -> flagged meta['pit_warning'].
        'first_message'    avail = time (ORACLE: nobody knows at the first message that the day will be bursty);
                           for diagnostics only, flagged meta['pit_warning'].
  * with the extracted TGDataset messages (<root>/crosspump/tgdataset_pd/burst_messages.csv, columns
    coin,time_ns,channel_id,text -- written by prepare(root, {'tgdataset': True})): a message-level stream, one event
    per message mentioning the coin inside the window (magnitude 1, text_score = keyword score, source = hashed channel
    id, avail = message time), plus Panel.text. The burst-level events are then not added.

ASSUMPTIONS (also in meta['assumptions'])
 A1 proxy label (above); the paper's own validation (RDD + Bonferroni + placebo + DiD) is not reproduced.
 A2 the coin -> exchange map is a present-day snapshot; a burst is usable only if Binance klines exist for its month.
 A3 burst-level events are knowable only after the detector's look-ahead (burst_avail='cfar').
 A4 magnitude = log1p(duration hours) (the spec's 'log duration' is undefined for the zero-length bursts in the file).
 A5 the peak of a pump episode = argmax high within peak_window_bars after the burst start (common.episode_panel);
    the label rule uses closes.
 A6 the TGDataset extraction in prepare() follows the layout published with the dataset (channel objects with a
    `text_messages` map of {message, date}); it was NOT run here (the 71 GB archive is not on c225) -> UNVERIFIED.
 A7 the RoBERTa crypto-relatedness classifier of the paper (filipeasm18/roberta-binary-crypto-related-classifier) is an
    optional offline step that this loader never requires nor runs.
"""
from __future__ import annotations
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from . import common
from . import text_utils as tu

SOURCE_URL = "https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027"
LICENCE = "MIT (Moura et al. repository); optional TGDataset messages: CC-BY-4.0 (Zenodo 7640712)"
SUBDIR = os.path.join("crosspump", "moura")
BURSTS = os.path.join(SUBDIR, "reference_data", "anomaly_time_intervals.json")
MARKETS = os.path.join(SUBDIR, "reference_data", "coins_exchanges_and_trades_normalized.json")
TG_DIR = os.path.join("crosspump", "tgdataset")
TG_MESSAGES = os.path.join("crosspump", "tgdataset_pd", "burst_messages.csv")
REQUIRED = [BURSTS, MARKETS]
RECIPE = """D5 Moura et al. (ICWSM 2027, arXiv:2609.01176) bursts over the TGDataset (La Morgia et al., KDD 2025)
  git clone --depth 1 https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027 data_raw/crosspump/moura
  needs reference_data/anomaly_time_intervals.json and reference_data/coins_exchanges_and_trades_normalized.json
  python3 scripts/xp_prepare_data.py --dataset tgdataset_pd --klines      # Binance 1-minute klines (BTC, else USDT pair) of the burst coins
  OPTIONAL raw messages (TGDataset, Zenodo record 7640712, CC-BY-4.0, ~71 GB zipped / ~460 GB):
    download and unzip the channel json files into data_raw/crosspump/tgdataset/, then
    python3 scripts/xp_prepare_data.py --dataset tgdataset_pd --tgdataset # writes data_raw/crosspump/tgdataset_pd/burst_messages.csv (coin,time_ns,channel_id,text)
  OPTIONAL paper classifier: filipeasm18/roberta-binary-crypto-related-classifier (HuggingFace) -- never required by the loader
  optional wallet modality: data_raw/crosspump/onchain/tgdataset_pd_transfers.csv (key,time_ns,wallet_id,size_adv,size_raw)"""
ASSUMPTIONS = [
    "A1 PROXY label: is_pump iff max close within peak_window_bars after the burst start >= (1 + min_rise) x the pre-burst 1-hour mean close AND the close "
    "reverts by at least half of the rise within post_bars; Moura et al.'s 47 validated events are not released",
    "A2 coin -> exchange map is a present-day snapshot; a burst is kept only when Binance klines exist (quote preference BTC then USDT, mixed quotes)",
    "A3 burst-level social events become available only after the CA-CFAR look-ahead (burst_avail='cfar': burst day end + 6 days); events not available "
    "inside the window are dropped, so the social modality is missing unless TGDataset messages are extracted",
    "A4 magnitude of a burst event = log1p(duration in hours)",
    "A5 peak = argmax high within peak_window_bars after the burst start; the label rule uses closes",
    "A6 TGDataset extraction (prepare --tgdataset) follows the published layout and is unverified (archive not available on the cluster)",
    "A7 the paper's RoBERTa classifier is optional and never run by the loader",
]


# ------------------------------------------------------------------------------------------------ raw tables
def read_bursts(root: str) -> pd.DataFrame:
    with open(os.path.join(root, BURSTS), "r") as f:
        d = json.load(f)
    rows = []
    for coin, lst in d.items():
        for b in lst or []:
            try:
                rows.append({"symbol": str(coin).strip().upper(), "date": str(b["date"]), "first_ns": int(float(b["first_message_unix"])) * 10 ** 9,
                             "last_ns": int(float(b["last_message_unix"])) * 10 ** 9})
            except (KeyError, TypeError, ValueError):
                continue
    df = pd.DataFrame(rows, columns=["symbol", "date", "first_ns", "last_ns"])
    return df.sort_values(["first_ns", "symbol"]).reset_index(drop=True)


def binance_quotes(root: str) -> Dict[str, List[str]]:
    with open(os.path.join(root, MARKETS), "r") as f:
        d = json.load(f)
    out: Dict[str, List[str]] = {}
    for coin, exs in d.items():
        if not isinstance(exs, dict):
            continue
        for ex, quotes in exs.items():
            if tu.norm_exchange(ex) == "binance":
                out[str(coin).strip().upper()] = [str(q).upper() for q in (quotes if isinstance(quotes, (list, tuple)) else [quotes])]
    return out


def pick_quotes(root: str, symbols: Iterable[str], listed: Dict[str, List[str]], prefs: List[str], require_listed: bool) -> Dict[str, str]:
    """symbol -> first preferred quote that has cached kline files (and, with require_listed, is in the Binance map)."""
    kdir = os.path.join(root, tu.KLINE_SUBDIR)
    out = {}
    for s in symbols:
        for q in prefs:
            if require_listed and q not in listed.get(s, []):
                continue
            if common.find_kline_files(kdir, s, q):
                out[s] = q; break
    return out


def proxy_label(kl: pd.DataFrame, start_ns: int, bar_minutes: int, peak_window_bars: int, post_bars: int, min_rise: float, revert_frac: float) -> Dict[str, Any]:
    """The documented proxy rule on bar closes. kl: bar_minutes bars indexed by open_time ns. Returns the decision and
    its ingredients (base, peak close, rise, trough after the peak)."""
    step = int(bar_minutes) * tu.MIN_NS
    a0 = (int(start_ns) // step) * step
    n_pre = max(1, int(round(60 / bar_minutes)))
    pre = kl["close"].reindex(a0 + (np.arange(-n_pre, 0, dtype=np.int64)) * step).to_numpy(dtype=np.float64)        # bars closed before the burst bar opens
    fut = kl["close"].reindex(a0 + np.arange(0, int(post_bars) + 1, dtype=np.int64) * step).to_numpy(dtype=np.float64)
    res = {"is_pump": False, "base": np.nan, "peak_close": np.nan, "rise": np.nan, "reverted": False}
    if not np.isfinite(pre).any():
        return res
    base = float(np.nanmean(pre))
    win = fut[: int(peak_window_bars) + 1]
    if base <= 0 or not np.isfinite(win).any():
        return res
    k = int(np.nanargmax(win)); pk = float(win[k])
    rise = pk / base - 1.0
    after = fut[k + 1:]
    trough = float(np.nanmin(after)) if np.isfinite(after).any() else np.nan
    reverted = bool(np.isfinite(trough) and trough <= pk - float(revert_frac) * (pk - base))
    res.update({"base": base, "peak_close": pk, "rise": float(rise), "reverted": reverted, "is_pump": bool(rise >= float(min_rise) and reverted)})
    return res


def burst_episodes(root: str, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], pd.DataFrame, Dict[str, str]]:
    c = {**tu.DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    bursts = read_bursts(root)
    listed = binance_quotes(root)
    prefs = [str(q).upper() for q in (cfg.get("quotes") or ["BTC", "USDT"])]
    stats: Dict[str, Any] = {"bursts": int(len(bursts)), "burst_coins": int(bursts["symbol"].nunique()),
                             "bursts_coin_listed_on_binance": int(bursts["symbol"].isin(listed).sum())}
    quote_of = pick_quotes(root, sorted(bursts["symbol"].unique()), listed, prefs, bool(cfg.get("require_listed", True)))
    kept = bursts[bursts["symbol"].isin(quote_of)].reset_index(drop=True)
    stats["bursts_with_binance_klines"] = int(len(kept)); stats["coins_with_binance_klines"] = len(quote_of)
    kept = pd.DataFrame(tu.subsample_evenly(kept.to_dict("records"), cfg.get("max_pumps")), columns=kept.columns)
    bm, pre, post = int(c["bar_minutes"]), int(c["pre_bars"]), int(c["post_bars"])
    kdir = os.path.join(root, tu.KLINE_SUBDIR)
    months = tu.episode_months([{"symbol": r.symbol, "anchor_ns": r.first_ns} for r in kept.itertuples(index=False)], pre, post, bm, pad_days=1.0)
    pumps, ctrls = [], []
    n_nolabel = 0
    for sym, g in kept.groupby("symbol", sort=True):
        kl = tu.load_bars(kdir, sym, quote_of[sym], months.get(sym), bm)
        for r in g.itertuples(index=False):
            lab = proxy_label(kl, r.first_ns, bm, int(c["peak_window_bars"]), post, float(cfg.get("min_rise", 0.10)), float(cfg.get("revert_frac", 0.5))) \
                if kl is not None else {"is_pump": False, "base": np.nan, "peak_close": np.nan, "rise": np.nan, "reverted": False}
            n_nolabel += int(not np.isfinite(lab["base"]))
            e = {"key": tu.pump_key(sym, quote_of[sym], int(r.first_ns)), "symbol": sym, "quote": quote_of[sym], "exchange": "binance", "channel": None, "channels": [],
                 "announcements": [], "anchor_ns": int(r.first_ns), "peak_ns": None, "is_pump": bool(lab["is_pump"]), "control_of": None,
                 "source": "burst_reaction_proxy" if lab["is_pump"] else "burst_no_reaction", "burst_date": r.date, "burst_last_ns": int(r.last_ns),
                 "proxy": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in lab.items()}}
            (pumps if lab["is_pump"] else ctrls).append(e)
    stats.update({"bursts_evaluated": int(len(kept)), "proxy_pumps": len(pumps), "burst_no_reaction": len(ctrls), "bursts_without_pre_burst_bars": n_nolabel})
    return pumps, ctrls, stats, bursts, quote_of


# ------------------------------------------------------------------------------------------------ social stream
def read_tg_messages(root: str) -> Optional[pd.DataFrame]:
    p = os.path.join(root, TG_MESSAGES)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype={"text": object})
    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in ("coin", "time_ns", "text") if c not in cols]
    if missing:
        raise ValueError(f"{p}: columns {missing} missing; expected coin,time_ns,channel_id,text")
    out = pd.DataFrame({"symbol": df[cols["coin"]].astype(str).str.upper().str.strip(), "time_ns": pd.to_numeric(df[cols["time_ns"]], errors="coerce"),
                        "channel": df[cols["channel_id"]].astype(str) if "channel_id" in cols else "", "text": df[cols["text"]].fillna("").astype(str)})
    out = out[out["time_ns"].notna()].copy(); out["time_ns"] = out["time_ns"].astype(np.int64)
    return out.sort_values(["symbol", "time_ns"], kind="stable").reset_index(drop=True)


def _social_fn(bursts: pd.DataFrame, tg: Optional[pd.DataFrame], cfg: Dict[str, Any]):
    mode = str(cfg.get("burst_avail", "cfar")).lower()
    fut_days = float(cfg.get("cfar_future_days", 6.0))
    by_sym = {s: g for s, g in bursts.groupby("symbol")}
    tg_by = {s: g for s, g in tg.groupby("symbol")} if tg is not None else {}

    def fn(episodes: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
        ev, docs, ents = [], [], {}
        n_b = 0
        for e in episodes:
            lo, hi = ctx["window"][e["key"]]
            if tg is not None:
                g = tg_by.get(e["symbol"])
                if g is None:
                    continue
                t = g["time_ns"].to_numpy(); i0, i1 = int(np.searchsorted(t, lo, "left")), int(np.searchsorted(t, hi, "left"))
                if i1 <= i0:
                    continue
                sub = g.iloc[i0:i1]
                sid = np.array([tu.stable_id(ch, "tgd:") if str(ch) else 0 for ch in sub["channel"]], dtype=np.int64)
                for s_, ch in zip(sid, sub["channel"]):
                    if s_:
                        ents[int(s_)] = f"tgdataset channel {ch}"
                ev.append(pd.DataFrame({"key": e["key"], "time_ns": sub["time_ns"].to_numpy(), "magnitude": 1.0, "reach": np.nan,
                                        "text_score": tu.text_scores(sub["text"].to_numpy()), "source_id": sid}))
                has = sub["text"].str.strip().str.len().to_numpy() > 0
                docs.append(pd.DataFrame({"key": e["key"], "time_ns": sub["time_ns"].to_numpy()[has], "text": sub["text"].to_numpy()[has]}))
                continue
            g = by_sym.get(e["symbol"])
            if g is None:
                continue
            m = (g["first_ns"].to_numpy() >= lo) & (g["first_ns"].to_numpy() < hi)
            if not m.any():
                continue
            sub = g[m]; n_b += int(m.sum())
            first = sub["first_ns"].to_numpy().astype(np.int64); last = sub["last_ns"].to_numpy().astype(np.int64)
            day_end = (pd.to_datetime(sub["date"], utc=True).values.astype("datetime64[ns]").astype(np.int64) + tu.NS_PER_DAY)
            if mode == "first_message":
                avail = first
            elif mode == "day_end":
                avail = np.maximum(last, day_end)
            else:
                avail = np.maximum(last, day_end) + int(fut_days * tu.NS_PER_DAY)
            ev.append(pd.DataFrame({"key": e["key"], "time_ns": first, "avail_ns": avail, "magnitude": np.log1p(np.maximum(last - first, 0) / tu.NS_PER_H),
                                    "reach": np.nan, "text_score": np.nan, "source_id": 0}))
        if tg is not None:
            info = {"kind": "message-level stream from the extracted TGDataset messages mentioning the coin (magnitude 1, keyword text_score, source = hashed channel)",
                    "coverage_note": "TGDataset messages mentioning the episode's coin inside the window", "text_note": "TGDataset message text mentioning the coin"}
        else:
            info = {"kind": "BURST-LEVEL stream (one event per CA-CFAR bursty day of the coin; no raw messages): time = first_message_unix, "
                            "magnitude = log1p(duration hours), text_score NaN, source 0",
                    "burst_avail": mode, "bursts_inside_windows": n_b,
                    "coverage_note": ("burst-level stream; with burst_avail='cfar' a burst is knowable only at day end + "
                                      f"{fut_days:g} days (two-sided CA-CFAR), after the window end -> events dropped, social modality MISSING") if mode == "cfar"
                    else f"burst-level stream with burst_avail='{mode}' (NOT point-in-time: the burst flag needs later data)",
                    "text_note": "burst-level stream only (TGDataset messages not extracted)"}
        return {"events": pd.concat(ev, ignore_index=True) if ev else None, "docs": pd.concat(docs, ignore_index=True) if docs else None, "info": info, "entities": ents}
    return fn


# ------------------------------------------------------------------------------------------------ prepare
def _iter_tg_channels(path: str):
    """Channel objects of one TGDataset json file (published layout: a list of channel dicts, or a dict of them)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        d = json.load(f)
    if isinstance(d, dict) and "text_messages" in d:
        yield d
    elif isinstance(d, dict):
        for v in d.values():
            if isinstance(v, dict):
                yield v
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, dict):
                yield v


def extract_tg_messages(root: str, coins: Iterable[str], out_csv: str, t_lo_ns: int, t_hi_ns: int) -> int:
    """Scan <root>/crosspump/tgdataset/**/*.json and write the messages that mention a burst coin (uppercase token or
    cashtag) to out_csv (coin,time_ns,channel_id,text). UNVERIFIED against the real archive (assumption A6)."""
    files = sorted(glob.glob(os.path.join(root, TG_DIR, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No TGDataset json under {os.path.join(root, TG_DIR)}.\n{RECIPE}")
    coins = sorted(set(coins)); n = 0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    first = True
    for fp in files:
        rows = []
        for ch in _iter_tg_channels(fp):
            cid = ch.get("_id", ch.get("id", ch.get("username", "")))
            msgs = ch.get("text_messages") or {}
            it = msgs.values() if isinstance(msgs, dict) else msgs
            for m in it:
                if not isinstance(m, dict):
                    continue
                txt = m.get("message") or ""; d = m.get("date")
                if not txt or d is None:
                    continue
                try:
                    t_ns = int(float(d)) * 10 ** 9 if not isinstance(d, str) or d.replace(".", "", 1).isdigit() else int(pd.Timestamp(d, tz="UTC").value)
                except (TypeError, ValueError):
                    continue
                if not (t_lo_ns <= t_ns < t_hi_ns):
                    continue
                for coin in tu.extract_tickers(txt, coins):
                    rows.append({"coin": coin, "time_ns": t_ns, "channel_id": cid, "text": txt})
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, mode="w" if first else "a", header=first, index=False); first = False; n += len(rows)
    return n


def prepare(root: str, args: Any) -> None:
    """Called by scripts/xp_prepare_data.py only: clones the repository when absent; --klines fetches the Binance
    klines of the burst coins (BTC pair first, USDT for coins the map lists with USDT only); --tgdataset extracts the
    messages mentioning burst coins from an already downloaded TGDataset (the 71 GB download itself is manual)."""
    a = tu.args_dict(args)
    base = os.path.join(root, SUBDIR)
    if not os.path.exists(os.path.join(root, BURSTS)):
        import subprocess
        os.makedirs(os.path.dirname(base), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SOURCE_URL, base], check=True)
    bursts = read_bursts(root); listed = binance_quotes(root)
    c = dict(tu.DEFAULTS)
    if a.get("klines"):
        try:
            from . import binance_fetch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"crosspump/data/loaders/binance_fetch.py is not importable ({type(e).__name__}: {e}); fetch the klines manually:\n{RECIPE}")
        for quote in ("BTC", "USDT"):
            sub = bursts[[quote in listed.get(s, []) and (quote == "BTC" or "BTC" not in listed.get(s, [])) for s in bursts["symbol"]]]
            eps = [{"symbol": r.symbol, "anchor_ns": int(r.first_ns)} for r in sub.itertuples(index=False)]
            months = tu.episode_months(eps, int(c["pre_bars"]), int(c["post_bars"]), int(c["bar_minutes"]), pad_days=float(c["control_span_days"]) + 1.0)
            st = binance_fetch.fetch_klines({s: sorted(m) for s, m in months.items()}, os.path.join(root, tu.KLINE_SUBDIR), quote=quote, workers=int(a.get("workers") or 8))
            print(f"[tgdataset_pd] {quote} klines: {pd.Series(list(st.values()), dtype=object).value_counts().to_dict()} for {len(months)} symbols")
    if a.get("tgdataset"):
        pad = (int(c["pre_bars"]) + int(c["post_bars"]) + 1) * int(c["bar_minutes"]) * tu.MIN_NS
        n = extract_tg_messages(root, bursts["symbol"].unique(), os.path.join(root, TG_MESSAGES), int(bursts["first_ns"].min()) - pad, int(bursts["first_ns"].max()) + pad)
        print(f"[tgdataset_pd] wrote {n} coin-mentioning messages to {os.path.join(root, TG_MESSAGES)}")


# ------------------------------------------------------------------------------------------------ builder
@common.register_dataset("xp_tgdataset_pd")
class TGDatasetPDBuilder(BaseDatasetBuilder):
    name = "xp_tgdataset_pd"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self) -> Panel:
        cfg = dict(self.cfg)
        root = str(cfg.get("root", "data_raw"))
        common.require(root, REQUIRED, RECIPE)
        pumps, ctrls, stats, bursts, quote_of = burst_episodes(root, cfg)
        if not pumps and not ctrls:
            raise FileNotFoundError(f"xp_tgdataset_pd: no burst coin has Binance klines under {os.path.join(root, tu.KLINE_SUBDIR)} (stats: {stats}).\n"
                                    f"Download recipe:\n{RECIPE}")
        tg = read_tg_messages(root)
        mode = "messages" if tg is not None else str(cfg.get("burst_avail", "cfar")).lower()
        all_times = {s: np.sort(g["first_ns"].to_numpy().astype(np.int64)) for s, g in bursts.groupby("symbol")}
        build_cfg = {**cfg, "max_pumps": None}          # the cap was applied to the burst list (pumps and source controls together)
        extra = {"paper": "Moura, Paoletti, Ferreira, Almeida, ICWSM 2027, arXiv:2609.01176 (bursts); La Morgia, Mei, Mongardini, KDD 2025 (TGDataset)",
                 "social_platforms": "Telegram (platform 0): " + ("message-level (TGDataset extraction)" if tg is not None else "burst-level"),
                 "quote_of": dict(quote_of), "social_stream_level": "message" if tg is not None else "burst",
                 "label_rule": {"type": "proxy (Moura et al. event list not released)", "min_rise": float(cfg.get("min_rise", 0.10)),
                                "revert_frac": float(cfg.get("revert_frac", 0.5)),
                                "rule": "is_pump iff max close within peak_window_bars after the burst start >= (1 + min_rise) x mean close of the hour before the "
                                        "burst start AND the close reverts by >= revert_frac of the rise within post_bars; other bursts = controls (burst_no_reaction)",
                                "label_on": "[anchor - lead, peak]", "bars_to_peak": "peak - row (exact)", "peak": "argmax high within peak_window_bars after the burst start"}}
        if tg is None and mode != "cfar":
            extra["pit_warning"] = (f"burst_avail='{mode}': burst-level social events are stamped available before the two-sided CA-CFAR detector could have "
                                    "flagged the day -- NOT point-in-time; use for diagnostics only")
        return tu.build_panel(build_cfg, pumps, name=self.name, dataset="tgdataset_pd", source=SOURCE_URL, licence=LICENCE, recipe=RECIPE,
                              social_fn=_social_fn(bursts, tg, cfg), pump_times=all_times, fixed_controls=ctrls, quote_of=quote_of,
                              assumptions=ASSUMPTIONS, table_stats=stats, extra_meta=extra)
