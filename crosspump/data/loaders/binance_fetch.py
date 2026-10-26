"""Shared Binance market-data layer for the group-A CEX loaders (D1 hu_pd, D2 sapienza_pd, D3 xu_pumpolymp, D6 ardia_pd).

It is the only module the four loaders share, so it holds three things (nothing here fabricates data: every function
downloads a public file or transforms tables read from disk):

1. Downloaders -- used ONLY by each loader's prepare(root, args), which scripts/xp_prepare_data.py calls. build_panel()
   never downloads.
       fetch_klines(symbols_months, cache_dir, quote='BTC', workers=8)
           monthly 1-minute klines from https://data.binance.vision/data/spot/monthly/klines/<SYM><QUOTE>/1m/
           <SYM><QUOTE>-1m-YYYY-MM.zip (12 header-less columns open_time, open, high, low, close, volume, close_time,
           quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore; open_time in ms, in us from 2025-01),
           unzipped to <cache_dir>/<SYM><QUOTE>-1m-YYYY-MM.csv; existing files are skipped; 404s (delisted pairs /
           months before listing) are logged to <cache_dir>/manifest.json under "missing".
       fetch_aggtrades(symbol, dates, cache_dir, quote='BTC', workers=4)
           daily aggTrades (https://data.binance.vision/data/spot/daily/aggTrades/<SYM><QUOTE>/<SYM><QUOTE>-aggTrades-
           YYYY-MM-DD.zip; columns agg_id, price, qty, first_id, last_id, time, is_buyer_maker, best_match) unzipped to
           <cache_dir>/<SYM><QUOTE>-aggTrades-YYYY-MM-DD.csv.
       prepare_market_data(root, pumps, args)      klines (+ optional aggTrades) for a list of pump episodes
       prepare_onchain(root, dataset, pumps, args)  OPTIONAL wallet platform: ERC-20 / BEP-20 transfers from the
           Etherscan V2 API (env ETHERSCAN_API_KEY) for tokens whose contract is listed in
           <root>/crosspump/onchain/token_contracts.csv (symbol,chain,contract[,coingecko_id,n_candidates]); the contract
           table is built from the free CoinGecko /coins/list?include_platform=true (one call) and /coins/{id} endpoints
           (rate limited). Writes <root>/crosspump/onchain/<dataset>_transfers.csv (key,time_ns,wallet_id,size_adv,
           size_raw[,to_exchange]). Coverage is PARTIAL by construction: only tokens with an Ethereum / BSC contract;
           native-chain coins (and most 2018-2019 Binance alts) have no transfers here. Exchange-deposit labelling needs
           the optional <root>/crosspump/onchain/exchange_addresses.csv (address,label).

2. Cache readers: klines_dir / aggtrades_dir / onchain_dir, months_for_anchors, load_bars (through
   common.find_kline_files / read_binance_klines / resample_bars, restricted to the months a dataset needs),
   read_aggtrades + aggtrades_bursts (1-minute taker-buy imbalance / volume bursts, causal EWMA z-scores).

3. build_cex_panel(cfg, pumps, ...): the stacked-episode Panel + EventStreams assembly shared by the four loaders:
   bars -> controls (common.sample_controls: same coin at a pump-free time, other coin at the same time; controls whose
   WINDOW contains a listed pump of their coin are redrawn / dropped) -> common.episode_panel -> social stream (one
   event per announcement row at its own time, source = stable hash of the channel id, mark [1, NaN, 1.0, prior]; plus
   loader-provided message events through `social_fn`, which also serves the control windows of the watched channel)
   -> optional wallet stream (<root>/crosspump/onchain/<dataset>_transfers.csv; ABSENT = missing modality, never zeros)
   -> order-flow stream from ONE source for every episode (aggTrades bursts only when the window days of ALL episodes
   are cached, else bursts derived causally from the bar features: a per-episode mix would let a model tell pumps from
   controls by the stream's provenance) -> optional Panel.text (common.hash_embed; a message is placed at the first
   row at which it is usable, ceil(time + social lag), and avail_text = that row) -> meta.
   prepare_cex(...) is the shared prepare(root, args) body: git clone of the source repo when the event table is
   missing, klines, then (optional) aggTrades / on-chain transfers for pumps AND controls.

PIT rule (crosspump/CONVENTIONS.md §2): an event is usable at decision row t iff avail <= t; EventStreams.rasterise
counts it at row ceil(avail). Row r spans bar-unit time [r, r + 1); a message at time 864.3 is first usable at row 865.

Layout defaults (cfg keys): bar_minutes 5, pre_bars 864 (72 h; Hu et al. 2023 report price / volume drift from 57-60 h
before the pump), post_bars 96, lead 2, peak_window_bars 12, end_bars_after_peak 24, horizons [1,2,3,6,12,24],
n_controls_per_pump 2, gap_hours 24, control_span_days 30, merge_hours 6, lookback 48, latency 1, seed 0.
Episode key: f"{SYMBOL}{quote}@{YYYY-MM-DDTHH:MMZ}" (anchor = announcement / coordination message, UTC); control windows
use the same format with the prefix "CTRL_".
"""
from __future__ import annotations
import glob
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.schema import Panel
from causalgate.data.pit import attach_availability
from ..streams import EventStreams, derive_orderflow_events, PLATFORMS
from . import common

BINANCE_VISION = "https://data.binance.vision/data/spot"
KLINE_SUBDIR = os.path.join("crosspump", "binance_klines")
AGG_SUBDIR = os.path.join("crosspump", "binance_aggtrades")
ONCHAIN_SUBDIR = os.path.join("crosspump", "onchain")
AGG_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker", "best_match"]
MIN_NS = 60 * 10 ** 9
DAY_NS = 86400 * 10 ** 9
DEFAULTS: Dict[str, Any] = {"bar_minutes": 5, "pre_bars": 864, "post_bars": 96, "lead": 2, "peak_window_bars": 12, "end_bars_after_peak": 24,
                            "horizons": [1, 2, 3, 6, 12, 24], "n_controls_per_pump": 2, "gap_hours": 24.0, "control_span_days": 30.0, "seed": 0,
                            "min_bars_frac": 0.5, "social_lag_bars": 0.0, "wallet_lag_bars": 1.0, "quote": "BTC", "max_pumps": None,
                            "text_slots": 4, "text_dim": 32, "text_max_age_bars": 1, "text_max_mb": 2048.0,
                            "merge_hours": 6.0, "orderflow_source": "auto", "orderflow_z": 2.5, "control_exclude_window": True}
_MONTH_RE = re.compile(r"-1m-(\d{4}-\d{2})\.csv$")


# ================================================================================================= small helpers
def _args(args) -> Dict[str, Any]:
    """prepare(root, args) accepts a dict or an argparse Namespace (or None)."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    return {k: v for k, v in vars(args).items()}


def stable_id(s: Any) -> int:
    """Stable, non-zero int64 entity id for a channel / wallet / group string (md5, 60 bits; 0 is reserved = unknown)."""
    h = int(hashlib.md5(str(s).strip().lower().encode("utf-8")).hexdigest()[:15], 16)
    return int(h) + 1


def episode_key(symbol: str, quote: str, anchor_ns: int) -> str:
    return f"{symbol}{quote}@{pd.Timestamp(int(anchor_ns), tz='UTC').strftime('%Y-%m-%dT%H:%MZ')}"


def epoch_to_ns(v: pd.Series) -> np.ndarray:
    """Numeric epoch in s / ms / us (unit inferred from magnitude) or datetime strings -> float64 UTC ns (NaN if bad)."""
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.5:
        x = num.to_numpy(dtype=np.float64); m = np.nanmedian(x)
        unit = 10 ** 3 if m >= 1e14 else (10 ** 6 if m >= 1e11 else 10 ** 9)
        return np.where(np.isfinite(x), x * unit, np.nan)
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    return ts.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)


def klines_dir(root: str) -> str:
    return os.path.join(root, KLINE_SUBDIR)


def aggtrades_dir(root: str) -> str:
    return os.path.join(root, AGG_SUBDIR)


def onchain_dir(root: str) -> str:
    return os.path.join(root, ONCHAIN_SUBDIR)


def month_str(ns: int) -> str:
    return pd.Timestamp(int(ns), tz="UTC").strftime("%Y-%m")


def months_between(t0_ns: int, t1_ns: int) -> List[str]:
    a = pd.Timestamp(int(t0_ns)).to_period("M"); b = pd.Timestamp(int(t1_ns)).to_period("M")          # naive ns = UTC
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def window_days(cfg: Dict[str, Any]) -> Tuple[float, float]:
    """(days before, days after) an anchor that a dataset needs in cache: the episode window plus the control span."""
    c = {**DEFAULTS, **cfg}
    bm = int(c["bar_minutes"])
    before = int(c["pre_bars"]) * bm / 1440.0 + float(c["control_span_days"]) + 1.0
    after = (int(c["post_bars"]) + 1) * bm / 1440.0 + float(c["control_span_days"]) + 1.0
    return before, after


def months_for_anchors(anchor_ns: Iterable[int], before_days: float, after_days: float) -> List[str]:
    out = set()
    for a in anchor_ns:
        out.update(months_between(int(a) - int(before_days * DAY_NS), int(a) + int(after_days * DAY_NS)))
    return sorted(out)


def days_for_window(anchor_ns: int, pre_bars: int, post_bars: int, bar_minutes: int) -> List[str]:
    step = bar_minutes * MIN_NS
    a0 = (int(anchor_ns) // step) * step
    lo, hi = a0 - pre_bars * step, a0 + (post_bars + 1) * step
    d0 = pd.Timestamp(lo, tz="UTC").floor("D"); d1 = pd.Timestamp(hi - 1, tz="UTC").floor("D")
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(d0, d1, freq="D")]


# ================================================================================================= downloaders
def kline_url(symbol: str, month: str, quote: str = "BTC") -> str:
    pair = f"{symbol}{quote}"
    return f"{BINANCE_VISION}/monthly/klines/{pair}/1m/{pair}-1m-{month}.zip"


def aggtrades_url(symbol: str, date: str, quote: str = "BTC") -> str:
    pair = f"{symbol}{quote}"
    return f"{BINANCE_VISION}/daily/aggTrades/{pair}/{pair}-aggTrades-{date}.zip"


def _download_zip_csv(url: str, out_csv: str, timeout: float = 120.0) -> str:
    """Download one data.binance.vision zip and unzip its csv member to out_csv. -> 'ok' | 'exists' | '404' | 'error: ...'."""
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return "exists"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        return "404" if e.code == 404 else f"error: HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            if not members:
                return "error: no csv member in zip"
            data = z.read(members[0])
    except zipfile.BadZipFile as e:
        return f"error: bad zip ({e})"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    tmp = out_csv + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_csv)
    return "ok"


def _manifest_update(cache_dir: str, section: str, entries: Dict[str, str]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "manifest.json")
    man: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                man = json.load(f)
        except Exception:  # noqa: BLE001
            man = {}
    sec = man.setdefault(section, {})
    sec.update(entries)
    man["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)


def fetch_klines(symbols_months, cache_dir: str, quote: str = "BTC", workers: int = 8) -> Dict[str, str]:
    """symbols_months: {symbol: [YYYY-MM, ...]} or an iterable of (symbol, YYYY-MM). Returns {filename: status}; 404s
    (delisted pairs, months before listing) go to <cache_dir>/manifest.json["missing"]."""
    jobs: List[Tuple[str, str]] = []
    if isinstance(symbols_months, dict):
        for s, months in symbols_months.items():
            jobs += [(str(s), str(m)) for m in months]
    else:
        jobs = [(str(s), str(m)) for s, m in symbols_months]
    jobs = sorted(set(jobs))
    os.makedirs(cache_dir, exist_ok=True)

    def one(job):
        s, m = job
        name = f"{s}{quote}-1m-{m}.csv"
        return name, _download_zip_csv(kline_url(s, m, quote), os.path.join(cache_dir, name))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        status = dict(ex.map(one, jobs))
    missing = {k: v for k, v in status.items() if v == "404"}
    errors = {k: v for k, v in status.items() if v.startswith("error")}
    if missing:
        _manifest_update(cache_dir, "missing", missing)
    if errors:
        _manifest_update(cache_dir, "errors", errors)
    _manifest_update(cache_dir, "fetched", {k: v for k, v in status.items() if v == "ok"})
    return status


def fetch_aggtrades(symbol: str, dates: Iterable[str], cache_dir: str, quote: str = "BTC", workers: int = 4) -> Dict[str, str]:
    """Daily aggTrades for one pair. dates: iterable of YYYY-MM-DD. Returns {filename: status}; 404s -> manifest."""
    os.makedirs(cache_dir, exist_ok=True)
    dates = sorted(set(str(d) for d in dates))

    def one(d):
        name = f"{symbol}{quote}-aggTrades-{d}.csv"
        return name, _download_zip_csv(aggtrades_url(symbol, d, quote), os.path.join(cache_dir, name))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        status = dict(ex.map(one, dates))
    missing = {k: v for k, v in status.items() if v == "404"}
    if missing:
        _manifest_update(cache_dir, "missing", missing)
    _manifest_update(cache_dir, "fetched", {k: v for k, v in status.items() if v == "ok"})
    return status


def prepare_market_data(root: str, pumps: Sequence[Dict[str, Any]], args: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Klines for every pump symbol over [anchor - before, anchor + after] (window + control span), optional aggTrades
    for the window days (args['aggtrades']). pumps: dicts with symbol, anchor_ns. Returns a summary dict."""
    c = {**DEFAULTS, **(cfg or {}), **{k: v for k, v in args.items() if v is not None}}
    quote = str(c["quote"]); workers = int(c.get("workers", 8))
    before, after = window_days(c)
    by_sym: Dict[str, List[int]] = {}
    for p in pumps:
        by_sym.setdefault(str(p["symbol"]), []).append(int(p["anchor_ns"]))
    sm = {s: months_for_anchors(a, before, after) for s, a in by_sym.items()}
    st = fetch_klines(sm, klines_dir(root), quote=quote, workers=workers)
    summary: Dict[str, Any] = {"klines": {k: sum(1 for v in st.values() if v == k) for k in ("ok", "exists", "404")},
                               "kline_errors": sum(1 for v in st.values() if v.startswith("error")), "symbols": len(by_sym)}
    if c.get("aggtrades"):
        n_ok = 0; n_404 = 0
        for p in pumps:
            days = days_for_window(int(p["anchor_ns"]), int(c["pre_bars"]), int(c["post_bars"]), int(c["bar_minutes"]))
            s2 = fetch_aggtrades(str(p["symbol"]), days, aggtrades_dir(root), quote=quote, workers=min(workers, 4))
            n_ok += sum(1 for v in s2.values() if v in ("ok", "exists")); n_404 += sum(1 for v in s2.values() if v == "404")
        summary["aggtrades"] = {"ok": n_ok, "404": n_404}
    return summary


# ------------------------------------------------------------------------------------------------- optional on-chain
COINGECKO = "https://api.coingecko.com/api/v3"
ETHERSCAN_V2 = "https://api.etherscan.io/v2/api"
CHAIN_IDS = {"ethereum": 1, "binance-smart-chain": 56}


def _get_json(url: str, timeout: float = 60.0, retries: int = 3, sleep: float = 2.0) -> Any:
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crosspump-loader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                time.sleep(sleep * 5); continue
            if e.code == 404:
                return None
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(sleep)
    raise RuntimeError(f"GET {url} failed: {last}")


def build_token_contracts(symbols: Iterable[str], out_csv: str, per_coin_lookups: int = 50, sleep: float = 2.5) -> pd.DataFrame:
    """CoinGecko /coins/list?include_platform=true (one call) -> ethereum / binance-smart-chain contracts per symbol.
    Symbols are ambiguous on CoinGecko (many coins share a ticker): the first candidate that has a contract on a supported
    chain is used and n_candidates is recorded so the user can prune the table by hand. Coins without a platform in the
    list are looked up with /coins/{id} (bounded by per_coin_lookups, rate limited by `sleep`)."""
    want = {str(s).upper() for s in symbols}
    lst = _get_json(f"{COINGECKO}/coins/list?include_platform=true") or []
    cands: Dict[str, List[Dict[str, Any]]] = {}
    for c in lst:
        sym = str(c.get("symbol", "")).upper()
        if sym in want:
            cands.setdefault(sym, []).append(c)
    rows = []; lookups = 0
    for sym in sorted(want):
        found = None
        for c in cands.get(sym, []):
            plats = c.get("platforms") or {}
            for chain in CHAIN_IDS:
                addr = plats.get(chain)
                if addr:
                    found = (chain, addr, c.get("id")); break
            if found:
                break
        if found is None and lookups < per_coin_lookups:
            for c in cands.get(sym, [])[:2]:
                lookups += 1; time.sleep(sleep)
                d = _get_json(f"{COINGECKO}/coins/{c.get('id')}?localization=false&tickers=false&market_data=false&community_data=false&developer_data=false")
                plats = (d or {}).get("platforms") or {}
                for chain in CHAIN_IDS:
                    if plats.get(chain):
                        found = (chain, plats[chain], c.get("id")); break
                if found:
                    break
        rows.append({"symbol": sym, "chain": found[0] if found else "", "contract": found[1] if found else "", "coingecko_id": found[2] if found else "",
                     "n_candidates": len(cands.get(sym, []))})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def _etherscan(params: Dict[str, Any], api_key: str, sleep: float = 0.25) -> Any:
    q = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
    time.sleep(sleep)
    d = _get_json(f"{ETHERSCAN_V2}?{q}&apikey={api_key}")
    if not isinstance(d, dict):
        return None
    if str(d.get("status")) != "1" and d.get("message") not in ("OK", "No transactions found"):
        if "rate limit" in str(d.get("result", "")).lower():
            time.sleep(5)
        return None
    return d.get("result")


def prepare_onchain(root: str, dataset: str, pumps: Sequence[Dict[str, Any]], args: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None,
                    bars: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    """Write <root>/crosspump/onchain/<dataset>_transfers.csv (key,time_ns,wallet_id,size_adv,size_raw,to_exchange) from
    Etherscan V2 token transfers inside every pump window. Needs env ETHERSCAN_API_KEY. Partial coverage (ERC-20 /
    BEP-20 tokens only); exchange-deposit flags need the optional exchange_addresses.csv (address,label)."""
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    if not key:
        raise RuntimeError("prepare_onchain needs the environment variable ETHERSCAN_API_KEY (free key from https://etherscan.io/apis; V2 keys cover BSC too)")
    c = {**DEFAULTS, **(cfg or {}), **{k: v for k, v in args.items() if v is not None}}
    odir = onchain_dir(root); os.makedirs(odir, exist_ok=True)
    quote = str(c["quote"]); bm = int(c["bar_minutes"]); step = bm * MIN_NS
    tpath = os.path.join(odir, "token_contracts.csv")
    if not os.path.exists(tpath):
        build_token_contracts(sorted({str(p["symbol"]) for p in pumps}), tpath, sleep=float(c.get("coingecko_sleep", 2.5)))
    tok = pd.read_csv(tpath, dtype=str).fillna("")
    tok = tok[(tok["contract"].str.len() > 0) & tok["chain"].isin(list(CHAIN_IDS))]
    contracts = {r.symbol.upper(): (r.chain, r.contract) for r in tok.itertuples(index=False)}
    xpath = os.path.join(odir, "exchange_addresses.csv")
    exch_addr = set()
    if os.path.exists(xpath):
        xa = pd.read_csv(xpath, dtype=str).fillna("")
        col = "address" if "address" in xa else xa.columns[0]
        exch_addr = set(xa[col].str.lower().str.strip())
    block_cache: Dict[Tuple[int, int], int] = {}

    def block_at(chain_id: int, ts_s: int, closest: str) -> Optional[int]:
        k = (chain_id, ts_s)
        if k not in block_cache:
            r = _etherscan({"chainid": chain_id, "module": "block", "action": "getblocknobytime", "timestamp": ts_s, "closest": closest}, key)
            block_cache[k] = int(r) if r is not None and str(r).isdigit() else None
        return block_cache[k]

    rows = []
    for p in pumps:
        sym = str(p["symbol"]).upper()
        if sym not in contracts:
            continue
        chain, contract = contracts[sym]; chain_id = CHAIN_IDS[chain]
        a0 = (int(p["anchor_ns"]) // step) * step
        lo_ns, hi_ns = a0 - int(c["pre_bars"]) * step, a0 + (int(c["post_bars"]) + 1) * step
        b0 = block_at(chain_id, lo_ns // 10 ** 9, "before"); b1 = block_at(chain_id, hi_ns // 10 ** 9, "after")
        if b0 is None or b1 is None:
            continue
        adv = np.nan
        if bars and sym in bars:
            kl = bars[sym]; w = kl[(kl.index >= lo_ns) & (kl.index < a0)]
            if len(w):
                adv = float(w["volume"].sum() / max(len(w) * bm / 1440.0, 1e-9))         # mean daily base volume in the pre-window
        page = 1
        while page <= int(c.get("etherscan_max_pages", 20)):
            res = _etherscan({"chainid": chain_id, "module": "account", "action": "tokentx", "contractaddress": contract, "startblock": b0, "endblock": b1,
                              "page": page, "offset": 10000, "sort": "asc"}, key)
            if not res or not isinstance(res, list):
                break
            for tx in res:
                try:
                    ts = int(tx["timeStamp"]) * 10 ** 9
                    dec = int(tx.get("tokenDecimal") or 18)
                    size = float(int(tx["value"]) / 10 ** dec)
                except (KeyError, ValueError, TypeError):
                    continue
                if not (lo_ns <= ts < hi_ns):
                    continue
                rows.append({"key": p["key"], "time_ns": ts, "wallet_id": stable_id(str(tx.get("from", ""))), "size_adv": size / adv if np.isfinite(adv) and adv > 0 else np.nan,
                             "size_raw": size, "to_exchange": int(str(tx.get("to", "")).lower() in exch_addr) if exch_addr else -1})
            if len(res) < 10000:
                break
            page += 1
    out = os.path.join(odir, f"{dataset}_transfers.csv")
    pd.DataFrame(rows, columns=["key", "time_ns", "wallet_id", "size_adv", "size_raw", "to_exchange"]).to_csv(out, index=False)
    return out


# ================================================================================================= cache readers
def select_month_files(files: Sequence[str], months: Optional[Iterable[str]]) -> List[str]:
    """Keep monthly files (<PAIR>-1m-YYYY-MM.csv) whose month is needed; files without a month suffix are kept."""
    if months is None:
        return list(files)
    want = set(months)
    out = []
    for f in files:
        m = _MONTH_RE.search(os.path.basename(f))
        if m is None or m.group(1) in want:
            out.append(f)
    return out


def load_bars(root: str, symbols: Iterable[str], quote: str, bar_minutes: int, months_by_symbol: Optional[Dict[str, Iterable[str]]] = None,
              keep_intervals: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> Dict[str, pd.DataFrame]:
    """<root>/crosspump/binance_klines -> {symbol: bar_minutes bars indexed by open_time ns}. Symbols without any cached
    file are simply absent from the result (the caller counts them). keep_intervals[symbol] = [(lo_ns, hi_ns), ...]
    restricts the 1-minute rows read into memory to the needed windows before resampling."""
    kdir = klines_dir(root)
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        files = common.find_kline_files(kdir, s, quote)
        if months_by_symbol is not None:
            files = select_month_files(files, months_by_symbol.get(s))
        if not files:
            continue
        kl = common.read_binance_klines(files)
        if keep_intervals and s in keep_intervals and len(kl):
            idx = kl.index.to_numpy()
            keep = np.zeros(len(kl), dtype=bool)
            for lo, hi in keep_intervals[s]:
                keep[np.searchsorted(idx, lo, side="left") : np.searchsorted(idx, hi, side="right")] = True
            kl = kl[keep]
        if len(kl):
            out[s] = common.resample_bars(kl, bar_minutes)
    return out


def read_aggtrades(paths: Sequence[str]) -> pd.DataFrame:
    """Daily aggTrades csv(s) (header-less 8 columns or headed) -> DataFrame[time_ns, price, qty, taker_buy] sorted."""
    frames = []
    for p in paths:
        with open(p, "r") as f:
            first = f.readline().strip()
        headed = not first.split(",")[0].strip().isdigit()
        df = pd.read_csv(p, header=0 if headed else None)
        if not headed:
            df.columns = AGG_COLS[: df.shape[1]] + [f"extra{k}" for k in range(max(0, df.shape[1] - len(AGG_COLS)))]
        cols = {str(c).lower().strip().replace(" ", "_"): c for c in df.columns}
        tcol = next((cols[k] for k in ("time", "transact_time", "timestamp") if k in cols), None)
        if tcol is None:
            raise ValueError(f"{p}: no time column in {list(df.columns)}")
        bm = df[cols.get("is_buyer_maker", "is_buyer_maker")] if "is_buyer_maker" in cols else pd.Series(False, index=df.index)
        bm = bm.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
        frames.append(pd.DataFrame({"time_ns": epoch_to_ns(df[tcol]), "price": pd.to_numeric(df[cols["price"]], errors="coerce").to_numpy(),
                                    "qty": pd.to_numeric(df[cols["qty"]], errors="coerce").to_numpy(), "taker_buy": (~bm).to_numpy()}))
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["time_ns"]) & np.isfinite(df["qty"])]
    df["time_ns"] = df["time_ns"].astype(np.int64)
    return df.sort_values("time_ns").reset_index(drop=True)


def _causal_z(x: np.ndarray, warmup: int, halflife: float) -> np.ndarray:
    T = len(x); z = np.full(T, np.nan); lam = 0.5 ** (1.0 / max(halflife, 1.0)); m = 0.0; v = 0.0; n = 0
    for t in range(T):
        xt = x[t]
        if n >= warmup and np.isfinite(xt) and v > 0:
            z[t] = (xt - m) / np.sqrt(v)
        if np.isfinite(xt):
            if n == 0:
                m, v = float(xt), 0.0
            else:
                d = xt - m; m = lam * m + (1 - lam) * xt; v = lam * v + (1 - lam) * d * d
            n += 1
    return z


def aggtrades_bursts(trades: pd.DataFrame, bucket_minutes: int = 1, z_thresh: float = 2.5, warmup: int = 30, halflife: float = 60.0) -> pd.DataFrame:
    """1-minute buckets of aggTrades -> burst events (DataFrame time_ns = bucket close, z, imbalance): an event when the
    causal EWMA z-score (statistics of strictly earlier buckets) of log bucket volume or of the taker-buy imbalance
    (taker_buy_qty / qty - 0.5) exceeds z_thresh. z = max of the two z-scores; imbalance = the bucket's imbalance."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["time_ns", "z", "imbalance"])
    step = bucket_minutes * MIN_NS
    b = (trades["time_ns"].to_numpy() // step) * step
    g = pd.DataFrame({"b": b, "qty": trades["qty"].to_numpy(), "tb": trades["qty"].to_numpy() * trades["taker_buy"].to_numpy()}).groupby("b").sum()
    idx = np.arange(int(g.index.min()), int(g.index.max()) + step, step, dtype=np.int64)
    g = g.reindex(idx).fillna(0.0)
    vol = g["qty"].to_numpy(); tb = g["tb"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lv = np.log1p(np.where(vol > 0, vol, np.nan)); imb = np.where(vol > 0, tb / np.where(vol > 0, vol, np.nan) - 0.5, np.nan)
    z1 = _causal_z(lv, warmup, halflife); z2 = _causal_z(imb, warmup, halflife)
    zmax = np.nanmax(np.stack([np.nan_to_num(z1, nan=-np.inf), np.nan_to_num(z2, nan=-np.inf)]), 0)
    hit = np.nonzero(zmax > z_thresh)[0]
    return pd.DataFrame({"time_ns": idx[hit] + step, "z": zmax[hit], "imbalance": imb[hit]})


# ================================================================================================= episode tables
def merge_announcements(rows: pd.DataFrame, merge_hours: float, quote: str, source: str, extra_cols: Sequence[str] = (), ann_cols: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """rows: DataFrame[symbol, exchange, channel, announce_ns (int64 UTC), peak_ns (float, NaN = none), channel_id (int)]
    (+ extra_cols kept from the FIRST row; ann_cols kept per announcement). Rows of the same (symbol, exchange) within merge_hours of the first row
    become ONE pump episode anchored at the earliest announcement, with every row kept in episode['announcements']
    (the social stream gets one event per row at its own time). Returns pump episode dicts sorted by anchor."""
    out: List[Dict[str, Any]] = []
    gap = float(merge_hours) * 3600 * 10 ** 9
    df = rows.sort_values("announce_ns").reset_index(drop=True)
    for (sym, ex), g in df.groupby(["symbol", "exchange"], sort=False):
        cur: Optional[Dict[str, Any]] = None
        for r in g.itertuples(index=False):
            rd = r._asdict()
            ann = {"channel": rd.get("channel"), "channel_id": int(rd.get("channel_id") or stable_id(rd.get("channel"))), "announce_ns": int(rd["announce_ns"])}
            for k in ("session_id", "message", *ann_cols):
                if k in rd:
                    ann[k] = rd[k]
            pk = rd.get("peak_ns", np.nan)
            if cur is not None and int(rd["announce_ns"]) - cur["anchor_ns"] <= gap:
                cur["announcements"].append(ann)
                cur["channels"].append(ann["channel"])
                if (cur.get("peak_ns") is None) and pk is not None and np.isfinite(pk):
                    cur["peak_ns"] = int(pk)
                continue
            if cur is not None:
                out.append(cur)
            cur = {"key": episode_key(str(sym), quote, int(rd["announce_ns"])), "symbol": str(sym), "exchange": str(ex), "channel": ann["channel"],
                   "channel_id": ann["channel_id"], "anchor_ns": int(rd["announce_ns"]), "announce_ns": int(rd["announce_ns"]),
                   "peak_ns": int(pk) if pk is not None and np.isfinite(pk) else None, "is_pump": True, "control_of": None, "source": source,
                   "announcements": [ann], "channels": [ann["channel"]]}
            for k in extra_cols:
                if k in rd:
                    cur[k] = rd[k]
        if cur is not None:
            out.append(cur)
    out.sort(key=lambda d: d["anchor_ns"])
    return out


def pump_times_by_symbol(rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Every announcement time per symbol (ALL exchanges) -> used so controls stay >= gap_hours from any pump of the coin."""
    return {str(s): np.sort(g["announce_ns"].to_numpy().astype(np.int64)) for s, g in rows.groupby("symbol")}


# ================================================================================================= text slots
def fill_text_slots(T: int, N: int, M: int, emb: np.ndarray, rows: np.ndarray, assets: np.ndarray, max_age_bars: int) -> Dict[str, np.ndarray]:
    """Place embedded messages into Panel.text slots at the bar containing their publication (row), repeated with a
    growing text_age for up to max_age_bars bars (the M most recent per (t, i) win); avail_text = the publication bar
    for every copy, so the PIT view masks nothing that was published at or before t."""
    D = int(emb.shape[1]) if len(emb) else 1
    text = np.zeros((T, N, M, D), dtype=np.float32); mask = np.zeros((T, N, M), dtype=bool); age = np.zeros((T, N, M), dtype=np.float32)
    avail = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None, None], (T, N, M)).copy()
    fill = np.zeros((T, N), dtype=int)
    order = np.argsort(rows, kind="stable")[::-1]
    for j in order:
        r = int(rows[j]); i = int(assets[j])
        if r < 0 or r >= T:
            continue
        for k in range(max_age_bars):
            t = r + k
            if t >= T:
                break
            slot = fill[t, i]
            if slot >= M:
                continue
            text[t, i, slot] = emb[j]; mask[t, i, slot] = True; age[t, i, slot] = k; avail[t, i, slot] = r
            fill[t, i] += 1
    return {"text": text, "text_mask": mask, "text_age": age, "avail_text": avail}




def cap_pumps(pumps: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    """max_pumps cap that keeps the chronological spread (evenly spaced picks of the anchor-sorted list)."""
    if not n or int(n) <= 0 or len(pumps) <= int(n):
        return list(pumps)
    pumps = sorted(pumps, key=lambda d: d["anchor_ns"])
    pick = np.unique(np.round(np.linspace(0, len(pumps) - 1, int(n))).astype(int))
    return [pumps[k] for k in pick]


def norm_exchange(x: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).strip().lower().replace(".com", ""))


# ================================================================================================= panel assembly
def _window_bounds(anchor_ns: int, pre_bars: int, post_bars: int, step: int) -> Tuple[int, int]:
    a0 = (int(anchor_ns) // step) * step
    return a0 - pre_bars * step, a0 + (post_bars + 1) * step


def _coverage_ok(kl: pd.DataFrame, lo: int, hi: int, need: int) -> bool:
    idx = kl.index.to_numpy()
    return int(np.searchsorted(idx, hi, side="left") - np.searchsorted(idx, lo, side="left")) >= need


def _repair_controls(controls: List[Dict[str, Any]], bars: Dict[str, pd.DataFrame], pump_times: Dict[str, np.ndarray], key_symbol: Dict[str, str],
                     key_anchor: Dict[str, int], gap_ns: int, span_ns: int, pre_bars: int, post_bars: int, step: int, need: int, exclude_window: bool,
                     rng: np.random.Generator, quote: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """common.sample_controls draws control anchors / other-coin symbols blindly. Keep a control only if its coin has
    bars over the window, has no listed pump within +-gap of the control anchor and (exclude_window) no listed pump
    anywhere INSIDE the control window (a real pump inside a window labelled all-zero would be a false negative label);
    otherwise redraw (same coin: another time inside +-span of the pump; other coin: another coin at the same time).
    Controls are re-keyed 'CTRL_<SYMBOL><quote>@<time>' and de-duplicated."""
    syms = sorted(bars)
    stats = {"sampled": len(controls), "redrawn": 0, "dropped": 0, "duplicates": 0}

    def ok(s: str, a: int) -> bool:
        if s not in bars:
            return False
        lo, hi = _window_bounds(a, pre_bars, post_bars, step)
        if not _coverage_ok(bars[s], lo, hi, need):
            return False
        pt = pump_times.get(s, np.zeros(0))
        if len(pt) == 0:
            return True
        if np.abs(pt - a).min() <= gap_ns:
            return False
        return not (exclude_window and ((pt >= lo) & (pt < hi)).any())

    out: List[Dict[str, Any]] = []; seen = set()
    for c in controls:
        pump_sym = key_symbol[c["control_of"]]; pa = key_anchor[c["control_of"]]
        s, a = str(c["symbol"]), int(c["anchor_ns"])
        same = s == pump_sym
        if not ok(s, a):
            found = False
            for _ in range(40):
                if same:
                    a2 = int(pa + rng.uniform(-span_ns, span_ns)); s2 = s
                else:
                    s2 = str(rng.choice(syms)); a2 = a
                    if s2 == pump_sym:
                        continue
                if ok(s2, a2):
                    s, a, found = s2, a2, True; break
            if not found:
                stats["dropped"] += 1; continue
            stats["redrawn"] += 1
        key = "CTRL_" + episode_key(s, quote, a)
        if key in seen:
            stats["duplicates"] += 1; continue
        seen.add(key)
        out.append({**c, "symbol": s, "anchor_ns": a, "key": key, "control_type": "same_coin_other_time" if same else "other_coin_same_time"})
    return out, stats


SocialFn = Callable[[List[Dict[str, Any]], Dict[str, Any]], Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]]


def build_cex_panel(cfg: Dict[str, Any], pumps: List[Dict[str, Any]], *, name: str, dataset: str, source: str, licence: str, recipe: str,
                    all_pump_times: Optional[Dict[str, np.ndarray]] = None, social_fn: Optional[SocialFn] = None, entities: Optional[Dict[str, Any]] = None,
                    assumptions: Optional[List[str]] = None, extra_meta: Optional[Dict[str, Any]] = None) -> Panel:
    """pumps: merged pump episode dicts (key, symbol, exchange, channel, channel_id, anchor_ns, peak_ns|None, is_pump True,
    announcements [...], channels [...]) already filtered by exchange / quote / max_pumps. `dataset` = short name used
    for the optional on-chain file <root>/crosspump/onchain/<dataset>_transfers.csv.
    social_fn(episodes, cfg) -> (events DataFrame[key, time_ns, reach, text_score, source_id] | None,
                                 docs DataFrame[key, time_ns, text] | None, info dict)
    is called with the FINAL episode list (pumps and controls; every episode carries 'watch_channels' = the channels that
    announced the pump it belongs to) so that a loader with raw messages can serve control windows as well."""
    c = {**DEFAULTS, **cfg}
    root = str(c.get("root", "data_raw")); quote = str(c["quote"])
    bm, pre, post, lead = int(c["bar_minutes"]), int(c["pre_bars"]), int(c["post_bars"]), int(c["lead"])
    step = bm * MIN_NS; T = pre + post + 1
    horizons = [int(h) for h in c["horizons"]]
    rng = np.random.default_rng(int(c["seed"]))
    need = max(10, int(float(c["min_bars_frac"]) * T))
    kdir = klines_dir(root)
    if not pumps:
        raise FileNotFoundError(f"{name}: no pump episode left after filtering (exchange={c.get('exchange')}, quote={quote}).\nDownload recipe:\n{recipe}")
    seen_keys = set(); uniq = []
    for p in sorted(pumps, key=lambda d: d["anchor_ns"]):
        if p["key"] not in seen_keys:
            seen_keys.add(p["key"]); uniq.append(p)
    n_dup_keys = len(pumps) - len(uniq); pumps = uniq
    # ---- market bars (only the months / intervals the episodes need)
    before, after = window_days(c)
    by_sym: Dict[str, List[int]] = {}
    for p in pumps:
        by_sym.setdefault(p["symbol"], []).append(int(p["anchor_ns"]))
    months = {s: months_for_anchors(a, before, after) for s, a in by_sym.items()}
    intervals = {s: [(int(x) - int(before * DAY_NS), int(x) + int(after * DAY_NS)) for x in a] for s, a in by_sym.items()}
    bars = load_bars(root, sorted(by_sym), quote, bm, months_by_symbol=months, keep_intervals=intervals)
    if not bars:
        raise FileNotFoundError(f"{name}: no Binance 1-minute klines for any of the {len(by_sym)} pump symbols under {kdir} "
                                f"(expected <SYMBOL>{quote}-1m-YYYY-MM.csv).\nDownload recipe:\n{recipe}\n(see scripts/xp_prepare_data.py)")
    pump_times = all_pump_times or pump_times_by_symbol(pd.DataFrame([{"symbol": p["symbol"], "announce_ns": p["anchor_ns"]} for p in pumps]))
    pumps_with_bars = [dict(p) for p in pumps if p["symbol"] in bars]
    n_no_bars = len(pumps) - len(pumps_with_bars)
    for p in pumps_with_bars:
        p["watch_channels"] = sorted({str(a.get("channel")) for a in p.get("announcements", []) if a.get("channel") is not None})
        lo, hi = _window_bounds(int(p["anchor_ns"]), pre, post, step)
        pt = pump_times.get(p["symbol"], np.zeros(0))
        p["other_pumps_in_window"] = int(((pt >= lo) & (pt < hi) & (np.abs(pt - int(p["anchor_ns"])) > float(c["merge_hours"]) * 3600e9)).sum()) if len(pt) else 0
    # ---- controls: same coin at a pump-free time + another pumped coin at the same time
    gap_ns = int(float(c["gap_hours"]) * 3600 * 10 ** 9); span_ns = int(float(c["control_span_days"]) * DAY_NS)
    controls = common.sample_controls(pumps_with_bars, pump_times, int(c["n_controls_per_pump"]), float(c["gap_hours"]), rng,
                                      span_days=float(c["control_span_days"]), other_symbols=sorted(bars) if len(bars) > 1 else None)
    by_key = {p["key"]: p for p in pumps_with_bars}
    controls, ctl_stats = _repair_controls(controls, bars, pump_times, {k: p["symbol"] for k, p in by_key.items()}, {k: int(p["anchor_ns"]) for k, p in by_key.items()},
                                           gap_ns, span_ns, pre, post, step, need, bool(c["control_exclude_window"]), rng, quote)
    for ctl in controls:
        ctl["channel_id"] = 0; ctl["announcements"] = []; ctl["channels"] = []
        ctl["watch_channels"] = list(by_key[ctl["control_of"]]["watch_channels"])
    episodes = pumps_with_bars + controls
    base_meta = {"dataset": name, "n_pumps_in_table": len(pumps), "pumps_without_klines": n_no_bars, "duplicate_pump_keys_dropped": n_dup_keys, "quote": quote,
                 "exchange_filter": c.get("exchange"),
                 "controls": {"n_per_pump": int(c["n_controls_per_pump"]), "gap_hours": float(c["gap_hours"]), "span_days": float(c["control_span_days"]),
                              "exclude_window": bool(c["control_exclude_window"]), "stats": ctl_stats,
                              "rule": "same coin uniformly inside +-span_days at >= gap_hours from any listed pump of that coin (all exchanges); another pumped coin at the "
                                      "same anchor time; both need klines over the window and (exclude_window) no listed pump of the coin anywhere inside the window"},
                 "peak_window_bars": int(c["peak_window_bars"]), "end_bars_after_peak": int(c["end_bars_after_peak"]), "seed": int(c["seed"]), "root": root}
    panel, skipped = common.episode_panel(bars, episodes, pre, post, bm, lead, horizons, peak_window_bars=int(c["peak_window_bars"]),
                                          end_bars_after_peak=int(c["end_bars_after_peak"]), min_bars_frac=float(c["min_bars_frac"]), dataset=name,
                                          source=source, licence=licence, extra_meta=base_meta)
    idx = common.asset_index(panel)
    eps = panel.meta["episodes"]
    lag = float(c["social_lag_bars"])
    # ---- social stream: one event per announcement row at its own time (source = stable hash of the channel id)
    ann_rows = []
    for e in eps:
        for a in e.get("announcements", []) or []:
            ann_rows.append({"key": e["key"], "time_ns": int(a["announce_ns"]), "reach": np.nan, "text_score": 1.0, "source_id": int(a["channel_id"])})
    parts: List[EventStreams] = []
    social_meta: Dict[str, Any] = {"announcements": "one event per announcement row of the pump table at its own time; mark [1, NaN, 1.0, prior]; source = stable md5 id of the channel"}
    if ann_rows:
        parts.append(common.social_stream(panel, pd.DataFrame(ann_rows), lag_bars=lag))
    n_msg = 0; text_docs = None; social_info: Dict[str, Any] = {}
    if social_fn is not None:
        extra_social, text_docs, social_info = social_fn(eps, c)
        if extra_social is not None and len(extra_social):
            es = extra_social[extra_social["key"].astype(str).isin(idx)]
            if len(es):
                parts.append(common.social_stream(panel, es, lag_bars=lag))
                n_msg = int(len(es))
                social_meta["messages"] = "messages of the watched organiser channel(s) inside the window, pumps and controls alike (loader-specific; see meta['assumptions'])"
    # ---- wallet stream: optional on-chain transfers (absent = missing modality)
    wpath = os.path.join(onchain_dir(root), f"{dataset}_transfers.csv")
    wallet_cov: Dict[str, Any]
    if os.path.exists(wpath):
        tr = pd.read_csv(wpath)
        tr = tr[tr["key"].astype(str).isin(idx)] if "key" in tr else tr.iloc[0:0]
        if len(tr):
            parts.append(common.wallet_stream(panel, tr, lag_bars=float(c["wallet_lag_bars"])))
        keys_w = sorted(set(tr["key"].astype(str))) if len(tr) else []
        wallet_cov = {"file": wpath, "n_transfers": int(len(tr)), "episodes": keys_w, "n_pump_episodes": int(sum(1 for k in keys_w if not k.startswith("CTRL_"))),
                      "n_control_episodes": int(sum(1 for k in keys_w if k.startswith("CTRL_"))),
                      "note": "ERC-20 / BEP-20 transfers only (partial coverage); episodes not listed have NO wallet modality (missing, not 'no activity')"}
    else:
        wallet_cov = {"file": None, "n_transfers": 0, "episodes": [], "n_pump_episodes": 0, "n_control_episodes": 0,
                      "note": f"missing modality: {wpath} absent (run scripts/xp_prepare_data.py --dataset {dataset} --onchain with ETHERSCAN_API_KEY; ERC-20 / BEP-20 tokens only)"}
    # ---- order-flow stream: ONE source for every episode (aggTrades bursts only when all window days are cached)
    adir = aggtrades_dir(root); mode = str(c.get("orderflow_source", "auto")).lower()
    files_of: Dict[str, List[str]] = {}
    covered = np.zeros(panel.N, dtype=bool)
    if mode != "bars" and os.path.isdir(adir):
        for e in eps:
            files = [os.path.join(adir, f"{e['symbol']}{quote}-aggTrades-{d}.csv") for d in days_for_window(int(e["anchor_ns"]), pre, post, bm)]
            if all(os.path.exists(f) for f in files):
                covered[idx[e["key"]]] = True; files_of[e["key"]] = files
    if mode == "aggtrades" and not covered.all():
        raise FileNotFoundError(f"{name}: orderflow_source='aggtrades' but only {int(covered.sum())}/{panel.N} episodes have every window day under {adir}.\n"
                                f"Download recipe:\n{recipe}\n(python3 scripts/xp_prepare_data.py --dataset {dataset} --aggtrades)")
    use_agg = bool(covered.all()) and mode in ("auto", "aggtrades")
    of_meta: Dict[str, Any] = {"source": "aggtrades" if use_agg else "bars", "episodes_with_all_aggtrades_days": int(covered.sum())}
    if use_agg:
        agg_rows = []
        for e in eps:
            b = aggtrades_bursts(read_aggtrades(files_of[e["key"]]), z_thresh=float(c["orderflow_z"]))
            lo, hi = _window_bounds(int(e["anchor_ns"]), pre, post, step)
            bb = b[(b["time_ns"] >= lo) & (b["time_ns"] < hi)]
            agg_rows += [{"key": e["key"], "time_ns": int(r.time_ns), "z": float(r.z), "imbalance": float(r.imbalance)} for r in bb.itertuples(index=False)]
        if agg_rows:
            parts.append(common.orderflow_stream_from_trades(panel, pd.DataFrame(agg_rows)))
        of_meta["note"] = (f"aggTrades ({adir}): 1-minute taker-buy imbalance / volume bursts, causal EWMA z > {float(c['orderflow_z'])} (statistics of strictly earlier "
                           "buckets, warm-up 30 buckets from the first cached day); event time = avail = bucket close")
    else:
        der = derive_orderflow_events(panel, z_thresh=float(c["orderflow_z"]))
        if len(der):
            parts.append(der)
        of_meta["derived_rule"] = der.meta.get("orderflow_rule")
        of_meta["note"] = ("bursts derived causally from the bar features of every episode (crosspump.data.streams.derive_orderflow_events; time = avail = close of the bar)"
                           + ("" if not covered.any() else f"; aggTrades exist for only {int(covered.sum())}/{panel.N} episodes and are NOT mixed in (one source for all episodes)"))
    common.finish_streams(panel, parts, derive_orderflow_if_missing=False, platform_lags={"social": lag, "wallet": float(c["wallet_lag_bars"]), "orderflow": 0.0},
                          entities=entities or {}, built={"social": social_meta, "wallet": wallet_cov["note"], "orderflow": of_meta["note"]})
    panel.meta["coverage"].update({"wallet": wallet_cov, "orderflow": of_meta,
                                   "social": {"announcement_events": len(ann_rows), "message_events": n_msg, **social_info,
                                              "note": "announcement events exist for pump episodes only (they define the anchor); message events, when a loader has raw "
                                                      "messages, cover pumps and controls of the watched channels"},
                                   "n_pumps": int(panel.meta["n_pumps"]), "n_controls": int(panel.meta["n_controls"])})
    # ---- optional text (hash embedding of the watched-channel messages inside the window)
    tcfg = c.get("text", {}) if isinstance(c.get("text", {}), dict) else {}
    panel.meta["text"] = "none (no message text for this dataset, or text disabled)"
    if text_docs is not None and len(text_docs) and tcfg.get("enabled", True):
        td = text_docs[text_docs["key"].astype(str).isin(idx)].reset_index(drop=True)
        M = int(tcfg.get("slots", c["text_slots"])); D = int(tcfg.get("dim", c["text_dim"])); max_age = max(1, int(tcfg.get("max_age_bars", c["text_max_age_bars"])))
        mb = T * panel.N * M * D * 4 / 2 ** 20
        if len(td) and M > 0 and mb > float(c["text_max_mb"]):
            panel.meta["text"] = f"skipped: Panel.text would need {mb:.0f} MB > text_max_mb={float(c['text_max_mb']):.0f} (lower text_slots / text_dim / max_pumps or raise text_max_mb)"
        elif len(td) and M > 0:
            emb = common.hash_embed(td["text"].astype(str).tolist(), dim=D)
            assets = np.array([idx[str(k)] for k in td["key"]], dtype=np.int64)
            tb = np.array([common.bar_units(panel, int(i), np.array([int(t)]))[0] for i, t in zip(assets, td["time_ns"].to_numpy())])
            rows = np.ceil(tb + lag - 1e-9).astype(np.int64)          # first row at which the message is usable (avail <= row)
            ok = (tb >= 0) & (rows < T)
            fs = fill_text_slots(T, panel.N, M, emb[ok], rows[ok], assets[ok], max_age)
            panel.text, panel.text_mask, panel.text_age = fs["text"], fs["text_mask"], fs["text_age"]
            attach_availability(panel, avail_text=fs["avail_text"])
            panel.meta["text"] = {"embedder": "hash_embed (hashed log-tf bag of words, nothing learned; cutoff n/a)", "dim": D, "slots": M, "max_age_bars": max_age,
                                  "n_docs": int(ok.sum()), "n_slots_filled": int(fs["text_mask"].sum()),
                                  "avail_text": "first row at which the message is usable = ceil(message time + social_lag_bars) in bar units"}
    panel.meta["pit_notes"] = {"avail_feat": "t (bar close)", "avail_label": "episode end row (peak + end_bars_after_peak) on event rows, else t + max(horizons)",
                               "avail_text": "ceil(message time + social_lag_bars): the first decision row at which the message is usable", "universe": "a kline exists at that bar",
                               "tradable": "kline exists and volume > 0",
                               "streams": "usable at row t iff avail <= t; social avail = message time + social_lag_bars; wallet = block time + wallet_lag_bars; orderflow = bucket / bar close"}
    panel.meta["assumptions"] = list(assumptions or []) + [
        "Controls: 'same_coin_other_time' and 'other_coin_same_time' windows (meta['episodes'][i]['control_type']); a control is only as clean as the pump table "
        "is complete -- an unlisted pump inside a control window is labelled 0.",
        f"{sum(1 for e in eps if e.get('is_pump') and e.get('other_pumps_in_window', 0) > 0)} pump episodes contain ANOTHER listed pump of the same coin inside their "
        "own window (labelled 0 there; flag other_pumps_in_window per episode).",
        "Announcement events exist only on pump episodes (the table defines the anchor), so the social platform is trivially informative from the first row after "
        "the anchor; pre-announcement rows are the detection target.",
        "Order-flow events come from one source for all episodes (aggTrades when every window day of every episode is cached, else bar-derived).",
        "Peak = curated time when the table has one, else argmax of the bar high within peak_window_bars after the anchor (meta['episodes'][i]['peak_source']).",
        "liq[..., 0] is a high-low proxy for the relative half-spread (no quote data in klines); taker_imb uses kline taker_buy_volume / volume."]
    panel.meta["required_layout"] = {"klines": f"<root>/{KLINE_SUBDIR}/<SYMBOL>{quote}-1m-YYYY-MM.csv", "aggtrades (optional)": f"<root>/{AGG_SUBDIR}/<SYMBOL>{quote}-aggTrades-YYYY-MM-DD.csv",
                                     "onchain (optional)": f"<root>/{ONCHAIN_SUBDIR}/{dataset}_transfers.csv (key,time_ns,wallet_id,size_adv,size_raw)"}
    panel.meta["skipped"] = skipped
    panel.meta.update(extra_meta or {})
    panel.validate()
    return panel


# ================================================================================================= shared prepare()
def clone_repo(url: str, target: str, branch: Optional[str] = None) -> None:
    """git clone --depth 1 (prepare-time only). Refuses to touch a non-empty directory."""
    import subprocess
    if os.path.isdir(target) and os.listdir(target):
        raise RuntimeError(f"{target} exists but lacks the expected files; remove it or clone {url} there by hand")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"] + (["--branch", branch] if branch else []) + [url, target]
    print("[xp_prepare]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare_cex(root: str, args, *, dataset: str, subdir: str, repo_url: str, required: Sequence[str], pumps_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
                builder: Callable[[Dict[str, Any]], Any], branch: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Shared prepare(root, args): (1) clone the source repo when the event table is missing; (2) klines for the pumped
    symbols (--klines, or when no flag is given); (3) --aggtrades: daily aggTrades for the window days of pumps AND
    controls; (4) --onchain: Etherscan transfers for pumps AND controls. Steps 3-4 build the panel from the cached
    klines to know the control windows (fall back to pumps only, with a warning, when klines are still missing)."""
    a = _args(args)
    c = dict(cfg or {}); c.update({"root": root, "subdir": subdir})
    if a.get("max_pumps"):
        c["max_pumps"] = int(a["max_pumps"])
    if not all(os.path.exists(os.path.join(root, r)) for r in required):
        clone_repo(repo_url, os.path.join(root, "crosspump", subdir), branch)
    pumps = pumps_fn(c)
    summary: Dict[str, Any] = {"dataset": dataset, "pumps": len(pumps)}
    workers = int(a.get("workers") or 8)
    if a.get("klines") or not any(a.get(k) for k in ("klines", "aggtrades", "onchain")):
        summary["market"] = prepare_market_data(root, pumps, {"workers": workers}, c)
    if a.get("aggtrades") or a.get("onchain"):
        cc = {**DEFAULTS, **c}
        try:
            episodes = builder({**c, "orderflow_source": "bars", "text": {"enabled": False}}).build_panel().meta["episodes"]
        except FileNotFoundError as e:
            print(f"[xp_prepare] {dataset}: cannot build the panel yet ({str(e).splitlines()[0]}); using pump windows only -- re-run after --klines "
                  "so that control windows get the same modalities", flush=True)
            episodes = pumps
        summary["episodes"] = len(episodes)
        if a.get("aggtrades"):
            n_ok = n_404 = 0
            for e in episodes:
                st = fetch_aggtrades(str(e["symbol"]), days_for_window(int(e["anchor_ns"]), int(cc["pre_bars"]), int(cc["post_bars"]), int(cc["bar_minutes"])),
                                     aggtrades_dir(root), quote=str(cc["quote"]), workers=min(workers, 4))
                n_ok += sum(1 for v in st.values() if v in ("ok", "exists")); n_404 += sum(1 for v in st.values() if v == "404")
            summary["aggtrades"] = {"ok": n_ok, "404": n_404}
        if a.get("onchain"):
            bars = load_bars(root, sorted({str(e["symbol"]) for e in episodes}), str(cc["quote"]), int(cc["bar_minutes"]))
            summary["onchain"] = prepare_onchain(root, dataset, episodes, {}, c, bars=bars)
    print(f"[xp_prepare] {json.dumps(summary, default=str)}", flush=True)
    return summary
