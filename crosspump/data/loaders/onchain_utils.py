"""On-chain / DEX helpers shared by the D8-D10 loaders (crosspump/CONVENTIONS.md section 1-3). Nothing here fabricates
data: every function transforms a trade / transfer table the loader read from disk.

    hash_address / hash_ids          stable, non-zero int64 entity ids from addresses or channel names (0 = unknown)
    causal_z                         EWMA z-score using statistics of strictly earlier bars (vectorised; NaN in warm-up)
    normalise_trades                 tolerant column mapping -> [time_ns, price, amount, taker_buy, buyer, seller, tx]
    trades_to_bars                   trades -> bar_minutes bars on the epoch grid: open/high/low/close/volume/
                                     taker_buy_volume/count (index = bar open_time ns), the layout common.episode_panel reads
    carry_forward_bars               dense grid between a token's first and last trade (last trade price carried, volume 0)
    trade_count_bursts               causal z-score bursts of per-bar trade counts -> orderflow events (time = bar close)
    wallet_inflow_events             every buy trade as a wallet inflow event (wallet_id = hashed buyer or taker address,
                                     size_adv = amount / trailing mean daily volume, size_raw = amount)
    wash_trade_flags                 cheap part of Victor & Weintraud's rule: self-trades and 2-cycles with matching amounts
    assign_episode_keys              tag events (symbol, time_ns) with the episode key whose window contains them
    chunked_filter_csv               read a multi-GB csv in pandas chunks keeping selected rows, cached under the dataset subdir
    optional_transfers               the OPTIONAL <root>/crosspump/onchain/<dataset>_transfers.csv wallet file (None if absent)
    tabular_panel                    degenerate stacked-episode Panel for tabular-only sources (one feature row at the anchor)

    make_keys_unique / iso_minute    episode-key helpers
    read_xlsx                        dependency-free reader of the first sheet of an .xlsx (zipfile + xml; no openpyxl)

Point-in-time: an event is usable at decision row t iff avail <= t (EventStreams.rasterise counts it at row ceil(avail)).
Bar-derived bursts therefore carry time = avail = the bar CLOSE; trade-level wallet events carry the trade time and get
their confirmation lag in common.wallet_stream.

Conventions: times are int64 UTC ns; `price` is quote per token; `amount` is the token (base) amount so that `volume`
in the bars is base volume like the Binance klines; `taker_buy` is True when the TAKER buys the token (the maker sells).
"""
from __future__ import annotations
import hashlib
import os
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.schema import Panel
from causalgate.data.pit import attach_availability, default_availability
from .common import FEAT_NAMES, MIN_NS, liquidity_fields

TRADE_COLS = ["time_ns", "price", "amount", "taker_buy", "buyer", "seller", "tx"]
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "time_ns": ("time_ns", "timestamp_ns", "block_time_ns"),
    "time": ("timestamp", "time", "block_time", "block_timestamp", "datetime", "date", "ts"),
    "price": ("price", "price_quote", "price_eth", "price_usd", "px"),
    "amount": ("amount", "amount_token", "token_amount", "qty", "quantity", "size", "base_amount"),
    "quote_amount": ("quote_amount", "amount_quote", "eth_amount", "amount_eth", "usd_amount", "value_quote"),
    "taker_buy": ("taker_buy", "is_taker_buy", "taker_is_buyer"),
    "side": ("side", "taker_side", "type"),
    "buyer": ("buyer", "buyer_address", "to", "recipient"),
    "seller": ("seller", "seller_address", "from", "sender"),
    "taker": ("taker", "taker_address"),
    "maker": ("maker", "maker_address"),
    "tx": ("tx", "tx_hash", "transaction_hash", "hash", "signature"),
}


# ------------------------------------------------------------------------------------------------ ids
def hash_address(addr: Any, salt: str = "") -> int:
    """Stable non-zero int64 id for an address / channel / account string (sha1, 60 bits). Unknown / empty -> 0."""
    if addr is None or (isinstance(addr, float) and np.isnan(addr)):
        return 0
    s = str(addr).strip().lower()
    if not s or s in ("nan", "none", "null"):
        return 0
    h = int(hashlib.sha1((salt + s).encode()).hexdigest(), 16) & ((1 << 60) - 1)
    return h or 1


def hash_ids(values: Iterable[Any], salt: str = "") -> Tuple[np.ndarray, Dict[int, str]]:
    """Vectorised hash_address with a table id -> 10-char prefix of the sha1 (an alias, never the raw address)."""
    ids, table = [], {}
    for v in values:
        i = hash_address(v, salt)
        ids.append(i)
        if i and i not in table:
            table[i] = hashlib.sha1((salt + str(v).strip().lower()).encode()).hexdigest()[:10]
    return np.asarray(ids, dtype=np.int64), table


# ------------------------------------------------------------------------------------------------ causal statistics
def causal_z(x: np.ndarray, warmup: int = 30, halflife: float = 60.0) -> np.ndarray:
    """EWMA z-score of x[t] against the mean / variance of x[:t] (strictly earlier samples, same recursion as
    crosspump.data.streams._causal_z but vectorised): NaN during the first `warmup` finite samples and where var == 0."""
    x = np.asarray(x, dtype=np.float64)
    s = pd.Series(x)
    alpha = 1.0 - 0.5 ** (1.0 / max(float(halflife), 1.0))
    m_prev = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().shift(1)
    d2 = (s - m_prev) ** 2
    first = s.first_valid_index()
    if first is not None:
        d2.iloc[first] = 0.0                                          # the reference recursion starts with variance 0 at the first sample
    v_prev = d2.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().shift(1)
    n_prev = s.notna().cumsum().shift(1).fillna(0)
    z = (s - m_prev) / np.sqrt(v_prev)
    z[(n_prev < warmup) | ~(v_prev > 0) | s.isna()] = np.nan
    return z.to_numpy()


# ------------------------------------------------------------------------------------------------ trades
def _epoch_to_ns(v: pd.Series) -> np.ndarray:
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.5:
        x = num.to_numpy(dtype=np.float64)
        m = np.nanmedian(x)
        unit = 1 if m >= 1e17 else (10 ** 3 if m >= 1e14 else (10 ** 6 if m >= 1e11 else 10 ** 9))
        return np.where(np.isfinite(x), x * unit, np.nan)
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    return ts.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)


def _pick(cols: Dict[str, str], key: str) -> Optional[str]:
    return next((cols[a] for a in _ALIASES[key] if a in cols), None)


def normalise_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Tolerant mapping of a trade table to TRADE_COLS. Accepts time_ns | timestamp (s/ms/us/ns or datetime), price or
    quote_amount (price = quote_amount / amount), taker_buy (bool/0/1) or side ('buy'/'sell' = the TAKER side), and
    buyer/seller or taker/maker (buyer = taker when taker_buy else maker). Rows without time, price or amount are dropped."""
    cols = {str(c).lower().strip().replace(" ", "_"): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    c = _pick(cols, "time_ns")
    if c is not None:
        out["time_ns"] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    else:
        c = _pick(cols, "time")
        if c is None:
            raise ValueError(f"trade table has no time column among {list(df.columns)}")
        out["time_ns"] = _epoch_to_ns(df[c])
    ca = _pick(cols, "amount")
    if ca is None:
        raise ValueError(f"trade table has no amount column among {list(df.columns)}")
    out["amount"] = pd.to_numeric(df[ca], errors="coerce").to_numpy(dtype=np.float64)
    cp = _pick(cols, "price")
    if cp is not None:
        out["price"] = pd.to_numeric(df[cp], errors="coerce").to_numpy(dtype=np.float64)
    else:
        cq = _pick(cols, "quote_amount")
        if cq is None:
            raise ValueError(f"trade table has neither price nor quote_amount among {list(df.columns)}")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["price"] = pd.to_numeric(df[cq], errors="coerce").to_numpy(dtype=np.float64) / out["amount"].to_numpy()
    ct = _pick(cols, "taker_buy")
    if ct is not None:
        v = df[ct]
        tb = v.astype(str).str.lower().isin(("1", "true", "t", "yes", "y", "buy")) if v.dtype == object else v.astype(float) > 0.5
        out["taker_buy"] = tb.to_numpy()
    else:
        cs = _pick(cols, "side")
        out["taker_buy"] = df[cs].astype(str).str.lower().str.startswith("b").to_numpy() if cs is not None else np.zeros(len(df), dtype=bool)
    cb, cs_ = _pick(cols, "buyer"), _pick(cols, "seller")
    if cb is not None and cs_ is not None:
        out["buyer"], out["seller"] = df[cb].astype(str).str.lower().to_numpy(), df[cs_].astype(str).str.lower().to_numpy()
    else:
        ctk, cmk = _pick(cols, "taker"), _pick(cols, "maker")
        if ctk is None or cmk is None:
            out["buyer"] = out["seller"] = np.array([""] * len(df), dtype=object)
        else:
            tk, mk = df[ctk].astype(str).str.lower().to_numpy(), df[cmk].astype(str).str.lower().to_numpy()
            tb = out["taker_buy"].to_numpy()
            out["buyer"], out["seller"] = np.where(tb, tk, mk), np.where(tb, mk, tk)
    cx = _pick(cols, "tx")
    out["tx"] = df[cx].astype(str).to_numpy() if cx is not None else np.array([""] * len(df), dtype=object)
    ok = np.isfinite(out["time_ns"].to_numpy()) & np.isfinite(out["price"].to_numpy()) & np.isfinite(out["amount"].to_numpy()) & (out["amount"].to_numpy() > 0) & (out["price"].to_numpy() > 0)
    out = out[ok].copy()
    out["time_ns"] = out["time_ns"].to_numpy().astype(np.int64)
    return out[TRADE_COLS].sort_values("time_ns", kind="stable").reset_index(drop=True)


def trades_to_bars(trades: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Normalised trades -> bars on the epoch-aligned grid (index = bar open_time ns; only bars with >= 1 trade):
    open/high/low/close from trade prices in time order, volume = sum(amount), taker_buy_volume = sum(amount | taker_buy),
    count = number of trades. This is the layout common.episode_panel / resample_bars use."""
    cols = ["open", "high", "low", "close", "volume", "taker_buy_volume", "count"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols, index=pd.Index([], dtype=np.int64, name="open_time"))
    step = int(bar_minutes) * MIN_NS
    t = trades.sort_values("time_ns", kind="stable")
    g = (t["time_ns"].to_numpy() // step) * step
    amt = t["amount"].to_numpy(dtype=np.float64)
    tb = t["taker_buy"].to_numpy().astype(bool)
    df = pd.DataFrame({"g": g, "price": t["price"].to_numpy(dtype=np.float64), "amount": amt, "tbv": np.where(tb, amt, 0.0)})
    out = df.groupby("g").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"), volume=("amount", "sum"),
                              taker_buy_volume=("tbv", "sum"), count=("price", "size"))
    out.index = out.index.astype(np.int64)
    out.index.name = "open_time"
    return out[cols]


def dense_bar_grid(bars: pd.DataFrame, bar_minutes: int, lo_ns: Optional[int] = None, hi_ns: Optional[int] = None) -> pd.DataFrame:
    """Reindex bars onto every bar between lo and hi (count / volume 0 where no trade; prices NaN)."""
    step = int(bar_minutes) * MIN_NS
    if len(bars) == 0:
        return bars
    lo = int(bars.index.min() if lo_ns is None else (lo_ns // step) * step)
    hi = int(bars.index.max() if hi_ns is None else (hi_ns // step) * step)
    grid = np.arange(lo, hi + step, step, dtype=np.int64)
    d = bars.reindex(grid)
    for k in ("volume", "taker_buy_volume", "count"):
        if k in d:
            d[k] = d[k].fillna(0.0)
    d.index.name = "open_time"
    return d


def carry_forward_bars(bars: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Bars with >= 1 trade -> every bar between the token's FIRST and LAST trade. A bar without trades carries the last
    traded price (open = high = low = close = previous close: a causal, last-observation-carried value, not a new price),
    volume = taker_buy_volume = count = 0, so common.episode_panel marks it untradable (volume 0) but the mid exists.
    Nothing is created before the first or after the last trade (NaN there: the market does not exist)."""
    if bars is None or len(bars) == 0:
        return bars
    had = bars["close"].notna()
    d = dense_bar_grid(bars, bar_minutes)
    c = d["close"].ffill()
    for k in ("open", "high", "low"):
        d[k] = d[k].where(d[k].notna(), c)
    d["close"] = c
    d["carried"] = ~had.reindex(d.index, fill_value=False).to_numpy()
    return d


def trade_count_bursts(bars: pd.DataFrame, bar_minutes: int, z_thresh: float = 2.5, warmup: int = 30, halflife: float = 60.0,
                       min_gap_bars: int = 1, lo_ns: Optional[int] = None, hi_ns: Optional[int] = None) -> pd.DataFrame:
    """Order-flow burst events from per-bar TRADE COUNTS, causally: an event at bar b when the EWMA z-score of
    log1p(count) (statistics from bars < b, zero-trade bars included) exceeds z_thresh. Returns columns
    time_ns (= the bar CLOSE: knowable only once the bar is complete), z, imbalance (taker_buy_volume / volume - 0.5, NaN if
    no volume), bar_open_ns. The z-score of a count is also the magnitude that goes into mark[:, 0]."""
    cols = ["time_ns", "z", "imbalance", "bar_open_ns"]
    if bars is None or len(bars) == 0:
        return pd.DataFrame(columns=cols)
    step = int(bar_minutes) * MIN_NS
    d = dense_bar_grid(bars, bar_minutes, lo_ns, hi_ns)
    z = causal_z(np.log1p(d["count"].to_numpy(dtype=np.float64)), warmup, halflife)
    hit = np.nonzero(np.nan_to_num(z, nan=-np.inf) > z_thresh)[0]
    keep, last = [], -10 ** 9
    for b in hit:
        if b - last >= min_gap_bars:
            keep.append(int(b)); last = int(b)
    if not keep:
        return pd.DataFrame(columns=cols)
    idx = np.asarray(keep)
    vol = d["volume"].to_numpy(dtype=np.float64)[idx]
    tbv = d["taker_buy_volume"].to_numpy(dtype=np.float64)[idx] if "taker_buy_volume" in d else np.full(len(idx), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(vol > 0, tbv / np.where(vol > 0, vol, np.nan) - 0.5, np.nan)
    opens = d.index.to_numpy()[idx].astype(np.int64)
    return pd.DataFrame({"time_ns": opens + step, "z": z[idx], "imbalance": imb, "bar_open_ns": opens})


def wallet_inflow_events(trades: pd.DataFrame, adv_days: float = 7.0, side: str = "buyer", salt: str = "", trailing: str = "days") -> pd.DataFrame:
    """Every trade in which somebody BUYS the token -> one wallet inflow event. side='buyer': the wallet is the buyer of
    every trade (taker when taker_buy, else maker); side='taker': only taker-buy trades, wallet = taker. Columns: time_ns,
    wallet_id (hashed address, 0 = unknown), size_adv = amount / trailing mean DAILY volume of the token over the preceding
    adv_days days (strictly before the trade's day; NaN when there is no earlier day), size_raw = amount, taker_buy.
    trailing='cumulative' (token launches, where no earlier day exists): size_adv = amount / cumulative volume of ALL trades
    of the token strictly before this trade (NaN for the first trade) - still causal, but a share of volume-so-far, not of ADV.
    Returns also an `entities` attribute (id -> sha1 prefix) in .attrs['entities']."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["time_ns", "wallet_id", "size_adv", "size_raw", "taker_buy"])
    t = trades.sort_values("time_ns", kind="stable").reset_index(drop=True)
    if side == "taker":
        t = t[t["taker_buy"].astype(bool)].reset_index(drop=True)
        who = t["buyer"]
    else:
        who = t["buyer"]
    if len(t) == 0:
        return pd.DataFrame(columns=["time_ns", "wallet_id", "size_adv", "size_raw", "taker_buy"])
    day = (t["time_ns"].to_numpy() // (86400 * 10 ** 9)).astype(np.int64)
    amt = t["amount"].to_numpy(dtype=np.float64)
    daily = pd.Series(amt).groupby(day).sum()
    days = daily.index.to_numpy(); vols = daily.to_numpy(); cum = np.concatenate([[0.0], np.cumsum(vols)])
    lo = np.searchsorted(days, day - int(adv_days), side="left"); hi = np.searchsorted(days, day, side="left")
    trail_vol = cum[hi] - cum[lo]
    n_days = np.minimum(int(adv_days), np.maximum(day - days[0], 0)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        adv = np.where(n_days > 0, trail_vol / np.where(n_days > 0, n_days, np.nan), np.nan)
        size_adv = np.where(adv > 0, amt / adv, np.nan)
    if trailing == "cumulative":
        full = trades.sort_values("time_ns", kind="stable")
        ft, fa = full["time_ns"].to_numpy(), full["amount"].to_numpy(dtype=np.float64)
        cumv = np.concatenate([[0.0], np.cumsum(fa)])
        prior = cumv[np.searchsorted(ft, t["time_ns"].to_numpy(), side="left")]          # volume of trades with an earlier timestamp
        with np.errstate(divide="ignore", invalid="ignore"):
            size_adv = np.where(prior > 0, amt / np.where(prior > 0, prior, np.nan), np.nan)
    ids, table = hash_ids(who.tolist(), salt)
    out = pd.DataFrame({"time_ns": t["time_ns"].to_numpy().astype(np.int64), "wallet_id": ids, "size_adv": size_adv, "size_raw": amt, "taker_buy": t["taker_buy"].to_numpy().astype(bool)})
    out.attrs["entities"] = table
    return out


def wash_trade_flags(trades: pd.DataFrame, window_s: float = 3600.0, tol: float = 0.05) -> np.ndarray:
    """Cheap subset of the Victor & Weintraud (WWW 2021) wash-trading rule: a trade is flagged when (a) buyer == seller
    (self-trade) or (b) it closes a 2-cycle: the SAME pair of addresses traded the token in the opposite direction within
    window_s seconds with amounts matching within `tol` (relative). 3-cycles are NOT implemented (documented in the loader).
    Returns a bool array aligned with `trades` (any order)."""
    n = len(trades)
    flag = np.zeros(n, dtype=bool)
    if n == 0:
        return flag
    b = trades["buyer"].astype(str).to_numpy(); s = trades["seller"].astype(str).to_numpy()
    flag |= (b == s) & (b != "")
    lo = np.where(b < s, b, s); hi = np.where(b < s, s, b)
    df = pd.DataFrame({"pair": pd.Series(lo).str.cat(pd.Series(hi), sep="|").to_numpy(), "t": trades["time_ns"].to_numpy().astype(np.int64),
                       "a": trades["amount"].to_numpy(dtype=np.float64), "dir": (b < s), "i": np.arange(n)})
    df = df[(b != "") & (s != "") & (b != s)].sort_values(["pair", "t"], kind="stable")
    if len(df) < 2:
        return flag
    same = df["pair"].to_numpy()[1:] == df["pair"].to_numpy()[:-1]
    opp = df["dir"].to_numpy()[1:] != df["dir"].to_numpy()[:-1]
    dt = (df["t"].to_numpy()[1:] - df["t"].to_numpy()[:-1]) <= window_s * 10 ** 9
    a0, a1 = df["a"].to_numpy()[:-1], df["a"].to_numpy()[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        close = np.abs(a1 - a0) <= tol * np.maximum(a0, a1)
    cyc = same & opp & dt & close
    idx = df["i"].to_numpy()
    flag[idx[:-1][cyc]] = True; flag[idx[1:][cyc]] = True
    return flag


# ------------------------------------------------------------------------------------------------ windows
def episode_window_ns(e: Dict[str, Any], pre_bars: int, post_bars: int, bar_minutes: int) -> Tuple[int, int]:
    """[open of row 0, close of row T-1) of an episode in UTC ns (same grid as common.episode_panel)."""
    step = int(bar_minutes) * MIN_NS
    a0 = (int(e["anchor_ns"]) // step) * step
    return a0 - pre_bars * step, a0 + (post_bars + 1) * step


def assign_episode_keys(df: pd.DataFrame, episodes: Sequence[Dict[str, Any]], pre_bars: int, post_bars: int, bar_minutes: int,
                        symbol_col: str = "symbol", max_per_episode: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """df (symbol, time_ns, ...) -> the same rows tagged with `key` for every episode whose window contains them (a row can
    appear under several episodes when windows overlap). max_per_episode: seeded random subsample when exceeded (the number
    dropped is in .attrs['subsampled'])."""
    parts, dropped = [], {}
    by_sym: Dict[str, pd.DataFrame] = {s: g.sort_values("time_ns") for s, g in df.groupby(symbol_col)} if len(df) else {}
    for e in episodes:
        g = by_sym.get(e["symbol"])
        if g is None:
            continue
        lo, hi = episode_window_ns(e, pre_bars, post_bars, bar_minutes)
        t = g["time_ns"].to_numpy()
        i0, i1 = np.searchsorted(t, lo, side="left"), np.searchsorted(t, hi, side="left")
        if i1 <= i0:
            continue
        w = g.iloc[i0:i1]
        if max_per_episode is not None and len(w) > max_per_episode:
            r = rng or np.random.default_rng(0)
            keep = np.sort(r.choice(len(w), max_per_episode, replace=False))
            dropped[e["key"]] = int(len(w) - max_per_episode)
            w = w.iloc[keep]
        w = w.copy(); w["key"] = e["key"]
        parts.append(w)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(df.columns) + ["key"])
    out.attrs["subsampled"] = dropped
    return out


# ------------------------------------------------------------------------------------------------ big csv
def chunked_filter_csv(path: str, keep: Callable[[pd.DataFrame], np.ndarray], usecols: Optional[Sequence[str]] = None, chunksize: int = 500_000,
                       cache_path: Optional[str] = None, dtype: Optional[Dict[str, Any]] = None, transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                       verbose: bool = True) -> pd.DataFrame:
    """Stream `path` in chunks, keep rows where keep(chunk) is True (optionally transform the kept rows), and cache the
    result as csv at cache_path (read back on the next call; delete the cache to rebuild). Works on a partially downloaded
    file (the last broken line is skipped)."""
    if cache_path and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return pd.read_csv(cache_path, low_memory=False)
    parts, n_in = [], 0
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=dtype, low_memory=False, on_bad_lines="skip")
    for ch in reader:
        n_in += len(ch)
        m = np.asarray(keep(ch), dtype=bool)
        if m.any():
            sub = ch[m]
            parts.append(transform(sub) if transform is not None else sub)
        if verbose and n_in % (chunksize * 4) == 0:
            print(f"[chunked_filter_csv] {os.path.basename(path)}: {n_in:,} rows read, {sum(len(p) for p in parts):,} kept", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(usecols) if usecols else [])
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_csv(cache_path, index=False)
    return out


# ------------------------------------------------------------------------------------------------ optional CEX wallet file
def optional_transfers(root: str, dataset: str, path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """<root>/crosspump/onchain/<dataset>_transfers.csv with columns key, time_ns, wallet_id, size_adv, size_raw (the shared
    optional wallet file for exchange-traded sets). None when absent: the wallet modality is then MISSING (coverage says so)."""
    p = path or os.path.join(root, "crosspump", "onchain", f"{dataset}_transfers.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    need = {"key", "time_ns", "wallet_id"}
    if not need <= set(df.columns):
        raise ValueError(f"{p}: columns must include {sorted(need)} (+ size_adv, size_raw); got {list(df.columns)}")
    df["time_ns"] = pd.to_numeric(df["time_ns"], errors="coerce").astype(np.int64)
    if df["wallet_id"].dtype == object:
        df["wallet_id"], _ = hash_ids(df["wallet_id"].tolist())
    return df


# ------------------------------------------------------------------------------------------------ degenerate panel
def tabular_panel(episodes: List[Dict[str, Any]], pre_bars: int, post_bars: int, bar_minutes: int, lead: int, horizons: Sequence[int],
                  feat_at_anchor: np.ndarray, mid_at_anchor: np.ndarray, dataset: str, source: str, licence: str,
                  extra_meta: Optional[Dict[str, Any]] = None, tradable_at_anchor: bool = False) -> Panel:
    """Stacked-episode Panel for a source that has ONE tabular row per project: row `pre_bars` (the anchor) carries the
    feature row and (if known) a price; every other row is NaN and untradable (the anchor row is tradable only when
    tradable_at_anchor=True AND a price is given; default False: a single tabular price is not an executable quote). Labels: 1 on [anchor - lead, anchor] for
    is_pump episodes (peak_row = anchor: no peak exists), bars_to_peak = anchor - row there, avail_label = anchor row.
    meta['tabular_only'] = True and meta['excluded_from'] = ['lead_time'] so the lead-time metrics skip this dataset.
    episodes: dicts with key, symbol, anchor_ns, is_pump (+ anything else, kept in meta['episodes']); feat_at_anchor (N, 5)
    in the FEAT_NAMES order (NaN = unknown); mid_at_anchor (N,) (NaN = no price knowable at the anchor)."""
    order = np.argsort([int(e["anchor_ns"]) for e in episodes], kind="stable")
    eps = [episodes[i] for i in order]
    F = np.asarray(feat_at_anchor, dtype=np.float64)[order]
    M = np.asarray(mid_at_anchor, dtype=np.float64)[order]
    T, N = pre_bars + post_bars + 1, len(eps)
    step = bar_minutes * MIN_NS
    mid = np.full((T, N), np.nan); mid[pre_bars] = M
    feats = np.full((T, N, len(FEAT_NAMES)), np.nan, dtype=np.float32); feats[pre_bars] = F[:, : len(FEAT_NAMES)]
    high = low = mid.copy()
    vol = np.full((T, N), np.nan); vol[pre_bars] = np.where(np.isfinite(F[:, 1]), np.expm1(np.clip(F[:, 1], 0, 50)), np.nan)
    liq = liquidity_fields(mid, vol, high, low)
    tradable = np.zeros((T, N), dtype=bool); tradable[pre_bars] = np.isfinite(M) & bool(tradable_at_anchor)
    labels = np.zeros((T, N), dtype=np.int8); btp = np.full((T, N), np.nan)
    hmax = int(max(horizons))
    avail_label = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None] + hmax, (T, N)).copy()
    is_pump = np.array([bool(e.get("is_pump", False)) for e in eps]); peak_row = np.full(N, pre_bars, dtype=np.int64)
    lo = max(0, pre_bars - lead)
    for i in range(N):
        if is_pump[i]:
            rows = np.arange(lo, pre_bars + 1); labels[rows, i] = 1; btp[rows, i] = pre_bars - rows; avail_label[rows, i] = pre_bars
    anchor_ns = np.array([(int(e["anchor_ns"]) // step) * step for e in eps], dtype=np.int64)
    bar_close_ns = anchor_ns[None, :] + (np.arange(T, dtype=np.int64) - pre_bars + 1)[:, None] * step
    episodes_meta = [{**e, "peak_row": int(pre_bars), "peak_source": "undefined (tabular: peak_row = anchor row)"} for e in eps]
    meta: Dict[str, Any] = {"freq": f"{bar_minutes}min", "bars_per_year": int(365 * 24 * 60 / bar_minutes), "horizons": list(horizons), "raw_features": True, "ret_col": 0,
                            "feat_names": list(FEAT_NAMES), "dataset": dataset, "source": source, "licence": licence, "bar_minutes": bar_minutes, "pre_bars": pre_bars,
                            "post_bars": post_bars, "lead": lead, "anchor_ns": anchor_ns, "peak_row": peak_row, "is_pump": is_pump, "episodes": episodes_meta,
                            "split_axis": "assets", "assets_time_aligned": False, "bar_close_ns": bar_close_ns, "n_pumps": int(is_pump.sum()), "n_controls": int((~is_pump).sum()),
                            "tabular_only": True, "excluded_from": ["lead_time"], "has_taker_buy": False, "has_trade_count": bool(np.isfinite(F[:, 4]).any()),
                            "label_rule": {"type": "tabular", "label_on": "[anchor - lead, anchor] (no peak exists)", "bars_to_peak": "anchor - row on the labelled rows (degenerate)", "peak": "undefined"},
                            "times": "bar index; per-episode UTC close times in meta['bar_close_ns'] (T, N); only row pre_bars carries data"}
    meta.update(extra_meta or {})
    panel = Panel(times=np.arange(T, dtype=np.int64), assets=[e["key"] for e in eps], mid=mid, price_feat=feats, liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp, meta=meta)
    panel.validate()
    attach_availability(panel, avail_label=avail_label, universe=np.isfinite(mid))
    default_availability(panel)
    warnings.warn(f"{dataset}: TABULAR-ONLY source -> degenerate panel (one feature row per episode at the anchor; no bars, no peak); "
                  f"meta['tabular_only'] = True, excluded from lead-time metrics", stacklevel=2)
    return panel


# ------------------------------------------------------------------------------------------------ xlsx without openpyxl
def read_xlsx(path: str, sheet: int = 1) -> pd.DataFrame:
    """First (or `sheet`-th, 1-based file order) worksheet of an .xlsx as a DataFrame of STRINGS (None = empty cell); row 1
    is the header (stripped; empty header cells dropped). Uses only zipfile + xml (openpyxl is not installed on the
    cluster). Numbers and Excel date serials come back as their stored text; convert with pd.to_numeric / excel_serial_to_ns."""
    import re
    import zipfile
    import xml.etree.ElementTree as ET
    M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            shared = ["".join(t.text or "" for t in si.iter(M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(M + "si")]
        sheets = sorted((n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        if len(sheets) < sheet:
            raise ValueError(f"{path}: no worksheet #{sheet} (found {sheets})")
        root = ET.fromstring(z.read(sheets[sheet - 1]))
    rows: List[Dict[str, Optional[str]]] = []
    for row in root.find(M + "sheetData").findall(M + "row"):
        d: Dict[str, Optional[str]] = {}
        for c in row.findall(M + "c"):
            col = re.sub(r"\d", "", c.attrib.get("r", ""))
            t = c.attrib.get("t")
            if t == "inlineStr":
                d[col] = "".join(x.text or "" for x in c.iter(M + "t"))
                continue
            v = c.find(M + "v")
            if v is None or v.text is None:
                continue
            d[col] = shared[int(v.text)] if t == "s" else v.text
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    header = {col: str(name).strip() for col, name in rows[0].items() if name is not None and str(name).strip()}
    return pd.DataFrame([{name: r.get(col) for col, name in header.items()} for r in rows[1:]], columns=list(header.values()))


def excel_serial_to_ns(v: Any) -> np.ndarray:
    """Excel date serial (days since 1899-12-30) -> int64-valued float UTC ns (NaN where not numeric)."""
    x = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=np.float64)
    return (x - 25569.0) * 86400.0 * 1e9


def to_number(v: Any) -> np.ndarray:
    """Spreadsheet text -> float (thousands separators and non-breaking spaces removed; '-' / text -> NaN)."""
    s = pd.Series(v).astype(str).str.replace("\u00a0", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def iso_minute(ns: int) -> str:
    """UTC ns -> 'YYYY-MM-DDTHH:MMZ' (the timestamp part of an episode key)."""
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").strftime("%Y-%m-%dT%H:%MZ")


def make_keys_unique(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Episode keys must be unique (they are the Panel's asset names): repeated keys get a '~k' suffix, in place."""
    seen: Dict[str, int] = {}
    for e in episodes:
        k = e["key"]
        if k in seen:
            seen[k] += 1
            e["key"] = f"{k}~{seen[k]}"
        else:
            seen[k] = 0
    return episodes
