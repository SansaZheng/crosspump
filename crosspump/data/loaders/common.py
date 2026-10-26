"""Helpers shared by the D1-D10 loaders (see crosspump/CONVENTIONS.md §1-§3).

    register_dataset / require                 tolerant registry + "missing raw file" error with the download recipe
    read_binance_klines / find_kline_files     data.binance.vision 1-minute klines (12 header-less columns or headed csv)
    resample_bars                              1-minute bars -> bar_minutes bars (OHLCV, taker buy volume, trade count)
    sample_controls                            control windows (same coin, no pump within +-gap; or another coin, same time)
    episode_panel                              episodes + bars -> Panel in the stacked-episode layout (labels, PIT fields, meta)
    social_stream / wallet_stream / orderflow_stream_from_trades   EventStreams parts from message / transfer / trade tables
    hash_embed                                 dependency-free hashed bag-of-words embedding for optional message text
Nothing here fabricates data: every function transforms tables the loader read from disk.
"""
from __future__ import annotations
import glob
import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.schema import Panel
from warn2trade.utils.registry import DATASETS
from causalgate.data.pit import attach_availability, default_availability
from ..streams import EventStreams, attach_streams, add_prior_involvement, derive_orderflow_events, PLATFORMS

MIN_NS = 60 * 10 ** 9
FEAT_NAMES = ["log_ret", "log_volume", "hl_range", "taker_imb", "log_trades"]
BINANCE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def register_dataset(name: str):
    def deco(cls):
        try:
            DATASETS.register(name)(cls)
        except KeyError:
            pass
        return cls
    return deco


def require(root: str, paths: Iterable[str], recipe: str) -> None:
    missing = [p for p in paths if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {root}.\nDownload recipe:\n{recipe}\n(see scripts/xp_prepare_data.py)")


def to_ns(x) -> np.ndarray:
    return pd.to_datetime(x, utc=True).values.astype("datetime64[ns]").astype(np.int64)


# ------------------------------------------------------------------------------------------------ bars
def _epoch_to_ns(v: pd.Series) -> np.ndarray:
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.5:
        x = num.to_numpy(dtype=np.float64); m = np.nanmedian(x)
        unit = 10 ** 3 if m >= 1e14 else (10 ** 6 if m >= 1e11 else 10 ** 9)
        return np.where(np.isfinite(x), x * unit, np.nan)
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    return ts.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)


def read_binance_klines(paths: Sequence[str]) -> pd.DataFrame:
    """-> DataFrame indexed by open_time (int64 UTC ns, minute grid) with open/high/low/close/volume[/taker_buy_volume][/count]."""
    frames = []
    for p in paths:
        with open(p, "r") as f:
            first = f.readline().strip()
        tok = first.split(",")[0].strip()
        headed = not tok.replace(".", "", 1).replace("-", "", 1).isdigit()
        df = pd.read_csv(p, header=0 if headed else None)
        if not headed:
            df.columns = BINANCE_COLS[: df.shape[1]] + [f"extra{k}" for k in range(max(0, df.shape[1] - len(BINANCE_COLS)))]
        cols = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
        out = pd.DataFrame()
        tcol = next((cols[k] for k in ("open_time", "timestamp", "time", "date", "datetime") if k in cols), None)
        if tcol is None:
            raise ValueError(f"{p}: no open_time column in {list(df.columns)}")
        out["open_time"] = _epoch_to_ns(df[tcol])
        for k, aliases in {"open": ("open",), "high": ("high",), "low": ("low",), "close": ("close",), "volume": ("volume", "vol"),
                           "taker_buy_volume": ("taker_buy_volume", "taker_buy_base_asset_volume", "taker_buy_base_volume"), "count": ("count", "trades", "number_of_trades", "n_trades")}.items():
            c = next((cols[a] for a in aliases if a in cols), None)
            if c is not None:
                out[k] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["open_time"])]
    df["open_time"] = (df["open_time"].to_numpy() // MIN_NS * MIN_NS).astype(np.int64)
    df = df.drop_duplicates("open_time", keep="last").set_index("open_time").sort_index()
    for k in ("open", "high", "low", "close", "volume"):
        if k not in df:
            raise ValueError(f"{paths[0]}: kline column {k} missing")
    return df


def find_kline_files(kdir: str, symbol: str, quote: str = "BTC") -> List[str]:
    pair = f"{symbol}{quote}"
    for pattern in (f"{pair}.csv", f"{pair}-1m*.csv", f"{pair}_1m*.csv", f"{pair}/*.csv", f"{symbol}_{quote}.csv", f"{symbol}-{quote}.csv", f"{symbol}.csv"):
        hits = sorted(glob.glob(os.path.join(kdir, pattern))) + sorted(glob.glob(os.path.join(kdir, pattern.lower())))
        if hits:
            return sorted(set(hits))
    return []


def resample_bars(kl: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """1-minute klines -> bar_minutes bars on a grid aligned to the epoch (index = bar open_time ns)."""
    if bar_minutes <= 1:
        return kl
    step = bar_minutes * MIN_NS
    g = (kl.index.to_numpy() // step) * step
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for k in ("taker_buy_volume", "count"):
        if k in kl:
            agg[k] = "sum"
    out = kl.groupby(g).agg(agg)
    out.index.name = "open_time"
    return out


# ------------------------------------------------------------------------------------------------ controls
def sample_controls(episodes: List[Dict[str, Any]], pump_times_by_symbol: Dict[str, np.ndarray], n_per_pump: int, gap_hours: float, rng: np.random.Generator,
                    span_days: float = 30.0, other_symbols: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Control windows: for every pump episode draw n_per_pump anchors (a) on the SAME coin, uniformly inside
    +-span_days around the pump but at least gap_hours away from any pump of that coin, and (b) when other_symbols is
    given, alternate with ANOTHER coin at the same anchor time (cross-sectional control). Returns episode dicts with
    is_pump False and control_of = key of the pump."""
    out = []
    gap = gap_hours * 3600 * 10 ** 9; span = span_days * 86400 * 10 ** 9
    for e in episodes:
        if not e.get("is_pump", True):
            continue
        for k in range(n_per_pump):
            if other_symbols is not None and k % 2 == 1:
                sym = str(rng.choice([s for s in other_symbols if s != e["symbol"]] or [e["symbol"]]))
                a = int(e["anchor_ns"])
            else:
                sym = e["symbol"]
                for _ in range(50):
                    a = int(e["anchor_ns"] + rng.uniform(-span, span))
                    pt = pump_times_by_symbol.get(sym, np.zeros(0))
                    if len(pt) == 0 or np.abs(pt - a).min() > gap:
                        break
                else:
                    continue
            out.append({"key": f"CTRL{len(out):05d}_{sym}", "symbol": sym, "exchange": e.get("exchange"), "channel": None, "anchor_ns": a, "peak_ns": None,
                        "is_pump": False, "control_of": e["key"], "source": e.get("source")})
    return out


# ------------------------------------------------------------------------------------------------ panel
def liquidity_fields(close: np.ndarray, volume: np.ndarray, high: np.ndarray, low: np.ndarray, adv_window: int = 288) -> np.ndarray:
    T, N = close.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        hs = np.clip(np.nan_to_num(0.1 * 0.5 * (high - low) / close, nan=5e-4), 1e-5, 0.05)
        dv = pd.DataFrame(np.nan_to_num(volume * close)).rolling(adv_window, min_periods=1).mean().to_numpy()
        med = np.nanmedian(dv[dv > 0]) if (dv > 0).any() else 1.0
        adv_frac = np.clip(np.where(dv > 0, 1.0 / np.maximum(dv / med, 1e-6), 0.02), 1e-4, 1.0)
    return np.stack([hs, np.ones((T, N)), adv_frac], -1).astype(np.float32)


def episode_panel(bars: Dict[str, pd.DataFrame], episodes: List[Dict[str, Any]], pre_bars: int, post_bars: int, bar_minutes: int, lead: int,
                  horizons: Sequence[int], peak_window_bars: int = 12, end_bars_after_peak: int = 12, min_bars_frac: float = 0.5,
                  dataset: str = "", source: str = "", licence: str = "", extra_meta: Optional[Dict[str, Any]] = None) -> Tuple[Panel, Dict[str, int]]:
    """episodes (dicts with key, symbol, anchor_ns, peak_ns|None, is_pump, ...) + bars[symbol] (bar_minutes bars indexed by
    open_time ns) -> Panel. Episodes whose symbol has no bars or too few bars inside the window are dropped (counted).
    Peak = curated peak_ns when given, else argmax high within peak_window_bars after the anchor (meta records which)."""
    T = pre_bars + post_bars + 1
    step = bar_minutes * MIN_NS
    skipped = {"no_bars": 0, "too_few_bars": 0, "no_peak": 0}
    cols: List[Dict[str, Any]] = []
    for e in sorted(episodes, key=lambda d: int(d["anchor_ns"])):
        kl = bars.get(e["symbol"])
        if kl is None or len(kl) == 0:
            skipped["no_bars"] += 1; continue
        a0 = (int(e["anchor_ns"]) // step) * step
        grid = a0 + (np.arange(T, dtype=np.int64) - pre_bars) * step
        w = kl.reindex(grid)
        if w["close"].notna().sum() < max(10, int(min_bars_frac * T)):
            skipped["too_few_bars"] += 1; continue
        close_ns = grid + step
        peak_t, src = -1, None
        if e.get("is_pump", True):
            if e.get("peak_ns"):
                peak_t = int(np.clip(np.searchsorted(close_ns, int(e["peak_ns"]), side="left"), 0, T - 1)); src = "curated"
            else:
                hi = w["high"].to_numpy()[pre_bars : min(T, pre_bars + peak_window_bars + 1)]
                if not np.isfinite(hi).any():
                    skipped["no_peak"] += 1; continue
                peak_t = pre_bars + int(np.nanargmax(hi)); src = "derived_argmax_high"
        cols.append({**e, "peak_row": peak_t, "peak_source": src, "close_ns": close_ns, "w": w})
    if not cols:
        raise FileNotFoundError(f"No episode with bars (skipped: {skipped})")
    N = len(cols)
    get = lambda k: np.stack([c["w"][k].to_numpy(dtype=np.float64) if k in c["w"] else np.full(T, np.nan) for c in cols], 1)
    close, high, low, vol = get("close"), get("high"), get("low"), get("volume")
    tbv, cnt = get("taker_buy_volume"), get("count")
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.full((T, N), np.nan); lr[1:] = np.log(close[1:]) - np.log(close[:-1])
        lv = np.log1p(np.where(vol > 0, vol, np.nan))
        hl = (high - low) / close
        timb = np.where(np.isfinite(tbv) & (vol > 0), tbv / np.where(vol > 0, vol, np.nan) - 0.5, 0.0)
        lt = np.log1p(np.where(cnt >= 0, cnt, np.nan))
    feats = np.stack([lr, lv, hl, timb, lt], -1).astype(np.float32)
    liq = liquidity_fields(close, vol, high, low)
    tradable = np.isfinite(close) & (np.nan_to_num(vol) > 0)
    labels = np.zeros((T, N), dtype=np.int8); btp = np.full((T, N), np.nan)
    hmax = int(max(horizons))
    avail_label = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None] + hmax, (T, N)).copy()
    is_pump = np.array([bool(c.get("is_pump", True)) for c in cols]); peak_row = np.array([c["peak_row"] for c in cols], dtype=np.int64)
    for i, c in enumerate(cols):
        if not is_pump[i]:
            continue
        pk = int(c["peak_row"]); end = min(T - 1, pk + end_bars_after_peak)
        lo = max(0, pre_bars - lead); labels[lo : pk + 1, i] = 1
        rows = np.arange(lo, end + 1); btp[rows, i] = pk - rows; avail_label[rows, i] = end
    anchor_ns = np.array([int(c["anchor_ns"]) for c in cols], dtype=np.int64)
    bar_close_ns = np.stack([c["close_ns"] for c in cols], 1)
    episodes_meta = [{k: v for k, v in c.items() if k not in ("w", "close_ns")} for c in cols]
    meta: Dict[str, Any] = {"freq": f"{bar_minutes}min", "bars_per_year": int(365 * 24 * 60 / bar_minutes), "horizons": list(horizons), "raw_features": True, "ret_col": 0,
                            "feat_names": list(FEAT_NAMES), "dataset": dataset, "source": source, "licence": licence, "bar_minutes": bar_minutes, "pre_bars": pre_bars,
                            "post_bars": post_bars, "lead": lead, "anchor_ns": anchor_ns, "peak_row": peak_row, "is_pump": is_pump, "episodes": episodes_meta,
                            "split_axis": "assets", "assets_time_aligned": False, "bar_close_ns": bar_close_ns, "n_pumps": int(is_pump.sum()), "n_controls": int((~is_pump).sum()),
                            "skipped": skipped, "has_taker_buy": bool(np.isfinite(tbv).any()), "has_trade_count": bool(np.isfinite(cnt).any()),
                            "label_rule": {"type": "event_table", "label_on": "[anchor - lead, peak]", "bars_to_peak": "peak - row (exact)", "peak": "curated peak_ns else argmax high within peak_window_bars"},
                            "times": "bar index; per-episode UTC close times in meta['bar_close_ns'] (T, N)"}
    meta.update(extra_meta or {})
    panel = Panel(times=np.arange(T, dtype=np.int64), assets=[c["key"] for c in cols], mid=close, price_feat=feats, liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp, meta=meta)
    panel.validate()
    attach_availability(panel, avail_label=avail_label, universe=np.isfinite(close))
    default_availability(panel)
    return panel, skipped


def bar_units(panel: Panel, asset: int, t_ns: np.ndarray) -> np.ndarray:
    """UTC ns -> fractional bar units of one episode column (row = floor): time inside the panel grid."""
    bc = panel.meta["bar_close_ns"][:, asset]
    step = bc[1] - bc[0]
    return (np.asarray(t_ns, dtype=np.float64) - (bc[0] - step)) / step


# ------------------------------------------------------------------------------------------------ stream parts
def asset_index(panel: Panel) -> Dict[str, int]:
    """episode key -> column index AFTER episode_panel sorted the episodes chronologically (always join through this)."""
    return {k: i for i, k in enumerate(panel.assets)}


def _stream_part(panel: Panel, df: pd.DataFrame, platform: int, lag_bars: float, mark_cols: Sequence[Optional[str]], source_col: Optional[str]) -> EventStreams:
    """df columns: `key` (episode key, preferred) or `asset` (column index), time_ns (int64); mark_cols -> mark[:, k];
    source_col -> source id. Events outside the window or for unknown keys are dropped."""
    if df is None or len(df) == 0:
        return EventStreams.empty()
    T = panel.T
    if "key" in df:
        idx = asset_index(panel)
        a = np.array([idx.get(str(k), -1) for k in df["key"]], dtype=np.int64)
        df = df[a >= 0].reset_index(drop=True); a = a[a >= 0]
        if len(df) == 0:
            return EventStreams.empty()
    else:
        a = df["asset"].to_numpy().astype(np.int64)
    tb = np.array([bar_units(panel, int(i), np.array([ts]))[0] for i, ts in zip(a, df["time_ns"].to_numpy())])
    ok = (tb >= 0) & (tb < T - 1e-6)
    marks = [[(float(df[c].iloc[k]) if c is not None and c in df else np.nan) for c in mark_cols] for k in np.nonzero(ok)[0]]
    src = df[source_col].to_numpy()[ok].astype(np.int64) if source_col is not None and source_col in df else np.zeros(int(ok.sum()), np.int64)
    return EventStreams.from_lists(a[ok], tb[ok], [platform] * int(ok.sum()), marks, np.minimum(tb[ok] + lag_bars, T - 1e-6), src)


def social_stream(panel: Panel, msgs: pd.DataFrame, lag_bars: float = 0.0) -> EventStreams:
    """msgs: key (or asset), time_ns, [reach], [text_score], [source_id]. mark = [1, reach, text_score, NaN]."""
    df = msgs.copy(); df["one"] = 1.0
    return _stream_part(panel, df, 0, lag_bars, ["one", "reach", "text_score", None], "source_id")


def wallet_stream(panel: Panel, transfers: pd.DataFrame, lag_bars: float = 1.0) -> EventStreams:
    """transfers: key (or asset), time_ns, [size_adv] (transfer size / ADV), [size_raw], [wallet_id]. mark = [log1p size_adv, size_raw, NaN, NaN]."""
    df = transfers.copy()
    if "size_adv" in df:
        df["lsize"] = np.log1p(np.clip(df["size_adv"].to_numpy(dtype=np.float64), 0, None))
    return _stream_part(panel, df, 1, lag_bars, ["lsize", "size_raw", None, None], "wallet_id")


def orderflow_stream_from_trades(panel: Panel, bursts: pd.DataFrame) -> EventStreams:
    """bursts: key (or asset), time_ns, z (burst magnitude), [imbalance]. mark = [z, imbalance, NaN, NaN]; avail = time (knowable at the trade)."""
    return _stream_part(panel, bursts, 2, 0.0, ["z", "imbalance", None, None], None)


def finish_streams(panel: Panel, parts: Sequence[EventStreams], derive_orderflow_if_missing: bool = True, **meta) -> Panel:
    parts = list(parts)
    if derive_orderflow_if_missing and not any(len(p) and (p.platform == 2).any() for p in parts):
        parts.append(derive_orderflow_events(panel))
        meta["orderflow"] = "derived causally from bar features (crosspump.data.streams.derive_orderflow_events)"
    es = EventStreams.concat(parts, **meta)
    es = add_prior_involvement(es, np.argsort(np.argsort(panel.meta["anchor_ns"])), np.asarray(panel.meta["is_pump"]))
    return attach_streams(panel, es)


# ------------------------------------------------------------------------------------------------ text
_TOK = re.compile(r"[a-z0-9$#@]{2,}")


def hash_embed(texts: Iterable[str], dim: int = 64) -> np.ndarray:
    """Hashed, log-tf bag-of-words embedding (dependency-free; declare cutoff n/a: nothing is learned)."""
    out = []
    for s in texts:
        v = np.zeros(dim, dtype=np.float32)
        for tok in _TOK.findall(str(s).lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(v)
        out.append(np.log1p(np.abs(v)) * np.sign(v) / (np.log1p(n) + 1e-6))
    return np.stack(out) if out else np.zeros((0, dim), np.float32)
