"""Shared message / text / cache helpers for the group-B CrossPump loaders (D4 PumpSense, D5 Moura + TGDataset,
D7 Mirtaheri). Everything here transforms what a loader read from disk; nothing fabricates messages, prices or labels.

    parse_telegram_text   Telegram-export text field -> plain string. Handles the stringified entity lists written by the
                          Telegram Desktop JSON export ("[{'type': 'bold', 'text': 'Coin:'}, ' $XYZ']"), json lists, NaN.
    text_score / text_scores
                          dependency-free keyword score in [0, 1]: how announcement-like a message reads
                          (1 - exp(-sum of lexicon weights)). It is a crude lexicon, not a classifier; labelled
                          announcements get 1.0 from the loader, never from this function.
    extract_tickers       uppercase / cashtag tokens of a message that belong to a given ticker universe
    stable_id             deterministic positive int64 id for a group / channel / sender name (0 = unknown is reserved)
    cached_hash_embed     common.hash_embed with an on-disk npz cache keyed by a content fingerprint (nothing is learned,
                          cutoff n/a); the cache is a pure speed-up and is rebuilt when the fingerprint differs
    place_text            in-window messages -> Panel.text slots (T, N, M, D) + text_mask / text_age / avail_text, keeping
                          per (row, episode) the M latest messages; a message sits at row ceil(avail) = the first row at
                          which its stream event is usable (usable at row t iff avail <= t), avail_text = that row
    text_budget_ok        size guard for Panel.text (the stacked layout can reach GBs for thousands of episodes)
    months_covering / episode_months / load_bars
                          Binance monthly kline cache helpers (<root>/crosspump/binance_klines/<SYMBOL><QUOTE>-1m-YYYY-MM.csv)
    window_ns / pump_key  episode window bounds and the shared key format f"{SYMBOL}{quote}@{YYYY-MM-DDTHH:MMZ}"
    optional_wallet_stream
                          the OPTIONAL <root>/crosspump/onchain/<dataset>_transfers.csv wallet modality (absent = missing)
    stream_part / social_part / finish / fast_prior_involvement
                          vectorised equivalents of common._stream_part / social_stream / finish_streams (identical output,
                          asserted by the tests) + an explicit availability column for events known only later
    build_panel           shared D4/D5/D7 assembly: kline coverage, controls, common.episode_panel, streams, text, meta
    subsample_evenly      deterministic max_pumps cap that keeps the chronological spread
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.schema import Panel
from causalgate.data.pit import attach_availability
from ..streams import EventStreams, attach_streams, derive_orderflow_events
from . import common
from .common import hash_embed, find_kline_files, read_binance_klines, resample_bars, asset_index, wallet_stream, MIN_NS

NS_PER_H = 3600 * 10 ** 9
NS_PER_DAY = 86400 * 10 ** 9


# ------------------------------------------------------------------------------------------------ message text
def _flatten(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, (str, list, tuple, dict)):
            yield from _flatten(t)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _flatten(o)


def parse_telegram_text(x: Any) -> str:
    """Telegram Desktop export `text` value -> plain string. The csv tables of PumpSense keep the export's python repr
    of the entity list for formatted messages; plain messages are plain strings; NaN -> ''."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, (list, tuple, dict)):
        return "".join(_flatten(x))
    s = str(x)
    st = s.strip()
    if st.startswith("[") and st.endswith("]"):
        obj = None
        try:
            obj = ast.literal_eval(st)
        except Exception:
            try:
                obj = json.loads(st)
            except Exception:
                obj = None
        if isinstance(obj, (list, tuple)):
            return "".join(_flatten(obj))
    return s


# weights of a crude pump-announcement lexicon (regex -> weight); score = 1 - exp(-sum of matched weights)
PUMP_LEXICON: Tuple[Tuple[str, float], ...] = (
    (r"\bpump(s|ed|ing)?\b", 1.0), (r"\bnext pump\b", 0.6), (r"\bthe coin (is|will be)\b", 1.0), (r"\bcoin (name|is|of the day)\b", 0.8),
    (r"\bbuy (now|order|zone|fast)\b", 0.8), (r"\bbuy\b", 0.35), (r"\bsell\b", 0.35), (r"\bhold\b", 0.3), (r"\btarget(s)?\b", 0.6),
    (r"\bsignal\b", 0.6), (r"\bcountdown\b", 0.7), (r"\bsats?\b|\bsatoshi\b", 0.6), (r"\bget ready\b|\bbe ready\b|\bready\b", 0.5),
    (r"\bmoon\b|\bto the moon\b", 0.4), (r"\b\d{2,4}\s?%", 0.4), (r"\bx\d+\b|\b\d+x\b", 0.4), (r"\bminutes?\b|\bmins?\b", 0.3),
    (r"\bexchange\b", 0.2), (r"\bbinance\b|\byobit\b|\bkucoin\b|\bbittrex\b|\bcryptopia\b|\bhotbit\b", 0.3), (r"\bprofit(s)?\b", 0.3),
    (r"\bvip\b|\bpremium\b", 0.3), (r"\bannounce(d|ment)?\b", 0.5), (r"\bstart(s|ed|ing)?\b", 0.3), (r"\bgo go\b|\bgogo\b", 0.5),
    (r"\bdump(ed|ing)?\b", 0.4), (r"\$[a-z0-9]{2,8}\b", 0.5), (r"\b[a-z0-9]{2,8}/btc\b|\b[a-z0-9]{2,8}btc\b", 0.6), (r"\bpremium\b", 0.2),
)
_LEX = [(re.compile(p, re.I), w) for p, w in PUMP_LEXICON]


def text_score(text: Any) -> float:
    """Keyword score in [0, 1] (NaN for an empty message). Deterministic, no model, no fitting."""
    s = parse_telegram_text(text) if not isinstance(text, str) else text
    if not s or not s.strip():
        return float("nan")
    w = 0.0
    for rx, wt in _LEX:
        if rx.search(s):
            w += wt
    return float(1.0 - math.exp(-w))


def text_scores(texts: Iterable[Any]) -> np.ndarray:
    return np.asarray([text_score(t) for t in texts], dtype=np.float64)


_TICKER = re.compile(r"\$?\b([A-Z0-9]{2,10})\b")


def extract_tickers(text: str, universe: Sequence[str], require_cashtag_below: int = 3) -> List[str]:
    """Tickers of `universe` mentioned in `text` (uppercase tokens or cashtags). Tokens shorter than
    `require_cashtag_below` characters must carry a '$' (avoids matching ordinary short words)."""
    uni = set(universe)
    out: List[str] = []
    for m in _TICKER.finditer(text or ""):
        tok = m.group(1)
        if tok in uni and tok not in out:
            if len(tok) < require_cashtag_below and not m.group(0).startswith("$"):
                continue
            out.append(tok)
    return out


def stable_id(name: Any, salt: str = "") -> int:
    """Deterministic positive int64 (< 2**62) for an entity name; 0 is reserved for 'unknown'."""
    h = int(hashlib.md5(f"{salt}{name}".encode("utf-8")).hexdigest()[:15], 16) % (2 ** 62)
    return int(h) or 1


# ------------------------------------------------------------------------------------------------ embedding cache
def embed_cache_path(root: str, dataset: str, dim: int) -> str:
    return os.path.join(root, "crosspump", "cache", f"{dataset}_hashembed_d{int(dim)}.npz")


def _fingerprint(texts: Sequence[str], dim: int) -> str:
    fp = hashlib.md5(f"hash_embed|dim={int(dim)}|n={len(texts)}|".encode())
    for t in texts:
        fp.update(hashlib.md5(str(t).encode("utf-8", "ignore")).digest())
    return fp.hexdigest()


def cached_hash_embed(texts: Sequence[str], dim: int, path: Optional[str] = None) -> np.ndarray:
    """common.hash_embed(texts, dim) with an optional npz cache (keys emb, fingerprint, meta). The embedding is a fixed
    hashed bag of words: nothing is fitted, so the cutoff is n/a and caching changes nothing but speed."""
    texts = [str(t) for t in texts]
    fp = _fingerprint(texts, dim)
    if path and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["fingerprint"]) == fp and z["emb"].shape == (len(texts), int(dim)):
                    return np.asarray(z["emb"], dtype=np.float32)
        except Exception:
            pass
    emb = hash_embed(texts, dim) if texts else np.zeros((0, int(dim)), np.float32)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, emb=emb, fingerprint=np.array(fp), meta=np.array(json.dumps({"embedder": "hash_embed", "dim": int(dim), "cutoff": "n/a (nothing learned)", "n": len(texts)})))
        except OSError:
            pass
    return emb


# ------------------------------------------------------------------------------------------------ Panel.text placement
def text_budget_ok(T: int, N: int, M: int, D: int, max_mb: float) -> bool:
    return (T * N * M * D * 4) / 1e6 <= float(max_mb)


def place_text(panel: Panel, keys: Sequence[str], time_ns: np.ndarray, emb: np.ndarray, slots: int, lag_bars: float = 0.0,
               avail_ns: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Place message embeddings into (T, N, M, D) slots. A message published at bar-unit time tb (availability
    av = max(tb, avail_ns in bar units) + lag_bars) is placed at row ceil(av): the FIRST decision row at which its stream
    event is usable (streams rule: usable at row t iff avail <= t), so text and stream agree and no model reads a
    message before its event. avail_text = that row (hence avail_text >= row and >= the publication bar); text_age =
    whole bars elapsed since the publication bar. Per (row, episode) the M latest messages are kept."""
    T, N = panel.T, panel.N
    M, D = int(slots), int(emb.shape[1]) if emb.ndim == 2 else 0
    idx = asset_index(panel)
    a = np.array([idx.get(str(k), -1) for k in keys], dtype=np.int64)
    time_ns = np.asarray(time_ns, dtype=np.int64)
    tb = np.full(len(a), -1.0); av = np.full(len(a), -1.0)
    has = a >= 0
    if has.any():
        tb[has] = bar_units_vec(panel, a[has], time_ns[has])
        av[has] = tb[has] if avail_ns is None else np.maximum(tb[has], bar_units_vec(panel, a[has], np.asarray(avail_ns, dtype=np.int64)[has]))
    av = av + float(lag_bars)
    rows = np.ceil(av).astype(np.int64)
    ok = has & (tb >= 0) & (rows <= T - 1)
    text = np.zeros((T, N, M, D), dtype=np.float32)
    mask = np.zeros((T, N, M), dtype=bool)
    age = np.zeros((T, N, M), dtype=np.float32)
    avt = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None, None], (T, N, M)).copy()
    stats = {"n_docs": int(len(a)), "placed": 0, "dropped_out_of_window": int((~ok).sum()), "dropped_overflow": 0, "slots": M, "dim": D,
             "rule": "row = ceil(availability in bar units) = first row at which the message is usable; avail_text = row"}
    order = np.lexsort((-time_ns, a, rows))
    fill = np.zeros((T, N), dtype=np.int64)
    for j in order:
        if not ok[j]:
            continue
        r, i = int(rows[j]), int(a[j])
        s = int(fill[r, i])
        if s >= M:
            stats["dropped_overflow"] += 1
            continue
        text[r, i, s] = emb[j]
        mask[r, i, s] = True
        age[r, i, s] = float(r - int(np.floor(tb[j])))
        fill[r, i] += 1
        stats["placed"] += 1
    return {"text": text, "text_mask": mask, "text_age": age, "avail_text": avt, "stats": stats}


def attach_text(panel: Panel, keys: Sequence[str], time_ns: np.ndarray, texts: Sequence[str], dim: int, slots: int, lag_bars: float,
                cache: Optional[str], max_mb: float, note: str, avail_ns: Optional[np.ndarray] = None) -> None:
    """Embed (cached hash_embed over the UNIQUE texts) and place in-window messages; records meta['text'] /
    meta['text_placement']. When the (T, N, M, D) tensor would exceed max_mb or there is no message, Panel.text stays
    None and meta['text'] says why."""
    T, N = panel.T, panel.N
    n = len(texts)
    if n == 0:
        panel.meta["text"] = "none: no message text inside the episode windows"
        return
    if not text_budget_ok(T, N, slots, dim, max_mb):
        panel.meta["text"] = (f"skipped: Panel.text would need {T * N * slots * dim * 4 / 1e6:.0f} MB > text_max_mb={max_mb} "
                              f"(raise text_max_mb, lower text_slots/text_dim or max_pumps); the social stream still carries text_score per message")
        return
    uniq, inv = np.unique(np.asarray([str(t) for t in texts], dtype=object), return_inverse=True)
    emb = cached_hash_embed(list(uniq), dim, cache)[inv]
    pl = place_text(panel, keys, np.asarray(time_ns, dtype=np.int64), emb, slots, lag_bars, avail_ns)
    panel.text, panel.text_mask, panel.text_age = pl["text"], pl["text_mask"], pl["text_age"]
    attach_availability(panel, avail_text=pl["avail_text"])
    panel.meta["text"] = {"embedder": "hash_embed (crosspump.data.loaders.common; hashed log-tf bag of words, nothing learned)", "cutoff": "n/a", "dim": int(dim),
                          "slots": int(slots), "n_messages": int(n), "n_unique_texts": int(len(uniq)), "cache": cache, "note": note}
    panel.meta["text_placement"] = pl["stats"]


def bar_units_vec(panel: Panel, assets: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Vectorised common.bar_units: UTC ns -> fractional bar units of each event's own episode column."""
    bc = panel.meta["bar_close_ns"]
    step = bc[1, 0] - bc[0, 0]
    origin = (bc[0, np.asarray(assets, dtype=np.int64)] - step).astype(np.float64)
    return (np.asarray(t_ns, dtype=np.float64) - origin) / step


# ------------------------------------------------------------------------------------------------ klines
def months_covering(start_ns: int, end_ns: int) -> List[str]:
    """'YYYY-MM' strings of every month touching [start_ns, end_ns]."""
    s = pd.Timestamp(int(start_ns), tz="UTC"); e = pd.Timestamp(int(end_ns), tz="UTC")
    out = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def window_ns(anchor_ns: int, pre_bars: int, post_bars: int, bar_minutes: int) -> Tuple[int, int]:
    step = int(bar_minutes) * MIN_NS
    a0 = (int(anchor_ns) // step) * step
    return a0 - int(pre_bars) * step, a0 + (int(post_bars) + 1) * step


def episode_months(episodes: Sequence[Dict[str, Any]], pre_bars: int, post_bars: int, bar_minutes: int, pad_days: float = 0.0) -> Dict[str, set]:
    """symbol -> set of 'YYYY-MM' months whose klines the episode windows (padded by pad_days on both sides) need."""
    out: Dict[str, set] = {}
    pad = int(pad_days * NS_PER_DAY)
    for e in episodes:
        lo, hi = window_ns(int(e["anchor_ns"]), pre_bars, post_bars, bar_minutes)
        out.setdefault(str(e["symbol"]), set()).update(months_covering(lo - pad, hi + pad))
    return out


_MONTH_RX = re.compile(r"(\d{4}-\d{2})")


def load_bars(kdir: str, symbol: str, quote: str, months: Optional[Iterable[str]], bar_minutes: int) -> Optional[pd.DataFrame]:
    """Read the cached 1-minute klines of <symbol><quote> (monthly files whose name carries YYYY-MM are filtered to
    `months`; files without a month token are read whole) and resample to bar_minutes. None when no file exists."""
    files = find_kline_files(kdir, symbol, quote)
    if not files:
        return None
    if months is not None:
        want = set(months)
        keep = []
        for f in files:
            m = _MONTH_RX.search(os.path.basename(f))
            if m is None or m.group(1) in want:
                keep.append(f)
        files = keep
    if not files:
        return None
    kl = read_binance_klines(files)
    if len(kl) == 0:
        return None
    return resample_bars(kl, int(bar_minutes))


def pump_key(symbol: str, quote: str, anchor_ns: int) -> str:
    return f"{symbol}{quote}@{pd.Timestamp(int(anchor_ns), tz='UTC').strftime('%Y-%m-%dT%H:%MZ')}"


def subsample_evenly(items: List[Any], n: Optional[int]) -> List[Any]:
    """Keep at most n items spread evenly over the (chronologically sorted) list -- a max_pumps cap that keeps every
    period represented instead of truncating the test years."""
    if n is None or n <= 0 or len(items) <= int(n):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, int(n)).round().astype(int))
    return [items[i] for i in idx]


# ------------------------------------------------------------------------------------------------ streams
def stream_part(panel: Panel, df: pd.DataFrame, platform: int, lag_bars: float, mark_cols: Sequence[Optional[str]], source_col: Optional[str],
                avail_col: Optional[str] = None) -> EventStreams:
    """Vectorised equivalent of common._stream_part (same formulas, same output; the per-event python loop there does
    not scale to the millions of messages / tweet buckets of D4 and D7). df: key, time_ns, <mark cols>, [source col].
    avail = time + lag_bars (clamped to the window end exactly like common). With avail_col (UTC ns at which the event
    became KNOWABLE, e.g. the end of a counting bucket or of a detection delay) avail = max(time, avail_col) + lag_bars
    and events whose availability falls after the window end are DROPPED, never clamped into the window."""
    if df is None or len(df) == 0:
        return EventStreams.empty()
    T = panel.T
    idx = asset_index(panel)
    a = np.array([idx.get(str(k), -1) for k in df["key"]], dtype=np.int64)
    keep = a >= 0
    if not keep.any():
        return EventStreams.empty()
    a = a[keep]
    tb = bar_units_vec(panel, a, df["time_ns"].to_numpy()[keep])
    ok = (tb >= 0) & (tb < T - 1e-6)
    if avail_col is not None and avail_col in df:
        av = np.maximum(tb, bar_units_vec(panel, a, df[avail_col].to_numpy()[keep])) + float(lag_bars)
        ok &= av <= T - 1e-6
    else:
        av = np.minimum(tb + float(lag_bars), T - 1e-6)
    E = int(ok.sum())
    mark = np.full((E, len(mark_cols)), np.nan, dtype=np.float32)
    for k, c in enumerate(mark_cols):
        if c is not None and c in df:
            mark[:, k] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)[keep][ok].astype(np.float32)
    src = df[source_col].to_numpy()[keep][ok].astype(np.int64) if source_col is not None and source_col in df else np.zeros(E, np.int64)
    return EventStreams(a[ok], tb[ok].astype(np.float64), np.full(E, platform, dtype=np.int8), mark, av[ok].astype(np.float64), src, {}).sorted()


def social_part(panel: Panel, events: pd.DataFrame, lag_bars: float = 0.0) -> EventStreams:
    """Social events: key, time_ns, [magnitude (default 1 = one message)], [reach], [text_score], [source_id], [avail_ns].
    mark = [magnitude, reach, text_score, NaN] -- identical to common.social_stream when magnitude is 1 and there is no
    avail_ns (tests assert the equality)."""
    if events is None or len(events) == 0:
        return EventStreams.empty()
    df = events
    if "magnitude" not in df:
        df = df.copy(); df["magnitude"] = 1.0
    return stream_part(panel, df, 0, lag_bars, ["magnitude", "reach", "text_score", None], "source_id", "avail_ns" if "avail_ns" in df else None)


def fast_prior_involvement(streams: EventStreams, anchor_order: np.ndarray, is_pump: np.ndarray) -> EventStreams:
    """Same result as crosspump.data.streams.add_prior_involvement (mark[:, 3] = number of EARLIER pump episodes, by
    anchor rank, in which the event's source id appears; NaN for source 0) without the per-event python loop."""
    if not len(streams):
        return streams
    m = streams.mark.copy()
    rank = np.asarray(anchor_order, dtype=np.int64)[streams.asset]
    src = streams.source
    known = src != 0
    m[~known, 3] = np.nan
    if known.any():
        us, sidx = np.unique(src[known], return_inverse=True)
        big = int(len(anchor_order)) + 1
        code = sidx.astype(np.int64) * big + rank[known]
        pump_ev = np.asarray(is_pump, dtype=bool)[streams.asset[known]]
        pump_codes = np.unique(code[pump_ev])                     # one entry per (source, pump episode)
        first = np.searchsorted(pump_codes, sidx.astype(np.int64) * big, side="left")
        m[known, 3] = (np.searchsorted(pump_codes, code, side="left") - first).astype(np.float32)
    return EventStreams(streams.asset, streams.time, streams.platform, m, streams.avail, streams.source, streams.meta)


def finish(panel: Panel, parts: Sequence[EventStreams], fast_threshold: int = 200000, **meta) -> Panel:
    """common.finish_streams for ordinary sizes; above fast_threshold events the same steps (derive order flow when no
    trade-level part exists, concat, prior involvement, attach) with the vectorised prior-involvement count."""
    parts = [p for p in parts if p is not None]
    if sum(len(p) for p in parts) <= int(fast_threshold):
        return common.finish_streams(panel, parts, **meta)
    if not any(len(p) and (p.platform == 2).any() for p in parts):
        parts.append(derive_orderflow_events(panel))
        meta["orderflow"] = "derived causally from bar features (crosspump.data.streams.derive_orderflow_events)"
    es = EventStreams.concat(parts, **meta)
    es = fast_prior_involvement(es, np.argsort(np.argsort(panel.meta["anchor_ns"])), np.asarray(panel.meta["is_pump"]))
    return attach_streams(panel, es)


def onchain_path(root: str, dataset: str) -> str:
    return os.path.join(root, "crosspump", "onchain", f"{dataset}_transfers.csv")


def optional_wallet_stream(panel: Panel, root: str, dataset: str, lag_bars: float) -> Tuple[Optional[EventStreams], Dict[str, Any]]:
    """The OPTIONAL on-chain transfer table <root>/crosspump/onchain/<dataset>_transfers.csv
    (columns key, time_ns, wallet_id, size_adv, size_raw). Absent -> (None, note saying the wallet modality is missing);
    never an empty stream pretending there was no activity."""
    p = onchain_path(root, dataset)
    if not os.path.exists(p):
        return None, {"status": "missing", "file": p, "note": "no on-chain transfer table: the wallet modality is MISSING for every episode (not 'no activity')"}
    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["key", "time_ns"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}; expected key,time_ns,wallet_id,size_adv,size_raw")
    out = pd.DataFrame({"key": df[cols["key"]].astype(str), "time_ns": pd.to_numeric(df[cols["time_ns"]], errors="coerce").astype("int64")})
    for k in ("wallet_id", "size_adv", "size_raw"):
        if k in cols:
            out[k] = pd.to_numeric(df[cols[k]], errors="coerce")
    if "wallet_id" in out:
        out["wallet_id"] = out["wallet_id"].fillna(0).astype(np.int64)
    es = wallet_stream(panel, out, lag_bars=lag_bars)
    keys_with = sorted(set(out["key"]) & set(panel.assets))
    return es, {"status": "present", "file": p, "n_transfers": int(len(out)), "episodes_with_transfers": len(keys_with),
                "note": "episodes absent from the transfer table have NO wallet coverage (missing modality), see coverage['episodes_with_platform']"}


# ------------------------------------------------------------------------------------------------ shared assembly (D4 / D5 / D7)
DEFAULTS: Dict[str, Any] = {"root": "data_raw", "quote": "BTC", "bar_minutes": 5, "pre_bars": 864, "post_bars": 96, "lead": 2, "peak_window_bars": 12,
                            "end_bars_after_peak": 24, "horizons": [1, 2, 3, 6, 12, 24], "n_controls_per_pump": 2, "gap_hours": 24.0,
                            "control_span_days": 30.0, "control_exclude_window": True, "seed": 0, "min_bars_frac": 0.5, "social_lag_bars": 0.0,
                            "wallet_lag_bars": 1.0, "max_pumps": None, "text_slots": 4, "text_dim": 32, "text_max_mb": 2048.0, "text_cache": True,
                            "fast_stream_threshold": 200000}
KLINE_SUBDIR = os.path.join("crosspump", "binance_klines")


def args_dict(args: Any) -> Dict[str, Any]:
    """prepare(root, args) accepts a dict, an argparse Namespace or None."""
    if args is None:
        return {}
    return dict(args) if isinstance(args, dict) else dict(vars(args))


def norm_exchange(x: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).strip().lower().replace(".com", ""))


def merge_events(rows: pd.DataFrame, merge_hours: float) -> List[List[int]]:
    """rows: DataFrame[symbol, time_ns] -> lists of row positions; rows of the same symbol within merge_hours of the
    FIRST row of a group form one pump episode (several channels relaying the same call are one market event)."""
    gap = int(float(merge_hours) * NS_PER_H)
    out: List[List[int]] = []
    t = rows["time_ns"].to_numpy().astype(np.int64)
    for _, pos in rows.groupby("symbol", sort=True).indices.items():
        pos = pos[np.argsort(t[pos], kind="stable")]
        cur = [int(pos[0])]
        for j in pos[1:]:
            if t[j] - t[cur[0]] <= gap:
                cur.append(int(j))
            else:
                out.append(cur); cur = [int(j)]
        out.append(cur)
    out.sort(key=lambda g: (int(t[g[0]]), g[0]))
    return out


def _coverage_ok(kl: pd.DataFrame, lo: int, hi: int, need: int) -> bool:
    idx = kl.index.to_numpy()
    return int(np.searchsorted(idx, hi, side="left") - np.searchsorted(idx, lo, side="left")) >= need


def make_controls(pumps: List[Dict[str, Any]], bars: Dict[str, pd.DataFrame], pump_times: Dict[str, np.ndarray], c: Dict[str, Any],
                  rng: np.random.Generator, quote_of: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """common.sample_controls (same coin at another time; another coin with klines at the same time) followed by a
    validity pass: a control is kept only if its coin has bars over the whole window, has no listed pump within
    gap_hours of the anchor and (control_exclude_window) none anywhere inside the window -- otherwise it is redrawn
    (up to 40 tries) or dropped. Controls are re-keyed 'CTRL_<SYMBOL><quote>@<time>' and de-duplicated; they inherit the
    channel list of their pump (the same channels are watched at the control time)."""
    n = int(c["n_controls_per_pump"])
    stats = {"sampled": 0, "redrawn": 0, "dropped": 0, "duplicates": 0}
    if n <= 0 or not pumps:
        return [], stats
    bm, pre, post = int(c["bar_minutes"]), int(c["pre_bars"]), int(c["post_bars"])
    need = max(10, int(float(c["min_bars_frac"]) * (pre + post + 1)))
    gap = int(float(c["gap_hours"]) * NS_PER_H); span = int(float(c["control_span_days"]) * NS_PER_DAY)
    syms = sorted(bars)
    raw = common.sample_controls(pumps, pump_times, n, float(c["gap_hours"]), rng, span_days=float(c["control_span_days"]),
                                 other_symbols=syms if len(syms) > 1 else None)
    stats["sampled"] = len(raw)
    by_key = {p["key"]: p for p in pumps}

    def ok(s: str, a: int) -> bool:
        if s not in bars:
            return False
        lo, hi = window_ns(a, pre, post, bm)
        if not _coverage_ok(bars[s], lo, hi, need):
            return False
        pt = pump_times.get(s, np.zeros(0))
        if len(pt) == 0:
            return True
        if np.abs(pt - a).min() <= gap:
            return False
        return not (bool(c["control_exclude_window"]) and ((pt >= lo) & (pt < hi)).any())

    out: List[Dict[str, Any]] = []; seen = set()
    for ctl in raw:
        pump = by_key[ctl["control_of"]]
        s, a = str(ctl["symbol"]), int(ctl["anchor_ns"])
        same = s == pump["symbol"]
        if not ok(s, a):
            found = False
            for _ in range(40):
                if same:
                    s2, a2 = s, int(int(pump["anchor_ns"]) + rng.uniform(-span, span))
                else:
                    s2, a2 = str(rng.choice(syms)), a
                    if s2 == pump["symbol"]:
                        continue
                if ok(s2, a2):
                    s, a, found = s2, a2, True; break
            if not found:
                stats["dropped"] += 1; continue
            stats["redrawn"] += 1
        key = "CTRL_" + pump_key(s, quote_of.get(s, str(c["quote"])), a)
        if key in seen:
            stats["duplicates"] += 1; continue
        seen.add(key)
        out.append({**ctl, "key": key, "symbol": s, "anchor_ns": a, "quote": quote_of.get(s, str(c["quote"])), "channel": None,
                    "channels": list(pump.get("channels", [])), "announcements": [],
                    "control_type": "same_coin_other_time" if same else "other_coin_same_time", "source": "sampled_control"})
    return out, stats


def build_panel(cfg: Dict[str, Any], pumps: List[Dict[str, Any]], *, name: str, dataset: str, source: str, licence: str, recipe: str,
                social_fn, pump_times: Optional[Dict[str, np.ndarray]] = None, fixed_controls: Optional[List[Dict[str, Any]]] = None,
                quote_of: Optional[Dict[str, str]] = None, assumptions: Optional[List[str]] = None, extra_meta: Optional[Dict[str, Any]] = None,
                table_stats: Optional[Dict[str, Any]] = None) -> Panel:
    """Shared assembly of the group-B loaders.

    pumps           pump episode dicts (key, symbol, exchange, channel, channels [...], announcements [...], anchor_ns,
                    peak_ns|None, is_pump True, control_of None, source), already filtered by exchange / notes
    fixed_controls  control episodes that come from the SOURCE itself (D5: bursts without a market reaction)
    pump_times      symbol -> every listed pump / burst time (all exchanges) so that sampled controls avoid them
    quote_of        symbol -> quote of its kline files (default cfg quote)
    social_fn(episodes, ctx) -> {"events": DataFrame[key, time_ns, magnitude?, reach?, text_score?, source_id?, avail_ns?] | None,
                                 "docs": DataFrame[key, time_ns, text, avail_ns?] | None, "info": dict, "entities": dict}
                    is called with the FINAL episode list (pumps + controls, each with its UTC window ctx['window'][key])
    Nothing is downloaded and nothing is generated: episodes without Binance klines are dropped and counted."""
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None or k == "max_pumps"}}
    root = str(c["root"]); kdir = os.path.join(root, KLINE_SUBDIR)
    bm, pre, post, lead = int(c["bar_minutes"]), int(c["pre_bars"]), int(c["post_bars"]), int(c["lead"])
    T = pre + post + 1
    need = max(10, int(float(c["min_bars_frac"]) * T))
    horizons = [int(h) for h in c["horizons"]]
    rng = np.random.default_rng(int(c["seed"]))
    quote_of = dict(quote_of or {})
    fixed_controls = list(fixed_controls or [])
    hint = f"(python3 scripts/xp_prepare_data.py --dataset {dataset} --klines)"
    if not pumps and not fixed_controls:
        raise FileNotFoundError(f"{name}: no pump episode left after filtering (exchange={c.get('exchange')}, quote={c.get('quote')}).\nDownload recipe:\n{recipe}")
    # ---- which symbols have cached Binance klines at all
    q = lambda s: quote_of.get(s, str(c["quote"]))
    has_files = {s: bool(find_kline_files(kdir, s, q(s))) for s in sorted({str(e["symbol"]) for e in pumps + fixed_controls})}
    n_table = len(pumps)
    pumps = sorted([p for p in pumps if has_files[p["symbol"]]], key=lambda d: int(d["anchor_ns"]))
    fixed_controls = [p for p in fixed_controls if has_files[p["symbol"]]]
    n_no_files = n_table - len(pumps)
    if not pumps and not fixed_controls:
        raise FileNotFoundError(f"{name}: none of the {len(has_files)} symbols has Binance 1-minute klines under {kdir} "
                                f"(expected <SYMBOL><QUOTE>-1m-YYYY-MM.csv).\nDownload recipe:\n{recipe}\n{hint}")
    n_before_cap = len(pumps)
    pumps = subsample_evenly(pumps, c.get("max_pumps"))
    # ---- bars: only the months the windows (and the control span) need
    pad = float(c["control_span_days"]) + 1.0 if int(c["n_controls_per_pump"]) > 0 else 1.0
    months = episode_months(pumps + fixed_controls, pre, post, bm, pad_days=pad)
    bars: Dict[str, pd.DataFrame] = {}
    for s in sorted(months):
        kl = load_bars(kdir, s, q(s), months[s], bm)
        if kl is not None and len(kl):
            bars[s] = kl

    def covered(e: Dict[str, Any]) -> bool:
        if e["symbol"] not in bars:
            return False
        lo, hi = window_ns(int(e["anchor_ns"]), pre, post, bm)
        return _coverage_ok(bars[e["symbol"]], lo, hi, need)

    n_sel = len(pumps)
    pumps = [p for p in pumps if covered(p)]
    fixed_n = len(fixed_controls)
    fixed_controls = [p for p in fixed_controls if covered(p)]
    if not pumps and not fixed_controls:
        raise FileNotFoundError(f"{name}: no episode window is covered by the cached klines under {kdir}.\nDownload recipe:\n{recipe}\n{hint}")
    if pump_times is None:
        pump_times = {}
        for p in pumps:
            pump_times.setdefault(p["symbol"], []).append(int(p["anchor_ns"]))
        pump_times = {s: np.sort(np.asarray(v, dtype=np.int64)) for s, v in pump_times.items()}
    controls, ctl_stats = make_controls(pumps, bars, pump_times, c, rng, {s: q(s) for s in bars})
    episodes = pumps + fixed_controls + controls
    keys = [e["key"] for e in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name}: duplicate episode keys {[k for k in set(keys) if keys.count(k) > 1][:5]}")
    table = {"pumps_in_table_after_filters": n_table, "pumps_without_any_binance_klines": n_no_files, "pumps_before_max_pumps": n_before_cap,
             "pumps_after_max_pumps": n_sel, "pumps_dropped_window_not_covered": n_sel - len(pumps), "source_controls_dropped_no_klines": fixed_n - len(fixed_controls),
             **(table_stats or {})}
    base_meta = {"quote": str(c["quote"]), "exchange_filter": c.get("exchange"), "root": root, "seed": int(c["seed"]), "table": table,
                 "controls": {"n_per_pump": int(c["n_controls_per_pump"]), "gap_hours": float(c["gap_hours"]), "span_days": float(c["control_span_days"]),
                              "exclude_window": bool(c["control_exclude_window"]), "stats": ctl_stats, "n_source_controls": len(fixed_controls),
                              "rule": "common.sample_controls: same coin uniformly inside +-span_days at >= gap_hours from any listed pump of the coin; another coin "
                                      "with klines at the same anchor time; both need bars over the window and (exclude_window) no listed pump inside it"},
                 "peak_window_bars": int(c["peak_window_bars"]), "end_bars_after_peak": int(c["end_bars_after_peak"])}
    panel, skipped = common.episode_panel(bars, episodes, pre, post, bm, lead, horizons, peak_window_bars=int(c["peak_window_bars"]),
                                          end_bars_after_peak=int(c["end_bars_after_peak"]), min_bars_frac=float(c["min_bars_frac"]), dataset=name,
                                          source=source, licence=licence, extra_meta=base_meta)
    eps = panel.meta["episodes"]
    ctx = {"cfg": c, "root": root, "window": {e["key"]: window_ns(int(e["anchor_ns"]), pre, post, bm) for e in eps}, "bar_minutes": bm}
    soc = social_fn(eps, ctx) or {}
    events, docs, info = soc.get("events"), soc.get("docs"), dict(soc.get("info") or {})
    lag = float(c["social_lag_bars"]); wlag = float(c["wallet_lag_bars"])
    parts = [social_part(panel, events, lag)]
    info["events_built"] = 0 if events is None else int(len(events)); info["events_in_stream"] = int(len(parts[0]))
    wal, wal_info = optional_wallet_stream(panel, root, dataset, wlag)
    if wal is not None:
        parts.append(wal)
    finish(panel, parts, fast_threshold=int(c["fast_stream_threshold"]),
           platform_lags={"social": lag, "wallet": wlag, "orderflow": "bar close (time = avail = t + 1)"},
           entities=soc.get("entities") or {}, social=info, wallet=wal_info)
    es = panel.streams
    cov = es.coverage(panel.N)["mask"]
    is_pump = np.asarray(panel.meta["is_pump"])
    panel.meta["coverage"].update({
        "pump_episodes_with_platform": {pl: int(cov[is_pump, k].sum()) for k, pl in enumerate(("social", "wallet", "orderflow"))},
        "social": info.get("coverage_note", "see streams.meta['social']"),
        "wallet": wal_info["note"] if wal_info["status"] == "missing" else f"present for {wal_info['episodes_with_transfers']} episodes; {wal_info['note']}",
        "orderflow": es.meta.get("orderflow", "trade-level"), "table": table, "skipped_by_episode_panel": skipped,
        "note": "an episode without social / wallet events has that modality MISSING (per-episode mask via panel.streams.coverage(N)['mask']); "
                "bar-derived order flow exists wherever bars exist (no event = no burst)"})
    # ---- optional message text
    if docs is not None and len(docs):
        cache = embed_cache_path(root, dataset, int(c["text_dim"])) if c.get("text_cache") else None
        attach_text(panel, docs["key"].tolist(), docs["time_ns"].to_numpy(), docs["text"].tolist(), int(c["text_dim"]), int(c["text_slots"]), lag, cache,
                    float(c["text_max_mb"]), note=str(info.get("text_note", "")), avail_ns=docs["avail_ns"].to_numpy() if "avail_ns" in docs else None)
    else:
        panel.meta["text"] = "none: " + str(info.get("text_note", "the source has no message text inside the episode windows"))
    panel.meta["assumptions"] = list(assumptions or [])
    panel.meta["pit"] = {"avail_feat": "row (bar close)", "avail_label": "episode end row (peak + end_bars_after_peak) on event rows, else row + max(horizon)",
                         "avail_text": "row = ceil(message availability): first row at which the message's stream event is usable",
                         "streams": "usable at decision row t iff avail <= t; social avail = message time + social_lag_bars; wallet = block time + wallet_lag_bars",
                         "universe": "kline exists at that bar", "tradable": "volume > 0"}
    panel.meta.update(extra_meta or {})
    panel.validate()
    return panel
