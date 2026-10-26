"""D1 -- Hu, Zhang, Lu, He, Li, "Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump", SIGMOD 2023 /
Proc. ACM Manag. Data 1(1), doi:10.1145/3588686, arXiv:2204.12929. Registry key: xp_hu_pd.

Raw files (git clone of https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency at <root>/crosspump/hu_pd),
structure as observed on the clone:
    Data/Telegram/Labeled/PD_logs_cleaned.txt   REQUIRED. TSV, header channel_id, session_id, coin, exchange, pair, timestamp
                                                ('2019-06-04 12:01:33+00:00'); 1,335 rows (binance 1003, yobit 207, hotbit 63,
                                                kucoin 26, gate.io 16, 9 others <= 10 rows); pair BTC 1211 / USDT 122 / ETH 2;
                                                108 channel ids; 2019-01-01 .. 2022-01-19. Binance x BTC: 981 rows, 135 coins.
    Data/Telegram/raw/<channel_id>.pkl          OPTIONAL (516 files, 845 MB). Plain-python pickles: a list of Telethon
                                                Message.to_dict() dicts (keys '_', id, date (tz-aware datetime), message,
                                                edit_date, views, forwards, ...). They do NOT need telethon; they are read with a
                                                restricted unpickler that only admits datetime / timezone / timedelta / bytes.
                                                102 of the 108 labelled channels have a raw file.
    Data/Telegram/Labeled/pred_pump_message.csv OPTIONAL fallback. TSV, header channel_id, sample_id (= Telegram message id),
                                                message (lower-cased, punctuation stripped), date, pred_label (all 1.0);
                                                36,696 messages of 213 channels that the authors' classifier flagged.
    Data/Telegram/Labeled/label.txt             not used (space separated 'label channel_id sample_id', no text / time).
    Data/Telegram/Labeled/<channel_id>_session.pkl  not used (pickled instances of the repo's own Session class).
Market data: Binance 1-minute klines (crosspump/binance_klines, see binance_fetch.py).

Episodes. One pump episode per (coin, exchange) announcement cluster: rows within merge_hours (6 h) of the first row are
the same pump relayed by several channels (the paper counts 709 distinct pumps over all exchanges). Anchor = earliest
announcement timestamp (UTC). Peak = argmax bar high within peak_window_bars after the anchor (the table has no peak).

Social stream.
  * one event per table row at its timestamp: mark [1, NaN, 1.0, prior], source = stable id of the channel id;
  * every message of the watched channel(s) inside the window (messages = 'auto': raw pickles when present, else the
    pred_pump_message.csv rows of that channel; 'raw' | 'pred' | 'none' force one) with mark [1, NaN, keyword score, prior].
    'Watched channels' of a control window = the channels that announced the pump it controls for, so control windows
    carry the real messages those channels posted at that time (the stream is coin-agnostic: before the announcement
    nobody outside the organisers knows the coin). On pump episodes a message of the announcing channel within 60 s of
    its table row is dropped (it IS the announcement event).
  * Panel.text: hash_embed of the same messages (dim 32, 4 slots), placed at the first row at which a message is usable.

Assumptions (also in meta['assumptions']).
  A1 timestamps are UTC as written; the table timestamp is the coin-release message time.
  A2 pair filter = quote (BTC); exchange filter 'binance' (exchange: all keeps other venues but prices are still Binance
     klines of the same symbol -- a cross-venue view, flagged per episode in 'exchange').
  A3 `views` / `forwards` in the raw pickles are counts at CRAWL time (not point-in-time) and are therefore NOT used: reach
     is NaN.
  A4 messages with edit_date != None may have been rewritten after publication: the event is kept at the original date but
     its keyword score is NaN and its text is not embedded (edited_text: drop; 'keep' uses the crawled text).
  A5 pred_pump_message.csv (fallback) was selected by the authors' classifier, trained on labels from the whole period:
     which messages EXIST in the fallback stream is therefore a mildly look-ahead selection; pred_label itself is unused.
  A6 keyword score = min(1, distinct lexicon hits / 4) over KEYWORDS below (a lexicon, not a classifier).
  A7 wallet platform: optional <root>/crosspump/onchain/hu_pd_transfers.csv; absent = missing modality.
"""
from __future__ import annotations
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from . import common
from . import binance_fetch as bf

SOURCE_URL = "https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency"
LICENCE = "no licence file in the repository (research use; cite Hu et al., SIGMOD 2023, doi:10.1145/3588686) -- redistribution unverified"
PAPER = "Hu et al., SIGMOD 2023 / PACMMOD 1(1), doi:10.1145/3588686, arXiv:2204.12929"
SUBDIR = "hu_pd"
TABLE = "Data/Telegram/Labeled/PD_logs_cleaned.txt"
REQUIRED = [f"crosspump/{SUBDIR}/{TABLE}"]
RECIPE = f"""git clone --depth 1 {SOURCE_URL} <root>/crosspump/{SUBDIR}
  (needs {TABLE}; optional Data/Telegram/raw/<channel_id>.pkl and Data/Telegram/Labeled/pred_pump_message.csv for messages)
python3 scripts/xp_prepare_data.py --dataset hu_pd --klines        # Binance 1m klines -> <root>/crosspump/binance_klines/<SYMBOL>BTC-1m-YYYY-MM.csv
python3 scripts/xp_prepare_data.py --dataset hu_pd --aggtrades     # optional: daily aggTrades for pump and control windows
ETHERSCAN_API_KEY=... python3 scripts/xp_prepare_data.py --dataset hu_pd --onchain   # optional wallet platform (ERC-20 / BEP-20 only)"""

KEYWORDS = [r"\bpump", r"\bsignal", r"\bcoin\b", r"\bbuy", r"\bhold", r"\btarget", r"\bprofit", r"\bbinance\b", r"\byobit\b", r"\bkucoin\b", r"\bhotbit\b",
            r"\bminutes?\b", r"\bhours?\b", r"\bcountdown\b", r"\bnext\b", r"\bbtc\b", r"\bpair", r"\bready\b", r"\bannounce", r"\bwhales?\b", r"\bgmt\b", r"\butc\b"]
_KW = [re.compile(k, re.I) for k in KEYWORDS]


def keyword_score(text: Any) -> float:
    s = str(text or "")
    return float(min(1.0, sum(1 for k in _KW if k.search(s)) / 4.0))


class _SafeUnpickler(pickle.Unpickler):
    """The raw pickles only need datetime objects; refuse every other global (never run code from a downloaded file)."""
    ALLOWED = {("datetime", "datetime"), ("datetime", "timezone"), ("datetime", "timedelta"), ("datetime", "date"), ("builtins", "bytes"), ("_codecs", "encode")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a raw Telegram pickle")


def read_table(root: str, subdir: str = SUBDIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """PD_logs_cleaned.txt -> DataFrame[symbol, exchange, pair, channel, channel_id, announce_ns, peak_ns, session_id]."""
    common.require(root, [os.path.join("crosspump", subdir, TABLE)], RECIPE)
    df = pd.read_csv(os.path.join(root, "crosspump", subdir, TABLE), sep="\t", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["channel_id", "session_id", "coin", "exchange", "pair", "timestamp"]
    if any(c not in df for c in need):
        raise ValueError(f"{TABLE}: expected columns {need}, got {list(df.columns)}")
    ts = pd.to_datetime(df["timestamp"].str.strip(), utc=True, errors="coerce")
    ok = ts.notna() & df["coin"].notna()
    out = pd.DataFrame({"symbol": df["coin"].str.strip().str.upper(), "exchange": df["exchange"].map(bf.norm_exchange), "pair": df["pair"].str.strip().str.upper(),
                        "channel": df["channel_id"].str.strip(), "session_id": df["session_id"].astype(str), "peak_ns": np.nan})[ok].reset_index(drop=True)
    out["announce_ns"] = ts[ok].values.astype("datetime64[ns]").astype(np.int64)
    out["channel_id"] = [bf.stable_id(c) for c in out["channel"]]
    return out, {"rows": int(len(df)), "rows_bad_timestamp": int((~ok).sum()), "exchanges": out["exchange"].value_counts().to_dict(), "channels": int(out["channel"].nunique())}


def load_pumps(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    c = {**bf.DEFAULTS, **cfg}
    root = str(c.get("root", "data_raw")); quote = str(c["quote"])
    rows, info = read_table(root, str(c.get("subdir", SUBDIR)))
    ex = bf.norm_exchange(c.get("exchange", "binance"))
    sel = rows[(rows["pair"] == quote) & ((rows["exchange"] == ex) | (ex == "all"))]
    pumps = bf.merge_announcements(sel, float(c["merge_hours"]), quote, "hu_pd", ann_cols=("session_id",))
    info.update({"rows_selected": int(len(sel)), "pumps_merged": len(pumps)})
    return bf.cap_pumps(pumps, c.get("max_pumps")), rows, info


# ------------------------------------------------------------------------------------------------ messages
def _channel_messages_raw(path: str) -> Optional[pd.DataFrame]:
    with open(path, "rb") as f:
        obj = _SafeUnpickler(f).load()
    rec = []
    for m in obj if isinstance(obj, list) else []:
        if not isinstance(m, dict) or m.get("_", "Message") != "Message":
            continue
        txt, d = m.get("message"), m.get("date")
        if not txt or d is None:
            continue
        rec.append((pd.Timestamp(d).tz_convert("UTC").value if pd.Timestamp(d).tzinfo else pd.Timestamp(d, tz="UTC").value, str(txt), m.get("edit_date") is not None))
    if not rec:
        return None
    df = pd.DataFrame(rec, columns=["time_ns", "text", "edited"]).sort_values("time_ns").reset_index(drop=True)
    return df


def _pred_messages(path: str) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df[ts.notna()].copy(); df["time_ns"] = ts[ts.notna()].values.astype("datetime64[ns]").astype(np.int64)
    df["text"] = df["message"].fillna("").astype(str); df["edited"] = False
    return {str(k).strip(): g[["time_ns", "text", "edited"]].sort_values("time_ns").reset_index(drop=True) for k, g in df.groupby("channel_id")}


def make_social_fn(root: str, subdir: str, mode: str, edited_text: str):
    tdir = os.path.join(root, "crosspump", subdir, "Data", "Telegram")

    def social_fn(eps: List[Dict[str, Any]], c: Dict[str, Any]):
        info: Dict[str, Any] = {"messages_mode": mode}
        if mode == "none":
            return None, None, info
        step = int(c["bar_minutes"]) * bf.MIN_NS; pre, post = int(c["pre_bars"]), int(c["post_bars"])
        channels = sorted({ch for e in eps for ch in e.get("watch_channels", [])})
        pred: Optional[Dict[str, pd.DataFrame]] = None
        ppath = os.path.join(tdir, "Labeled", "pred_pump_message.csv")
        store: Dict[str, pd.DataFrame] = {}; src: Dict[str, str] = {}
        for ch in channels:
            rp = os.path.join(tdir, "raw", f"{ch}.pkl")
            if mode in ("auto", "raw") and os.path.exists(rp):
                m = _channel_messages_raw(rp)
                if m is not None:
                    store[ch] = m; src[ch] = "raw"; continue
            if mode in ("auto", "pred") and os.path.exists(ppath):
                if pred is None:
                    pred = _pred_messages(ppath)
                if ch in pred:
                    store[ch] = pred[ch]; src[ch] = "pred"
        ev, docs = [], []
        n_edit = 0; with_msgs = set()
        for e in eps:
            lo, hi = bf._window_bounds(int(e["anchor_ns"]), pre, post, step)
            ann = {}
            for a in e.get("announcements", []) or []:
                ann.setdefault(str(a.get("channel")), []).append(int(a["announce_ns"]))
            for ch in e.get("watch_channels", []):
                m = store.get(ch)
                if m is None:
                    continue
                t = m["time_ns"].to_numpy()
                i0, i1 = int(np.searchsorted(t, lo, side="left")), int(np.searchsorted(t, hi, side="left"))
                sid = bf.stable_id(ch)
                for j in range(i0, i1):
                    tj = int(t[j]); txt = m["text"].iat[j]; edited = bool(m["edited"].iat[j])
                    use_text = (not edited) or edited_text == "keep"
                    n_edit += int(edited)
                    if use_text:
                        docs.append({"key": e["key"], "time_ns": tj, "text": txt})
                    if any(abs(tj - x) <= 60 * 10 ** 9 for x in ann.get(ch, [])):
                        continue                                                        # this message is the announcement event itself
                    ev.append({"key": e["key"], "time_ns": tj, "reach": np.nan, "text_score": keyword_score(txt) if use_text else np.nan, "source_id": sid})
                    with_msgs.add(e["key"])
        info.update({"channels_watched": len(channels), "channels_with_raw_messages": sum(1 for v in src.values() if v == "raw"),
                     "channels_with_pred_messages_only": sum(1 for v in src.values() if v == "pred"), "channels_without_messages": len(channels) - len(src),
                     "episodes_with_message_events": len(with_msgs), "edited_messages_in_windows": n_edit, "edited_text": edited_text})
        cols_e = ["key", "time_ns", "reach", "text_score", "source_id"]
        return pd.DataFrame(ev, columns=cols_e), pd.DataFrame(docs, columns=["key", "time_ns", "text"]), info

    return social_fn


def prepare(root: str, args) -> None:
    bf.prepare_cex(root, args, dataset="hu_pd", subdir=SUBDIR, repo_url=SOURCE_URL, required=REQUIRED, pumps_fn=lambda c: load_pumps(c)[0], builder=HuPD)


@common.register_dataset("xp_hu_pd")
class HuPD(BaseDatasetBuilder):
    name = "xp_hu_pd"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self):
        cfg = dict(self.cfg); cfg.setdefault("horizons", list(self.horizons))
        root = str(cfg.get("root", "data_raw")); subdir = str(cfg.get("subdir", SUBDIR))
        pumps, rows, info = load_pumps(cfg)
        mode = str(cfg.get("messages", "auto")).lower(); edited = str(cfg.get("edited_text", "drop")).lower()
        entities = {"channels": {int(bf.stable_id(ch)): str(ch) for ch in sorted(rows["channel"].unique())}}
        assumptions = [
            "A1 table timestamps are UTC and equal the coin-release message time; anchor = earliest row of a (coin, exchange) cluster within merge_hours.",
            "A2 only rows with pair == quote; exchange filter '%s' (with 'all', prices are still Binance klines of the same symbol)." % cfg.get("exchange", "binance"),
            "A3 Telegram views / forwards are crawl-time counts (not point-in-time): reach is NaN.",
            "A4 edited messages keep their event but lose text score and text embedding unless edited_text == 'keep'.",
            "A5 the pred_pump_message.csv fallback is a classifier-selected subset (classifier trained on whole-period labels): message EXISTENCE in that fallback is mildly look-ahead.",
            "A6 keyword text_score = min(1, distinct lexicon hits / 4) (d1_hu_pd.KEYWORDS); announcement rows get 1.0.",
            "A7 control windows carry the messages of the channels that announced the pump they control for (coin-agnostic social stream).",
            "A8 no curated peak in this table: peak = argmax bar high within peak_window_bars after the anchor."]
        panel = bf.build_cex_panel(cfg, pumps, name=self.name, dataset="hu_pd", source=f"{SOURCE_URL} ({PAPER})", licence=LICENCE, recipe=RECIPE,
                                   all_pump_times=bf.pump_times_by_symbol(rows), social_fn=make_social_fn(root, subdir, mode, edited), entities=entities,
                                   assumptions=assumptions, extra_meta={"table": info, "paper": PAPER})
        return panel
