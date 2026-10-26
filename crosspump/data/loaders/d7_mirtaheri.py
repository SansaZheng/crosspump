"""D7 Mirtaheri et al. -- Mirtaheri, Abu-El-Haija, Morstatter, Ver Steeg, Galstyan, "Identifying and Analyzing
Cryptocurrency Manipulations in Social Media", IEEE Trans. Computational Social Systems 8(3), 2021,
doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110. Repo https://github.com/Mehrnoom/Cryptocurrency-Pump-Dump
(commit 86e86b9, 2019-03-13; no licence file). Registry key: xp_mirtaheri.

Raw layout (observed on c225, 2026-10-03, clone at <root>/crosspump/mirtaheri):
    Telegram/classified/coin-pump.csv      62,850 rows, columns ",Channel ID,Message ID,Date,Time,Coin"
                                           (first column = unnamed row index; Date 2017-05-18 .. 2018-07-23; 209 channels;
                                            543 coins; 47,992 distinct messages, 14,858 of them listed with two coins)
    Telegram/classified/price_extract.csv  55,751 rows, ",Channel ID,Message ID,Date,Time,Coin,Sell,Buy" (not used)
    Telegram/channels/<channel_id>.jl      420 files, 1.19 GB; one Telegram message json per line:
                                           {"_": "Message", "id": 2, "to_id": {...}, "date": {"$date": 1443140961000},
                                            "message": "...", "views": 194458, "from_id": null, ...}; "MessageService"
                                           lines have no "message". Every csv channel has its .jl.
    Twitter/tweets-ts/$COIN.json           1,518 files  {"data": [[{"$date": ms}, n_tweets], ...]}
    Twitter/users-ts/$COIN.json            1,518 files  same layout, unique users
    Twitter/sentiment-ts/COIN.json         1,517 files  same layout, mean sentiment in [-1, 1]; NO '$' prefix
                                           1-MINUTE cadence (stamps are multiples of 60 s; only minutes with tweets are
                                           listed; NOT sorted), 2017-09-01 .. 2018-08-25.
    coin-pump.csv Date/Time are UTC (they equal the .jl "$date" of the same message id).

Episodes: classified pump messages of the same coin within merge_hours (default 6) form one pump attempt anchored at
the earliest message; kept iff the coin has cached Binance klines covering the window (the table has NO exchange
column: most 2017-18 attempts ran on YoBit / Cryptopia / Bittrex and are dropped -- counted in meta['coverage']).
Social stream, both on platform 0 (meta['social_platforms'] says how to tell them apart: source id 1 = Twitter):
  (a) Twitter: one event per twitter_agg_minutes bucket (default 5; 1 = the native cadence) with tweets > 0 for the
      episode's coin: magnitude = tweet count, reach = users (sum of the per-minute unique users = upper bound of the
      bucket's unique users), text_score = tweet-weighted mean sentiment, source 1. time = avail = END of the bucket
      (a count over a minute is knowable only when the minute is over).
  (b) Telegram: every message of the episode's channel .jl files inside the window: magnitude 1, reach NaN,
      source = stable hash of the channel id, text_score = 1.0 for the classified pump message(s) of the episode's coin
      (the paper's classifier output = an ORACLE-like mark), else the keyword score of text_utils.
Panel.text = hash_embed of the Telegram message texts. Wallet: optional onchain/mirtaheri_transfers.csv, else missing.

ASSUMPTIONS (also in meta['assumptions'])
 A1 coin-pump.csv is a classifier output with ticker matching: 'BTC' (9,027 rows) is the quote of 'XYZ/BTC' calls and
    'HOLD' (2,725) / 'CAN' / 'TIME' / 'OK' are English words. cfg exclude_coins (default BTC, ETH, USDT, BNB, HOLD) are
    dropped; with require_ticker_in_text (default true) a row is kept only if the coin appears in the message as an
    UPPERCASE token or cashtag (text_utils.extract_tickers) -- 'have fun' does not make a FUN pump.
 A2 no exchange column: an attempt is assumed tradable on Binance when Binance klines exist for <COIN><quote>; the
    venue named in the message (binance / bittrex / yobit / cryptopia / kucoin ...) is kept in episode['exchange'] for
    information; cfg exchange: 'Binance' additionally REQUIRES 'binance' to be mentioned, default 'any'.
 A3 the Telegram `views` counter is a crawl-time total (it keeps growing after posting) -> NOT point-in-time -> reach
    is NaN for Telegram events.
 A4 Twitter minute stamps are taken as the START of the counting minute; events are stamped at the bucket END.
 A5 peak = argmax high within peak_window_bars after the anchor (the repo has no peak time).
 A6 these are pump ATTEMPTS / trading signals found by a classifier, not verified pump-and-dump events.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from . import common
from . import text_utils as tu

SOURCE_URL = "https://github.com/Mehrnoom/Cryptocurrency-Pump-Dump"
LICENCE = "no licence file in the repository (research data released with arXiv:1902.03110 / IEEE TCSS 2021; cite the paper)"
SUBDIR = os.path.join("crosspump", "mirtaheri")
PUMP_CSV = os.path.join(SUBDIR, "Telegram", "classified", "coin-pump.csv")
CHANNEL_DIR = os.path.join(SUBDIR, "Telegram", "channels")
TWITTER_DIR = os.path.join(SUBDIR, "Twitter")
REQUIRED = [PUMP_CSV, CHANNEL_DIR, os.path.join(TWITTER_DIR, "tweets-ts")]
RECIPE = """D7 Mirtaheri et al. (IEEE TCSS 8(3) 2021, doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110)
  git clone --depth 1 https://github.com/Mehrnoom/Cryptocurrency-Pump-Dump data_raw/crosspump/mirtaheri      # 7.4 GB with .git
  needs Telegram/classified/coin-pump.csv, Telegram/channels/<channel_id>.jl, Twitter/{tweets-ts,users-ts,sentiment-ts}/
  python3 scripts/xp_prepare_data.py --dataset mirtaheri --klines     # Binance 1-minute klines 2017-2018 -> data_raw/crosspump/binance_klines/
  optional wallet modality: data_raw/crosspump/onchain/mirtaheri_transfers.csv (key,time_ns,wallet_id,size_adv,size_raw)"""
TWITTER_SOURCE = 1
EXCLUDE = ("BTC", "ETH", "USDT", "BNB", "HOLD")
VENUES = ("binance", "bittrex", "yobit", "cryptopia", "kucoin", "poloniex", "hitbtc", "bitfinex", "huobi", "okex", "coinexchange")
ASSUMPTIONS = [
    "A1 coin-pump.csv is a classifier + ticker-matching output: exclude_coins (BTC, ETH, USDT, BNB, HOLD) dropped; require_ticker_in_text keeps a row only "
    "if the coin appears in the message as an uppercase token or cashtag",
    "A2 the table has no exchange column: an attempt is kept when Binance klines exist for the coin; the venue named in the message is informational "
    "(cfg exchange 'Binance' requires the mention, default 'any')",
    "A3 Telegram `views` is a crawl-time total, not point-in-time: reach is NaN for Telegram events",
    "A4 Twitter stamps = start of the counting minute; tweet events are stamped (time = avail) at the END of their twitter_agg_minutes bucket; "
    "reach = sum of per-minute unique users (upper bound); text_score = tweet-weighted mean sentiment",
    "A5 peak = argmax high within peak_window_bars after the anchor",
    "A6 episodes are classifier-detected pump ATTEMPTS / signals, not verified pump-and-dump events",
    "A7 classified messages of the same coin within merge_hours are one episode (anchor = earliest message)",
    "A8 Twitter and Telegram both map to platform 0 (social); source id 1 = Twitter, any other id = hashed Telegram channel",
    "A9 text_score = 1.0 on the classified pump message(s) of the episode's coin (classifier label of the source), else keyword score",
    "A10 coin-pump.csv Date/Time are UTC (equal to the .jl $date of the same message)",
]


# ------------------------------------------------------------------------------------------------ raw tables
def read_pump_table(root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, PUMP_CSV))
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["channel id", "message id", "date", "time", "coin"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"{PUMP_CSV}: columns {missing} missing (found {list(df.columns)})")
    t = pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str), utc=True, errors="coerce")
    out = pd.DataFrame({"channel": pd.to_numeric(df[cols["channel id"]], errors="coerce"), "msg_id": pd.to_numeric(df[cols["message id"]], errors="coerce"),
                        "symbol": df[cols["coin"]].astype(str).str.strip().str.upper(), "t": t})
    out = out[out["t"].notna() & out["channel"].notna() & out["msg_id"].notna()].copy()
    out["time_ns"] = out["t"].values.astype("datetime64[ns]").astype(np.int64)
    out["channel"] = out["channel"].astype(np.int64); out["msg_id"] = out["msg_id"].astype(np.int64)
    return out.drop(columns="t").reset_index(drop=True)


class ChannelStore:
    """Lazy per-channel message arrays from Telegram/channels/<id>.jl: time_ns (sorted), msg_id, text."""

    def __init__(self, root: str):
        self.dir = os.path.join(root, CHANNEL_DIR)
        self._c: Dict[int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._pos: Dict[int, Dict[int, int]] = {}

    def get(self, channel: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        channel = int(channel)
        if channel not in self._c:
            p = os.path.join(self.dir, f"{channel}.jl")
            if not os.path.exists(p):
                self._c[channel] = None
            else:
                ts, ids, txt = [], [], []
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if '"message"' not in line:
                            continue
                        try:
                            o = json.loads(line)
                        except ValueError:
                            continue
                        d = o.get("date")
                        ms = d.get("$date") if isinstance(d, dict) else None
                        if ms is None or o.get("id") is None:
                            continue
                        if isinstance(ms, str):
                            ms = pd.Timestamp(ms).value // 10 ** 6
                        ts.append(int(ms) * 10 ** 6); ids.append(int(o["id"])); txt.append(o.get("message") or "")
                order = np.argsort(np.asarray(ts, dtype=np.int64), kind="stable")
                self._c[channel] = (np.asarray(ts, dtype=np.int64)[order], np.asarray(ids, dtype=np.int64)[order], np.asarray(txt, dtype=object)[order])
        return self._c[channel]

    def text_of(self, channel: int, msg_id: int) -> Optional[str]:
        channel = int(channel)
        got = self.get(channel)
        if got is None:
            return None
        if channel not in self._pos:
            self._pos[channel] = {int(i): k for k, i in enumerate(got[1].tolist())}
        k = self._pos[channel].get(int(msg_id))
        return None if k is None else str(got[2][k])


def _read_ts(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        d = json.load(f)
    arr = d.get("data", []) if isinstance(d, dict) else d
    ts = np.asarray([int(a[0]["$date"]) if isinstance(a[0], dict) else int(a[0]) for a in arr], dtype=np.int64) * 10 ** 6
    v = np.asarray([np.nan if a[1] is None else float(a[1]) for a in arr], dtype=np.float64)
    return ts, v


class TwitterStore:
    """Lazy per-coin Twitter buckets: (bucket_end_ns sorted, tweets, users, sentiment)."""

    def __init__(self, root: str, agg_minutes: int):
        self.dir = os.path.join(root, TWITTER_DIR); self.agg = max(1, int(agg_minutes))
        self._c: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    def _path(self, sub: str, sym: str) -> Optional[str]:
        for name in (f"${sym}.json", f"{sym}.json"):
            p = os.path.join(self.dir, sub, name)
            if os.path.exists(p):
                return p
        return None

    def get(self, sym: str):
        if sym not in self._c:
            p = self._path("tweets-ts", sym)
            if p is None:
                self._c[sym] = None
            else:
                ts, n = _read_ts(p)
                df = pd.DataFrame({"tweets": n}, index=ts).groupby(level=0).sum()
                for sub, col in (("users-ts", "users"), ("sentiment-ts", "sent")):
                    q = self._path(sub, sym)
                    if q is not None:
                        t2, v2 = _read_ts(q)
                        s = pd.Series(v2, index=t2)
                        df[col] = s.groupby(level=0).sum().reindex(df.index) if col == "users" else s.groupby(level=0).mean().reindex(df.index)
                    else:
                        df[col] = np.nan
                df = df[df["tweets"] > 0]
                step = self.agg * tu.MIN_NS
                bucket = (df.index.to_numpy() // step) * step
                w = df["tweets"].to_numpy()
                sw = np.where(np.isfinite(df["sent"].to_numpy()), w, 0.0)
                g = pd.DataFrame({"b": bucket, "tweets": w, "users": df["users"].to_numpy(), "sx": np.nan_to_num(df["sent"].to_numpy()) * sw, "sw": sw,
                                  "has_u": np.isfinite(df["users"].to_numpy())}).groupby("b").sum()
                sent = np.where(g["sw"].to_numpy() > 0, g["sx"].to_numpy() / np.where(g["sw"].to_numpy() > 0, g["sw"].to_numpy(), 1.0), np.nan)
                users = np.where(g["has_u"].to_numpy() > 0, g["users"].to_numpy(), np.nan)
                self._c[sym] = (g.index.to_numpy().astype(np.int64) + step, g["tweets"].to_numpy(dtype=np.float64), users, sent)
        return self._c[sym]


def pump_rows(root: str, cfg: Dict[str, Any], store: ChannelStore) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tab = read_pump_table(root)
    stats: Dict[str, Any] = {"rows": int(len(tab)), "channels": int(tab["channel"].nunique()), "coins": int(tab["symbol"].nunique())}
    excl = {str(x).upper() for x in (cfg.get("exclude_coins") or EXCLUDE)}
    m = tab["symbol"].isin(excl) | ~tab["symbol"].str.fullmatch(r"[A-Z0-9]{1,12}")
    stats["dropped_excluded_coins"] = int(m.sum()); tab = tab[~m].reset_index(drop=True)
    kdir = os.path.join(root, tu.KLINE_SUBDIR); quote = str(cfg.get("quote", "BTC")).upper()
    has = {s: bool(common.find_kline_files(kdir, s, quote)) for s in tab["symbol"].unique()}
    m = tab["symbol"].map(has)
    stats["rows_coin_without_binance_klines"] = int((~m).sum()); stats["coins_with_binance_klines"] = int(sum(has.values()))
    tab = tab[m].reset_index(drop=True)
    need_text = bool(cfg.get("require_ticker_in_text", True)); ex = tu.norm_exchange(cfg.get("exchange", "any"))
    texts, venue, ok = [], [], np.ones(len(tab), dtype=bool)
    for k, r in enumerate(tab.itertuples(index=False)):
        txt = store.text_of(r.channel, r.msg_id)
        texts.append(txt)
        low = (txt or "").lower()
        v = [x for x in VENUES if x in low]
        venue.append("|".join(v))
        if need_text and (txt is None or r.symbol not in tu.extract_tickers(txt, [r.symbol])):
            ok[k] = False
    stats["dropped_ticker_not_in_message"] = int((~ok).sum()); stats["rows_message_not_in_jl"] = int(sum(t is None for t in texts))
    tab["venue"] = venue
    tab = tab[ok].reset_index(drop=True)
    if ex not in ("any", "all", ""):
        m = tab["venue"].str.contains(ex)
        stats["dropped_venue_not_mentioned"] = int((~m).sum()); tab = tab[m].reset_index(drop=True)
    stats["rows_kept"] = int(len(tab)); stats["venues_mentioned"] = {k or "(none)": int(v) for k, v in tab["venue"].value_counts().head(12).items()}
    return tab, stats


def pump_episodes(tab: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    quote = str(cfg.get("quote", "BTC")).upper()
    out = []
    if not len(tab):
        return out
    for grp in tu.merge_events(tab[["symbol", "time_ns"]], float(cfg.get("merge_hours", 6.0))):
        g = tab.iloc[grp]; first = g.iloc[0]
        anns = [{"channel": int(r.channel), "channel_id": tu.stable_id(int(r.channel), "tg:"), "announce_ns": int(r.time_ns), "msg_id": int(r.msg_id), "venue": str(r.venue)}
                for r in g.itertuples(index=False)]
        venues = sorted({v for a in anns for v in a["venue"].split("|") if v})
        out.append({"key": tu.pump_key(str(first["symbol"]), quote, int(first["time_ns"])), "symbol": str(first["symbol"]), "quote": quote,
                    "exchange": "|".join(venues) if venues else "unknown (no exchange column; Binance klines exist)", "channel": int(first["channel"]),
                    "channels": sorted({a["channel"] for a in anns}), "announcements": anns, "anchor_ns": int(first["time_ns"]), "peak_ns": None,
                    "is_pump": True, "control_of": None, "source": "mirtaheri_classified"})
    return out


# ------------------------------------------------------------------------------------------------ social stream
def _social_fn(store: ChannelStore, tw: TwitterStore, cfg: Dict[str, Any]):
    use_tw = bool(cfg.get("twitter", True)); use_tg = bool(cfg.get("telegram", True))
    oracle = str(cfg.get("announcement_score", "label")).lower() == "label"

    def fn(episodes: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
        ev: List[pd.DataFrame] = []; docs: List[pd.DataFrame] = []
        n_tw = n_tg = 0; ents: Dict[int, str] = {TWITTER_SOURCE: "twitter (all accounts; per-coin cashtag time series)"}
        kw_cache: Dict[int, np.ndarray] = {}
        for e in episodes:
            lo, hi = ctx["window"][e["key"]]
            if use_tw:
                got = tw.get(str(e["symbol"]))
                if got is not None:
                    end, n, users, sent = got
                    step = tw.agg * tu.MIN_NS
                    i0 = int(np.searchsorted(end, lo + step, side="left")); i1 = int(np.searchsorted(end, hi, side="left"))   # bucket inside the window
                    if i1 > i0:
                        n_tw += 1
                        ev.append(pd.DataFrame({"key": e["key"], "time_ns": end[i0:i1], "magnitude": n[i0:i1], "reach": users[i0:i1],
                                                "text_score": sent[i0:i1], "source_id": TWITTER_SOURCE}))
            if use_tg:
                own = {(int(a["channel"]), int(a["msg_id"])) for a in e.get("announcements", [])} if (e.get("is_pump") and oracle) else set()
                any_msg = False
                for ch in e.get("channels") or []:
                    got = store.get(int(ch))
                    if got is None:
                        continue
                    t, ids, txt = got
                    i0 = int(np.searchsorted(t, lo, side="left")); i1 = int(np.searchsorted(t, hi, side="left"))
                    if i1 <= i0:
                        continue
                    any_msg = True
                    if int(ch) not in kw_cache:
                        kw_cache[int(ch)] = np.full(len(t), np.inf)
                    kw = kw_cache[int(ch)]
                    todo = np.arange(i0, i1)[np.isinf(kw[i0:i1])]
                    if len(todo):
                        kw[todo] = tu.text_scores(txt[todo])
                    score = kw[i0:i1].copy()
                    if own:
                        mine = np.array([(int(ch), int(i)) in own for i in ids[i0:i1]])
                        score[mine] = 1.0
                    sid = tu.stable_id(int(ch), "tg:"); ents[sid] = f"telegram channel {int(ch)}"
                    ev.append(pd.DataFrame({"key": e["key"], "time_ns": t[i0:i1], "magnitude": 1.0, "reach": np.nan, "text_score": score, "source_id": sid}))
                    has = np.array([bool(str(x).strip()) for x in txt[i0:i1]])
                    if has.any():
                        docs.append(pd.DataFrame({"key": e["key"], "time_ns": t[i0:i1][has], "text": txt[i0:i1][has]}))
                n_tg += int(any_msg)
        info = {"kind": "Twitter per-coin tweet-count buckets + Telegram channel messages, both on platform 0 (source 1 = Twitter)",
                "twitter": {"agg_minutes": tw.agg, "episodes_with_events": n_tw, "marks": "magnitude = tweets, reach = users (sum of per-minute unique users), "
                            "text_score = tweet-weighted mean sentiment; time = avail = bucket end", "enabled": use_tw},
                "telegram": {"episodes_with_events": n_tg, "marks": "magnitude 1, reach NaN (views are crawl-time totals, not PIT), text_score 1.0 = classified pump "
                             f"message of the episode's coin ({'on' if oracle else 'off'}), else keyword score", "enabled": use_tg},
                "coverage_note": f"Twitter series for {n_tw} episodes, Telegram channel messages for {n_tg} episodes (controls watch the channels of their pump)",
                "text_note": "Telegram message text of the episode's channels (tweets are released as time series only, no text)"}
        return {"events": pd.concat(ev, ignore_index=True) if ev else None, "docs": pd.concat(docs, ignore_index=True) if docs else None,
                "info": info, "entities": ents}
    return fn


# ------------------------------------------------------------------------------------------------ prepare
def prepare(root: str, args: Any) -> None:
    """Called by scripts/xp_prepare_data.py only: clones the repository when absent and, with args['klines'], fetches
    the Binance 1-minute klines for every coin of coin-pump.csv over the months of its classified messages (pairs that
    never traded on Binance return 404 and are recorded in the kline manifest)."""
    a = tu.args_dict(args)
    base = os.path.join(root, SUBDIR)
    if not os.path.exists(os.path.join(root, PUMP_CSV)):
        import subprocess
        os.makedirs(os.path.dirname(base), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SOURCE_URL, base], check=True)
    if a.get("klines"):
        tab = read_pump_table(root)
        tab = tab[~tab["symbol"].isin(set(EXCLUDE))]
        c = dict(tu.DEFAULTS)
        eps = [{"symbol": s, "anchor_ns": int(t)} for s, t in zip(tab["symbol"], tab["time_ns"])]
        months = tu.episode_months(eps, int(c["pre_bars"]), int(c["post_bars"]), int(c["bar_minutes"]), pad_days=float(c["control_span_days"]) + 1.0)
        try:
            from . import binance_fetch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"crosspump/data/loaders/binance_fetch.py is not importable ({type(e).__name__}: {e}); fetch the klines manually:\n{RECIPE}")
        st = binance_fetch.fetch_klines({s: sorted(m) for s, m in months.items()}, os.path.join(root, tu.KLINE_SUBDIR), quote="BTC", workers=int(a.get("workers") or 8))
        print(f"[mirtaheri] klines: {pd.Series(list(st.values())).value_counts().to_dict()} for {len(months)} symbols")


# ------------------------------------------------------------------------------------------------ builder
@common.register_dataset("xp_mirtaheri")
class MirtaheriBuilder(BaseDatasetBuilder):
    name = "xp_mirtaheri"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self) -> Panel:
        cfg = dict(self.cfg)
        root = str(cfg.get("root", "data_raw"))
        common.require(root, REQUIRED, RECIPE)
        store = ChannelStore(root)
        tab, stats = pump_rows(root, cfg, store)
        pumps = pump_episodes(tab, cfg)
        stats["pump_attempts_after_merge"] = len(pumps)
        if not pumps:
            raise FileNotFoundError(f"xp_mirtaheri: no classified pump attempt has Binance klines under {os.path.join(root, tu.KLINE_SUBDIR)} "
                                    f"(table stats: {stats}).\nDownload recipe:\n{RECIPE}")
        all_times = {s: np.sort(g["time_ns"].to_numpy().astype(np.int64)) for s, g in tab.groupby("symbol")}
        tw = TwitterStore(root, int(cfg.get("twitter_agg_minutes", 5)))
        return tu.build_panel(cfg, pumps, name=self.name, dataset="mirtaheri", source=SOURCE_URL, licence=LICENCE, recipe=RECIPE,
                              social_fn=_social_fn(store, tw, cfg), pump_times=all_times, assumptions=ASSUMPTIONS, table_stats=stats,
                              extra_meta={"paper": "Mirtaheri et al., IEEE TCSS 8(3) 2021, doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110",
                                          "social_platforms": "Twitter AND Telegram, both platform 0 (social); source id 1 = Twitter, other ids = hashed Telegram channel",
                                          "merge_hours": float(cfg.get("merge_hours", 6.0)), "twitter_agg_minutes": tw.agg})
