"""D8 xp_dex_wash: on-chain DEX trades (EtherDelta, IDEX; 2017-2018) of tokens that were Telegram pump targets.

Source   Victor & Weintraud, "Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges", The Web
         Conference 2021, doi:10.1145/3442381.3449824; data: Zenodo record 4540223 (CC-BY-4.0).
Raw      <root>/crosspump/victor_dex/EtherDeltaTrades.csv   transaction_hash, block_number, timestamp (s), tokenGet, amountGet,
                                                            tokenGive, amountGive, get, give
         <root>/crosspump/victor_dex/IDEXTrades.csv         transaction_hash, status, block_number, gas, gas_price, timestamp (s),
                                                            amountBuy, amountSell, expires, nonce, amount, tradeNonce, feeMake,
                                                            feeTake, tokenBuy, tokenSell, maker, taker
         <root>/crosspump/victor_dex/token_decimals.json    {SYMBOL: {name, decimals, address[, slug]}}
         pump tables (owned by the D2 / D7 loaders, read here only for the pump times):
         <root>/crosspump/sapienza_pd/pump_telegram.csv     symbol, group, date, hour, exchange
         <root>/crosspump/mirtaheri/Telegram/classified/coin-pump.csv   (index), Channel ID, Message ID, Date, Time, Coin  [optional]
Cache    <root>/crosspump/victor_dex/cache/xp_dex_wash_<exchange>_<selection hash>.csv: the normalised trades of the selected
         tokens inside their time envelope (written by build_panel on first use from the multi-GB csvs read in pandas
         chunks; cfg cache_dir overrides the directory; delete the file to rebuild).

Episodes  A pump episode = one (token, announcement cluster): every row of the pump tables whose symbol is an ERC-20 token of
          token_decimals.json; rows of the same symbol closer than dedup_hours are ONE episode anchored at the earliest row.
          Only episodes with at least min_trades_in_window on-chain trades inside the window survive. Controls: the same
          token at times with no pump within +-gap_hours (common.sample_controls), same trade-count filter.
Bars      built here from the trades (onchain_utils.trades_to_bars; bar_minutes 15): price = ETH per token, volume = token
          amount (decimals from token_decimals.json), taker_buy_volume from the taker side, count = trades; bars without a
          trade between the token's first and last trade carry the last traded price with volume 0 (untradable).
Streams   wallet     every taker-BUY trade = one wallet inflow event (wallet_id = hashed taker address, size_adv = amount /
                     trailing 7-day mean daily volume of the token, size_raw = amount); avail = trade time + wallet_lag_minutes
          orderflow  causal EWMA z-score bursts of per-bar trade counts (time = avail = bar close)
          social     the pump announcement rows / classified pump messages inside the window (source = hashed group or
                     channel id; reach and text_score unknown = NaN)
Extra     meta['wash_labels']: per episode the number / share of trades in the window flagged by the cheap part of the
          authors' rule (self-trades and 2-cycles with matching amounts within wash_window_s). 3-cycles and the authors'
          SCC-based volume matching are NOT implemented; these are NOT the authors' released labels. Never used as features.

ASSUMPTIONS (also in meta['assumptions'])
 A1 EtherDelta Trade event semantics: `get` = maker (receives tokenGet), `give` = taker (msg.sender, receives tokenGive);
    amounts are the traded amounts. tokenGet == ETH (zero address) => the taker buys the token. Token-token trades are dropped.
 A2 IDEX semantics: the maker's order buys tokenBuy and sells tokenSell; `amount` is the filled amount in tokenBuy units, the
    counter amount is amountSell * amount / amountBuy. tokenBuy == ETH => the taker buys the token. Rows with status 0.0
    (failed transactions; 2,361 rows observed) are dropped; status 1.0 (5,341,213 rows) and empty status (575 early rows) are kept.
 A3 The pump tables list pumps announced for CENTRALISED exchanges (Binance, Yobit, Cryptopia, Bittrex, ...); neither table
    has EtherDelta / IDEX pumps. The episode is therefore the on-chain DEX market of the SAME token around the announcement
    (cross-venue evidence), matched by ticker symbol. Ticker collisions between a CEX coin and an ERC-20 token with the same
    symbol are possible and not resolved.
 A4 Pump table times are UTC (Sapienza: date + hour; Mirtaheri: Date + Time of the Telegram message).
 A5 The Mirtaheri table holds messages classified as pump-related with an extracted coin; a cluster of such messages is
    treated as one announcement anchored at its first message (it may include pre-announcement teasers).
 A6 Peak = argmax of the bar high within peak_window_bars after the anchor on the DEX bars (no curated peak exists).
 A7 Trailing ADV for size_adv uses whole UTC days strictly before the trade's day (NaN on the token's first day).
"""
from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from . import common
from . import onchain_utils as ou

SOURCE_URL = "https://zenodo.org/records/4540223 (doi:10.1145/3442381.3449824)"
LICENCE = "CC-BY-4.0 (Zenodo record 4540223)"
ZENODO_FILES = ["EtherDeltaTrades.csv", "IDEXTrades.csv", "EtherDollarPrice.csv", "token_decimals.json"]
REQUIRED = ["crosspump/victor_dex/token_decimals.json", "crosspump/victor_dex/EtherDeltaTrades.csv", "crosspump/victor_dex/IDEXTrades.csv",
            "crosspump/sapienza_pd/pump_telegram.csv"]
RECIPE = """D8 xp_dex_wash (Victor & Weintraud, WWW 2021; Zenodo 4540223, CC-BY-4.0)
  python3 scripts/xp_prepare_data.py --dataset dex_wash          # downloads the four Zenodo files (3.2 GB)
or by hand:
  mkdir -p data_raw/crosspump/victor_dex && cd data_raw/crosspump/victor_dex
  for f in EtherDeltaTrades.csv IDEXTrades.csv EtherDollarPrice.csv token_decimals.json; do
    curl -L -C - -o $f "https://zenodo.org/records/4540223/files/$f?download=1"; done
Pump times come from the D2 / D7 raw files:
  git clone https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset data_raw/crosspump/sapienza_pd      (pump_telegram.csv, required)
  data_raw/crosspump/mirtaheri/Telegram/classified/coin-pump.csv   (optional; see the D7 recipe: xp_prepare_data.py --dataset mirtaheri)"""

ETH = "0x0000000000000000000000000000000000000000"
NORM_COLS = ["symbol", "exchange", "time_ns", "price", "amount", "taker_buy", "buyer", "seller", "tx"]
ASSUMPTIONS = [
    "A1 EtherDelta: get = maker, give = taker; tokenGet == ETH means the taker buys the token; token-token trades dropped",
    "A2 IDEX: amount is in tokenBuy units, counter amount = amountSell * amount / amountBuy; tokenBuy == ETH means the taker buys the token; rows with a non-success status dropped",
    "A3 pump tables list CEX pumps (no EtherDelta/IDEX pumps exist in them): episode = the on-chain DEX market of the same ticker around the announcement; ticker collisions not resolved",
    "A4 pump table times are UTC",
    "A5 Mirtaheri classified pump messages of one coin closer than dedup_hours form one episode anchored at the first message",
    "A6 peak = argmax bar high within peak_window_bars after the anchor (DEX bars); no curated peak",
    "A7 size_adv = amount / mean daily token volume over the 7 whole UTC days before the trade's day (NaN on the first day)",
    "bars without trades between a token's first and last trade carry the last traded price with volume 0 (untradable)",
    "wash_labels = self-trades + 2-cycles only (cheap subset of the authors' rule), meta only, not the authors' labels",
]


def prepare(root: str, args: Optional[Dict[str, Any]] = None) -> None:
    """Download the Zenodo files (resumable is NOT attempted: a partial file is re-downloaded). Called only by scripts/xp_prepare_data.py."""
    import urllib.request
    d = os.path.join(root, "crosspump", "victor_dex")
    os.makedirs(d, exist_ok=True)
    for f in ZENODO_FILES:
        dst = os.path.join(d, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[dex_wash] have {dst}"); continue
        url = f"https://zenodo.org/records/4540223/files/{f}?download=1"
        try:
            print(f"[dex_wash] downloading {url}", flush=True)
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
        except Exception as e:  # network / proxy problems must not look like success
            print(f"[dex_wash] FAILED to download {f}: {e!r}\nFetch it by hand:\n{RECIPE}")
    for p in REQUIRED[3:]:
        if not os.path.exists(os.path.join(root, p)):
            print(f"[dex_wash] pump table missing: {os.path.join(root, p)} (run xp_prepare_data.py --dataset sapienza_pd)")


# ------------------------------------------------------------------------------------------------ raw readers
def load_token_table(path: str) -> pd.DataFrame:
    """token_decimals.json -> DataFrame[symbol (upper), address (lower), decimals]; ETH and entries without decimals dropped."""
    with open(path, "r") as f:
        d = json.load(f)
    rows = []
    for sym, v in d.items():
        try:
            addr, dec = str(v["address"]).lower(), int(v["decimals"])
        except (KeyError, TypeError, ValueError):
            continue
        if addr == ETH or not addr.startswith("0x"):
            continue
        rows.append((str(sym).upper().strip(), addr, dec))
    t = pd.DataFrame(rows, columns=["symbol", "address", "decimals"])
    return t.drop_duplicates("symbol", keep="first").drop_duplicates("address", keep="first").reset_index(drop=True)


def load_pump_rows(root: str, use_mirtaheri: bool = True) -> pd.DataFrame:
    """Rows of the pump tables -> DataFrame[symbol, time_ns, source_name, table, exchange] (UTC; unparsable rows dropped)."""
    parts = []
    p = os.path.join(root, "crosspump", "sapienza_pd", "pump_telegram.csv")
    if os.path.exists(p):
        s = pd.read_csv(p, dtype=str)
        s.columns = [c.strip().lower() for c in s.columns]
        ts = pd.to_datetime(s["date"].str.strip() + " " + s["hour"].str.strip(), utc=True, errors="coerce")
        parts.append(pd.DataFrame({"symbol": s["symbol"].str.upper().str.strip(), "ts": ts, "source_name": "sapienza:" + s["group"].astype(str).str.strip(),
                                   "table": "sapienza", "exchange": s["exchange"].astype(str).str.lower().str.strip()}))
    p = os.path.join(root, "crosspump", "mirtaheri", "Telegram", "classified", "coin-pump.csv")
    if use_mirtaheri and os.path.exists(p):
        m = pd.read_csv(p, dtype=str)
        m.columns = [c.strip().lower() for c in m.columns]
        ts = pd.to_datetime(m["date"].str.strip() + " " + m["time"].str.strip(), utc=True, errors="coerce")
        parts.append(pd.DataFrame({"symbol": m["coin"].str.upper().str.strip(), "ts": ts, "source_name": "mirtaheri:" + m["channel id"].astype(str).str.strip(),
                                   "table": "mirtaheri", "exchange": "telegram"}))
    if not parts:
        return pd.DataFrame(columns=["symbol", "time_ns", "source_name", "table", "exchange"])
    df = pd.concat(parts, ignore_index=True)
    df = df[df["ts"].notna() & df["symbol"].notna()].copy()
    df["time_ns"] = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    return df.drop(columns="ts").sort_values(["symbol", "time_ns"], kind="stable").reset_index(drop=True)


def cluster_pumps(rows: pd.DataFrame, dedup_hours: float, quote: str) -> List[Dict[str, Any]]:
    """Rows of one symbol closer than dedup_hours (chained) -> one episode anchored at the earliest row."""
    eps: List[Dict[str, Any]] = []
    gap = dedup_hours * 3600 * 10 ** 9
    for sym, g in rows.groupby("symbol", sort=True):
        t = g["time_ns"].to_numpy()
        cid = np.concatenate([[0], np.cumsum(np.diff(t) > gap)])
        for c in np.unique(cid):
            sub = g[cid == c]
            a = int(sub["time_ns"].iloc[0])
            eps.append({"key": f"{sym}{quote}@{ou.iso_minute(a)}", "symbol": sym, "exchange": "etherdelta+idex", "channel": str(sub["source_name"].iloc[0]),
                        "anchor_ns": a, "peak_ns": None, "is_pump": True, "control_of": None, "source": "+".join(sorted(set(sub["table"]))),
                        "announced_on": "+".join(sorted(set(sub["exchange"]))), "n_messages": int(len(sub))})
    return eps


def _envelope(addr: pd.Series, t_s: pd.Series, lo_s: Dict[str, float], hi_s: Dict[str, float]) -> np.ndarray:
    lo, hi = addr.map(lo_s), addr.map(hi_s)
    t = pd.to_numeric(t_s, errors="coerce")
    return (lo.notna() & (t >= lo) & (t <= hi)).to_numpy()


def _finish_norm(token: np.ndarray, t_s: pd.Series, tok_raw: np.ndarray, eth_raw: np.ndarray, taker_buy: np.ndarray, taker: np.ndarray, maker: np.ndarray, tx: pd.Series,
                 sym_of: Dict[str, str], dec_of: Dict[str, int], exchange: str) -> pd.DataFrame:
    dec = pd.Series(token).map(dec_of).to_numpy(dtype=np.float64)
    amount = tok_raw / np.power(10.0, dec)
    with np.errstate(divide="ignore", invalid="ignore"):
        price = (eth_raw / 1e18) / amount
    out = pd.DataFrame({"symbol": pd.Series(token).map(sym_of).to_numpy(), "exchange": exchange, "time_ns": pd.to_numeric(t_s, errors="coerce").to_numpy(dtype=np.float64) * 1e9,
                        "price": price, "amount": amount, "taker_buy": taker_buy, "buyer": np.where(taker_buy, taker, maker), "seller": np.where(taker_buy, maker, taker),
                        "tx": tx.astype(str).to_numpy()})
    ok = np.isfinite(out["price"].to_numpy()) & (out["price"].to_numpy() > 0) & np.isfinite(out["amount"].to_numpy()) & (out["amount"].to_numpy() > 0) & np.isfinite(out["time_ns"].to_numpy())
    out = out[ok].copy()
    out["time_ns"] = out["time_ns"].to_numpy().astype(np.int64)
    return out[NORM_COLS]


def scan_etherdelta(path: str, sym_of: Dict[str, str], dec_of: Dict[str, int], lo_s: Dict[str, float], hi_s: Dict[str, float], cache: Optional[str], chunksize: int) -> pd.DataFrame:
    def token_of(ch):
        tg, tv = ch["tokenGet"].str.lower(), ch["tokenGive"].str.lower()
        return tg, tv, tv.where(tg == ETH, tg)

    def keep(ch):
        tg, tv, tok = token_of(ch)
        return ((tg == ETH) ^ (tv == ETH)).to_numpy() & _envelope(tok, ch["timestamp"], lo_s, hi_s)

    def transform(ch):
        tg, tv, tok = token_of(ch)
        tb = (tg == ETH).to_numpy()                                   # maker receives ETH -> the taker buys the token (A1)
        ag, av = ch["amountGet"].to_numpy(dtype=np.float64), ch["amountGive"].to_numpy(dtype=np.float64)
        return _finish_norm(tok.to_numpy(), ch["timestamp"], np.where(tb, av, ag), np.where(tb, ag, av), tb, ch["give"].str.lower().to_numpy(), ch["get"].str.lower().to_numpy(),
                            ch["transaction_hash"], sym_of, dec_of, "etherdelta")
    cols = ["transaction_hash", "timestamp", "tokenGet", "amountGet", "tokenGive", "amountGive", "get", "give"]
    return ou.chunked_filter_csv(path, keep, usecols=cols, chunksize=chunksize, cache_path=cache, transform=transform,
                                 dtype={"transaction_hash": str, "tokenGet": str, "tokenGive": str, "get": str, "give": str, "amountGet": np.float64, "amountGive": np.float64})


def scan_idex(path: str, sym_of: Dict[str, str], dec_of: Dict[str, int], lo_s: Dict[str, float], hi_s: Dict[str, float], cache: Optional[str], chunksize: int) -> pd.DataFrame:
    def token_of(ch):
        tb_, ts_ = ch["tokenBuy"].str.lower(), ch["tokenSell"].str.lower()
        return tb_, ts_, ts_.where(tb_ == ETH, tb_)

    def keep(ch):
        tb_, ts_, tok = token_of(ch)
        st = ch["status"].astype(str).str.strip().str.lower()
        ok_status = ch["status"].isna().to_numpy() | st.isin(("1", "1.0", "true", "success", "")).to_numpy()
        return ((tb_ == ETH) ^ (ts_ == ETH)).to_numpy() & ok_status & _envelope(tok, ch["timestamp"], lo_s, hi_s)

    def transform(ch):
        tb_, ts_, tok = token_of(ch)
        tb = (tb_ == ETH).to_numpy()                                  # maker buys ETH -> the taker buys the token (A2)
        amt, ab, as_ = (ch[k].to_numpy(dtype=np.float64) for k in ("amount", "amountBuy", "amountSell"))
        with np.errstate(divide="ignore", invalid="ignore"):
            counter = as_ * amt / ab
        return _finish_norm(tok.to_numpy(), ch["timestamp"], np.where(tb, counter, amt), np.where(tb, amt, counter), tb, ch["taker"].str.lower().to_numpy(), ch["maker"].str.lower().to_numpy(),
                            ch["transaction_hash"], sym_of, dec_of, "idex")
    cols = ["transaction_hash", "status", "timestamp", "amountBuy", "amountSell", "amount", "tokenBuy", "tokenSell", "maker", "taker"]
    return ou.chunked_filter_csv(path, keep, usecols=cols, chunksize=chunksize, cache_path=cache, transform=transform,
                                 dtype={"transaction_hash": str, "status": str, "tokenBuy": str, "tokenSell": str, "maker": str, "taker": str, "amountBuy": np.float64, "amountSell": np.float64, "amount": np.float64})


def _count_in_windows(t_sorted: np.ndarray, eps: List[Dict[str, Any]], pre: int, post: int, bm: int) -> np.ndarray:
    return np.array([np.searchsorted(t_sorted, hi, "left") - np.searchsorted(t_sorted, lo, "left") for lo, hi in (ou.episode_window_ns(e, pre, post, bm) for e in eps)], dtype=np.int64)


@common.register_dataset("xp_dex_wash")
class DexWashBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "xp_dex_wash", "15min", 365 * 24 * 4, (1, 2, 3, 6, 12, 24)

    def build_panel(self) -> Panel:
        c = self.cfg
        root = c.get("root", "data_raw")
        sub = os.path.join(root, c.get("subdir", "crosspump/victor_dex"))
        exchanges = [str(x).lower() for x in c.get("dex", ["etherdelta", "idex"])]
        files = {"etherdelta": "EtherDeltaTrades.csv", "idex": "IDEXTrades.csv"}
        common.require(root, [REQUIRED[0]] + [f"crosspump/victor_dex/{files[x]}" for x in exchanges] + [REQUIRED[3]], RECIPE)
        quote = str(c.get("quote", "ETH"))
        bm, pre, post, lead = int(c.get("bar_minutes", 15)), int(c.get("pre_bars", 288)), int(c.get("post_bars", 96)), int(c.get("lead", 2))
        horizons = list(c.get("horizons", self.horizons))
        gap_h, span_d, adv_days = float(c.get("gap_hours", 96)), float(c.get("control_span_days", 30)), float(c.get("adv_days", 7))
        min_trades = int(c.get("min_trades_in_window", 20))
        rng = np.random.default_rng(int(c.get("seed", 0)))

        # ---- pump candidates restricted to ERC-20 tokens of the token table
        tok = load_token_table(os.path.join(sub, "token_decimals.json"))
        rows = load_pump_rows(root, bool(c.get("use_mirtaheri", True)))
        rows = rows[rows["symbol"].isin(set(tok["symbol"]))].reset_index(drop=True)
        if len(rows) == 0:
            raise FileNotFoundError(f"No pump-table symbol matches an ERC-20 token of {sub}/token_decimals.json.\n{RECIPE}")
        cand = cluster_pumps(rows, float(c.get("dedup_hours", 24)), quote)
        step = bm * common.MIN_NS
        margin_lo = (pre * step + (adv_days + 1 + span_d) * 86400 * 10 ** 9) / 1e9
        margin_hi = ((post + 1) * step + span_d * 86400 * 10 ** 9) / 1e9
        a_min = pd.Series({s: min(e["anchor_ns"] for e in cand if e["symbol"] == s) for s in {e["symbol"] for e in cand}}) / 1e9
        a_max = pd.Series({s: max(e["anchor_ns"] for e in cand if e["symbol"] == s) for s in a_min.index}) / 1e9
        tk = tok[tok["symbol"].isin(a_min.index)]
        sym_of, dec_of = dict(zip(tk["address"], tk["symbol"])), dict(zip(tk["address"], tk["decimals"]))
        lo_s = {a: float(a_min[s] - margin_lo) for a, s in sym_of.items()}
        hi_s = {a: float(a_max[s] + margin_hi) for a, s in sym_of.items()}

        # ---- trades of the selected tokens (chunked scan of the multi-GB csvs, cached)
        sel = hashlib.sha1(json.dumps([sorted(lo_s.items()), sorted(hi_s.items())]).encode()).hexdigest()[:12]
        cache_dir = c.get("cache_dir") or os.path.join(sub, "cache")
        chunk = int(c.get("chunksize", 500_000))
        parts = []
        for x in exchanges:
            scan = scan_etherdelta if x == "etherdelta" else scan_idex
            cache = os.path.join(cache_dir, f"xp_dex_wash_{x}_{sel}.csv") if c.get("use_cache", True) else None
            parts.append(scan(os.path.join(sub, files[x]), sym_of, dec_of, lo_s, hi_s, cache, chunk))
        trades = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame(columns=NORM_COLS)
        if len(trades) == 0:
            raise FileNotFoundError(f"No on-chain trade of any pump-table token inside its window in {sub}.\n{RECIPE}")
        trades["taker_buy"] = trades["taker_buy"].astype(str).str.lower().isin(("true", "1"))
        for k in ("buyer", "seller", "tx"):
            trades[k] = trades[k].fillna("").astype(str)
        trades = trades.sort_values(["symbol", "time_ns"], kind="stable").reset_index(drop=True)
        by_sym = {s: g.reset_index(drop=True) for s, g in trades.groupby("symbol", sort=True)}

        # ---- episodes with enough on-chain trades, then controls on the same tokens
        n_cand = len(cand)
        pumps = [e for e in cand if e["symbol"] in by_sym]
        pumps = [e for e, n in zip(pumps, [int(_count_in_windows(by_sym[e["symbol"]]["time_ns"].to_numpy(), [e], pre, post, bm)[0]) for e in pumps]) if n >= min_trades]
        pumps.sort(key=lambda e: e["anchor_ns"])
        if c.get("max_pumps"):
            pumps = pumps[: int(c["max_pumps"])]
        if not pumps:
            raise FileNotFoundError(f"No pump episode with >= {min_trades} on-chain trades in its window ({n_cand} candidates).\n{RECIPE}")
        pump_times = {s: np.array(sorted(e["anchor_ns"] for e in cand if e["symbol"] == s), dtype=np.float64) for s in {e["symbol"] for e in pumps}}
        ctrl = common.sample_controls(pumps, pump_times, int(c.get("n_controls_per_pump", 2)), gap_h, rng, span_days=span_d)
        ctrl = [e for e in ctrl if int(_count_in_windows(by_sym[e["symbol"]]["time_ns"].to_numpy(), [e], pre, post, bm)[0]) >= min_trades]
        for k, e in enumerate(ctrl):
            e["key"] = f"CTRL{k:05d}_{e['symbol']}{quote}@{ou.iso_minute(e['anchor_ns'])}"
        episodes = pumps + ctrl
        syms = sorted({e["symbol"] for e in episodes})

        # ---- bars, panel
        raw_bars = {s: ou.trades_to_bars(by_sym[s], bm) for s in syms}
        bars = {s: ou.carry_forward_bars(raw_bars[s], bm) for s in syms}
        panel, skipped = common.episode_panel(bars, episodes, pre, post, bm, lead, horizons, peak_window_bars=int(c.get("peak_window_bars", 12)),
                                              end_bars_after_peak=int(c.get("end_bars_after_peak", 24)), min_bars_frac=float(c.get("min_bars_frac", 0.25)),
                                              dataset=self.name, source=SOURCE_URL, licence=LICENCE)
        kept = [e for e in episodes if e["key"] in set(panel.assets)]

        # ---- streams
        cap = c.get("max_wallet_events_per_episode", 5000)
        wal_parts, burst_parts, entities, wash = [], [], {}, {}
        for s in syms:
            t = by_sym[s]
            w = ou.wallet_inflow_events(t, adv_days=adv_days, side=str(c.get("wallet_side", "taker")))
            entities.update(w.attrs.get("entities", {}))
            w["symbol"] = s; wal_parts.append(w)
            b = ou.trade_count_bursts(raw_bars[s], bm, z_thresh=float(c.get("burst_z", 2.5)), warmup=int(c.get("burst_warmup", 30)), halflife=float(c.get("burst_halflife", 60)))
            b["symbol"] = s; burst_parts.append(b)
            flags = ou.wash_trade_flags(t, window_s=float(c.get("wash_window_s", 3600)), tol=float(c.get("wash_tol", 0.05)))
            tt = t["time_ns"].to_numpy(); cs = np.concatenate([[0], np.cumsum(flags)])
            for e in kept:
                if e["symbol"] == s:
                    lo, hi = ou.episode_window_ns(e, pre, post, bm)
                    i0, i1 = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "left")
                    wash[e["key"]] = {"n_trades": int(i1 - i0), "n_wash": int(cs[i1] - cs[i0]), "share": float((cs[i1] - cs[i0]) / max(i1 - i0, 1))}
        wal = ou.assign_episode_keys(pd.concat(wal_parts, ignore_index=True), kept, pre, post, bm, max_per_episode=int(cap) if cap else None, rng=rng)
        bur = ou.assign_episode_keys(pd.concat(burst_parts, ignore_index=True), kept, pre, post, bm)
        msg = rows[["symbol", "time_ns", "source_name"]].copy()
        ids, names = ou.hash_ids(msg["source_name"].tolist(), salt="tg:")
        msg["source_id"] = ids; msg["reach"] = np.nan; msg["text_score"] = np.nan
        soc = ou.assign_episode_keys(msg, kept, pre, post, bm)
        wallet_lag = float(c.get("wallet_lag_minutes", 3.0)) / bm
        social_lag = float(c.get("social_lag_minutes", 0.0)) / bm
        common.finish_streams(panel, [common.social_stream(panel, soc, lag_bars=social_lag), common.wallet_stream(panel, wal, lag_bars=wallet_lag),
                                      common.orderflow_stream_from_trades(panel, bur)], derive_orderflow_if_missing=False,
                              platform_lags={"social_bars": social_lag, "wallet_bars": wallet_lag, "orderflow": "bar close"},
                              entities={"wallet": "sha1 prefix per hashed taker address (meta['wallet_entities'])", "social": names},
                              wallet_subsampled=wal.attrs.get("subsampled", {}),
                              how={"wallet": "taker-buy DEX trades", "orderflow": "causal z-score bursts of per-bar trade counts", "social": "pump-table rows inside the window"})
        panel.meta["wallet_entities"] = entities
        panel.meta["coverage"]["note"] = "wallet/orderflow from on-chain DEX trades; social = pump-table rows (controls normally have none: missing, not zero)"
        panel.meta["wash_labels"] = {"rule": "self-trade or 2-cycle with matching amounts (tol, window) - cheap subset of Victor & Weintraud; 3-cycles not implemented",
                                     "per_episode": [wash.get(k, {"n_trades": 0, "n_wash": 0, "share": float("nan")}) for k in panel.assets]}
        panel.meta.update({"quote": quote, "exchange": "+".join(exchanges), "assumptions": list(ASSUMPTIONS), "n_candidate_pumps": n_cand, "n_pumps_with_trades": len(pumps),
                           "n_trades_selected": int(len(trades)), "tokens": syms, "skipped": {**skipped, "candidates_without_enough_trades": n_cand - len(pumps)},
                           "lookback": int(c.get("lookback", 48)), "latency": int(c.get("latency", 1)), "price_unit": "ETH per token", "volume_unit": "token units"})
        return panel
