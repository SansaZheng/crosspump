"""D10 xp_memechain: MemeChain meme-coin launches (Ethereum, BNB Smart Chain, Solana, Base).

Source   Mongardini & Mei, "MemeChain: A Multimodal Cross-Chain Dataset for Meme Coin Forensics and Risk Analysis",
         arXiv:2601.22185 (2026); Zenodo record 18246856 (CC-BY-4.0): data.zip (677 MB) + README.md.
Raw      <root>/crosspump/memechain/data.zip (read in place) or the extracted <root>/crosspump/memechain/data/*.csv.
         Tables inside data.zip (observed 2026-10-03; join key = (address, platform)):
           data/confirmed_meme_coins.csv        34,988 rows  symbol, name, platform, address
           data/final_filtered_meme_coins.csv   31,811 rows  symbol, name, address, platform, creation_date (unix s; deployment)
           data/economic_info_data.csv          20,692 rows  name, platform, price, market_cap, volume, symbol, creation_date, address
                                                             (ONE market snapshot per token at crawl time)
           data/social_info.csv                 31,811 rows  name, platform, website, twitter, telegram, discord, symbol, address
           data/web_page_analysis.csv           23,837 rows  ..., chain_patrol_status (UNKNOWN 12,682 / ALLOWED 634 / BLOCKED 9 / empty), ...
           data/html_web_pages_path.csv, data/memes_with_image_paths.csv, data/html_web_pages.zip, data/logos_images.zip
         platforms: bsc 15,455 | solana 11,809 | ethereum 3,720 | base 827; creation dates 2017-06-20 .. 2024-10-24.

*** THE RELEASE CONTAINS NO TRANSACTION / TRADE TABLE. *** The paper's "1,801 tokens (5.15 %) cease trading within 24 h of
launch" is a statistic of the authors' own on-chain crawl; no per-token flag or trade history for it is in data.zip. So:

MODE 'tabular' (what the public release supports; the default when no trade file exists) — DEGENERATE panel
    episodes  every token of final_filtered_meme_coins.csv; anchor = deployment timestamp; 15-minute rows, row pre_bars
              carries the (empty) feature row, everything is NaN and untradable; meta['tabular_only'] = True,
              meta['excluded_from'] = ['lead_time'].
    is_pump   chain_patrol_status == 'BLOCKED' (the only manipulation / scam flag the data provides: 9 tokens). The 24-hour
              cessation label CANNOT be built in this mode. Controls = all other tokens (seeded subsample max_controls).
    streams   social only: ONE static event at launch, reach = number of linked accounts among website / twitter /
              telegram / discord, source = hashed twitter (else telegram, else website) URL. wallet and orderflow MISSING.
MODE 'trades' (used automatically when the OPTIONAL user-supplied file(s) <root>/crosspump/memechain/trades.csv or
    trades/*.csv exist; columns address, platform, timestamp (s/ms/ns or ISO), price, amount, side ('buy'/'sell' = taker
    side) or taker_buy, and buyer/seller or taker/maker — e.g. exported from a DEX indexer for the registry's addresses)
    episodes  token launches; anchor = first trade; bar_minutes 15, pre_bars 4, post_bars 96 (24 h).
    is_pump   the token ceases trading within cease_hours (24) of its first trade — last trade < first + 24 h AND the trade
              file extends at least quiet_hours beyond that last trade (so "no more trades" is observed, not truncated) — or
              it is ChainPatrol-BLOCKED (label_rule 'cease_or_blocked'; 'cease_24h' uses the first criterion only).
    peak      time of the highest traded price (necessarily before cessation); the label covers [anchor - lead, peak].
              LEAD TIME HERE IS RELATIVE TO THE COLLAPSE (peak / last trade, meta['cease_row']), NOT to the launch: the
              anchor of a launch is not an announcement, an alarm "before the anchor" is impossible by construction.
    streams   wallet = every taker-buy trade (wallet_id = hashed buyer, size_adv = amount / cumulative volume traded before
              it, size_raw = amount); orderflow = causal z-score bursts of per-bar trade counts; social = the static event.

ASSUMPTIONS (also in meta['assumptions'])
 A1 creation_date is the contract deployment time in UTC seconds; in 'trades' mode the launch is the first trade instead.
 A2 addresses are joined case-insensitively (lower-cased), platform as given.
 A3 the social links were collected at crawl time; placing them at launch treats them as STATIC evidence (they may have been
    created later) — this is the weakest point-in-time link of the dataset and is flagged in meta['coverage'].
 A4 economic_info_data.csv and web_page_analysis.csv are crawl-time snapshots (after the outcome) and are never features;
    only chain_patrol_status is used, as a label.
 A5 'trades' mode: price / amount units are whatever the supplied file uses (one quote per token assumed); bars without
    trades between first and last trade carry the last price with volume 0; tokens with fewer than 10 bars are dropped by
    common.episode_panel (counted in meta['skipped']) — this removes the very fastest collapses.
"""
from __future__ import annotations
import glob
import io
import os
import zipfile
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from . import common
from . import onchain_utils as ou

SOURCE_URL = "https://zenodo.org/records/18246856 (arXiv:2601.22185)"
LICENCE = "CC-BY-4.0 (Zenodo record 18246856)"
REQUIRED = ["crosspump/memechain/data.zip"]
TABLES = ["final_filtered_meme_coins.csv", "social_info.csv", "web_page_analysis.csv"]
RECIPE = """D10 xp_memechain (MemeChain, arXiv:2601.22185; Zenodo 18246856, CC-BY-4.0)
  python3 scripts/xp_prepare_data.py --dataset memechain        # downloads data.zip (677 MB) + README.md
or by hand:
  mkdir -p data_raw/crosspump/memechain && cd data_raw/crosspump/memechain
  curl -L -C - -o data.zip "https://zenodo.org/records/18246856/files/data.zip?download=1"
  curl -L -o README.md "https://zenodo.org/records/18246856/files/README.md?download=1"
The release has NO trade table. For the time-series mode supply your own DEX trade export of the registry addresses as
  data_raw/crosspump/memechain/trades.csv  (address, platform, timestamp, price, amount, side, buyer, seller)."""
ASSUMPTIONS_COMMON = [
    "A2 addresses joined case-insensitively, platform as given",
    "A3 social links were collected at crawl time; the launch-time social event is STATIC evidence only",
    "A4 economic_info_data / web_page_analysis are crawl-time snapshots, never features; chain_patrol_status is a label only",
]


def prepare(root: str, args: Optional[Dict[str, Any]] = None) -> None:
    """Download data.zip + README.md from Zenodo and extract the csv tables (not the nested html / logo zips)."""
    import urllib.request
    d = os.path.join(root, "crosspump", "memechain")
    os.makedirs(d, exist_ok=True)
    for f in ("README.md", "data.zip"):
        dst = os.path.join(d, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[memechain] have {dst}"); continue
        try:
            print(f"[memechain] downloading {f} from Zenodo 18246856", flush=True)
            urllib.request.urlretrieve(f"https://zenodo.org/records/18246856/files/{f}?download=1", dst + ".part")
            os.replace(dst + ".part", dst)
        except Exception as e:
            print(f"[memechain] FAILED to download {f}: {e!r}\nFetch it by hand:\n{RECIPE}")
    z = os.path.join(d, "data.zip")
    if os.path.exists(z):
        try:
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".csv") and not os.path.exists(os.path.join(d, n)):
                        zf.extract(n, d)
        except Exception as e:
            print(f"[memechain] could not extract the csv tables from {z}: {e!r} (the loader can also read them from the zip)")


def read_table(sub: str, name: str) -> pd.DataFrame:
    p = os.path.join(sub, "data", name)
    if os.path.exists(p):
        return pd.read_csv(p, low_memory=False)
    with zipfile.ZipFile(os.path.join(sub, "data.zip")) as z:
        member = next((n for n in z.namelist() if n.endswith("/" + name) or n == name), None)
        if member is None:
            raise FileNotFoundError(f"{name} not found inside {sub}/data.zip.\n{RECIPE}")
        return pd.read_csv(io.BytesIO(z.read(member)), low_memory=False)


def _tok(df: pd.DataFrame) -> pd.Series:
    return df["address"].astype(str).str.strip().str.lower() + "|" + df["platform"].astype(str).str.strip().str.lower()


def load_registry(sub: str) -> pd.DataFrame:
    """Token registry joined with the social links and the ChainPatrol flag: tok, symbol, platform, address, creation_ns,
    n_social, social_src, blocked."""
    f = read_table(sub, "final_filtered_meme_coins.csv")
    f["tok"] = _tok(f)
    f = f.drop_duplicates("tok").copy()
    f["creation_ns"] = pd.to_numeric(f["creation_date"], errors="coerce") * 1e9
    s = read_table(sub, "social_info.csv")
    s["tok"] = _tok(s)
    s = s.drop_duplicates("tok")
    link_cols = [k for k in ("website", "twitter", "telegram", "discord") if k in s]
    has = s[link_cols].apply(lambda col: col.notna() & (col.astype(str).str.strip() != ""))
    s["n_social"] = has.sum(axis=1).astype(float)
    first = pd.Series("", index=s.index)
    for k in ("twitter", "telegram", "website"):
        if k in s:
            first = first.where(first != "", s[k].where(has[k], "").astype(str).str.strip().str.lower())
    s["social_src"] = first
    w = read_table(sub, "web_page_analysis.csv")
    w["tok"] = _tok(w)
    blocked = set(w.loc[w["chain_patrol_status"].astype(str).str.upper().str.strip() == "BLOCKED", "tok"])
    out = f.merge(s[["tok", "n_social", "social_src"]], on="tok", how="left")
    out["has_social_row"] = out["n_social"].notna()
    out["blocked"] = out["tok"].isin(blocked)
    out["symbol"] = out["symbol"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return out[["tok", "symbol", "platform", "address", "creation_ns", "n_social", "social_src", "has_social_row", "blocked"]]


def find_trade_files(sub: str) -> List[str]:
    return sorted(([os.path.join(sub, "trades.csv")] if os.path.exists(os.path.join(sub, "trades.csv")) else []) + glob.glob(os.path.join(sub, "trades", "*.csv")))


def _social_frame(reg: pd.DataFrame, key_of: Dict[str, str], time_of: Dict[str, int]) -> pd.DataFrame:
    r = reg[reg["tok"].isin(key_of) & (reg["n_social"].fillna(0) > 0)]
    ids, names = ou.hash_ids(r["social_src"].tolist(), salt="social:")
    df = pd.DataFrame({"key": [key_of[t] for t in r["tok"]], "time_ns": [int(time_of[t]) for t in r["tok"]], "reach": r["n_social"].to_numpy(dtype=np.float64),
                       "text_score": np.nan, "source_id": ids})
    df.attrs["entities"] = names
    return df


@common.register_dataset("xp_memechain")
class MemeChainBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "xp_memechain", "15min", 365 * 24 * 4, (1, 2, 3, 6, 12, 24)

    def build_panel(self) -> Panel:
        c = self.cfg
        root = c.get("root", "data_raw")
        sub = os.path.join(root, c.get("subdir", "crosspump/memechain"))
        if not all(os.path.exists(os.path.join(sub, "data", t)) for t in TABLES):
            common.require(root, [os.path.join(c.get("subdir", "crosspump/memechain"), "data.zip")], RECIPE)
        reg = load_registry(sub)
        files = find_trade_files(sub)
        mode = str(c.get("mode", "auto"))
        if mode == "trades" and not files:
            raise FileNotFoundError(f"mode 'trades' needs {sub}/trades.csv or {sub}/trades/*.csv (not part of the MemeChain release).\n{RECIPE}")
        return self._from_trades(reg, files) if (files and mode != "tabular") else self._tabular(reg)

    # ------------------------------------------------------------------ tabular (the public release)
    def _tabular(self, reg: pd.DataFrame) -> Panel:
        c = self.cfg
        quote = str(c.get("quote", "USD"))
        bm, pre, post, lead = int(c.get("bar_minutes", 15)), int(c.get("pre_bars", 4)), int(c.get("post_bars", 96)), int(c.get("lead", 2))
        horizons = list(c.get("horizons", self.horizons))
        r = reg[np.isfinite(reg["creation_ns"].to_numpy(dtype=np.float64))].reset_index(drop=True)
        n_all, n_blocked = len(r), int(r["blocked"].sum())
        pos, neg = np.nonzero(r["blocked"].to_numpy())[0], np.nonzero(~r["blocked"].to_numpy())[0]
        if c.get("max_pumps"):
            pos = pos[np.argsort(r["creation_ns"].to_numpy()[pos], kind="stable")][: int(c["max_pumps"])]
        if c.get("max_controls") and len(neg) > int(c["max_controls"]):
            neg = np.sort(np.random.default_rng(int(c.get("seed", 0))).choice(neg, int(c["max_controls"]), replace=False))     # seeded SUBSAMPLE of real tokens
        r = r.iloc[np.sort(np.concatenate([pos, neg]))].reset_index(drop=True)
        if len(r) == 0:
            raise FileNotFoundError(f"MemeChain registry is empty.\n{RECIPE}")
        episodes = [{"key": f"{sym or 'TOKEN'}{quote}@{ou.iso_minute(int(a))}#{addr[:8]}", "symbol": tok, "ticker": sym, "exchange": plat, "channel": None, "anchor_ns": int(a),
                     "peak_ns": None, "is_pump": bool(b), "control_of": None, "source": "memechain"}
                    for tok, sym, plat, addr, a, b in zip(r["tok"], r["symbol"], r["platform"], r["address"].astype(str), r["creation_ns"], r["blocked"])]
        ou.make_keys_unique(episodes)
        N = len(episodes)
        feat = np.full((N, len(common.FEAT_NAMES)), np.nan); feat[:, 3] = 0.0            # nothing market-side is knowable at deployment
        panel = ou.tabular_panel(episodes, pre, post, bm, lead, horizons, feat, np.full(N, np.nan), self.name, SOURCE_URL, LICENCE)
        key_of = {e["symbol"]: e["key"] for e in episodes}
        step = bm * common.MIN_NS
        soc = _social_frame(r, key_of, {e["symbol"]: (e["anchor_ns"] // step) * step for e in episodes})
        common.finish_streams(panel, [common.social_stream(panel, soc, lag_bars=0.0)], derive_orderflow_if_missing=False,
                              platform_lags={"social_bars": 0.0, "wallet_bars": "missing", "orderflow": "missing"}, entities={"social": soc.attrs.get("entities", {})},
                              how={"social": "ONE static event at launch, reach = number of linked accounts (crawl-time links)", "wallet": "MISSING", "orderflow": "MISSING"})
        panel.meta["coverage"]["note"] = ("tabular mode: wallet and orderflow MISSING (the release has no transaction table); social = static launch event "
                                          "(links collected at crawl time, not point-in-time)")
        panel.meta.update({"mode": "tabular", "quote": quote, "exchange": "on-chain (ethereum/bsc/solana/base)", "label_source": "chain_patrol_status == BLOCKED",
                           "n_registry": int(n_all), "n_blocked_in_registry": n_blocked, "lookback": int(c.get("lookback", 1)), "latency": int(c.get("latency", 1)),
                           "assumptions": ["TABULAR ONLY: the public release has no trade table -> degenerate panel, excluded from lead-time metrics",
                                           "A1 anchor = creation_date (contract deployment, UTC seconds)",
                                           "is_pump = ChainPatrol BLOCKED (9 tokens in the release); the paper's 24-hour-cessation statistic is NOT reproducible from the released files",
                                           "controls = all other registry tokens (seeded subsample when max_controls is set)"] + ASSUMPTIONS_COMMON})
        return panel

    # ------------------------------------------------------------------ time series (optional user-supplied trades)
    def _from_trades(self, reg: pd.DataFrame, files: List[str]) -> Panel:
        c = self.cfg
        quote = str(c.get("quote", "USD"))
        bm, pre, post, lead = int(c.get("bar_minutes", 15)), int(c.get("pre_bars", 4)), int(c.get("post_bars", 96)), int(c.get("lead", 2))
        horizons = list(c.get("horizons", self.horizons))
        hour = 3600 * 10 ** 9
        cease, quiet = float(c.get("cease_hours", 24)) * hour, float(c.get("quiet_hours", 168)) * hour
        rule = str(c.get("label_rule", "cease_or_blocked"))
        raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
        raw.columns = [str(x).strip() for x in raw.columns]
        if not {"address", "platform"} <= set(raw.columns):
            raise ValueError(f"{files[0]}: trade file needs address and platform columns; got {list(raw.columns)}")
        raw["tok"] = _tok(raw)
        info = reg.set_index("tok")
        by_tok: Dict[str, pd.DataFrame] = {}
        for tok, g in raw.groupby("tok", sort=True):
            t = ou.normalise_trades(g.drop(columns=["tok", "address", "platform"]))
            if len(t):
                by_tok[tok] = t
        if not by_tok:
            raise FileNotFoundError(f"{files}: no usable trade row.\n{RECIPE}")
        t_end = max(int(t["time_ns"].iloc[-1]) for t in by_tok.values())
        episodes, cease_ns = [], {}
        for tok, t in by_tok.items():
            first, last = int(t["time_ns"].iloc[0]), int(t["time_ns"].iloc[-1])
            ceased = (last - first < cease) and (t_end - last >= quiet)
            blocked = bool(info["blocked"].get(tok, False)) if tok in info.index else False
            is_pump = ceased or (blocked and rule == "cease_or_blocked")
            sym = str(info["symbol"].get(tok, "")) if tok in info.index else ""
            px = t["price"].to_numpy()
            episodes.append({"key": f"{sym or 'TOKEN'}{quote}@{ou.iso_minute(first)}#{tok[:8]}", "symbol": tok, "ticker": sym, "exchange": tok.split("|")[-1], "channel": None,
                             "anchor_ns": first, "peak_ns": int(t["time_ns"].iloc[int(np.argmax(px))]) if is_pump else None, "is_pump": bool(is_pump), "control_of": None,
                             "source": "memechain+user_trades", "ceased_within_24h": bool(ceased), "chainpatrol_blocked": blocked, "last_trade_ns": last,
                             "in_registry": bool(tok in info.index)})
            cease_ns[tok] = last
        episodes.sort(key=lambda e: e["anchor_ns"])
        ou.make_keys_unique(episodes)
        if c.get("max_pumps"):
            keep_p = {e["key"] for e in [e for e in episodes if e["is_pump"]][: int(c["max_pumps"])]}
            episodes = [e for e in episodes if not e["is_pump"] or e["key"] in keep_p]
        raw_bars = {tok: ou.trades_to_bars(t, bm) for tok, t in by_tok.items()}
        bars = {tok: ou.carry_forward_bars(b, bm) for tok, b in raw_bars.items()}
        panel, skipped = common.episode_panel(bars, episodes, pre, post, bm, lead, horizons, peak_window_bars=int(c.get("peak_window_bars", post)),
                                              end_bars_after_peak=int(c.get("end_bars_after_peak", 24)), min_bars_frac=float(c.get("min_bars_frac", 0.0)),
                                              dataset=self.name, source=SOURCE_URL, licence=LICENCE)
        kept = [e for e in episodes if e["key"] in set(panel.assets)]
        key_of = {e["symbol"]: e["key"] for e in kept}
        idx = common.asset_index(panel)
        cease_row = np.full(panel.N, -1, dtype=np.int64)
        for e in kept:
            cease_row[idx[e["key"]]] = int(np.clip(np.floor(common.bar_units(panel, idx[e["key"]], np.array([cease_ns[e["symbol"]]]))[0]), 0, panel.T - 1)) if e["ceased_within_24h"] else -1
        wal_parts, bur_parts, entities = [], [], {}
        for tok in key_of:
            w = ou.wallet_inflow_events(by_tok[tok], side=str(c.get("wallet_side", "taker")), trailing="cumulative")
            entities.update(w.attrs.get("entities", {}))
            w["symbol"] = tok; wal_parts.append(w)
            b = ou.trade_count_bursts(raw_bars[tok], bm, z_thresh=float(c.get("burst_z", 2.5)), warmup=int(c.get("burst_warmup", 8)), halflife=float(c.get("burst_halflife", 24)))
            b["symbol"] = tok; bur_parts.append(b)
        rng = np.random.default_rng(int(c.get("seed", 0)))
        cap = c.get("max_wallet_events_per_episode", 5000)
        wal = ou.assign_episode_keys(pd.concat(wal_parts, ignore_index=True), kept, pre, post, bm, max_per_episode=int(cap) if cap else None, rng=rng)
        bur = ou.assign_episode_keys(pd.concat(bur_parts, ignore_index=True), kept, pre, post, bm)
        soc = _social_frame(reg, key_of, {e["symbol"]: e["anchor_ns"] for e in kept})
        wallet_lag = float(c.get("wallet_lag_minutes", 1.0)) / bm
        common.finish_streams(panel, [common.social_stream(panel, soc, lag_bars=0.0), common.wallet_stream(panel, wal, lag_bars=wallet_lag), common.orderflow_stream_from_trades(panel, bur)],
                              derive_orderflow_if_missing=False, platform_lags={"social_bars": 0.0, "wallet_bars": wallet_lag, "orderflow": "bar close"},
                              entities={"social": soc.attrs.get("entities", {}), "wallet": "sha1 prefix per hashed buyer address (meta['wallet_entities'])"},
                              wallet_subsampled=wal.attrs.get("subsampled", {}),
                              how={"social": "ONE static event at launch, reach = number of linked accounts (crawl-time links)", "wallet": "taker-buy trades of the user-supplied trade file",
                                   "orderflow": "causal z-score bursts of per-bar trade counts"})
        panel.meta["wallet_entities"] = entities
        panel.meta["coverage"]["note"] = "trades mode: wallet/orderflow from the user-supplied trade file; social = static launch event (crawl-time links, not point-in-time)"
        panel.meta.update({"mode": "trades", "quote": quote, "exchange": "on-chain (ethereum/bsc/solana/base)", "label_source": rule, "cease_row": cease_row,
                           "lead_time_reference": "collapse (peak / last trade, meta['cease_row']), NOT the launch anchor", "trade_files": [os.path.basename(f) for f in files],
                           "skipped": skipped, "lookback": int(c.get("lookback", 4)), "latency": int(c.get("latency", 1)),
                           "assumptions": ["trades mode: the trade file is USER-SUPPLIED (not part of the MemeChain release)", "A1 launch = first trade of the token in the trade file",
                                           f"is_pump = last trade < first + {c.get('cease_hours', 24)} h and the file extends >= {c.get('quiet_hours', 168)} h past it"
                                           + (" OR ChainPatrol BLOCKED" if rule == "cease_or_blocked" else ""),
                                           "peak = time of the highest traded price; lead time is relative to the collapse, not the launch",
                                           "A5 carried bars (volume 0) between first and last trade; tokens with < 10 bars dropped; size_adv = amount / cumulative earlier volume"] + ASSUMPTIONS_COMMON})
        return panel
