"""D4 PumpSense -- Mahrous & Di Pietro, "PumpSense: Detection and Target Extraction of Cryptocurrency Pump-and-Dump on
Telegram", IEEE ICBC 2026, arXiv:2605.09431. Data: Zenodo doi:10.5281/zenodo.19616802 (MIT), file
AhmedMahrous00/icbc_2026_reproducibility-v1.1.zip (1.87 GB, mostly photos). Registry key: xp_pumpsense.

Raw layout (observed on c225, 2026-10-03; the zip is read in place, extraction is optional):
    <root>/crosspump/pumpsense/icbc_2026_reproducibility-v1.1.zip
        AhmedMahrous00-icbc_2026_reproducibility-2fdffd8/chat_data/<Group title>/result11.csv     (43 group folders)
        .../chat_data/<Group title>/photos/*.jpg (ignored), cryptocurrencies.json, *.py (the paper's scripts)
    result11.csv columns: date,text,Pump,Exchange,Coin,Notes[,photo]   (only 14 of the 43 files carry `photo`)
        date      naive local time of the Telegram-Desktop export, e.g. 2022-04-26T17:00:31 (NOT UTC, see assumptions)
        text      plain string, or the python repr of the export's entity list "['Coin: ', {'type': 'bold', 'text': 'XYZ'}]",
                  or empty (photo-only message: 34,518 rows)
        Pump      1.0 on the 2,246 manually labelled pump-start announcements, empty elsewhere (283,017 rows in total)
        Exchange  lower-case venue of a labelled row: binance 1036, yobit 583, cryptopia 194, kucoin 118, bittrex 105,
                  hotbit 81, pancakeswap 21, gate.io 10, coinexchange 9, ... (84 empty)
        Coin      'sym/quote' lower case ('vib/btc': btc 1922, usdt 229), a few bare symbols, 86 empty
        Notes     free text: cancelled / CANCELLED / postponed / futures / early / late / 'not preannounced' ...
    There is no sender column (broadcast channels) -> the only entity is the group; source id = stable hash of its title.

Episodes: labelled announcements of the same coin within merge_hours (default 1 h; many groups relay the same call:
1,036 Binance rows -> 563 market events) form ONE pump episode anchored at the earliest announcement, with every
relaying group in episode['channels']. Kept: exchange == cfg exchange ('Binance'; 'all' keeps every venue whose coin
has Binance klines and flags the venue mismatch), quote == cfg quote, Notes not matching drop_notes.
Social stream = EVERY message of the episode's groups inside the window (magnitude 1, reach NaN, source = group id);
text_score = 1.0 for the labelled announcement(s) of the episode's own coin (an ORACLE mark: the manual label), else
the keyword score of text_utils.text_score (NaN for photo-only messages). Controls (common.sample_controls) inherit the
groups of their pump: a same-coin control sees the same channels at another time; an other-coin control sees the very
same channel messages as its pump (social evidence is channel-level -- the coin is only revealed by the announcement),
where the announcement keeps its keyword score because it names another coin.
Panel.text = hash_embed of the message texts (nothing learned, cutoff n/a). Wallet modality: optional
<root>/crosspump/onchain/pumpsense_transfers.csv, otherwise missing. Order flow: derived causally from the bars.

ASSUMPTIONS (also in meta['assumptions'])
 A1 export time zone: the `date` column is local export time. Evidence gathered on c225: (i) groups announce
    "pump at 2 PM GMT" and the labelled message is stamped 17:00:31; (ii) the Binance 1-minute volume at
    (date - k hours) explodes (x100-x9000 over the previous hour's median) for exactly one k per group and is flat for
    the other k. Result: UTC+2 for most groups, UTC+3 for 'Alt the Way', 'Bitcoin Pump Group', 'Cryptic Trading',
    'Crypto Coin Pump Signals', 'Big Pump Signal'; no daylight-saving pattern. cfg export_utc_offset_hours (default 2)
    and group_utc_offset_hours (the five overrides). Groups without Binance pumps and without a GMT phrase
    (e.g. 'Crypto Profits' has text evidence +2; 'Cake Moon Pumps', 'Crown Signal Notifier', 'Fat Cat Crypto',
    'Wisdom Community ( Crypto Won )' have none) fall back to the default -- unverified, but they carry no Binance
    episode under the default exchange filter.
 A2 rows whose Notes match drop_notes (default 'cancel|postpon') are not pump episodes (the call did not happen);
    rows noted 'futures' are dropped with drop_futures (spot klines would be the wrong market).
 A3 peak = argmax high within peak_window_bars after the anchor (the dataset has no peak time).
 A4 text_score = 1.0 on labelled announcements is an oracle mark (cfg announcement_score: 'keyword' removes it).
 A5 the keyword text_score is a crude lexicon, not the paper's classifier (BGE / LightGBM / LLM scripts ship in the zip).
 A6 the release holds 43 group folders although the paper reports 39 groups; all 43 are read.
"""
from __future__ import annotations
import glob
import io
import os
import zipfile
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from . import common
from . import text_utils as tu

SOURCE_URL = "https://doi.org/10.5281/zenodo.19616802"
LICENCE = "MIT (Zenodo record 19616802, LICENSE inside the archive)"
ZIP_NAME = "icbc_2026_reproducibility-v1.1.zip"
ZIP_URL = f"https://zenodo.org/records/19616802/files/AhmedMahrous00/{ZIP_NAME}?download=1"
SUBDIR = os.path.join("crosspump", "pumpsense")
REQUIRED = [os.path.join(SUBDIR, ZIP_NAME)]
RECIPE = f"""D4 PumpSense (Mahrous & Di Pietro, IEEE ICBC 2026, arXiv:2605.09431; Zenodo 10.5281/zenodo.19616802, MIT)
  mkdir -p data_raw/crosspump/pumpsense
  curl -L -o data_raw/crosspump/pumpsense/{ZIP_NAME} "{ZIP_URL}"       # 1.87 GB
  # the loader reads chat_data/<group>/result11.csv straight from the zip; `python3 scripts/xp_prepare_data.py --dataset pumpsense`
  # downloads the zip when absent and extracts only the csv tables (a few MB) next to it
  python3 scripts/xp_prepare_data.py --dataset pumpsense --klines   # Binance 1-minute klines of the pumped pairs -> data_raw/crosspump/binance_klines/
  optional wallet modality: data_raw/crosspump/onchain/pumpsense_transfers.csv (key,time_ns,wallet_id,size_adv,size_raw)"""
GROUP_OFFSETS = {"Alt the Way": 3.0, "Bitcoin Pump Group": 3.0, "Cryptic Trading": 3.0, "Crypto Coin Pump Signals": 3.0, "Big Pump Signal": 3.0}
COLS = ["date", "text", "Pump", "Exchange", "Coin", "Notes"]
ASSUMPTIONS = [
    "A1 export time zone: `date` is local export time; UTC = date - export_utc_offset_hours (default 2), with group_utc_offset_hours overrides (+3 for "
    "'Alt the Way', 'Bitcoin Pump Group', 'Cryptic Trading', 'Crypto Coin Pump Signals', 'Big Pump Signal'); calibrated per group on announced GMT times and on "
    "the Binance 1-minute volume burst at the labelled minute; groups without evidence use the default (unverified)",
    "A2 labelled rows whose Notes match drop_notes ('cancel|postpon') are not pump episodes; 'futures' rows dropped (drop_futures)",
    "A3 peak = argmax high within peak_window_bars after the anchor (no peak time in the dataset)",
    "A4 text_score = 1.0 on the labelled announcement(s) of the episode's coin is an oracle mark (manual label); other messages get the keyword score",
    "A5 keyword text_score (text_utils.PUMP_LEXICON) is a lexicon, not the paper's classifier",
    "A6 43 group folders are released (paper: 39 groups); all are read",
    "A7 announcements of the same coin within merge_hours are one market event (anchor = earliest announcement)",
    "A8 controls inherit the groups of their pump; an other-coin control shares the channel stream of its pump (channel-level social evidence)",
    "A9 no sender column: the group is the only social entity (source id = stable hash of the group title)",
]


# ------------------------------------------------------------------------------------------------ raw tables
def _find_tables(base: str) -> List[Tuple[str, Any]]:
    """[(group title, opener)] from an extracted tree (**/chat_data/<group>/result*.csv) or from the zip itself."""
    hits = sorted(glob.glob(os.path.join(base, "**", "chat_data", "*", "result*.csv"), recursive=True))
    if hits:
        return [(os.path.basename(os.path.dirname(h)), h) for h in hits]
    zp = os.path.join(base, ZIP_NAME)
    if not os.path.exists(zp):
        zips = sorted(glob.glob(os.path.join(base, "*.zip")))
        zp = zips[0] if zips else zp
    if not os.path.exists(zp):
        return []
    out = []
    z = zipfile.ZipFile(zp)
    for n in sorted(z.namelist()):
        parts = n.split("/")
        if n.endswith(".csv") and len(parts) >= 3 and parts[-3] == "chat_data" and parts[-1].startswith("result"):
            out.append((parts[-2], (z, n)))
    return out


def read_messages(root: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Every message of every group: group, gid (stable id), time_ns (UTC), text (plain), pump (bool), exchange, coin, notes."""
    base = os.path.join(root, SUBDIR)
    tables = _find_tables(base)
    if not tables:
        common.require(root, REQUIRED, RECIPE)
        raise FileNotFoundError(f"No chat_data/<group>/result*.csv inside {base}.\nDownload recipe:\n{RECIPE}")
    default_off = float(cfg.get("export_utc_offset_hours", 2.0))
    offs = {**GROUP_OFFSETS, **(cfg.get("group_utc_offset_hours") or {})}
    frames = []
    for group, src in tables:
        if isinstance(src, tuple):
            df = pd.read_csv(io.BytesIO(src[0].read(src[1])), dtype={"text": object, "Exchange": object, "Coin": object, "Notes": object})
        else:
            df = pd.read_csv(src, dtype={"text": object, "Exchange": object, "Coin": object, "Notes": object})
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"pumpsense {group}: columns {missing} missing (found {list(df.columns)})")
        t = pd.to_datetime(df["date"], errors="coerce")
        df = df[t.notna()].reset_index(drop=True); t = t[t.notna()]
        off = float(offs.get(group, default_off))
        out = pd.DataFrame({"group": group, "gid": tu.stable_id(group, "pumpsense:"),
                            "time_ns": t.values.astype("datetime64[ns]").astype(np.int64) - int(off * tu.NS_PER_H),
                            "text": [tu.parse_telegram_text(x) for x in df["text"]],
                            "pump": pd.to_numeric(df["Pump"], errors="coerce").fillna(0).to_numpy() == 1,
                            "exchange": df["Exchange"].fillna("").astype(str).str.strip().str.lower(),
                            "coin": df["Coin"].fillna("").astype(str).str.strip().str.lower(),
                            "notes": df["Notes"].fillna("").astype(str).str.strip(), "utc_offset": off})
        frames.append(out)
    msgs = pd.concat(frames, ignore_index=True).sort_values(["group", "time_ns"], kind="stable").reset_index(drop=True)
    return msgs


def labelled_rows(msgs: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Labelled announcements -> rows kept as pump candidates (symbol, quote, exchange, time_ns, group, gid, msg index)."""
    lab = msgs[msgs["pump"]].copy()
    lab["msg"] = lab.index
    stats: Dict[str, Any] = {"messages": int(len(msgs)), "groups": int(msgs["group"].nunique()), "labelled_announcements": int(len(lab)),
                             "by_exchange": {k or "(empty)": int(v) for k, v in lab["exchange"].value_counts().items()}}
    sym = lab["coin"].str.split("/").str[0].str.upper().str.strip()
    quo = lab["coin"].apply(lambda s: s.split("/")[1].upper().strip() if "/" in s else "")
    lab["symbol"], lab["quote"] = sym, quo
    ok = lab["symbol"].str.fullmatch(r"[A-Z0-9]{1,12}")
    stats["dropped_no_coin"] = int((~ok).sum())
    lab = lab[ok]
    rx = str(cfg.get("drop_notes", "cancel|postpon") or "")
    if rx:
        m = lab["notes"].str.contains(rx, case=False, regex=True)
        stats["dropped_notes"] = int(m.sum()); lab = lab[~m]
    if cfg.get("drop_futures", True):
        m = lab["notes"].str.contains("futures", case=False)
        stats["dropped_futures"] = int(m.sum()); lab = lab[~m]
    all_times = {s: np.sort(g["time_ns"].to_numpy().astype(np.int64)) for s, g in lab.groupby("symbol")}
    ex = tu.norm_exchange(cfg.get("exchange", "Binance"))
    if ex != "all":
        m = lab["exchange"].map(tu.norm_exchange) == ex
        stats["dropped_other_exchange"] = int((~m).sum()); lab = lab[m]
    quote = str(cfg.get("quote", "BTC")).upper()
    m = (lab["quote"] == quote) | ((lab["quote"] == "") & (ex == "all"))
    stats["dropped_other_quote"] = int((~m).sum()); lab = lab[m]
    stats["announcements_kept"] = int(len(lab))
    return lab.reset_index(drop=True), {"stats": stats, "all_times": all_times}


def pump_episodes(lab: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    quote = str(cfg.get("quote", "BTC")).upper()
    ex_cfg = tu.norm_exchange(cfg.get("exchange", "Binance"))
    out = []
    if not len(lab):
        return out
    for grp in tu.merge_events(lab[["symbol", "time_ns"]], float(cfg.get("merge_hours", 1.0))):
        g = lab.iloc[grp]
        first = g.iloc[0]
        anns = [{"channel": str(r.group), "channel_id": int(r.gid), "announce_ns": int(r.time_ns), "msg": int(r.msg), "exchange": str(r.exchange), "notes": str(r.notes)}
                for r in g.itertuples(index=False)]
        venues = sorted({a["exchange"] for a in anns if a["exchange"]})
        out.append({"key": tu.pump_key(str(first["symbol"]), quote, int(first["time_ns"])), "symbol": str(first["symbol"]), "quote": quote,
                    "exchange": venues[0] if len(venues) == 1 else "|".join(venues), "channel": str(first["group"]),
                    "channels": sorted({a["channel"] for a in anns}), "announcements": anns, "anchor_ns": int(first["time_ns"]), "peak_ns": None,
                    "is_pump": True, "control_of": None, "source": "pumpsense_label",
                    "venue_mismatch": bool(ex_cfg == "all" and any(tu.norm_exchange(v) != "binance" for v in venues))})
    return out


# ------------------------------------------------------------------------------------------------ social stream
def _social_fn(msgs: pd.DataFrame, cfg: Dict[str, Any]):
    by_group: Dict[str, Tuple[int, int]] = {}
    g = msgs["group"].to_numpy()
    starts = np.r_[0, np.nonzero(g[1:] != g[:-1])[0] + 1, len(g)]
    for a, b in zip(starts[:-1], starts[1:]):
        by_group[str(g[a])] = (int(a), int(b))
    t_all = msgs["time_ns"].to_numpy().astype(np.int64)
    gid_all = msgs["gid"].to_numpy().astype(np.int64)
    text_all = msgs["text"].to_numpy()
    kw = np.full(len(msgs), np.nan); kw_done = np.zeros(len(msgs), dtype=bool)
    oracle = str(cfg.get("announcement_score", "label")).lower() == "label"

    def fn(episodes: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
        keys, idxs, override = [], [], []
        n_no_msgs = 0
        for e in episodes:
            lo, hi = ctx["window"][e["key"]]
            sel = []
            for grp in e.get("channels") or []:
                if grp not in by_group:
                    continue
                a, b = by_group[grp]
                i0 = a + int(np.searchsorted(t_all[a:b], lo, side="left")); i1 = a + int(np.searchsorted(t_all[a:b], hi, side="left"))
                if i1 > i0:
                    sel.append(np.arange(i0, i1))
            if not sel:
                n_no_msgs += 1; continue
            ix = np.concatenate(sel)
            own = {int(a["msg"]) for a in e.get("announcements", [])} if (e.get("is_pump") and oracle) else set()
            keys.append(np.full(len(ix), e["key"], dtype=object)); idxs.append(ix)
            override.append(np.isin(ix, list(own)) if own else np.zeros(len(ix), dtype=bool))
        if not idxs:
            return {"events": None, "docs": None, "info": {"coverage_note": "no group message inside any episode window"}}
        key = np.concatenate(keys); ix = np.concatenate(idxs); ov = np.concatenate(override)
        need = np.unique(ix[~kw_done[ix]])
        kw[need] = tu.text_scores(text_all[need]); kw_done[need] = True
        score = np.where(ov, 1.0, kw[ix])
        events = pd.DataFrame({"key": key, "time_ns": t_all[ix], "magnitude": 1.0, "reach": np.nan, "text_score": score, "source_id": gid_all[ix]})
        has_text = np.array([bool(str(x).strip()) for x in text_all[ix]])
        docs = pd.DataFrame({"key": key[has_text], "time_ns": t_all[ix][has_text], "text": text_all[ix][has_text]})
        info = {"kind": "message-level Telegram stream: every message of the episode's groups inside the window",
                "marks": "magnitude 1 per message, reach NaN (no view counts), text_score 1.0 = labelled announcement of the episode's coin "
                         f"({'oracle label' if oracle else 'disabled: keyword score'}), else keyword score (NaN for photo-only messages)",
                "episodes_without_group_messages": n_no_msgs, "n_oracle_marks": int(ov.sum()), "n_photo_only": int((~has_text).sum()),
                "coverage_note": "Telegram messages of the announcing groups (controls: the groups of their pump); sender ids are not in the release",
                "text_note": "Telegram message text of the episode's groups"}
        ents = {int(i): str(n) for n, i in msgs[["group", "gid"]].drop_duplicates().itertuples(index=False)}
        return {"events": events, "docs": docs, "info": info, "entities": ents}
    return fn


# ------------------------------------------------------------------------------------------------ prepare
def prepare(root: str, args: Any) -> None:
    """Called by scripts/xp_prepare_data.py only. Downloads the Zenodo zip when absent, extracts the csv tables
    (not the photos) and, with args['klines'], fetches the Binance 1-minute klines the pump windows need."""
    a = tu.args_dict(args)
    base = os.path.join(root, SUBDIR); os.makedirs(base, exist_ok=True)
    zp = os.path.join(base, ZIP_NAME)
    if not _find_tables(base):
        import urllib.request
        print(f"[pumpsense] downloading {ZIP_URL} -> {zp} (1.87 GB)")
        urllib.request.urlretrieve(ZIP_URL, zp)
    if os.path.exists(zp) and not glob.glob(os.path.join(base, "**", "chat_data", "*", "result*.csv"), recursive=True):
        with zipfile.ZipFile(zp) as z:
            members = [n for n in z.namelist() if n.endswith(".csv") or n.endswith("cryptocurrencies.json") or n.endswith("LICENSE") or n.endswith("README.md")]
            z.extractall(base, members=members)
        print(f"[pumpsense] extracted {len(members)} tables under {base}")
    if a.get("klines"):
        cfg = {"root": root, "max_pumps": a.get("max_pumps")}
        msgs = read_messages(root, cfg)
        lab, _ = labelled_rows(msgs, cfg)
        pumps = pump_episodes(lab, cfg)
        c = {**tu.DEFAULTS, **cfg}
        months = tu.episode_months(pumps, int(c["pre_bars"]), int(c["post_bars"]), int(c["bar_minutes"]), pad_days=float(c["control_span_days"]) + 1.0)
        try:
            from . import binance_fetch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"crosspump/data/loaders/binance_fetch.py is not importable ({type(e).__name__}: {e}); fetch the klines manually:\n{RECIPE}")
        st = binance_fetch.fetch_klines({s: sorted(m) for s, m in months.items()}, os.path.join(root, tu.KLINE_SUBDIR), quote="BTC")
        print(f"[pumpsense] klines: {pd.Series(list(st.values())).value_counts().to_dict()} for {len(months)} symbols / {len(pumps)} pump events")


# ------------------------------------------------------------------------------------------------ builder
@common.register_dataset("xp_pumpsense")
class PumpSenseBuilder(BaseDatasetBuilder):
    name = "xp_pumpsense"
    freq = "5min"
    bars_per_year = 105120
    horizons = (1, 2, 3, 6, 12, 24)

    def build_panel(self) -> Panel:
        cfg = dict(self.cfg)
        root = str(cfg.get("root", "data_raw"))
        msgs = read_messages(root, cfg)
        lab, extra = labelled_rows(msgs, cfg)
        pumps = pump_episodes(lab, cfg)
        stats = extra["stats"]; stats["pump_events_after_merge"] = len(pumps)
        offs = msgs[["group", "utc_offset"]].drop_duplicates()
        return tu.build_panel(cfg, pumps, name=self.name, dataset="pumpsense", source=SOURCE_URL, licence=LICENCE, recipe=RECIPE,
                              social_fn=_social_fn(msgs, cfg), pump_times=extra["all_times"], assumptions=ASSUMPTIONS, table_stats=stats,
                              extra_meta={"paper": "Mahrous & Di Pietro, PumpSense, IEEE ICBC 2026, arXiv:2605.09431",
                                          "social_platforms": "Telegram (platform 0)",
                                          "group_utc_offset_hours": {str(g): float(o) for g, o in offs.itertuples(index=False)},
                                          "merge_hours": float(cfg.get("merge_hours", 1.0))})
