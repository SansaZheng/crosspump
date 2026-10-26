"""D9 xp_rugpull1k: TM-RugPull / RugPull1K token projects — TABULAR ONLY (one row per project) -> DEGENERATE panel.

Source   Shoaei, Pishdar, Bag-Mohammadi, Karami, Lagaisse, "TMRugPull: A Temporally Sound Multimodal Dataset for Early
         RugPull Detection", arXiv:2602.21529 (2026). HuggingFace dataset fati-80-Sho/RugPull1K (files: README.md,
         dataset.xlsx, "SOURCE CODE.zip"); the same dataset.xlsx is in https://github.com/mohamadpishdar/RugPull-Defender
         under src/DataSet/. Companion method: LROO, arXiv:2603.11324.
Licence  CC BY-NC 4.0 according to BOTH the HuggingFace card (license:cc-by-nc-4.0) and the repository's Licence file
         (the project brief said CC BY 4.0: that is not what the sources state — non-commercial research use only).
Raw      <root>/crosspump/rugpull1k/dataset.xlsx, or (fallback) <root>/crosspump/rugpull_defender/src/DataSet/dataset.xlsx.
         One sheet, 1,000 project rows, 27 named columns (observed 2026-10-03):
           Project Title | MaxPrice (Quarter 1..4) | Blockchain | the number of Transactions | Token concentration ratio per
           holder | Total Variance | Variance of holders with more than 1% tokens | Token balance | Sign (ticker) | first
           deposits | Blockchain Type | Smart Contract (online) | smart Contract (offline) | website | x profile | class
           (scam 599 / normal 401) | project starting date | project end date (Excel serials) | Google results for project
           {title, website, x profile} x {first day, project duration/2}
         Read with a stdlib xlsx reader (onchain_utils.read_xlsx): no openpyxl needed.

*** THERE ARE NO PER-PROJECT TIME SERIES in the release (no daily bars, no transfers, no holder history). ***
The panel is therefore DEGENERATE: daily rows (bar_minutes 1440), row pre_bars = the project MIDPOINT (the dataset's own
observation cut-off: (start + end) / 2) carries one feature row, every other row is NaN and untradable, no row is
tradable. meta['tabular_only'] = True and meta['excluded_from'] = ['lead_time'] (a lead time cannot be measured), a
warning is emitted at build time. Use it for episode-level detection (AUC-PR at the anchor row) only.

Features at the anchor (FEAT_NAMES order): log_ret = log(MaxPrice Q2 / MaxPrice Q1) (both quarters lie in the first half),
log_volume NaN (no volume column), hl_range NaN, taker_imb 0 (unknown), log_trades = log1p(number of transactions).
mid at the anchor = MaxPrice (Quarter 2) when > 0: a price-level proxy only, NOT an executable price (tradable False).
All first-half tabular columns are in meta['tabular_features'] (N, K) with meta['tabular_feature_names']; the post-midpoint
columns (MaxPrice Quarter 3 / 4, project end date) are OUTCOMES and only stored in meta['outcomes'] (never features).

Streams (SUMMARY events at the anchor, not transfer-level data; meta['streams_are_summaries'] = True)
    social     OSINT: one event at the midpoint with reach = log1p(sum of the three "duration/2" Google result counts) and
               text_score = log1p(mid sum) - log1p(first-day sum) (growth of the search footprint); the first-day OSINT event
               sits at the project start and is kept only when the start falls inside the window; source = hashed x profile
               (else website).
    wallet     one event at the midpoint with size_raw = the "Token concentration ratio per holder" column (magnitude NaN: no
               ADV exists); source unknown (0). Disable with wallet_summary_event: false.
    orderflow  MISSING (no volume series) — coverage says so; nothing is derived.

ASSUMPTIONS (also in meta['assumptions'])
 A1 anchor = midpoint of [project starting date, project end date] at 00:00 UTC (Excel serial dates, day resolution).
 A2 every non-outcome column was measured within the first half of the project (the authors' "temporal hygiene" claim);
    this loader cannot verify it. Note that the midpoint itself depends on the end date, which for scams is the collapse.
 A3 "the number of Transactions", holder statistics and "first deposits" are first-half aggregates (the sheet does not say).
 A4 the meaning of "Token concentration ratio per holder" is not documented (values look like holder counts); it is passed
    through verbatim as size_raw of the wallet summary event and as a tabular feature.
 A5 is_pump = (class == 'scam'); peak_row = anchor row (no peak exists); labels 1 on [anchor - lead, anchor].
 A6 rows with end date <= start date (5 observed) or unparsable dates are dropped and counted in meta['skipped'].
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from . import common
from . import onchain_utils as ou

SOURCE_URL = "https://huggingface.co/datasets/fati-80-Sho/RugPull1K (arXiv:2602.21529); code https://github.com/mohamadpishdar/RugPull-Defender"
LICENCE = "CC BY-NC 4.0 (HuggingFace card and repository Licence file)"
REQUIRED = ["crosspump/rugpull1k/dataset.xlsx"]
FALLBACK = "crosspump/rugpull_defender/src/DataSet/dataset.xlsx"
RECIPE = """D9 xp_rugpull1k (TM-RugPull / RugPull1K, arXiv:2602.21529; CC BY-NC 4.0)
  python3 scripts/xp_prepare_data.py --dataset rugpull1k        # huggingface_hub.snapshot_download of dataset.xlsx + README.md
or by hand (either location is accepted):
  mkdir -p data_raw/crosspump/rugpull1k && curl -L -o data_raw/crosspump/rugpull1k/dataset.xlsx \\
      https://huggingface.co/datasets/fati-80-Sho/RugPull1K/resolve/main/dataset.xlsx
  git clone https://github.com/mohamadpishdar/RugPull-Defender data_raw/crosspump/rugpull_defender     # src/DataSet/dataset.xlsx"""

COLS = {"title": "Project Title", "p1": "MaxPrice (Quarter 1)", "p2": "MaxPrice (Quarter 2)", "p3": "MaxPrice (Quarter 3)", "p4": "MaxPrice (Quarter 4)",
        "chain": "Blockchain", "n_tx": "the number of Transactions", "holders": "Token concentration ratio per holder", "var_total": "Total Variance",
        "var_top": "Variance of holders with more than 1% tokens", "balance": "Token balance", "sign": "Sign", "first_deposits": "first deposits",
        "chain_type": "Blockchain Type", "website": "website", "x": "x profile", "cls": "class", "start": "project starting date", "end": "project end date",
        "g_title_0": "Google results for project title (first day)", "g_title_m": "Google results for project title (project duration/2)",
        "g_web_0": "Google results for project website (first day)", "g_web_m": "Google results for project website (duration/2)",
        "g_x_0": "Google results for project x profile (first days)", "g_x_m": "Google results for project x profile (duration/2)"}
TAB_FEATURES = ["p1", "p2", "n_tx", "holders", "var_total", "var_top", "balance", "first_deposits", "g_title_0", "g_title_m", "g_web_0", "g_web_m", "g_x_0", "g_x_m"]
ASSUMPTIONS = [
    "TABULAR ONLY: one row per project, no time series -> degenerate panel (feature row at the anchor), excluded from lead-time metrics",
    "A1 anchor = midpoint of [project starting date, project end date], 00:00 UTC",
    "A2 non-outcome columns are first-half measurements (authors' claim, not verifiable here); the midpoint depends on the end date",
    "A3 transaction count, holder statistics and first deposits are treated as first-half aggregates",
    "A4 'Token concentration ratio per holder' is undocumented; passed through verbatim",
    "A5 is_pump = class == 'scam'; peak_row = anchor row; labels on [anchor - lead, anchor]",
    "A6 rows with end <= start or unparsable dates dropped (counted)",
    "mid at the anchor = MaxPrice (Quarter 2) as a price-level proxy, never tradable",
    "streams are summary events at the anchor (social = Google result counts, wallet = holder column); orderflow missing",
    "licence is CC BY-NC 4.0 (not CC BY 4.0)",
]


def prepare(root: str, args: Optional[Dict[str, Any]] = None) -> None:
    """Fetch dataset.xlsx (+ README) from HuggingFace into <root>/crosspump/rugpull1k/. Called only by scripts/xp_prepare_data.py."""
    d = os.path.join(root, "crosspump", "rugpull1k")
    dst = os.path.join(d, "dataset.xlsx")
    if os.path.exists(dst) or os.path.exists(os.path.join(root, FALLBACK)):
        print(f"[rugpull1k] dataset.xlsx already present under {root}/crosspump"); return
    os.makedirs(d, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fati-80-Sho/RugPull1K", repo_type="dataset", local_dir=d, allow_patterns=["dataset.xlsx", "README.md"])
    except Exception as e:
        print(f"[rugpull1k] huggingface_hub download failed ({e!r}); trying the plain https file ...")
        try:
            import urllib.request
            urllib.request.urlretrieve("https://huggingface.co/datasets/fati-80-Sho/RugPull1K/resolve/main/dataset.xlsx", dst + ".part")
            os.replace(dst + ".part", dst)
        except Exception as e2:
            print(f"[rugpull1k] FAILED to download dataset.xlsx: {e2!r}\nFetch it by hand:\n{RECIPE}")


def find_xlsx(root: str, subdir: Optional[str] = None) -> Optional[str]:
    cands = ([os.path.join(root, subdir, "dataset.xlsx")] if subdir else []) + [os.path.join(root, REQUIRED[0]), os.path.join(root, FALLBACK)]
    return next((p for p in cands if os.path.exists(p)), None)


def load_projects(path: str) -> pd.DataFrame:
    """dataset.xlsx -> one row per project with numeric columns named by the COLS keys (+ row = sheet order)."""
    raw = ou.read_xlsx(path)
    norm = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in raw.columns}
    out = pd.DataFrame({"row": np.arange(len(raw))})
    for k, name in COLS.items():
        src = norm.get(name.lower())
        if src is None:
            if k in ("title", "cls", "start", "end"):
                raise ValueError(f"{path}: column {name!r} missing; found {list(raw.columns)}")
            out[k] = np.nan; continue
        out[k] = raw[src].to_numpy() if k in ("title", "chain", "sign", "chain_type", "website", "x", "cls") else ou.to_number(raw[src])
    out["start_ns"], out["end_ns"] = ou.excel_serial_to_ns(out["start"]), ou.excel_serial_to_ns(out["end"])
    return out


def _clean_url(v: Any) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in ("", "none", "nan", "-", "null") else s.lower()


@common.register_dataset("xp_rugpull1k")
class RugPull1KBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "xp_rugpull1k", "daily", 365, (1, 2, 3, 5, 10)

    def build_panel(self) -> Panel:
        c = self.cfg
        root = c.get("root", "data_raw")
        path = find_xlsx(root, c.get("subdir"))
        if path is None:
            common.require(root, REQUIRED, RECIPE)
        quote = str(c.get("quote", "USD"))
        bm, pre, post, lead = int(c.get("bar_minutes", 1440)), int(c.get("pre_bars", 30)), int(c.get("post_bars", 30)), int(c.get("lead", 2))
        horizons = list(c.get("horizons", self.horizons))
        df = load_projects(path)
        n_rows = len(df)
        cls = df["cls"].astype(str).str.strip().str.lower()
        ok_cls = cls.isin(("scam", "normal"))
        ok_date = np.isfinite(df["start_ns"].to_numpy()) & np.isfinite(df["end_ns"].to_numpy())
        ok_dur = ok_date & (df["end_ns"].to_numpy() > df["start_ns"].to_numpy())
        skipped = {"unknown_class": int((~ok_cls).sum()), "bad_dates": int((~ok_date).sum()), "end_not_after_start": int((ok_date & ~ok_dur).sum())}
        df = df[ok_cls.to_numpy() & ok_dur].reset_index(drop=True)
        cls = cls[ok_cls.to_numpy() & ok_dur].reset_index(drop=True)
        if len(df) == 0:
            raise FileNotFoundError(f"{path}: no usable project row.\n{RECIPE}")
        day = 86400 * 10 ** 9
        start = (df["start_ns"].to_numpy() // day * day).astype(np.int64)
        anchor = ((df["start_ns"].to_numpy() + df["end_ns"].to_numpy()) / 2 // day * day).astype(np.int64)
        if c.get("max_pumps"):                                       # chronological cap on the number of projects (smoke runs)
            keep = np.sort(np.argsort(anchor, kind="stable")[: int(c["max_pumps"])])
            df, cls, start, anchor = df.iloc[keep].reset_index(drop=True), cls.iloc[keep].reset_index(drop=True), start[keep], anchor[keep]

        episodes: List[Dict[str, Any]] = []
        for i in range(len(df)):
            sign = re.sub(r"[^A-Z0-9]", "", str(df["sign"].iloc[i] if df["sign"].iloc[i] is not None else "").upper())
            title = re.sub(r"[^A-Z0-9]", "", str(df["title"].iloc[i]).upper())[:16]
            sym = sign if 1 <= len(sign) <= 12 and "TOKEN" not in sign else (title or f"P{int(df['row'].iloc[i])}")
            episodes.append({"key": f"{sym}{quote}@{ou.iso_minute(anchor[i])}#{int(df['row'].iloc[i])}", "symbol": sym, "exchange": str(df["chain"].iloc[i]), "channel": None,
                             "anchor_ns": int(anchor[i]), "peak_ns": None, "is_pump": bool(cls.iloc[i] == "scam"), "control_of": None, "source": "rugpull1k",
                             "title": str(df["title"].iloc[i]), "start_ns": int(start[i]), "sheet_row": int(df["row"].iloc[i])})
        ou.make_keys_unique(episodes)
        p1, p2 = df["p1"].to_numpy(dtype=np.float64), df["p2"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where((p1 > 0) & (p2 > 0), np.log(p2 / p1), np.nan)
        N = len(df)
        feat = np.stack([lr, np.full(N, np.nan), np.full(N, np.nan), np.zeros(N), np.log1p(np.clip(df["n_tx"].to_numpy(dtype=np.float64), 0, None))], 1)
        mid = np.where(p2 > 0, p2, np.nan)
        panel = ou.tabular_panel(episodes, pre, post, bm, lead, horizons, feat, mid, self.name, SOURCE_URL, LICENCE, tradable_at_anchor=False)

        # ---- per-episode tables re-ordered like panel.assets (tabular_panel sorts by anchor)
        pos = {e["key"]: i for i, e in enumerate(episodes)}
        order = np.array([pos[k] for k in panel.assets])
        d = df.iloc[order].reset_index(drop=True)
        panel.meta["tabular_features"] = d[TAB_FEATURES].to_numpy(dtype=np.float64)
        panel.meta["tabular_feature_names"] = [COLS[k] for k in TAB_FEATURES]
        panel.meta["outcomes"] = {"note": "post-midpoint columns: never use as features", "max_price_q3": d["p3"].to_numpy(dtype=np.float64), "max_price_q4": d["p4"].to_numpy(dtype=np.float64),
                                  "end_ns": d["end_ns"].to_numpy(dtype=np.float64)}

        # ---- summary streams
        g0 = np.nansum(d[["g_title_0", "g_web_0", "g_x_0"]].to_numpy(dtype=np.float64), 1); has0 = d[["g_title_0", "g_web_0", "g_x_0"]].notna().any(axis=1).to_numpy()
        gm = np.nansum(d[["g_title_m", "g_web_m", "g_x_m"]].to_numpy(dtype=np.float64), 1); hasm = d[["g_title_m", "g_web_m", "g_x_m"]].notna().any(axis=1).to_numpy()
        src_txt = [(_clean_url(x) or _clean_url(w)) for x, w in zip(d["x"], d["website"])]
        sid, names = ou.hash_ids(src_txt, salt="osint:")
        keys = np.array(panel.assets, dtype=object)
        a_ns = np.asarray(panel.meta["anchor_ns"], dtype=np.int64)
        s_ns = start[order]
        soc = pd.concat([
            pd.DataFrame({"key": keys[hasm], "time_ns": a_ns[hasm], "reach": np.log1p(gm[hasm]), "text_score": np.where(has0[hasm], np.log1p(gm[hasm]) - np.log1p(g0[hasm]), np.nan), "source_id": sid[hasm]}),
            pd.DataFrame({"key": keys[has0], "time_ns": s_ns[has0], "reach": np.log1p(g0[has0]), "text_score": np.nan, "source_id": sid[has0]}),      # dropped when the start is outside the window
        ], ignore_index=True)
        parts = [common.social_stream(panel, soc, lag_bars=0.0)]
        if c.get("wallet_summary_event", True):
            h = d["holders"].to_numpy(dtype=np.float64); okh = np.isfinite(h)
            parts.append(common.wallet_stream(panel, pd.DataFrame({"key": keys[okh], "time_ns": a_ns[okh], "size_raw": h[okh], "wallet_id": np.zeros(int(okh.sum()), np.int64)}), lag_bars=0.0))
        common.finish_streams(panel, parts, derive_orderflow_if_missing=False, platform_lags={"social_bars": 0.0, "wallet_bars": 0.0, "orderflow": "missing"},
                              entities={"social": names}, how={"social": "Google result counts (OSINT) at project start / midpoint", "wallet": "holder-column summary at the midpoint", "orderflow": "MISSING"})
        panel.meta["coverage"]["note"] = "summary events only (one per project and platform at the anchor); orderflow MISSING (no volume series in the release)"
        panel.meta.update({"quote": quote, "exchange": "on-chain (ETH/BSC/Polygon/Arbitrum/Fantom/...)", "assumptions": list(ASSUMPTIONS), "streams_are_summaries": True,
                           "n_rows_in_sheet": int(n_rows), "skipped": skipped, "raw_path": path, "lookback": int(c.get("lookback", 1)), "latency": int(c.get("latency", 1)),
                           "class_counts": {"scam": int(panel.meta["is_pump"].sum()), "normal": int((~panel.meta["is_pump"]).sum())}})
        return panel
