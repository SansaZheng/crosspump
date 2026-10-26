"""Group-B loader tests (D4 pumpsense, D5 tgdataset_pd, D7 mirtaheri + text_utils), plain-script mode:

    python3 tests/crosspump_parts/test_loaders_b.py [-k substring]

Everything runs on the SYNTHETIC FIXTURES under tests/crosspump_parts/fixtures_b/root (written by make_fixtures.py: fake
coins, fake groups / channels / messages, seeded random-walk klines with planted spikes). No real data is read and
nothing is downloaded.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from warn2trade.utils import DATASETS                                   # noqa: E402
from crosspump.data.streams import get_streams, EventStreams, add_prior_involvement   # noqa: E402
from crosspump.data.loaders import common                               # noqa: E402
from crosspump.data.loaders import text_utils as tu                     # noqa: E402
from crosspump.data.loaders import d4_pumpsense, d5_tgdataset_pd, d7_mirtaheri   # noqa: E402

FIX = os.path.join(ROOT, "tests", "crosspump_parts", "fixtures_b")
FROOT = os.path.join(FIX, "root")
CFG_DIR = os.path.join(ROOT, "configs", "crosspump", "datasets")
MODS = {"pumpsense": d4_pumpsense, "tgdataset_pd": d5_tgdataset_pd, "mirtaheri": d7_mirtaheri}
# small layout for the fixtures (the yaml defaults need 72 h of klines before every anchor)
SMALL = {"root": FROOT, "pre_bars": 48, "post_bars": 24, "control_span_days": 1.0, "gap_hours": 3, "end_bars_after_peak": 6, "horizons": [1, 2, 3, 6],
         "text_cache": False}
EXTRA = {"pumpsense": {"group_utc_offset_hours": {"Fixture Group Three": 3.0}}, "mirtaheri": {"max_pumps": None}, "tgdataset_pd": {}}
YAML_KEYS = ["name", "root", "subdir", "quote", "bar_minutes", "pre_bars", "post_bars", "lead", "n_controls_per_pump", "gap_hours", "exchange", "max_pumps", "text_slots",
             "lookback", "horizons", "latency", "split", "pit", "assumptions"]
NS_H = 3600 * 10 ** 9
_CACHE = {}


def ns(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value)


def ensure_fixtures():
    if not os.path.exists(os.path.join(FROOT, "SYNTHETIC_FIXTURES.txt")):
        spec = importlib.util.spec_from_file_location("make_fixtures_b", os.path.join(FIX, "make_fixtures.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.build(FROOT)
    assert "SYNTHETIC FIXTURES" in open(os.path.join(FIX, "README.md")).read()


def cfg_for(short: str, **over):
    with open(os.path.join(CFG_DIR, f"{short}.yaml")) as f:
        c = yaml.safe_load(f)
    c.update(SMALL); c.update(EXTRA[short]); c.update(over)
    return c


def panel_for(short: str, **over):
    k = (short, json.dumps(over, sort_keys=True, default=str))
    if k not in _CACHE:
        c = cfg_for(short, **over)
        _CACHE[k] = (DATASETS.get(c["name"])(c).build_panel(), c)
    return _CACHE[k]


def check_invariants(p, c):
    """The contract of crosspump/CONVENTIONS.md §1-§3 on a built panel."""
    m = p.meta
    p.validate()
    T, N = p.mid.shape
    pre, post, lead = int(c["pre_bars"]), int(c["post_bars"]), int(c["lead"])
    assert T == pre + post + 1 and N == len(p.assets) == len(set(p.assets))
    assert m["split_axis"] == "assets" and m["assets_time_aligned"] is False
    is_pump = np.asarray(m["is_pump"]); peak = np.asarray(m["peak_row"])
    assert is_pump.dtype == bool and is_pump.shape == (N,) and is_pump.any() and (~is_pump).any()
    assert (np.diff(m["anchor_ns"]) >= 0).all(), "episodes must be sorted by anchor time"
    rows = np.arange(T)
    for i in range(N):
        e = m["episodes"][i]
        assert e["key"] == p.assets[i]
        if is_pump[i]:
            pk = int(peak[i])
            assert pre <= pk <= pre + int(c["peak_window_bars"])
            lab = np.zeros(T, dtype=int); lab[pre - lead: pk + 1] = 1
            assert (p.labels[:, i] == lab).all(), f"labels of {p.assets[i]} not on [anchor - lead, peak]"
            end = min(T - 1, pk + int(c["end_bars_after_peak"]))
            ev = np.arange(pre - lead, end + 1)
            assert np.array_equal(p.bars_to_peak[ev, i], pk - ev), "bars_to_peak must be exact"
            assert np.isnan(p.bars_to_peak[: pre - lead, i]).all()
            assert (p.avail_label[ev, i] == end).all()
            assert not p.assets[i].startswith("CTRL_") and "@" in p.assets[i] and p.assets[i].endswith("Z")
            sym_q, ts = p.assets[i].split("@")
            assert sym_q == e["symbol"] + e["quote"] and ts == pd.Timestamp(int(e["anchor_ns"]), tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
        else:
            assert (p.labels[:, i] == 0).all() and np.isnan(p.bars_to_peak[:, i]).all() and peak[i] == -1
    assert (p.avail_feat == rows[:, None]).all() and (p.avail_label >= rows[:, None]).all()
    assert p.universe.shape == (T, N) and p.tradable.dtype == bool
    assert m["feat_names"] == common.FEAT_NAMES and p.price_feat.shape == (T, N, 5)
    # streams
    es = get_streams(p)
    assert es is not None and len(es) > 0
    es.validate(T, N)
    assert (es.avail >= es.time - 1e-9).all() and (es.avail < T).all()
    cov = m["coverage"]
    assert isinstance(cov, dict) and {"episodes_with_platform", "n_events", "pump_episodes_with_platform", "social", "wallet", "orderflow", "table"} <= set(cov)
    assert cov["episodes_with_platform"]["orderflow"] > 0
    assert isinstance(m["assumptions"], list) and len(m["assumptions"]) >= 5
    # PIT of the rasters: an event is counted at ceil(avail) and is usable at row t iff avail <= t
    ras = es.rasterise(T, N)
    for i in range(N):
        for t in (pre - 5, pre, pre + 3, T - 1):
            assert ras[: t + 1, i].sum() == len(es.before(i, t)), "rasterise / before disagree"
    if p.text is not None:
        assert p.text.shape[:2] == (T, N) and p.text_mask.shape == p.text.shape[:3] and p.avail_text.shape == p.text.shape[:3]
        tt = np.broadcast_to(rows[:, None, None], p.text_mask.shape)
        assert (p.avail_text[p.text_mask] >= tt[p.text_mask]).all(), "avail_text must be >= row where text exists"
        assert p.text_mask.any() and np.isfinite(p.text).all()
        # every text row must be a row at which a social event of that episode is already usable
        soc = es.subset(es.platform == 0)
        first = np.full(N, T, dtype=float)
        np.minimum.at(first, soc.asset, np.ceil(soc.avail))
        t_idx, i_idx, _ = np.nonzero(p.text_mask)
        assert (t_idx >= first[i_idx]).all(), "text placed before any social event of the episode is usable"
    return es


# ====================================================================================================== text_utils
def test_text_utils_parsing():
    assert tu.parse_telegram_text("['Coin: ', {'type': 'bold', 'text': '$XYZ'}, ' go']") == "Coin: $XYZ go"
    assert tu.parse_telegram_text('["a", {"type": "link", "text": "b"}]') == "ab"
    assert tu.parse_telegram_text(float("nan")) == "" and tu.parse_telegram_text(None) == "" and tu.parse_telegram_text("plain [x]") == "plain [x]"
    assert tu.parse_telegram_text("[not a list") == "[not a list"
    s_hi = tu.text_score("The coin is $XYZ buy now on Binance, target 300%, pump starts"); s_lo = tu.text_score("good morning everyone")
    assert 0.0 <= s_lo < 0.3 < s_hi <= 1.0 and np.isnan(tu.text_score("")) and np.isnan(tu.text_score(float("nan")))
    assert tu.text_score("x") == tu.text_score("x")                                     # deterministic
    assert tu.extract_tickers("buy $AB and XYZ, not xyz2 or have fun", ["AB", "XYZ", "FUN"]) == ["AB", "XYZ"]
    assert tu.extract_tickers("AB is up", ["AB"]) == []                                 # 2-letter tokens need a cashtag
    a, b = tu.stable_id("group one"), tu.stable_id("group two")
    assert a == tu.stable_id("group one") and a != b and a > 1 and b > 1
    assert tu.months_covering(ns("2021-01-30"), ns("2021-03-02")) == ["2021-01", "2021-02", "2021-03"]
    assert tu.pump_key("AAA", "BTC", ns("2021-03-03 17:00:20")) == "AAABTC@2021-03-03T17:00Z"
    rows = pd.DataFrame({"symbol": ["A", "A", "B", "A"], "time_ns": [0, NS_H // 2, NS_H, 3 * NS_H]})
    assert tu.merge_events(rows, 1.0) == [[0, 1], [2], [3]]
    assert tu.subsample_evenly(list(range(10)), 3) == [0, 4, 9] or len(tu.subsample_evenly(list(range(10)), 3)) == 3


def test_embed_cache():
    d = tempfile.mkdtemp()
    try:
        path = tu.embed_cache_path(d, "unit", 16)
        texts = ["buy now", "pump $XYZ", "hello"]
        e1 = tu.cached_hash_embed(texts, 16, path)
        assert os.path.exists(path) and e1.shape == (3, 16) and np.allclose(e1, common.hash_embed(texts, 16))
        with np.load(path) as z:
            assert "n/a" in json.loads(str(z["meta"]))["cutoff"]
        np.savez_compressed(path, emb=np.zeros((3, 16), np.float32), fingerprint=np.array("stale"), meta=np.array("{}"))
        assert np.allclose(tu.cached_hash_embed(texts, 16, path), e1), "a stale cache (other fingerprint) must be ignored"
        assert tu.cached_hash_embed([], 16, None).shape == (0, 16)
    finally:
        shutil.rmtree(d)


def test_stream_helpers_match_common():
    p, c = panel_for("pumpsense")
    es = get_streams(p)
    rng = np.random.default_rng(0)
    n = 300
    keys = rng.choice(p.assets + ["UNKNOWN@x"], n)
    bc = p.meta["bar_close_ns"]
    t = rng.integers(int(bc[0].min()) - 3 * NS_H, int(bc[-1].max()) + NS_H, n)
    df = pd.DataFrame({"key": keys, "time_ns": t, "reach": rng.uniform(0, 5, n), "text_score": rng.uniform(0, 1, n), "source_id": rng.integers(0, 6, n)})
    for lag in (0.0, 1.5):
        a = common.social_stream(p, df, lag_bars=lag); b = tu.social_part(p, df, lag)
        assert len(a) == len(b) > 0
        for f in ("asset", "time", "platform", "avail", "source"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
        assert np.array_equal(a.mark, b.mark, equal_nan=True)
    # availability column: later availability moves avail, events not available inside the window are dropped
    df2 = df.copy(); df2["avail_ns"] = df2["time_ns"] + 2 * NS_H
    b2 = tu.social_part(p, df2, 0.0)
    assert len(b2) < len(b) and (b2.avail >= b2.time + 2 * 12 - 1e-6).all() and (b2.avail < p.T).all()
    # prior involvement: vectorised == reference
    rank = np.argsort(np.argsort(p.meta["anchor_ns"])); isp = np.asarray(p.meta["is_pump"])
    big = EventStreams.concat([a, es])
    ref = add_prior_involvement(big, rank, isp); fast = tu.fast_prior_involvement(big, rank, isp)
    assert np.array_equal(ref.mark, fast.mark, equal_nan=True)
    assert np.nanmax(fast.mark[:, 3]) >= 1
    # finish: the fast path gives the same streams as common.finish_streams
    import copy
    p1 = copy.copy(p); p1.meta = dict(p.meta); p2 = copy.copy(p); p2.meta = dict(p.meta)
    common.finish_streams(p1, [a]); tu.finish(p2, [a], fast_threshold=0)
    s1, s2 = get_streams(p1), get_streams(p2)
    for f in ("asset", "time", "platform", "avail", "source"):
        assert np.array_equal(getattr(s1, f), getattr(s2, f)), f
    assert np.array_equal(s1.mark, s2.mark, equal_nan=True)


def test_place_text_rule():
    p, c = panel_for("pumpsense")
    key = p.assets[0]; bc = p.meta["bar_close_ns"][:, 0]; step = int(bc[1] - bc[0])
    open10 = int(bc[10] - step)
    t = np.array([open10 + step // 3, open10, open10 + step - 1, int(bc[-1]) - 1, open10 + step // 2])
    emb = np.eye(5, 8, dtype=np.float32)
    pl = tu.place_text(p, [key] * 4 + ["UNKNOWN"], t, emb, slots=2, lag_bars=0.0)
    m = pl["text_mask"][:, 0]
    assert m[11].sum() == 2 and m[10].sum() == 1, "a message inside bar 10 is first usable at row 11; one exactly at the open of bar 10 at row 10"
    assert pl["stats"]["placed"] == 3 and pl["stats"]["dropped_out_of_window"] == 2
    assert (pl["avail_text"][:, 0][m] == np.nonzero(m)[0]).all()
    assert np.allclose(pl["text"][11, 0, 0], emb[2]), "the latest message takes slot 0"
    pl2 = tu.place_text(p, [key], t[:1], emb[:1], slots=2, lag_bars=2.0)
    assert pl2["text_mask"][13, 0].sum() == 1 and pl2["text_age"][13, 0, 0] == 3.0
    pl3 = tu.place_text(p, [key], t[:1], emb[:1], slots=2, lag_bars=0.0, avail_ns=np.array([open10 + 5 * step]))
    assert pl3["text_mask"][15, 0].sum() == 1


# ====================================================================================================== registry / config
def test_registry_and_configs():
    for short, mod in MODS.items():
        with open(os.path.join(CFG_DIR, f"{short}.yaml")) as f:
            y = yaml.safe_load(f)
        missing = [k for k in YAML_KEYS if k not in y]
        assert not missing, f"{short}.yaml misses {missing}"
        assert y["name"] == f"xp_{short}" and y["split"] == {"train_frac": 0.6, "val_frac": 0.15, "purge_hours": 96}
        assert (y["bar_minutes"], y["pre_bars"], y["post_bars"], y["lead"], y["lookback"], y["latency"]) == (5, 864, 96, 2, 48, 1)
        assert y["horizons"] == [1, 2, 3, 6, 12, 24] and y["n_controls_per_pump"] == 2 and y["gap_hours"] == 24
        cls = DATASETS.get(y["name"])
        assert cls.name == y["name"] and cls.freq == "5min" and cls.bars_per_year == 105120 and tuple(cls.horizons) == (1, 2, 3, 6, 12, 24)
        for attr in ("SOURCE_URL", "LICENCE", "RECIPE", "REQUIRED"):
            assert getattr(mod, attr), f"{short}: {attr}"
        assert callable(mod.prepare) and isinstance(mod.REQUIRED, list)
        assert all(r.startswith(os.path.join("crosspump", "")) for r in mod.REQUIRED)


def test_missing_raw_files_raise_with_recipe():
    d = tempfile.mkdtemp()
    try:
        for short, mod in MODS.items():
            c = cfg_for(short, root=d)
            try:
                DATASETS.get(c["name"])(c).build_panel()
            except FileNotFoundError as e:
                assert mod.RECIPE.strip().splitlines()[0] in str(e), f"{short}: the error must contain the download recipe"
            else:
                raise AssertionError(f"{short}: missing raw files must raise FileNotFoundError")
        # raw tables present but no klines at all -> FileNotFoundError with the recipe, never synthetic prices
        for sub in ("pumpsense", "mirtaheri", "moura"):
            shutil.copytree(os.path.join(FROOT, "crosspump", sub), os.path.join(d, "crosspump", sub))
        for short, mod in MODS.items():
            c = cfg_for(short, root=d)
            try:
                DATASETS.get(c["name"])(c).build_panel()
            except FileNotFoundError as e:
                assert mod.RECIPE.strip().splitlines()[0] in str(e) and "klines" in str(e), short
            else:
                raise AssertionError(f"{short}: no klines must raise FileNotFoundError")
    finally:
        shutil.rmtree(d)


# ====================================================================================================== D4 PumpSense
def test_pumpsense_panel():
    p, c = panel_for("pumpsense")
    es = check_invariants(p, c)
    m = p.meta
    assert m["dataset"] == "xp_pumpsense" and m["n_pumps"] == 3 and m["n_controls"] >= 3
    tab = m["table"]
    assert tab["labelled_announcements"] == 8 and tab["dropped_notes"] == 1 and tab["dropped_futures"] == 1 and tab["dropped_other_exchange"] == 1
    assert tab["announcements_kept"] == 5 and tab["pump_events_after_merge"] == 3
    # local export time -> UTC: group one/two UTC+2, group three UTC+3 (config override)
    pumps = [e for e in m["episodes"] if e["is_pump"]]
    assert [e["key"] for e in pumps] == ["AAABTC@2021-03-03T17:00Z", "BBBBTC@2021-03-04T16:00Z", "AAABTC@2021-03-06T18:00Z"]
    assert pumps[0]["anchor_ns"] == ns("2021-03-03 17:00:20") and pumps[0]["channels"] == ["Fixture Group One", "Fixture Group Two"]
    assert len(pumps[0]["announcements"]) == 2 and pumps[2]["channels"] == ["Fixture Group One", "Fixture Group Three"]
    assert pumps[2]["announcements"][1]["announce_ns"] == ns("2021-03-06 18:00:45")
    assert m["group_utc_offset_hours"]["Fixture Group Three"] == 3.0 and m["group_utc_offset_hours"]["Fixture Group One"] == 2.0
    # the planted spike starts at the announced minute: the derived peak is right after the anchor bar
    pre = int(c["pre_bars"])
    assert all(pre <= int(m["peak_row"][p.assets.index(e["key"])]) <= pre + 3 for e in pumps)
    # social stream of the first pump: every message of both groups in the window, announcement mark = 1.0 (oracle)
    i = p.assets.index("AAABTC@2021-03-03T17:00Z")
    s = es.for_asset(i); soc = s.subset(s.platform == 0)
    assert len(soc) == 8, len(soc)                               # 5 of group one + 3 of group two inside [anchor - 4 h, anchor + 2 h]
    ann = soc.subset(soc.mark[:, 2] == 1.0)
    assert len(ann) == 2 and (np.floor(ann.time) == pre).all() and (ann.time > pre).all(), "announcements sit inside the anchor bar"
    assert len(es.before(i, pre).subset(es.before(i, pre).mark[:, 2] == 1.0)) == 0, "the announcement is NOT usable at the anchor row (avail > row)"
    assert len(es.before(i, pre + 1).subset(es.before(i, pre + 1).mark[:, 2] == 1.0)) == 2
    assert np.isnan(soc.mark[:, 1]).all() and (soc.mark[:, 0] == 1.0).all()
    assert set(soc.source.tolist()) == {tu.stable_id("Fixture Group One", "pumpsense:"), tu.stable_id("Fixture Group Two", "pumpsense:")}
    assert set(es.meta["entities"].values()) >= {"Fixture Group One", "Fixture Group Two", "Fixture Group Three"}
    # photo-only labelled announcement: event exists, oracle mark 1.0, no text document
    j = p.assets.index("BBBBTC@2021-03-04T16:00Z")
    sj = es.for_asset(j); sj = sj.subset(sj.platform == 0)
    assert len(sj) == 1 and sj.mark[0, 2] == 1.0 and p.text_mask[:, j].sum() == 0
    # other-coin control of the first pump shares the channel stream but NOT the oracle mark
    ctl = [e for e in m["episodes"] if (not e["is_pump"]) and e["control_of"] == "AAABTC@2021-03-03T17:00Z" and e["control_type"] == "other_coin_same_time"]
    assert len(ctl) == 1 and ctl[0]["key"].startswith("CTRL_") and ctl[0]["channels"] == pumps[0]["channels"]
    k = p.assets.index(ctl[0]["key"]); sk = es.for_asset(k); sk = sk.subset(sk.platform == 0)
    assert len(sk) == len(soc) and not (sk.mark[:, 2] == 1.0).any() and np.nanmax(sk.mark[:, 2]) < 1.0
    # prior involvement: the third pump's groups already ran pump 1 (group one) and pumps 1+2
    l = p.assets.index("AAABTC@2021-03-06T18:00Z"); sl = es.for_asset(l); sl = sl.subset(sl.platform == 0)
    g1 = tu.stable_id("Fixture Group One", "pumpsense:")
    assert (sl.mark[sl.source == g1, 3] == 2.0).all() and (soc.mark[:, 3] == 0.0).all()
    # wallet: present only for the episode of the optional table, missing elsewhere
    cov = m["coverage"]
    assert cov["episodes_with_platform"]["wallet"] == 1 and "present" in cov["wallet"] and es.meta["wallet"]["status"] == "present"
    w = s.subset(s.platform == 1)
    assert len(w) == 3 and np.allclose(w.avail, w.time + 1.0) and set(w.source.tolist()) == {11, 12}
    # text
    assert p.text is not None and p.text.shape[2:] == (4, 32) and m["text"]["cutoff"] == "n/a"
    assert m["text_placement"]["placed"] > 0
    assert p.text_mask[pre + 1, i].sum() == 2 and p.text_mask[pre, i].sum() == 0, "announcement text appears at the first row where it is usable"
    assert any("A1" in a for a in m["assumptions"])


def test_pumpsense_options_and_zip():
    p0, _ = panel_for("pumpsense")
    # keyword mode removes the oracle mark
    p1, _ = panel_for("pumpsense", announcement_score="keyword")
    es1 = get_streams(p1); soc = es1.subset(es1.platform == 0)
    assert not (soc.mark[:, 2] == 1.0).any()
    # exchange 'all' keeps the yobit row only if its coin had klines (it has none) -> same pumps, counted
    p2, _ = panel_for("pumpsense", exchange="all")
    assert p2.meta["n_pumps"] == 3 and p2.meta["table"]["pumps_without_any_binance_klines"] == 1
    # wrong offset shifts the anchors by an hour (the time-zone assumption matters)
    p3, _ = panel_for("pumpsense", export_utc_offset_hours=3.0)
    assert "AAABTC@2021-03-03T16:00Z" in p3.assets
    # same tables read straight from a zip with the Zenodo layout
    d = tempfile.mkdtemp()
    try:
        base = os.path.join(d, "crosspump", "pumpsense"); os.makedirs(base)
        src = os.path.join(FROOT, "crosspump", "pumpsense")
        with zipfile.ZipFile(os.path.join(base, d4_pumpsense.ZIP_NAME), "w") as z:
            for dp, _, fs in os.walk(src):
                for f in fs:
                    full = os.path.join(dp, f)
                    z.write(full, os.path.join("AhmedMahrous00-icbc_2026_reproducibility-2fdffd8", os.path.relpath(full, src)))
            z.writestr("AhmedMahrous00-icbc_2026_reproducibility-2fdffd8/chat_data/Fixture Group One/photos/photo_1.jpg", b"not a table")
        os.symlink(os.path.join(FROOT, "crosspump", "binance_klines"), os.path.join(d, "crosspump", "binance_klines"))
        c = cfg_for("pumpsense", root=d)
        pz = DATASETS.get(c["name"])(c).build_panel()
        assert pz.assets == p0.assets and np.array_equal(pz.labels, p0.labels) and np.allclose(pz.mid, p0.mid, equal_nan=True)
        ez, e0 = get_streams(pz), get_streams(p0)
        assert (ez.platform == 0).sum() == (e0.platform == 0).sum() and pz.meta["coverage"]["episodes_with_platform"]["wallet"] == 0
        assert "MISSING" in pz.meta["coverage"]["wallet"]
        # prepare(): extracts only the tables (no photos), downloads nothing because the zip exists
        d4_pumpsense.prepare(d, {})
        ext = os.path.join(base, "AhmedMahrous00-icbc_2026_reproducibility-2fdffd8", "chat_data")
        assert os.path.exists(os.path.join(ext, "Fixture Group One", "result11.csv")) and not os.path.exists(os.path.join(ext, "Fixture Group One", "photos"))
        pe = DATASETS.get(c["name"])(c).build_panel()
        assert pe.assets == p0.assets
    finally:
        shutil.rmtree(d)


# ====================================================================================================== D7 Mirtaheri
def test_mirtaheri_panel():
    p, c = panel_for("mirtaheri")
    es = check_invariants(p, c)
    m = p.meta
    assert m["dataset"] == "xp_mirtaheri" and m["n_pumps"] == 2
    tab = m["table"]
    assert tab["rows"] == 7 and tab["dropped_excluded_coins"] == 2 and tab["rows_coin_without_binance_klines"] == 1 and tab["dropped_ticker_not_in_message"] == 1
    assert tab["rows_kept"] == 3 and tab["pump_attempts_after_merge"] == 2
    pumps = [e for e in m["episodes"] if e["is_pump"]]
    assert [e["key"] for e in pumps] == ["MMABTC@2018-03-03T12:00Z", "MMBBTC@2018-03-05T09:30Z"]
    assert pumps[0]["anchor_ns"] == ns("2018-03-03 12:00:30") and pumps[0]["channels"] == [1001, 1002] and pumps[0]["exchange"] == "binance"
    assert pumps[1]["exchange"] == "bittrex"
    pre = int(c["pre_bars"]); bm = int(c["bar_minutes"])
    i = p.assets.index(pumps[0]["key"])
    s = es.for_asset(i); soc = s.subset(s.platform == 0)
    tw = soc.subset(soc.source == d7_mirtaheri.TWITTER_SOURCE); tg = soc.subset(soc.source != d7_mirtaheri.TWITTER_SOURCE)
    assert len(tw) > 10 and len(tg) == 5                                               # 4 of channel 1001 + 1 of channel 1002 in the window
    # Twitter: bucket-end stamps on the bar grid, time == avail, counts and users summed, sentiment in [-1, 1]
    assert np.allclose(tw.time, np.round(tw.time)) and np.array_equal(tw.time, tw.avail)
    assert (tw.mark[:, 0] >= 1).all() and (tw.mark[:, 1] >= 1).all() and (tw.mark[:, 1] <= tw.mark[:, 0] * 1.0 + 1e-6).all()
    assert np.nanmin(tw.mark[:, 2]) >= -1 and np.nanmax(tw.mark[:, 2]) <= 1
    # check one bucket against the raw fixture series
    raw = json.load(open(os.path.join(FROOT, "crosspump", "mirtaheri", "Twitter", "tweets-ts", "$MMA.json")))["data"]
    bc = p.meta["bar_close_ns"][:, i]; r = pre - 3                                      # the bar closing at bc[r] <-> event time r + 1
    lo_ms, hi_ms = (int(bc[r]) - bm * 60 * 10 ** 9) // 10 ** 6, int(bc[r]) // 10 ** 6
    want = sum(v for d_, v in raw if lo_ms <= d_["$date"] < hi_ms)
    got = tw.mark[tw.time == r + 1, 0]
    assert want > 0 and len(got) == 1 and got[0] == want, (want, got)
    assert len(es.before(i, r).subset((es.before(i, r).source == 1) & (es.before(i, r).time == r + 1))) == 0, "a bucket is usable only from the row after it closes"
    # the planted tweet burst before the pump shows up in the raster (every minute of [11:30, 12:15) has tweets)
    burst = tw.subset((tw.time > pre - 6) & (tw.time <= pre)); before = tw.subset(tw.time <= pre - 12)
    assert burst.mark[:, 0].mean() > 2 * before.mark[:, 0].mean()
    # Telegram: classified message = 1.0, reach NaN (views are not PIT), keyword scores elsewhere
    assert np.isnan(tg.mark[:, 1]).all() and (tg.mark[:, 0] == 1).all()
    assert (tg.mark[:, 2] == 1.0).sum() == 2 and (np.floor(tg.time[tg.mark[:, 2] == 1.0]) >= pre).all()
    assert set(tg.source.tolist()) == {tu.stable_id(1001, "tg:"), tu.stable_id(1002, "tg:")}
    assert "source id 1 = Twitter" in m["social_platforms"] and es.meta["entities"][1].startswith("twitter")
    # other-coin control: Twitter series of ITS coin (MMB), Telegram of the pump's channels without the 1.0 mark
    ctl = [e for e in m["episodes"] if e.get("control_of") == pumps[0]["key"] and e.get("control_type") == "other_coin_same_time"][0]
    k = p.assets.index(ctl["key"]); sk = es.for_asset(k); sk = sk.subset(sk.platform == 0)
    assert ctl["symbol"] == "MMB" and not (sk.mark[sk.source != 1, 2] == 1.0).any() and (sk.source == 1).sum() > 0
    assert not np.array_equal(sk.subset(sk.source == 1).mark[:, 0], tw.mark[:, 0])
    assert m["coverage"]["episodes_with_platform"]["wallet"] == 0 and "MISSING" in m["coverage"]["wallet"]
    assert p.text is not None and m["text_placement"]["placed"] > 0
    # native 1-minute cadence option: more, finer events with fractional stamps
    p1, _ = panel_for("mirtaheri", twitter_agg_minutes=1, telegram=False)
    e1 = get_streams(p1).for_asset(p1.assets.index(pumps[0]["key"])); t1 = e1.subset(e1.platform == 0)
    assert (t1.source == 1).all() and len(t1) > len(tw) and t1.mark[:, 0].sum() == tw.mark[:, 0].sum() and p1.text is None
    assert np.array_equal(t1.time, t1.avail) and (np.abs(t1.time * 5 - np.round(t1.time * 5)) < 1e-6).all()
    # require_ticker_in_text off keeps the 'have funx' row; exchange: Binance keeps only the message naming Binance
    p2, _ = panel_for("mirtaheri", require_ticker_in_text=False)
    assert p2.meta["n_pumps"] == 3 and any(a.startswith("FUNXBTC@") for a in p2.assets)
    p3, _ = panel_for("mirtaheri", exchange="Binance", n_controls_per_pump=1)
    assert p3.meta["n_pumps"] == 1 and p3.meta["table"]["dropped_venue_not_mentioned"] == 2


# ====================================================================================================== D5 Moura / TGDataset
def test_tgdataset_pd_panel():
    p, c = panel_for("tgdataset_pd", burst_avail="day_end", post_bars=240)             # 20 h after the burst start: the day end falls inside the window
    es = check_invariants(p, c)
    m = p.meta
    assert m["dataset"] == "xp_tgdataset_pd" and m["label_rule"]["type"] == "proxy (Moura et al. event list not released)"
    tab = m["table"]
    assert tab["bursts"] == 5 and tab["bursts_with_binance_klines"] == 3 and tab["proxy_pumps"] == 1 and tab["burst_no_reaction"] == 2
    eps = {e["key"]: e for e in m["episodes"]}
    assert eps["QQABTC@2022-03-03T06:02Z"]["is_pump"] and eps["QQABTC@2022-03-03T06:02Z"]["source"] == "burst_reaction_proxy"
    pr = eps["QQABTC@2022-03-03T06:02Z"]["proxy"]
    assert pr["rise"] >= 0.10 and pr["reverted"] is True
    assert eps["QQABTC@2022-03-05T03:00Z"]["source"] == "burst_no_reaction" and not eps["QQABTC@2022-03-05T03:00Z"]["is_pump"]
    assert eps["QQBUSDT@2022-03-04T10:00Z"]["quote"] == "USDT" and m["quote_of"] == {"QQA": "BTC", "QQB": "USDT"}
    assert any(e["source"] == "sampled_control" for e in m["episodes"])
    assert "pit_warning" in m and m["social_stream_level"] == "burst"
    soc = es.subset(es.platform == 0)
    assert len(soc) >= 1 and (soc.source == 0).all() and np.isnan(soc.mark[:, 2]).all() and np.isnan(soc.mark[:, 3]).all()
    i = p.assets.index("QQABTC@2022-03-03T06:02Z"); si = soc.subset(soc.asset == i)
    assert len(si) == 1 and np.floor(si.time[0]) == int(c["pre_bars"])
    assert np.isclose(si.mark[0, 0], np.log1p((ns("2022-03-03 20:10:00") - ns("2022-03-03 06:02:10")) / NS_H), atol=1e-5)
    assert si.avail[0] - si.time[0] > 12 * 17, "the burst is knowable only at the end of its UTC day"
    assert p.text is None and "burst-level" in str(m["text"])
    # default (strict PIT): the two-sided CA-CFAR needs 6 future days -> no burst event is available inside the window
    p0, c0 = panel_for("tgdataset_pd")
    e0 = check_invariants(p0, c0)
    assert (e0.platform == 0).sum() == 0 and p0.meta["coverage"]["episodes_with_platform"]["social"] == 0
    assert "MISSING" in p0.meta["coverage"]["social"] and "pit_warning" not in p0.meta
    assert e0.meta["social"]["events_built"] >= 3 and e0.meta["social"]["events_in_stream"] == 0
    assert p0.meta["n_pumps"] == 1 and (np.asarray(p0.meta["is_pump"]) == [a == "QQABTC@2022-03-03T06:02Z" for a in p0.assets]).all()
    # oracle mode is flagged
    p2, _ = panel_for("tgdataset_pd", burst_avail="first_message")
    e2 = get_streams(p2); s2 = e2.subset(e2.platform == 0)
    assert len(s2) >= 3 and np.array_equal(s2.time, s2.avail) and "NOT point-in-time" in p2.meta["pit_warning"]
    # a stricter proxy threshold turns the pump into a control (the label is a documented proxy)
    px, _ = panel_for("tgdataset_pd", min_rise=5.0, n_controls_per_pump=0)
    assert px.meta["n_pumps"] == 0 and px.meta["n_controls"] == 3


def test_tgdataset_messages_and_prepare():
    d = tempfile.mkdtemp()
    try:
        X = os.path.join(d, "crosspump"); os.makedirs(X)
        shutil.copytree(os.path.join(FROOT, "crosspump", "moura"), os.path.join(X, "moura"))
        os.symlink(os.path.join(FROOT, "crosspump", "binance_klines"), os.path.join(X, "binance_klines"))
        # a TGDataset-like json (published layout: channel objects with a text_messages map) -- SYNTHETIC
        os.makedirs(os.path.join(X, "tgdataset"))
        u = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10 ** 9)
        chans = [{"_id": 77, "username": "fixture_a", "text_messages": {"1": {"message": "QQA pump today, buy now", "date": u("2022-03-03 05:30:00")},
                                                                         "2": {"message": "nothing about coins", "date": u("2022-03-03 05:40:00")},
                                                                         "3": {"message": "$QQA to the moon, target 200%", "date": u("2022-03-03 06:02:10")}}},
                 {"_id": 78, "username": "fixture_b", "text_messages": {"9": {"message": "QQB looks fine", "date": u("2022-03-04 09:00:00")},
                                                                         "10": {"message": "old QQA talk", "date": u("2019-01-01 00:00:00")}}}]
        with open(os.path.join(X, "tgdataset", "part_0.json"), "w") as f:
            json.dump(chans, f)
        d5_tgdataset_pd.prepare(d, {"tgdataset": True})                                  # no clone (files exist), no klines, extraction only
        out = pd.read_csv(os.path.join(d, d5_tgdataset_pd.TG_MESSAGES))
        assert list(out.columns) == ["coin", "time_ns", "channel_id", "text"] and sorted(out["coin"]) == ["QQA", "QQA", "QQB"]
        c = cfg_for("tgdataset_pd", root=d)
        p = DATASETS.get(c["name"])(c).build_panel()
        es = check_invariants(p, c)
        m = p.meta
        assert m["social_stream_level"] == "message" and "pit_warning" not in m
        i = p.assets.index("QQABTC@2022-03-03T06:02Z"); s = es.for_asset(i); soc = s.subset(s.platform == 0)
        assert len(soc) == 2 and np.array_equal(soc.time, soc.avail) and (soc.source == tu.stable_id("77", "tgd:")).all()
        assert (soc.mark[:, 0] == 1).all() and (soc.mark[:, 2] > 0.3).all()
        assert p.text is not None and p.text_mask[:, i].sum() == 2
        # prepare() without flags is a no-op when the repository is already there
        d5_tgdataset_pd.prepare(d, {})
        shutil.copytree(os.path.join(FROOT, "crosspump", "mirtaheri"), os.path.join(X, "mirtaheri"))
        d7_mirtaheri.prepare(d, None)
    finally:
        shutil.rmtree(d)


def test_proxy_label_unit():
    step = 5 * 60 * 10 ** 9
    idx = np.arange(0, 200) * step
    base = np.full(200, 100.0)
    up = base.copy(); up[50:53] = [108, 115, 112]; up[53:60] = 104                     # +15 % then back below the half-way level
    kl = pd.DataFrame({"close": up}, index=idx)
    r = d5_tgdataset_pd.proxy_label(kl, 50 * step + 7, 5, 12, 96, 0.10, 0.5)
    assert r["is_pump"] and np.isclose(r["rise"], 0.15) and r["reverted"]
    stay = base.copy(); stay[50:] = 115                                                # rises and stays: no reversion -> not a pump
    assert not d5_tgdataset_pd.proxy_label(pd.DataFrame({"close": stay}, index=idx), 50 * step, 5, 12, 96, 0.10, 0.5)["is_pump"]
    late = base.copy(); late[80:83] = 130                                              # reaction outside the peak window
    assert not d5_tgdataset_pd.proxy_label(pd.DataFrame({"close": late}, index=idx), 50 * step, 5, 12, 96, 0.10, 0.5)["is_pump"]
    small = base.copy(); small[51] = 105; small[52:] = 100
    assert not d5_tgdataset_pd.proxy_label(pd.DataFrame({"close": small}, index=idx), 50 * step, 5, 12, 96, 0.10, 0.5)["is_pump"]
    # the base uses only bars that closed before the burst bar opens
    lead = base.copy(); lead[50:52] = 200
    assert d5_tgdataset_pd.proxy_label(pd.DataFrame({"close": lead}, index=idx), 50 * step, 5, 12, 96, 0.10, 0.5)["base"] == 100.0
    assert not d5_tgdataset_pd.proxy_label(pd.DataFrame({"close": base[:10]}, index=idx[:10]), 150 * step, 5, 12, 96, 0.10, 0.5)["is_pump"]


def test_no_fabrication_in_loaders():
    """Rule zero: the real loaders never generate data (the only RNG is the seeded control sampler of text_utils)."""
    for mod in (d4_pumpsense, d5_tgdataset_pd, d7_mirtaheri):
        src = open(mod.__file__).read()
        for bad in ("np.random", "default_rng", "rng.", "random_walk", "synthetic"):
            assert bad not in src, f"{os.path.basename(mod.__file__)} contains {bad!r}"
    src = open(tu.__file__).read()
    assert src.count("default_rng") == 1 and "rng.normal" not in src and "rng.random" not in src


TESTS = [test_text_utils_parsing, test_embed_cache, test_registry_and_configs, test_missing_raw_files_raise_with_recipe, test_pumpsense_panel,
         test_stream_helpers_match_common, test_place_text_rule, test_pumpsense_options_and_zip, test_mirtaheri_panel, test_tgdataset_pd_panel,
         test_tgdataset_messages_and_prepare, test_proxy_label_unit, test_no_fabrication_in_loaders]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-k", default=None)
    a = ap.parse_args()
    ensure_fixtures()
    failed = 0
    for fn in TESTS:
        if a.k and a.k not in fn.__name__:
            continue
        t0 = time.time()
        try:
            fn(); print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{'ALL PASSED' if not failed else str(failed) + ' FAILED'} ({len(TESTS)} tests)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
