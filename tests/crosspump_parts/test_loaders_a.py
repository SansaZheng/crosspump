"""Group-A loader tests (D1 hu_pd, D2 sapienza_pd, D3 xu_pumpolymp, D6 ardia_pd + binance_fetch), plain-script mode:

    python3 tests/crosspump_parts/test_loaders_a.py [-k substring]

Everything runs on the SYNTHETIC FIXTURES under tests/crosspump_parts/fixtures_a/root (written by make_fixtures.py: fake
coins, fake channels, seeded random-walk klines). No real data is read and nothing is downloaded (the downloader is
exercised against file:// URLs in a temporary directory).
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
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from warn2trade.utils import DATASETS                                   # noqa: E402
from crosspump.data.streams import get_streams                          # noqa: E402
from crosspump.data.loaders import common                               # noqa: E402
from crosspump.data.loaders import binance_fetch as bf                  # noqa: E402
from crosspump.data.loaders import d1_hu_pd, d2_sapienza_pd, d3_xu_pumpolymp, d6_ardia_pd   # noqa: E402

FIX = os.path.join(ROOT, "tests", "crosspump_parts", "fixtures_a")
FROOT = os.path.join(FIX, "root")
CFG_DIR = os.path.join(ROOT, "configs", "crosspump", "datasets")
MODS = {"hu_pd": d1_hu_pd, "sapienza_pd": d2_sapienza_pd, "xu_pumpolymp": d3_xu_pumpolymp, "ardia_pd": d6_ardia_pd}
# small layout for the fixtures (the yaml defaults need 72 h of klines before every anchor)
SMALL = {"root": FROOT, "pre_bars": 48, "post_bars": 24, "control_span_days": 1.0, "gap_hours": 3, "end_bars_after_peak": 6, "horizons": [1, 2, 3, 6]}
YAML_KEYS = ["name", "root", "subdir", "quote", "bar_minutes", "pre_bars", "post_bars", "lead", "n_controls_per_pump", "gap_hours", "exchange", "max_pumps", "text_slots",
             "lookback", "horizons", "latency", "split", "pit", "assumptions"]
_CACHE = {}


def ensure_fixtures():
    if not os.path.exists(os.path.join(FROOT, "SYNTHETIC_FIXTURES.txt")):
        spec = importlib.util.spec_from_file_location("make_fixtures_a", os.path.join(FIX, "make_fixtures.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.build(FROOT)
    assert "SYNTHETIC FIXTURES" in open(os.path.join(FIX, "README.md")).read()


def cfg_for(short: str, **over):
    with open(os.path.join(CFG_DIR, f"{short}.yaml")) as f:
        c = yaml.safe_load(f)
    c.update(SMALL); c.update(over)
    return c


def build(short: str, **over):
    k = (short, json.dumps(over, sort_keys=True, default=str))
    if k not in _CACHE:
        c = cfg_for(short, **over)
        _CACHE[k] = DATASETS.get(c["name"])(c).build_panel()
    return _CACHE[k]


def col(panel, key):
    return common.asset_index(panel)[key]


# ----------------------------------------------------------------------------------------------- generic invariants
def check_panel(panel, name, n_pumps, has_text):
    m = panel.meta; T, N = panel.mid.shape
    pre, post, lead = m["pre_bars"], m["post_bars"], m["lead"]
    panel.validate()
    assert T == pre + post + 1 == 73, T
    assert m["dataset"] == name and m["split_axis"] == "assets" and m["assets_time_aligned"] is False
    assert m["feat_names"] == ["log_ret", "log_volume", "hl_range", "taker_imb", "log_trades"] and m["ret_col"] == 0 and panel.price_feat.shape == (T, N, 5)
    is_pump = np.asarray(m["is_pump"]); assert is_pump.dtype == bool and int(is_pump.sum()) == n_pumps and int((~is_pump).sum()) >= 1, is_pump
    assert (np.diff(m["anchor_ns"]) >= 0).all(), "episodes must be sorted by anchor time"
    assert len(set(panel.assets)) == N and len(m["episodes"]) == N
    end_after = m["end_bars_after_peak"]; hmax = max(m["horizons"])
    for i, e in enumerate(m["episodes"]):
        assert e["key"] == panel.assets[i] and bool(e["is_pump"]) == bool(is_pump[i])
        lab = panel.labels[:, i]; btp = panel.bars_to_peak[:, i]
        if not is_pump[i]:
            assert e["key"].startswith("CTRL_") and e["control_of"] in panel.assets and e["control_type"] in ("same_coin_other_time", "other_coin_same_time")
            assert lab.sum() == 0 and np.isnan(btp).all() and m["peak_row"][i] == -1
            assert (panel.avail_label[:, i] == np.arange(T) + hmax).all()
            continue
        sym, ts = e["key"].split("@"); assert sym == e["symbol"] + m["quote"] and ts.endswith("Z") and len(ts) == 17, e["key"]
        pk = int(m["peak_row"][i]); lo = pre - lead; end = min(T - 1, pk + end_after)
        assert pk >= pre, (e["key"], pk)
        exp = np.zeros(T, dtype=int); exp[lo : pk + 1] = 1
        assert (lab == exp).all(), f"{e['key']}: labels must be 1 exactly on [anchor - lead, peak]"
        rows = np.arange(lo, end + 1)
        assert np.array_equal(btp[rows], (pk - rows).astype(float)), "bars_to_peak must be exact on [anchor - lead, end]"
        assert np.isnan(btp[:lo]).all() and np.isnan(btp[end + 1 :]).all()
        assert (panel.avail_label[rows, i] == end).all() and (panel.avail_label[:lo, i] == np.arange(lo) + hmax).all()
        assert e["peak_source"] in ("curated", "derived_argmax_high")
        a0 = (int(e["anchor_ns"]) // (5 * 60 * 10 ** 9)) * 5 * 60 * 10 ** 9
        assert m["bar_close_ns"][pre, i] == a0 + 5 * 60 * 10 ** 9, "row pre_bars must be the anchor bar"
    assert (panel.avail_feat == np.arange(T)[:, None]).all() and (panel.avail_label >= np.arange(T)[:, None]).all()
    assert np.array_equal(panel.universe, np.isfinite(panel.mid)) and not (panel.tradable & ~np.isfinite(panel.mid)).any()
    assert set(m["pit_fields"]) >= {"avail_feat", "avail_label", "universe"}
    # streams
    s = get_streams(panel); assert s is not None and len(s) > 0
    s.validate(T, N)
    assert (s.avail >= s.time - 1e-9).all()
    plats = set(s.platforms_present()); assert {"social", "orderflow"} <= plats, plats
    soc = s.subset(s.platform == 0)
    assert set(np.unique(soc.asset[np.isclose(soc.mark[:, 2], 1.0) & (soc.source != 0)])) >= set(np.nonzero(is_pump)[0]), "every pump episode needs its announcement event"
    assert (soc.source != 0).all() and np.isfinite(soc.mark[:, 3]).all(), "social events carry a channel id and a prior-involvement count"
    of = s.subset(s.platform == 2); assert np.allclose(of.time, of.avail) and np.isfinite(of.mark[:, 0]).all()
    # PIT rule: usable at row t iff avail <= t; rasterise counts at ceil(avail)
    cnt = s.rasterise(T, N)
    assert cnt.shape == (T, N, 3) and cnt.sum() == (np.ceil(s.avail) < T).sum()
    j = int(np.nonzero(soc.avail % 1 > 1e-6)[0][0]) if (soc.avail % 1 > 1e-6).any() else None
    if j is not None:
        i, av = int(soc.asset[j]), float(soc.avail[j]); r = int(np.ceil(av))
        n_before = (s.before(i, r - 1).platform == 0).sum(); n_at = (s.before(i, r).platform == 0).sum()
        assert n_at > n_before and cnt[: r, i, 0].sum() == n_before and cnt[: r + 1, i, 0].sum() == n_at
    cov = m["coverage"]; assert isinstance(cov, dict)
    for k in ("episodes_with_platform", "n_events", "wallet", "orderflow", "social", "n_pumps", "n_controls"):
        assert k in cov, k
    assert cov["n_pumps"] == n_pumps and cov["episodes_with_platform"]["social"] >= n_pumps
    assert cov["orderflow"]["source"] in ("bars", "aggtrades")
    assert isinstance(m["assumptions"], list) and len(m["assumptions"]) >= 5 and m["licence"] and m["source"]
    # text
    if has_text:
        assert panel.text is not None and panel.text.shape[:2] == (T, N) and panel.text.shape[2:] == (4, 32) and "avail_text" in m["pit_fields"]
        tt = np.arange(T)[:, None, None]
        assert panel.text_mask.any() and (panel.avail_text[panel.text_mask] >= np.broadcast_to(tt, panel.text_mask.shape)[panel.text_mask]).all()
        assert (np.abs(panel.text[~panel.text_mask]).sum() == 0)
    else:
        assert panel.text is None and isinstance(m["text"], str)
    return s


# ----------------------------------------------------------------------------------------------- tests
def test_registry_and_configs():
    for short, mod in MODS.items():
        with open(os.path.join(CFG_DIR, f"{short}.yaml")) as f:
            c = yaml.safe_load(f)
        assert c["name"] == f"xp_{short}" and DATASETS.get(c["name"]) is not None, c["name"]
        for k in YAML_KEYS:
            assert k in c, (short, k)
        assert c["bar_minutes"] == 5 and c["pre_bars"] == 864 and c["post_bars"] == 96 and c["lead"] == 2 and c["lookback"] == 48 and c["latency"] == 1
        assert c["horizons"] == [1, 2, 3, 6, 12, 24] and c["split"] == {"train_frac": 0.6, "val_frac": 0.15, "purge_hours": 96} and c["n_controls_per_pump"] == 2 and c["gap_hours"] == 24
        cls = DATASETS.get(c["name"])
        assert cls.name == c["name"] and cls.freq == "5min" and cls.bars_per_year == 105120 and tuple(cls.horizons) == (1, 2, 3, 6, 12, 24)
        for attr in ("SOURCE_URL", "LICENCE", "RECIPE", "REQUIRED"):
            assert getattr(mod, attr), (short, attr)
        assert callable(mod.prepare) and all(r.startswith("crosspump/") for r in mod.REQUIRED)
        assert os.path.exists(os.path.join(FROOT, mod.REQUIRED[0])), mod.REQUIRED[0]
        assert c["subdir"] == mod.SUBDIR


def test_hu_pd():
    p = build("hu_pd")
    s = check_panel(p, "xp_hu_pd", n_pumps=3, has_text=True)
    m = p.meta
    assert m["table"]["rows"] == 7 and m["table"]["pumps_merged"] == 4 and m["pumps_without_klines"] == 1          # ZZZ has no klines; yobit / USDT rows filtered
    i = col(p, "AAABTC@2021-03-03T12:01Z"); e = m["episodes"][i]
    assert len(e["announcements"]) == 2 and e["watch_channels"] == ["111", "222"] and m["peak_row"][i] == 49 and e["peak_source"] == "derived_argmax_high"
    soc = s.for_asset(i); soc = soc.subset(soc.platform == 0)
    ann = soc.subset(np.isclose(soc.mark[:, 2], 1.0) & (soc.time >= 48))
    assert len(ann) == 2 and np.allclose(ann.time, [48 + 93 / 300, 48 + 130 / 300]) and set(ann.source) == {bf.stable_id("111"), bf.stable_id("222")}
    # announcement at 12:01:33 is inside bar 48 -> not usable at row 48, usable at row 49
    assert not (s.before(i, 48).time >= 48).any() or (s.before(i, 48).platform[s.before(i, 48).time >= 48] != 0).all()
    assert ((s.before(i, 49).platform == 0) & (s.before(i, 49).time >= 48)).sum() == 2
    # raw-pickle messages of channel 111 (09:30, 11:32:07) and the pred fallback of channel 222 (10:02); the announcement messages themselves are de-duplicated
    pre_msgs = soc.subset(soc.time < 48)
    assert len(pre_msgs) == 3 and np.allclose(np.sort(pre_msgs.time), [48 - 150 / 5, 48 - 118 / 5, 48 - (27 * 60 + 53) / 300])
    assert (pre_msgs.mark[:, 2] > 0).all() and (pre_msgs.mark[:, 2] <= 1).all() and np.isnan(pre_msgs.mark[:, 1]).all(), "keyword score in (0, 1], reach unknown"
    edited = soc.subset(np.isclose(soc.time, 54.0)); assert len(edited) == 1 and np.isnan(edited.mark[0, 2]), "edited message: event kept, score dropped"
    assert (soc.mark[:, 3] == 0).all(), "first pump of both channels: no prior involvement"
    j = col(p, "BBBBTC@2021-03-04T15:00Z"); sj = s.for_asset(j); sj = sj.subset((sj.platform == 0) & (sj.source == bf.stable_id("111")))
    assert (sj.mark[:, 3] == 1).all(), "channel 111 was involved in one earlier pump episode"
    # text: message at 11:32:07 (bar-unit time 42.42) is first usable at row 43; the edited message has no text; avail_text == row
    assert p.text_mask[43, i].sum() == 1 and p.text_mask[42, i].sum() == 0 and p.avail_text[43, i][p.text_mask[43, i]][0] == 43
    assert p.text_mask[49, i].sum() == 2 and p.text_mask[54, i].sum() == 0 and p.text_mask[48, i].sum() == 0
    assert m["text"]["n_docs"] == int(p.text_mask.sum()) and m["coverage"]["social"]["edited_messages_in_windows"] >= 1
    assert m["coverage"]["social"]["channels_with_raw_messages"] == 1 and m["coverage"]["social"]["channels_with_pred_messages_only"] == 1
    # control windows carry the watched channels' messages (coin-agnostic social stream) but no announcement event
    k = col(p, "CTRL_BBBBTC@2021-03-03T12:01Z"); sk = s.for_asset(k); sk = sk.subset(sk.platform == 0)
    assert m["episodes"][k]["control_type"] == "other_coin_same_time" and len(sk) >= 4 and m["episodes"][k]["watch_channels"] == ["111", "222"]
    assert sorted(set(sk.source)) == sorted({bf.stable_id("111"), bf.stable_id("222")})
    assert len(m["episodes"][k]["announcements"]) == 0 and np.isclose(sk.time, 48 + 93 / 300).sum() == 1, "the control sees the raw 12:01:33 message, not a table announcement"
    # wallet: optional file present for one pump episode only; avail = block time + 1 bar; unknown keys are ignored
    w = s.subset(s.platform == 1)
    assert len(w) == 2 and set(w.asset) == {i} and np.allclose(w.avail - w.time, 1.0) and set(w.source) == {9001, 9002}
    wc = m["coverage"]["wallet"]; assert wc["n_transfers"] == 2 and wc["n_pump_episodes"] == 1 and wc["n_control_episodes"] == 0 and "partial coverage" in wc["note"]
    assert m["coverage"]["episodes_with_platform"]["wallet"] == 1
    assert s.meta["entities"]["channels"][bf.stable_id("111")] == "111" and s.meta["platform_lags"] == {"social": 0.0, "wallet": 1.0, "orderflow": 0.0}


def test_hu_pd_message_modes():
    p_none = build("hu_pd", messages="none")
    assert p_none.text is None and p_none.meta["coverage"]["social"]["message_events"] == 0 and p_none.meta["coverage"]["social"]["announcement_events"] == 4
    p_pred = build("hu_pd", messages="pred")
    c = p_pred.meta["coverage"]["social"]; assert c["channels_with_raw_messages"] == 0 and c["channels_with_pred_messages_only"] == 1 and c["message_events"] >= 1
    p_keep = build("hu_pd", edited_text="keep"); i = col(p_keep, "AAABTC@2021-03-03T12:01Z")
    assert p_keep.text_mask[54, i].sum() == 1
    p_lag = build("hu_pd", social_lag_bars=1.0); s = get_streams(p_lag); soc = s.subset(s.platform == 0)
    assert np.allclose(soc.avail - soc.time, 1.0) and p_lag.text_mask[44, i].sum() == 1 and p_lag.text_mask[43, i].sum() == 0
    import io
    import pickle
    try:                                    # the raw Telegram pickles are read with a restricted unpickler (datetime only)
        d1_hu_pd._SafeUnpickler(io.BytesIO(pickle.dumps(os.getcwd))).load()
    except pickle.UnpicklingError as e:
        assert "not allowed" in str(e), e
    else:
        raise AssertionError("restricted unpickler admitted a foreign global")


def test_sapienza_pd():
    p = build("sapienza_pd")
    s = check_panel(p, "xp_sapienza_pd", n_pumps=3, has_text=False)
    m = p.meta
    assert m["table"]["anchors_refined"] == 1 and m["table"]["anchor_shift_s_median"] == -25.0 and m["pumps_without_klines"] == 1
    i = col(p, "AAABTC@2021-03-03T11:59Z"); e = m["episodes"][i]
    assert e["anchor_source"] == "labeled_features_gt" and e["anchor_shift_s"] == -25.0 and len(e["announcements"]) == 2
    soc = s.for_asset(i); soc = soc.subset(soc.platform == 0)
    assert len(soc) == 2 and np.allclose(soc.time, 49.0), "the social event stays at the table time (12:00 = one bar after the corrected 11:55 anchor bar)"
    assert m["peak_row"][i] == 50
    j = col(p, "BBBBTC@2021-03-04T15:00Z"); assert m["episodes"][j]["anchor_source"] == "table"
    wc = m["coverage"]["wallet"]; assert wc["file"] is None and "missing modality" in wc["note"] and m["coverage"]["episodes_with_platform"]["wallet"] == 0
    assert "wallet" not in s.platforms_present()
    ent = s.meta["entities"]["channels"][bf.stable_id("G1")]; assert ent["name"] == "Synthetic Group One"
    p2 = build("sapienza_pd", refine_anchor=False)
    assert "AAABTC@2021-03-03T12:00Z" in p2.assets and p2.meta["table"]["anchors_refined"] == 0


def test_xu_pumpolymp():
    p = build("xu_pumpolymp")
    s = check_panel(p, "xp_xu_pumpolymp", n_pumps=2, has_text=False)
    m = p.meta
    assert m["table"]["records"] == 5 and m["table"]["records_bad_time"] == 1 and m["table"]["exchanges"] == {"binance": 3, "cryptopia": 1}
    i = col(p, "AAABTC@2021-03-03T12:00Z"); e = m["episodes"][i]
    assert e["peak_source"] == "curated" and m["peak_row"][i] == 49, "peak = highest trade at/after the signal (12:07:30), not the 11:20 print"
    assert len(e["announcements"]) == 2 and e["pumpolymp_id"] == "synthetic-AAA-2021-03-03" and abs(e["price_before"] - 2.0e-5) < 1e-12
    j = col(p, "BBBBTC@2021-03-04T15:00Z"); assert m["episodes"][j]["peak_source"] == "derived_argmax_high"
    pc = m["price_check"]; assert pc["episodes_checked"] == 2 and 0 <= pc["close_within_25pct_of_priceBeforePump"] <= 2
    assert s.meta["entities"]["channels"][bf.stable_id("https://t.me/synthetic_pumps")]["title"] == "Synthetic Pumps"
    assert not any(a.startswith("CCC") for a in p.assets if not a.startswith("CTRL_")), "Cryptopia record must be filtered with exchange: binance"
    p_all = build("xu_pumpolymp", exchange="all")
    assert "CCCBTC@2021-03-05T09:00Z" in p_all.assets and p_all.meta["episodes"][col(p_all, "CCCBTC@2021-03-05T09:00Z")]["exchange"] == "cryptopia"


def test_ardia_pd():
    p = build("ardia_pd")
    s = check_panel(p, "xp_ardia_pd", n_pumps=3, has_text=False)
    m = p.meta
    i = col(p, "AAABTC@2021-03-03T12:01Z"); e = m["episodes"][i]
    assert e["success"] is True and e["peak_source"] == "curated" and m["peak_row"][i] == 49 and e["anchor_source"] == "table" and len(e["announcements"]) == 2
    j = col(p, "BBBBTC@2021-03-04T14:58Z"); ej = m["episodes"][j]
    assert ej["anchor_source"] == "pump_date_before_announcement" and m["peak_row"][j] == 48 and ej["peak_source"] == "curated"
    sj = s.for_asset(j); sj = sj.subset(sj.platform == 0)
    assert len(sj) == 1 and np.isclose(sj.time[0], 49 + 133 / 300), "the announcement keeps its recorded time (after the curated peak)"
    k = col(p, "CCCBTC@2021-03-05T09:00Z"); ek = m["episodes"][k]
    assert ek["success"] is False and ek["peak_source"] == "derived_argmax_high"
    assert np.array_equal(m["success"], np.array([bool(x.get("success", False)) for x in m["episodes"]])) and int(m["success"].sum()) == 2
    assert m["table"]["anchors_moved_to_pump_date"] == 1 and m["table"]["pumps_success"] == 2
    p2 = build("ardia_pd", success_only=True)
    assert int(p2.meta["is_pump"].sum()) == 2 and not any(a.startswith("CCCBTC@") for a in p2.assets)
    p3 = build("ardia_pd", max_pumps=1)
    assert int(p3.meta["is_pump"].sum()) == 1


def test_controls_and_determinism():
    p = build("hu_pd"); m = p.meta
    c = cfg_for("hu_pd"); q = DATASETS.get(c["name"])(c).build_panel()
    assert q.assets == p.assets and np.array_equal(np.nan_to_num(q.mid), np.nan_to_num(p.mid)) and len(get_streams(q)) == len(get_streams(p))
    pump_t = {}
    for e in m["episodes"]:
        if e["is_pump"]:
            pump_t.setdefault(e["symbol"], []).append(int(e["anchor_ns"]))
    step = 5 * 60 * 10 ** 9
    for e in m["episodes"]:
        if e["is_pump"]:
            continue
        a0 = (int(e["anchor_ns"]) // step) * step; lo, hi = a0 - 48 * step, a0 + 25 * step
        for t in pump_t.get(e["symbol"], []):
            assert abs(t - int(e["anchor_ns"])) > 3 * 3600 * 10 ** 9 and not (lo <= t < hi), f"control {e['key']} overlaps a pump of its coin"
    types = {e["control_type"] for e in m["episodes"] if not e["is_pump"]}
    assert types == {"same_coin_other_time", "other_coin_same_time"}, types
    p0 = build("hu_pd", n_controls_per_pump=0, max_pumps=1)
    assert p0.N == 1 and p0.meta["n_controls"] == 0


def test_orderflow_sources():
    tr = bf.read_aggtrades([os.path.join(FROOT, "crosspump", "binance_aggtrades", "AAABTC-aggTrades-2021-03-03.csv")])
    assert list(tr.columns) == ["time_ns", "price", "qty", "taker_buy"] and len(tr) > 2000 and tr["time_ns"].is_monotonic_increasing and 0.4 < tr["taker_buy"].mean() < 0.8
    b = bf.aggtrades_bursts(tr)
    day = int(np.datetime64("2021-03-03T00:00:00", "ns").astype(np.int64)); mins = (b["time_ns"].to_numpy() - day) / 6e10
    assert len(b) and ((mins > 715) & (mins <= 731)).any() and (b["z"] > 2.5).all() and (b["time_ns"] % (60 * 10 ** 9) == 0).all()
    # auto: aggTrades cover only one of nine episodes -> ONE source (bars) for all, nothing mixed in
    p = build("hu_pd"); of = p.meta["coverage"]["orderflow"]
    assert of["source"] == "bars" and of["episodes_with_all_aggtrades_days"] == 1 and "NOT mixed" in of["note"]
    try:
        build("hu_pd", orderflow_source="aggtrades"); raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "aggtrades" in str(e) and "git clone" in str(e)
    # single-episode panel whose window days are all cached -> aggTrades bursts (1-minute resolution, avail = bucket close)
    p1 = build("hu_pd", n_controls_per_pump=0, max_pumps=1)
    assert p1.assets == ["AAABTC@2021-03-03T12:01Z"] and p1.meta["coverage"]["orderflow"]["source"] == "aggtrades"
    s = get_streams(p1); o = s.subset(s.platform == 2)
    assert len(o) and np.allclose(o.time, o.avail) and np.allclose((o.time * 5) % 1, 0, atol=1e-6) and ((o.time > 47) & (o.time <= 50.2)).any()
    assert np.isfinite(o.mark[:, 1]).all(), "imbalance mark of aggTrades bursts"


def test_missing_raw_files():
    tmp = tempfile.mkdtemp(prefix="xp_loaders_a_")
    try:
        for short, mod in MODS.items():
            try:
                DATASETS.get(f"xp_{short}")(cfg_for(short, root=tmp)).build_panel(); raise AssertionError(f"{short}: expected FileNotFoundError")
            except FileNotFoundError as e:
                assert mod.RECIPE in str(e) and "git clone" in str(e) and mod.SUBDIR in str(e), short
            # event table present but no klines -> FileNotFoundError with the recipe as well
            src = os.path.join(FROOT, "crosspump", mod.SUBDIR); dst = os.path.join(tmp, "crosspump", mod.SUBDIR)
            shutil.copytree(src, dst)
            try:
                DATASETS.get(f"xp_{short}")(cfg_for(short, root=tmp)).build_panel(); raise AssertionError(f"{short}: expected FileNotFoundError (klines)")
            except FileNotFoundError as e:
                assert "klines" in str(e) and mod.RECIPE in str(e), short
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fetch_and_prepare_offline():
    """The downloader against file:// URLs (no network): unzip, skip existing, log failures, never called by build_panel."""
    tmp = tempfile.mkdtemp(prefix="xp_fetch_a_")
    old = bf.BINANCE_VISION
    try:
        vis = os.path.join(tmp, "vision"); zdir = os.path.join(vis, "monthly", "klines", "AAABTC", "1m"); os.makedirs(zdir)
        src = os.path.join(FROOT, "crosspump", "binance_klines", "AAABTC-1m-2021-03.csv")
        with zipfile.ZipFile(os.path.join(zdir, "AAABTC-1m-2021-03.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(src, "AAABTC-1m-2021-03.csv")
        bf.BINANCE_VISION = "file://" + vis
        assert bf.kline_url("AAA", "2021-03").endswith("/monthly/klines/AAABTC/1m/AAABTC-1m-2021-03.zip")
        assert bf.aggtrades_url("AAA", "2021-03-03").endswith("/daily/aggTrades/AAABTC/AAABTC-aggTrades-2021-03-03.zip")
        cache = os.path.join(tmp, "cache")
        st = bf.fetch_klines({"AAA": ["2021-03", "2021-04"]}, cache, workers=2)
        assert st["AAABTC-1m-2021-03.csv"] == "ok" and st["AAABTC-1m-2021-04.csv"].startswith("error"), st
        assert open(os.path.join(cache, "AAABTC-1m-2021-03.csv")).read() == open(src).read()
        assert bf.fetch_klines([("AAA", "2021-03")], cache)["AAABTC-1m-2021-03.csv"] == "exists"
        man = json.load(open(os.path.join(cache, "manifest.json"))); assert "AAABTC-1m-2021-03.csv" in man["fetched"] and "AAABTC-1m-2021-04.csv" in man["errors"]
        st = bf.fetch_aggtrades("AAA", ["2021-03-03"], os.path.join(tmp, "agg")); assert list(st.values())[0].startswith("error")
        # prepare(): table already present -> no clone; klines fetched into <root>/crosspump/binance_klines
        root = os.path.join(tmp, "root"); shutil.copytree(os.path.join(FROOT, "crosspump", "ardia"), os.path.join(root, "crosspump", "ardia"))
        d6_ardia_pd.prepare(root, {"klines": True, "aggtrades": False, "onchain": False, "workers": 2, "max_pumps": None})
        assert os.path.exists(os.path.join(root, "crosspump", "binance_klines", "AAABTC-1m-2021-03.csv"))
        assert bf.months_for_anchors([int(np.datetime64("2021-03-03T12:00", "ns").astype(np.int64))], 2.0, 30.0) == ["2021-03", "2021-04"]
        os.environ.pop("ETHERSCAN_API_KEY", None)
        try:
            bf.prepare_onchain(root, "ardia_pd", [], {}); raise AssertionError("expected RuntimeError without ETHERSCAN_API_KEY")
        except RuntimeError as e:
            assert "ETHERSCAN_API_KEY" in str(e)
    finally:
        bf.BINANCE_VISION = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_fabrication_in_loaders():
    """Rule zero: the only random draw in the loader path is the seeded choice of CONTROL WINDOWS (times / coins)."""
    import re
    for f in ("d1_hu_pd.py", "d2_sapienza_pd.py", "d3_xu_pumpolymp.py", "d6_ardia_pd.py"):
        src = open(os.path.join(ROOT, "crosspump", "data", "loaders", f)).read()
        assert not re.search(r"np\.random|default_rng|rng\.|random\.", src), f
    src = open(os.path.join(ROOT, "crosspump", "data", "loaders", "binance_fetch.py")).read()
    uses = re.findall(r"rng\.\w+\(", src)
    assert set(uses) <= {"rng.uniform(", "rng.choice("} and "np.random.default_rng(int(c[\"seed\"]))" in src, uses


TESTS = [test_registry_and_configs, test_hu_pd, test_hu_pd_message_modes, test_sapienza_pd, test_xu_pumpolymp, test_ardia_pd, test_controls_and_determinism,
         test_orderflow_sources, test_missing_raw_files, test_fetch_and_prepare_offline, test_no_fabrication_in_loaders]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-k", default=None)
    a = ap.parse_args()
    ensure_fixtures()
    failed = []
    for t in TESTS:
        if a.k and a.k not in t.__name__:
            continue
        t0 = time.time()
        try:
            t(); print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:  # noqa: BLE001
            failed.append(t.__name__); print(f"FAIL {t.__name__}\n{traceback.format_exc()}", flush=True)
    n = len([t for t in TESTS if not a.k or a.k in t.__name__])
    print(f"\n{n - len(failed)}/{n} passed" + (f"; FAILED: {failed}" if failed else " -- all group-A loader tests passed (SYNTHETIC fixtures only)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
