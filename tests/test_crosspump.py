"""CrossPump unit + smoke tests, plain-script mode:  python3 tests/test_crosspump.py [-k substring]
Everything here runs on the synthetic testbed (xp_synth); no real data is touched."""
from __future__ import annotations
import argparse, os, sys, tempfile, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warn2trade.utils import DATASETS
from warn2trade.backtest.metrics import auc_pr
import crosspump.data.synthetic  # noqa: F401
import crosspump.baselines.common, crosspump.models.crosspump  # noqa: F401
from crosspump.data.streams import EventStreams, get_streams, derive_orderflow_events
from crosspump.models.gate import DelayedACIGate
from crosspump.models.hawkes import MarkedHawkes
from crosspump.models.crosspump import CrossPump
from crosspump.eval.splits import EpisodeSplitter, EpisodeDataset
from crosspump.eval.metrics import lead_time, precision_at_far
from crosspump.eval.follower import follower_overlay
from crosspump.eval.runner import XPRunner

CFG = {"seed": 1, "n_pumps": 24, "n_controls": 48}
_P = {}


def panel():
    if "p" not in _P:
        _P["p"] = DATASETS.get("xp_synth")(CFG).build_panel()
    return _P["p"]


def test_panel_and_streams():
    p = panel(); p.validate(); es = get_streams(p)
    assert es is not None and len(es) > 1000
    es.validate(p.T, p.N)
    assert p.meta["split_axis"] == "assets" and p.meta["is_pump"].sum() == 24
    cov = es.coverage(p.N)["episodes_with_platform"]
    assert cov["social"] == p.N and cov["wallet"] < p.N, cov          # missing wallet modality on some episodes
    i = int(np.nonzero(p.meta["is_pump"])[0][0]); pre = p.meta["pre_bars"]
    assert p.labels[pre - p.meta["lead"]: p.meta["peak_row"][i] + 1, i].all() and p.labels[:, ~p.meta["is_pump"]].sum() == 0
    assert np.nanmax(p.bars_to_peak[:, i]) == p.meta["peak_row"][i] - (pre - p.meta["lead"])


def test_pit_rule_before_and_rasterise_agree():
    p = panel(); es = get_streams(p)
    r = es.rasterise(p.T, p.N, pit=True)
    for i in (0, 5, 17):
        for t in (10, 60, 150):
            assert len(es.before(i, t)) == int(r[: t + 1, i].sum()), (i, t)
    of = derive_orderflow_events(p)
    assert (of.avail >= of.time).all() and (of.platform == 2).all()


def test_gate_alarm_rate_under_shift():
    rng = np.random.default_rng(0)
    g = DelayedACIGate(alpha=0.05, gamma=0.01, tau=3).calibrate(rng.standard_normal(500))
    al = [g.step(float(x), t, 0.0)[0] for t, x in enumerate(rng.standard_normal(4000) + 0.8)]
    assert abs(np.mean(al[1000:]) - 0.05) < 0.02, np.mean(al[1000:])


def test_hawkes_recovers_cross_excitation():
    p = panel(); es = get_streams(p); ip = p.meta["is_pump"]
    hp = MarkedHawkes().fit(es, np.nonzero(ip)[0][:12], p.T, iters=80)
    hc = MarkedHawkes().fit(es, np.nonzero(~ip)[0][:20], p.T, iters=80)
    assert hp.history[-1] < hp.history[0]
    bp, bc = hp.branching_matrix(), hc.branching_matrix()
    assert bp[1, 2] > 0.1 and bp[0, 2] > bc[0, 2], (bp, bc)          # wallet -> orderflow and social -> orderflow planted in pumps
    test = np.arange(p.N - 20, p.N)
    llr = np.cumsum(hp.row_state(es, p.T, p.N, test)["ll_inc"] - hc.row_state(es, p.T, p.N, test)["ll_inc"], 0)[p.meta["pre_bars"] - 1]
    assert llr[test][ip[test]].mean() > llr[test][~ip[test]].mean() + 5, llr[test]


def test_hawkes_row_state_is_pit():
    p = panel(); es = get_streams(p)
    h = MarkedHawkes().fit(es, np.arange(10), p.T, iters=20)
    i, t0 = 30, 100
    full = h.row_state(es, p.T, p.N, [i])
    keep = ~((es.asset == i) & (es.avail > t0))
    trunc = h.row_state(es.subset(keep), p.T, p.N, [i])
    assert np.allclose(full["lam"][: t0 + 1, i], trunc["lam"][: t0 + 1, i]) and np.allclose(full["ll_inc"][: t0 + 1, i], trunc["ll_inc"][: t0 + 1, i])


def _fold(p):
    return list(EpisodeSplitter(p.meta["anchor_ns"], 0.6, 0.15, purge_ns=int(p.T * 5 * 60 * 1e9), n_folds=1))[0]


def test_splitter_is_chronological_and_purged():
    p = panel(); f = _fold(p); a = p.meta["anchor_ns"]
    assert a[f.train].max() < a[f.val].min() < a[f.test].min()
    assert a[f.test].min() - a[f.val].max() >= int(p.T * 5 * 60 * 1e9)
    assert not set(f.train) & set(f.test)


def test_crosspump_detects_and_warns():
    p = panel(); f = _fold(p)
    m = CrossPump({"panel": p, "lookback": 24, "alpha": 0.01, "model": {"hawkes_iters": 60}}, p.dims(), p.meta["horizons"], torch.device("cpu"))
    m._fit(p, f.train, f.val)
    ds = EpisodeDataset(p, f.test, 24, p.meta["horizons"]); t, i = ds.index[:, 0], ds.index[:, 1]
    s = m._score_rows(t, i); y = p.labels[t, i]; ok = np.isfinite(s)
    c = CrossPump({"panel": p, "lookback": 24, "alpha": 0.01, "model": {"head_features": "counts_only"}}, p.dims(), p.meta["horizons"], torch.device("cpu"))
    c._fit(p, f.train, f.val); sc = c._score_rows(t, i)
    ap, apc = auc_pr(y[ok], s[ok]), auc_pr(y[ok], sc[ok])
    assert ap > 5 * y.mean() and apc > 0, (ap, apc, y.mean())      # no claim that the Hawkes head beats counts-only: see docs/crosspump_results.md
    tv, iv = m.rows_of(f.val); sv = m._score_rows(tv, iv); yv = p.labels[tv, iv]
    alarm = ok & (s > np.quantile(sv[(yv == 0) & np.isfinite(sv)], 0.99))
    rows = pd.DataFrame({"t": t, "asset": i, "y": y})
    lt = lead_time(rows, alarm, p.meta)
    assert lt["event_recall"] >= 0.8, lt
    fo = follower_overlay(rows, alarm, p, hold_late=12)
    assert fo["loss_avoided_bps"] > 0 and fo["follower_pnl_bps"] < 0, fo
    pf = precision_at_far(y[ok], s[ok], s_cal=sv, y_cal=yv)
    assert pf["P@FAR0.01"] > 0.1, pf
    _P["cp_scores"] = (m, f, t, i, s)


def test_crosspump_scores_are_pit():
    p = panel(); f = _fold(p)
    kw = lambda pan: CrossPump({"panel": pan, "lookback": 24, "alpha": 0.01, "model": {"hawkes_iters": 15}}, pan.dims(), pan.meta["horizons"], torch.device("cpu"))
    i0, t0 = int(f.test[0]), 120
    m = kw(p); m._fit(p, f.train, f.val)
    import copy
    q = copy.copy(p); es = get_streams(p)
    q.streams = es.subset(~((es.asset == i0) & (es.avail > t0)))
    q.price_feat = p.price_feat.copy(); q.price_feat[t0 + 1:, i0] = 0.0
    m2 = kw(q); m2._fit(q, f.train, f.val)
    t = np.arange(24, t0 + 1); i = np.full(len(t), i0)
    assert np.allclose(m._score_rows(t, i), m2._score_rows(t, i), atol=1e-6, equal_nan=True)


def test_runner_end_to_end():
    p = panel(); out = tempfile.mkdtemp()
    R = XPRunner(p, p.meta["horizons"], p.meta["bars_per_year"], {"lookback": 24, "latency": 1, "exec_max": 4}, {"train_frac": 0.6, "val_frac": 0.15, "n_folds": 1},
                 {"commission_bps": 5.0}, {"cost_grid_bps": [1, 5, 30]}, torch.device("cpu"), out, 42, "xp_synth")
    a = R.run("crosspump", {"alpha": 0.01, "model": {"hawkes_iters": 40}})["agg"]
    b = R.run("xp_counts_rule", {"alpha": 0.01})["agg"]
    assert a["AUC_PR_rows"] > b["AUC_PR_rows"] and a["alarm_budget_error"] < 0.03 and np.isfinite(a["loss_avoided_bps"]), (a, b)
    d = os.path.join(out, "xp_synth", "crosspump", "main", "seed42")
    assert {"fold0.json", "trades_fold0.csv", "alerts_fold0.csv"} <= set(os.listdir(d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-k", default=""); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and a.k in n]
    fails = 0; t00 = time.time()
    for n, f in tests:
        t0 = time.time()
        try:
            f(); print(f"PASS {n} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            fails += 1; print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed in {time.time() - t00:.0f}s"); sys.exit(1 if fails else 0)
