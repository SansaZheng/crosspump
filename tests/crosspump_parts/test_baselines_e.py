"""Group-E baseline tests (B6-B10), plain-script mode:  python3 tests/crosspump_parts/test_baselines_e.py [-k substring]

Everything runs on the SYNTHETIC cross-platform testbed (xp_synth, crosspump/data/synthetic.py); no real data is read.
Per baseline (smoke config from configs/crosspump/baselines/<key>.yaml):
    1. _fit on episodes 0-44 (val 45-54), _score_rows on every row of the test episodes 55-71;
    2. scores are finite wherever the modality the baseline reads has been observed (causal coverage);
    3. AUC-PR on the test rows exceeds the positive rate (abstentions ranked last);
    4. causality: for one test pump episode and two cut bars t0, every stream event with avail > t0 is deleted and
       the bar features after t0 are blanked; the baseline is rebuilt and refitted on the truncated panel and the
       scores at rows <= t0 must be identical (atol 1e-6);
    5. the full scaffold path (fit with loaders is not needed) is covered by gate calibration on the val rows.
Plus unit checks of the causal CA-CFAR against a literal port of the notebook's left-window rule, and of the registry.
"""
from __future__ import annotations
import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from warn2trade.utils import DATASETS                                  # noqa: E402
from warn2trade.backtest.metrics import auc_pr                         # noqa: E402
import crosspump.data.synthetic                                        # noqa: E402,F401  (registers xp_synth)
from crosspump.data.streams import get_streams                         # noqa: E402
from crosspump.baselines import _feats_e as FE                         # noqa: E402
from crosspump.baselines.b6_bolz24 import Bolz24Baseline               # noqa: E402
from crosspump.baselines.b7_perseus25 import Perseus25Baseline         # noqa: E402
from crosspump.baselines.b8_pumpsense26 import PumpSense26Baseline     # noqa: E402
from crosspump.baselines.b9_moura26 import Moura26Baseline             # noqa: E402
from crosspump.baselines.b10_lroo26 import Lroo26Baseline              # noqa: E402

TRAIN, VAL, TEST = np.arange(0, 45), np.arange(45, 55), np.arange(55, 72)
BASELINES = [("bolz24", Bolz24Baseline, "xp_bolz24"), ("perseus25", Perseus25Baseline, "xp_perseus25"), ("pumpsense26", PumpSense26Baseline, "xp_pumpsense26"),
             ("moura26", Moura26Baseline, "xp_moura26"), ("lroo26", Lroo26Baseline, "xp_lroo26")]
REQUIRED_KEYS = ["baseline", "alpha", "gate", "lookback", "latency", "hold_index", "n_exec", "w_max", "model", "smoke"]
MAX_SECONDS = 90.0


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def smoke_cfg(key: str, panel):
    with open(os.path.join(ROOT, "configs", "crosspump", "baselines", f"{key}.yaml")) as f:
        raw = yaml.safe_load(f)
    for k in REQUIRED_KEYS:
        assert k in raw, f"{key}.yaml misses '{k}'"
    cfg = _merge({k: v for k, v in raw.items() if k != "smoke"}, raw.get("smoke") or {})
    cfg["panel"] = panel
    return raw, cfg


def build(cls, key, panel):
    raw, cfg = smoke_cfg(key, panel)
    torch.manual_seed(0); np.random.seed(0)
    b = cls(cfg, panel.dims(), panel.meta["horizons"], torch.device("cpu"))
    b._fit(panel, TRAIN, VAL)
    return raw, b


def truncated(panel, episode: int, t0: int):
    """Copy of the panel in which nothing after bar t0 of `episode` exists: stream events with avail > t0 are deleted
    and the bar features of rows > t0 are blanked (labels are untouched: `episode` is a test episode, never trained on)."""
    p = copy.copy(panel)
    p.meta = dict(panel.meta)
    es = get_streams(panel)
    keep = ~((es.asset == episode) & (es.avail > t0))
    p.streams = es.subset(keep)
    pf = np.array(panel.price_feat, copy=True)
    pf[t0 + 1 :, episode] = np.nan
    p.price_feat = pf
    return p, int((~keep).sum())


def modality_rows(b, key, t, i):
    """Rows on which the baseline must produce a finite score (its modality has been observed by bar t)."""
    seen = FE.seen_mask(b.counts)
    if key == "bolz24":
        return np.ones(len(t), dtype=bool)                              # bar features exist on every synthetic row (warm-up < lookback)
    if key == "lroo26":
        return seen.any(-1)[t, i]
    if key == "perseus25":
        return seen[(t // b.k) * b.k, i, 0]
    return seen[t, i, 0]


def run_baseline(key, cls, reg, panel):
    t_start = time.time(); c_start = time.process_time()
    raw, b = build(cls, key, panel)
    assert raw["baseline"] == reg == b.name, (raw["baseline"], reg, b.name)
    assert b.modalities <= {"social", "wallet", "orderflow", "text", "graph"} and b.paper and b.year and b.unspecified
    t, i = b.rows_of(TEST)
    s = np.asarray(b._score_rows(t, i), dtype=np.float64)
    fit_score_s = time.time() - t_start; fit_score_cpu = time.process_time() - c_start
    assert s.shape == t.shape
    need = modality_rows(b, key, t, i)
    assert need.mean() > 0.9, f"{key}: modality coverage unexpectedly low ({need.mean():.3f})"
    assert np.isfinite(s[need]).all(), f"{key}: {int((~np.isfinite(s[need])).sum())} non-finite scores where the modality exists"
    y = panel.labels[t, i].astype(int)
    fin = np.isfinite(s)
    s_rank = np.where(fin, s, np.nanmin(s) - 1.0)                       # abstain = ranked last
    ap = auc_pr(y, s_rank); base = float(y.mean())
    assert ap > base, f"{key}: AUC-PR {ap:.4f} <= positive rate {base:.4f}"
    # --- gate calibration path of the scaffold on the validation rows (no loaders needed)
    tv, iv = b.rows_of(VAL)
    b.gate.calibrate(b._score_rows(tv, iv), labels=panel.labels[tv, iv])
    # --- causality: delete the future of one test pump episode, rebuild + refit, compare the past
    pumps = [int(e) for e in TEST if panel.meta["is_pump"][e]]
    ep = pumps[0]
    worst = 0.0; n_del = 0
    for t0 in (130, 145):
        p2, nd = truncated(panel, ep, t0)
        assert nd > 0, "nothing was deleted: the causality check would be vacuous"
        _, b2 = build(cls, key, p2)
        rows = np.arange(b.L, t0 + 1); cols = np.full(len(rows), ep)
        a = np.asarray(b._score_rows(rows, cols), dtype=np.float64); c = np.asarray(b2._score_rows(rows, cols), dtype=np.float64)
        assert (np.isfinite(a) == np.isfinite(c)).all(), f"{key}: abstention pattern at rows <= {t0} changed after deleting the future"
        d = float(np.nanmax(np.abs(a - c))) if np.isfinite(a).any() else 0.0
        assert d <= 1e-6, f"{key}: scores at rows <= {t0} changed by {d:.3e} after deleting events with avail > {t0} (PIT violation)"
        # other test episodes are untouched as well
        o = np.asarray(b2._score_rows(t, i), dtype=np.float64); other = i != ep
        assert np.allclose(o[other], s[other], atol=1e-6, equal_nan=True), f"{key}: other episodes changed"
        worst = max(worst, d); n_del += nd
    total = time.time() - t_start
    # CPU seconds, not wall-clock: the shared machine runs other tracks concurrently, wall time is reported only
    assert fit_score_cpu < MAX_SECONDS, f"{key}: smoke fit+score took {fit_score_cpu:.1f} CPU-s"
    print(f"  {reg:<16s} status={b.status:<22s} AUC-PR={ap:.4f} (pos rate {base:.4f})  finite={fin.mean():.3f}  causality max|d|={worst:.1e} "
          f"({n_del} events deleted)  fit+score={fit_score_cpu:.1f} CPU-s ({fit_score_s:.1f}s wall) total={total:.1f}s wall")
    extra = {k: b.diag[k] for k in ("backend", "fallback_reason", "combine", "alpha_cfar", "train_graphs", "n_features") if k in b.diag}
    if extra:
        print(f"      diag: {extra}")
    return {"key": key, "ap": ap, "base": base, "seconds": fit_score_cpu, "status": b.status}


# ---------------------------------------------------------------------------------------------- unit checks
def _notebook_cfar_left(signal, n_train, n_guard, p_fa):
    """Literal port of ca_cfar_1d (02_anomaly_detection.ipynb, cell 17) restricted to the LEFT window."""
    n = len(signal); thr = np.full(n, np.inf)
    alpha = n_train * ((p_fa ** (-1.0 / n_train)) - 1)
    for k in range(n):
        left_start = k - n_train - n_guard; left_end = k - n_guard - 1
        left = signal[max(0, left_start) : left_end + 1] if left_end >= 0 else np.array([])
        if len(left):
            thr[k] = alpha * left.mean()
    return thr


def test_cfar_matches_notebook():
    rng = np.random.default_rng(0)
    sig = rng.poisson(3.0, size=(60, 4)).astype(float)
    assert abs(FE.cfar_alpha(5, 0.05) - 5 * (0.05 ** (-0.2) - 1)) < 1e-12 and abs(FE.cfar_alpha(5, 0.05) - 4.1028) < 1e-3
    cut, thr, nv = FE.ca_cfar_causal(sig, cell=1, n_train=5, n_guard=1, p_fa=0.05)
    assert np.allclose(cut, sig)
    for j in range(sig.shape[1]):
        ref = _notebook_cfar_left(sig[:, j], 5, 1, 0.05)
        got = np.where(np.isnan(thr[:, j]), np.inf, thr[:, j])
        assert np.allclose(got, ref), "causal CA-CFAR differs from the notebook's left-window rule"
    # cells of 3 bars: the threshold at t only uses bars <= t - 3 * (1 + n_guard)
    cut3, thr3, _ = FE.ca_cfar_causal(sig, cell=3, n_train=5, n_guard=1, p_fa=0.05)
    sig2 = sig.copy(); sig2[40:] += 100.0
    cut3b, thr3b, _ = FE.ca_cfar_causal(sig2, cell=3, n_train=5, n_guard=1, p_fa=0.05)
    assert np.allclose(thr3[:46], thr3b[:46], equal_nan=True) and np.allclose(cut3[:40], cut3b[:40])
    print("  ca_cfar_causal == notebook left-window rule (cell 17), alpha(5, 0.05) = %.4f" % FE.cfar_alpha(5, 0.05))


def test_feature_helpers_causal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((80, 3, 2)); x2 = x.copy(); x2[50:] = rng.standard_normal((30, 3, 2)) * 10
    z, z2 = FE.causal_ewma_z(x, warmup=10, halflife=20), FE.causal_ewma_z(x2, warmup=10, halflife=20)
    assert np.allclose(z[:50], z2[:50], equal_nan=True) and np.isnan(z[:10]).all() and np.isfinite(z[10:]).all()
    a = rng.random((30, 2))
    assert np.allclose(FE.trailing_sum(a, 4)[10], a[7:11].sum(0)) and np.allclose(FE.trailing_max(a, 4)[10], a[7:11].max(0))
    print("  causal_ewma_z / trailing_sum / trailing_max are causal and exact")


def test_registry():
    from warn2trade.baselines.event_base import BASELINES as EVENT_BASELINES
    for _, cls, reg in BASELINES:
        got = EVENT_BASELINES.get(reg)
        assert got is cls, f"{reg} is not registered to {cls.__name__}"
        for k in ("name", "paper", "year", "venue", "official_repo", "status", "modalities", "unspecified"):
            assert hasattr(cls, k), f"{cls.__name__} misses class attr {k}"
    for key, _, _ in BASELINES:
        assert os.path.exists(os.path.join(ROOT, "crosspump", "baselines", "sheets", f"{key}.md")), f"missing sheet {key}.md"
    print("  registry keys, class attributes and convention sheets present")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-k", default=""); a = ap.parse_args()
    t0 = time.time()
    print("[unit]")
    for fn in (test_cfar_matches_notebook, test_feature_helpers_causal, test_registry):
        if a.k in fn.__name__ or not a.k:
            fn()
    panel = DATASETS.get("xp_synth")({"seed": 1, "n_pumps": 24, "n_controls": 48}).build_panel()
    assert panel.N == 72 and get_streams(panel) is not None
    n_tp = int(panel.meta["is_pump"][TEST].sum())
    assert n_tp >= 2, "the test split needs pump episodes"
    print(f"[baselines] synthetic panel T={panel.T} N={panel.N} (train 0-44 / val 45-54 / test 55-71, {n_tp} test pumps)")
    res = []
    for key, cls, reg in BASELINES:
        if a.k and a.k not in key and a.k not in reg:
            continue
        res.append(run_baseline(key, cls, reg, panel))
    print(f"ALL GROUP-E BASELINE TESTS PASSED ({len(res)} baselines, {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
