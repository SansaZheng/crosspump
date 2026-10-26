"""Plain-script tests for the group-D baselines B1-B5 (run: python3 tests/crosspump_parts/test_baselines_d.py [-k name]).

Everything runs on the SYNTHETIC cross-platform testbed xp_synth (crosspump/data/synthetic.py); no real data is read.
Per baseline: smoke config from configs/crosspump/baselines/<key>.yaml, _fit on episodes 0-44, scores on the rows of the
test episodes 55-71, finite scores where the modality exists, AUC-PR above the positive rate, and a causality check:
scores at rows <= t0 of one episode must not change when every stream event of that episode with avail > t0 (and its bar
features after t0) is deleted and the baseline is re-built and re-fitted on the truncated panel.
"""
from __future__ import annotations
import argparse, copy, os, sys, time
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from warn2trade.utils import DATASETS                                    # noqa: E402
from warn2trade.backtest.metrics import auc_pr                           # noqa: E402
import crosspump.data.synthetic                                          # noqa: E402,F401
from crosspump.data.streams import get_streams, attach_streams, EventStreams   # noqa: E402
from crosspump.baselines.common import XPBaseline                        # noqa: E402
from crosspump.baselines import b1_mirtaheri21, b2_nghiem21, b3_victor21, b4_lamorgia23, b5_hu23   # noqa: E402
from warn2trade.baselines.event_base import BASELINES                    # noqa: E402

CLASSES = {"mirtaheri21": b1_mirtaheri21.Mirtaheri21Baseline, "nghiem21": b2_nghiem21.Nghiem21Baseline, "victor21": b3_victor21.Victor21Baseline,
           "lamorgia23": b4_lamorgia23.LaMorgia23Baseline, "hu23": b5_hu23.Hu23Baseline}
TRAIN, VAL, TEST = np.arange(0, 45), np.arange(45, 55), np.arange(55, 72)
_P = {}


def panel():
    if "p" not in _P:
        _P["p"] = DATASETS.get("xp_synth")({"seed": 1, "n_pumps": 24, "n_controls": 48}).build_panel()
    return _P["p"]


def smoke_cfg(key: str, pan) -> dict:
    with open(os.path.join(ROOT, "configs", "crosspump", "baselines", f"{key}.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["baseline"] == f"xp_{key}" and cfg["alpha"] == 0.01 and cfg["gate"] == {"type": "delayed_aci", "gamma": 0.005, "window": 2000}, cfg
    for k, v in {"lookback": 48, "latency": 1, "hold_index": 2, "n_exec": 4, "w_max": 1.0}.items():
        assert cfg[k] == v, (key, k, cfg[k])
    assert isinstance(cfg.get("model"), dict) and isinstance(cfg.get("smoke", {}).get("model"), dict), "model / smoke.model blocks required"
    smoke = cfg.pop("smoke")
    cfg["model"] = {**cfg["model"], **smoke.get("model", {})}
    cfg.update({k: v for k, v in smoke.items() if k != "model"})
    cfg.update({"panel": pan, "lookback": 48, "alpha": 0.01})
    return cfg


def build(key: str, pan) -> XPBaseline:
    torch.manual_seed(0); np.random.seed(0)
    b = CLASSES[key](smoke_cfg(key, pan), pan.dims(), pan.meta["horizons"], torch.device("cpu"))
    b._fit(pan, TRAIN, VAL)
    return b


def truncate(pan, episode: int, t0: int):
    """Copy of the panel in which episode `episode` knows nothing after row t0: stream events with avail > t0 deleted and
    bar features of rows > t0 set to NaN. Other episodes are untouched."""
    q = copy.copy(pan)
    q.meta = dict(pan.meta)
    q.price_feat = pan.price_feat.copy(); q.price_feat[t0 + 1 :, episode] = np.nan
    es = get_streams(pan)
    keep = ~((es.asset == episode) & (es.avail > t0))
    attach_streams(q, EventStreams(es.asset[keep], es.time[keep], es.platform[keep], es.mark[keep], es.avail[keep], es.source[keep], dict(es.meta)))
    assert len(get_streams(q)) < len(es) and len(get_streams(pan)) == len(es)
    return q


def check(key: str) -> dict:
    pan = panel()
    cls = CLASSES[key]
    t_start, c_start = time.time(), time.process_time()
    # ---- declarations and registry
    assert issubclass(cls, XPBaseline) and cls.name == f"xp_{key}" and BASELINES.get(f"xp_{key}") is cls
    assert cls.paper and cls.year >= 2021 and cls.venue and cls.status and isinstance(cls.unspecified, list) and len(cls.unspecified) >= 3
    assert cls.modalities and cls.modalities <= {"social", "wallet", "orderflow", "text", "graph"}
    assert os.path.exists(os.path.join(ROOT, "crosspump", "baselines", "sheets", f"{key}.md")), "convention sheet missing"
    assert "fit" not in cls.__dict__ and "predict" not in cls.__dict__, "only _fit / _score_rows may be implemented"
    # ---- fit + score
    b = build(key, pan)
    t, i = b.rows_of(TEST)
    s = b._score_rows(t, i)
    y = pan.labels[t, i]
    assert s.shape == t.shape
    has_wallet = np.array([e["has_wallet"] for e in pan.meta["episodes"]])
    exists = has_wallet[i] if cls.modalities == {"wallet"} else np.ones(len(t), dtype=bool)
    assert np.isfinite(s[exists]).all(), f"{key}: non-finite scores where the modality exists ({(~np.isfinite(s[exists])).sum()} rows)"
    if cls.modalities == {"wallet"}:
        assert np.isnan(s[~exists]).all(), "episodes without a wallet stream must abstain (NaN)"
    fin = np.isfinite(s)
    ap, base = auc_pr(y[fin], s[fin]), float(y[fin].mean())
    assert y[fin].sum() > 0 and ap > base, f"{key}: AUC-PR {ap:.4f} not above the positive rate {base:.4f}"
    # ---- scaffold integration: gate calibration + predict on a batch
    cal_t, cal_i = b.rows_of(VAL)
    b.gate.calibrate(b._score_rows(cal_t, cal_i), labels=pan.labels[cal_t, cal_i])
    sel = np.nonzero(i == TEST[0])[0]
    out = b.predict({"t": torch.as_tensor(t[sel]), "asset": torch.as_tensor(i[sel]), "y": torch.as_tensor(y[sel].astype(np.float32))})
    assert set(out) >= {"a_prob", "w", "abstain", "hold_p", "kappa_p", "score"} and out["w"].shape == (len(sel),)
    # ---- causality: delete the future of one pump episode (with a wallet stream) and re-fit
    is_pump = np.asarray(pan.meta["is_pump"])
    cand = [int(e) for e in TEST if is_pump[e] and has_wallet[e]]
    assert cand, "no test pump episode with a wallet stream"
    worst = 0.0
    for e, t0 in ((cand[0], 120), (cand[-1], pan.meta["pre_bars"] + 1)):
        rows = np.arange(b.L, t0 + 1); col = np.full(len(rows), e)
        ref = b._score_rows(rows, col)
        b2 = build(key, truncate(pan, e, t0))
        got = b2._score_rows(rows, col)
        assert np.array_equal(np.isnan(ref), np.isnan(got)), f"{key}: abstention pattern changed after deleting the future (episode {e}, t0 {t0})"
        d = float(np.nanmax(np.abs(ref - got))) if np.isfinite(ref).any() else 0.0
        assert d <= 1e-6, f"{key}: PIT violation, scores at rows <= {t0} changed by {d:.3g} when later events were deleted (episode {e})"
        worst = max(worst, d)
        assert np.isfinite(ref).any(), f"{key}: causality check is vacuous (all NaN)"
    dt, cpu = time.time() - t_start, time.process_time() - c_start
    # budget on CPU time of this process (1 fit + 2 re-fits + scoring): wall-clock is reported but not asserted, because
    # several tracks share this machine and the wall time of an idle-waiting process says nothing about the baseline
    assert cpu < 90, f"{key}: smoke run used {cpu:.1f} s CPU (> 90 s)"
    return {"key": key, "auc_pr": ap, "pos_rate": base, "finite": float(fin.mean()), "pit_max_diff": worst, "seconds": dt, "cpu": cpu, "status": b.status}


def test_no_label_access_outside_fit():
    """Static check: panel.labels is never read by the baseline modules (labels only through labels_for inside _fit)."""
    for mod in (b1_mirtaheri21, b2_nghiem21, b3_victor21, b4_lamorgia23, b5_hu23):
        src = open(mod.__file__).read()
        assert ".labels[" not in src and "panel.labels" not in src and "bars_to_peak" not in src and "peak_row" not in src and "is_pump" not in src, mod.__name__
        body = src.split("def _score_rows")[1]
        assert "labels_for" not in body, mod.__name__


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("-k", default=""); a = ap_.parse_args()
    torch.set_num_threads(1)                                              # deterministic and comparable CPU budget
    pan = panel()
    print(f"[synthetic xp_synth] T={pan.T} N={pan.N} pumps={int(np.sum(pan.meta['is_pump']))} events={len(get_streams(pan))} train/val/test={len(TRAIN)}/{len(VAL)}/{len(TEST)}")
    test_no_label_access_outside_fit(); print("ok   static label-access check")
    rows = []
    for key in CLASSES:
        if a.k and a.k not in key:
            continue
        r = check(key); rows.append(r)
        print(f"ok   xp_{r['key']:<12} AUC-PR {r['auc_pr']:.3f} (positive rate {r['pos_rate']:.3f})  finite {r['finite']:.2f}  PIT max diff {r['pit_max_diff']:.1e}  cpu {r['cpu']:.1f} s / wall {r['seconds']:.1f} s  [{r['status']}]")
    print(f"ALL PASSED ({len(rows)} baselines, cpu {sum(r['cpu'] for r in rows):.1f} s, wall {sum(r['seconds'] for r in rows):.1f} s)")
