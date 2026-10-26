"""CrossPump metrics on the (t, asset, y, score, alarm) rows of test episodes.

    lead_time          per pump episode: first alarm row vs anchor row and vs peak row, in bars and minutes (positive = early)
    precision_at_far   precision / recall at thresholds set so that the per-bar false-alarm rate on NORMAL rows equals a budget
                       (thresholds taken from calibration rows when given, else from the test rows themselves - reported as 'oracle')
    event_metrics      event recall (share of pump episodes with >= 1 alarm inside [anchor - early_window, peak]), false alarms per
                       control episode-day, alarm rate on control rows, alarm-budget error |rate - alpha|
    hit_rate_at_k      cross-sectional target-coin ranking at the anchor row (Hu et al. 2023 task) when several episodes share an
                       anchor time: HR@k over the pump episode's rank among its same-time controls
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd


def _first_alarm_rows(rows: pd.DataFrame, alarm: np.ndarray, lo: int, hi: int) -> Dict[int, int]:
    """asset -> first row t in [lo, hi] with an alarm (only rows present in `rows`)."""
    out: Dict[int, int] = {}
    sel = alarm & (rows["t"].to_numpy() >= lo) & (rows["t"].to_numpy() <= hi)
    for a, t in zip(rows["asset"].to_numpy()[sel], rows["t"].to_numpy()[sel]):
        a = int(a); t = int(t)
        if a not in out or t < out[a]:
            out[a] = t
    return out


def lead_time(rows: pd.DataFrame, alarm: np.ndarray, meta: Dict[str, Any], early_window: int = 0, assets: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """rows: DataFrame with t, asset; alarm: bool per row. meta from the panel (pre_bars, peak_row, is_pump, bar_minutes).
    Lead = anchor_row - first alarm row inside [lookback, peak] (bars); early_window limits how far before the anchor an alarm
    still counts (0 = whole window)."""
    pre, bm = int(meta["pre_bars"]), float(meta.get("bar_minutes", 1))
    is_pump, peak = np.asarray(meta["is_pump"]), np.asarray(meta["peak_row"])
    pumps = [a for a in np.unique(rows["asset"].to_numpy()) if is_pump[int(a)]]
    if assets is not None:
        pumps = [a for a in pumps if a in set(int(x) for x in assets)]
    lo = pre - early_window if early_window > 0 else 0
    firsts = {}
    for a in pumps:
        f = _first_alarm_rows(rows[rows["asset"] == a], alarm[rows["asset"].to_numpy() == a], lo, int(peak[int(a)]))
        if int(a) in f:
            firsts[int(a)] = f[int(a)]
    n = len(pumps)
    if n == 0:
        return {"n_pump_episodes": 0}
    lead_anchor = np.array([pre - firsts[a] for a in firsts], dtype=float)
    lead_peak = np.array([peak[a] - firsts[a] for a in firsts], dtype=float)
    out = {"n_pump_episodes": n, "event_recall": len(firsts) / n, "pre_anchor_recall": float(np.mean([pre - firsts[a] > 0 for a in firsts])) * (len(firsts) / n) if firsts else 0.0}
    if len(firsts):
        out.update({"lead_anchor_bars_mean": float(lead_anchor.mean()), "lead_anchor_bars_median": float(np.median(lead_anchor)), "lead_anchor_min_mean": float(lead_anchor.mean() * bm),
                    "lead_peak_bars_mean": float(lead_peak.mean()), "lead_peak_min_median": float(np.median(lead_peak) * bm), "share_early_vs_anchor": float((lead_anchor > 0).mean())})
    else:
        out.update({"lead_anchor_bars_mean": float("nan"), "lead_anchor_bars_median": float("nan"), "lead_anchor_min_mean": float("nan"), "lead_peak_bars_mean": float("nan"), "lead_peak_min_median": float("nan"), "share_early_vs_anchor": float("nan")})
    return out


def precision_at_far(y: np.ndarray, s: np.ndarray, far_grid: Sequence[float] = (0.001, 0.005, 0.01, 0.05), s_cal: Optional[np.ndarray] = None,
                     y_cal: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Threshold = (1 - far) quantile of scores on NORMAL rows (calibration rows if given, else test rows = 'oracle')."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    out: Dict[str, float] = {"far_threshold_source": 0.0 if s_cal is None else 1.0}
    ref_s, ref_y = (s[ok], y[ok]) if s_cal is None else (np.asarray(s_cal, dtype=np.float64), np.asarray(y_cal))
    ref = ref_s[(ref_y == 0) & np.isfinite(ref_s)]
    for far in far_grid:
        if ref.size == 0:
            out[f"P@FAR{far:g}"] = float("nan"); out[f"R@FAR{far:g}"] = float("nan"); continue
        thr = np.quantile(ref, 1 - far)
        pred = ok & (s > thr)
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        out[f"P@FAR{far:g}"] = float(tp / max(tp + fp, 1)) if pred.any() else float("nan")
        out[f"R@FAR{far:g}"] = float(tp / max((y == 1).sum(), 1))
        out[f"realisedFAR@{far:g}"] = float(fp / max((y == 0).sum(), 1))
    return out


def event_metrics(rows: pd.DataFrame, alarm: np.ndarray, meta: Dict[str, Any], alpha: Optional[float] = None) -> Dict[str, float]:
    is_pump = np.asarray(meta["is_pump"]); bm = float(meta.get("bar_minutes", 1))
    a = rows["asset"].to_numpy().astype(int); y = rows["y"].to_numpy()
    ctrl = ~is_pump[a]
    normal = (y == 0)
    out = {"alarm_rate_normal_rows": float(alarm[normal].mean()) if normal.any() else float("nan"),
           "alarm_rate_control_rows": float(alarm[ctrl].mean()) if ctrl.any() else float("nan"),
           "false_alarms_per_control_day": float(alarm[ctrl].sum() / max((ctrl.sum() * bm) / 1440.0, 1e-9)) if ctrl.any() else float("nan")}
    if alpha is not None and normal.any():
        out["alarm_budget_error"] = abs(out["alarm_rate_normal_rows"] - float(alpha))
    return out


def hit_rate_at_k(rows: pd.DataFrame, s: np.ndarray, meta: Dict[str, Any], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Among episodes sharing the same anchor timestamp (a pump and its same-time controls), rank by the score at the anchor
    row minus one (the last pre-announcement bar); HR@k = share of pumps ranked within the top k."""
    pre = int(meta["pre_bars"]); is_pump = np.asarray(meta["is_pump"]); anchor = np.asarray(meta["anchor_ns"])
    t = rows["t"].to_numpy(); a = rows["asset"].to_numpy().astype(int)
    sel = t == pre - 1
    sc = {int(i): float(v) for i, v in zip(a[sel], s[sel])}
    groups: Dict[int, list] = {}
    for i, v in sc.items():
        groups.setdefault(int(anchor[i]), []).append((i, v))
    hits = {k: [] for k in ks}; n = 0
    for g in groups.values():
        pumps_g = [i for i, _ in g if is_pump[i]]
        if not pumps_g or len(g) < 2:
            continue
        order = sorted(g, key=lambda x: -(x[1] if np.isfinite(x[1]) else -np.inf))
        ranks = {i: r for r, (i, _) in enumerate(order)}
        for i in pumps_g:
            n += 1
            for k in ks:
                hits[k].append(ranks[i] < k)
    out = {"HR_groups": float(n)}
    for k in ks:
        out[f"HR@{k}"] = float(np.mean(hits[k])) if hits[k] else float("nan")
    return out
