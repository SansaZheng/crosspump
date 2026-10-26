"""Follower overlay: the economic value of an avoid / exit signal for a follower of pump promotions, after Ardia & Bluteau
(IRFA 2024): investors informed through Twitter buy at the promotion and sell LATE, losing on average.

Benchmark follower F: for every pump episode buys at the close of the anchor bar + latency (the announcement is the entry
signal) and sells at anchor + hold_late bars (late exit). Policy follower with alerts:
    * an alert at row t_alert < anchor  ->  AVOID: no position, PnL 0 (the follower sees the warning before the promotion)
    * an alert at anchor <= t_alert < exit_late  ->  EXIT at t_alert + latency instead of the late exit
    * no alert  ->  identical to F
Loss avoided = PnL_policy - PnL_F per episode (log return units, net of the shared cost model's round-trip estimate at the
given bps), reported in mean bps per episode, as the share of F's losses avoided, and as the forgone gain on episodes where F
made money. Control episodes carry no follower position; false alarms there are reported by the detection metrics.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd


def follower_overlay(rows: pd.DataFrame, alarm: np.ndarray, panel, hold_late: int = 12, latency: int = 1, cost_bps: float = 5.0,
                     assets: Optional[Sequence[int]] = None) -> Dict[str, float]:
    meta = panel.meta
    pre = int(meta["pre_bars"]); is_pump = np.asarray(meta["is_pump"]); T = panel.T
    lp = np.log(panel.mid)
    a_rows = rows["asset"].to_numpy().astype(int); t_rows = rows["t"].to_numpy().astype(int)
    pumps = [int(a) for a in np.unique(a_rows) if is_pump[int(a)]]
    if assets is not None:
        pumps = [a for a in pumps if a in set(int(x) for x in assets)]
    cost = 2 * cost_bps * 1e-4 + 2 * float(np.nanmedian(panel.liq[:, :, 0]))          # round trip: commission + half-spread both ways
    pnl_f, pnl_p, kind = [], [], []
    for a in pumps:
        entry = pre + latency; exit_late = min(T - 1, pre + hold_late)
        if entry >= T or not np.isfinite(lp[entry, a]) or not np.isfinite(lp[exit_late, a]):
            continue
        r_f = lp[exit_late, a] - lp[entry, a] - cost
        sel = (a_rows == a) & alarm
        ts = np.sort(t_rows[sel])
        if ts.size and ts[0] < pre:
            r_p = 0.0; k = "avoid"
        elif ts.size and ts[0] < exit_late:
            t_exit = min(T - 1, int(ts[0]) + latency)
            t_exit = max(t_exit, entry)
            r_p = lp[t_exit, a] - lp[entry, a] - cost; k = "exit"
        else:
            r_p = r_f; k = "none"
        pnl_f.append(r_f); pnl_p.append(r_p); kind.append(k)
    if not pnl_f:
        return {"follower_n": 0}
    pf, pp = np.array(pnl_f), np.array(pnl_p)
    gain = pp - pf
    losses_f = np.clip(-pf, 0, None)
    out = {"follower_n": len(pf), "follower_pnl_bps": float(pf.mean() * 1e4), "policy_pnl_bps": float(pp.mean() * 1e4),
           "loss_avoided_bps": float(gain.mean() * 1e4), "loss_avoided_share": float(np.clip(gain, 0, None).sum() / losses_f.sum()) if losses_f.sum() > 0 else float("nan"),
           "forgone_gain_bps": float(np.clip(-gain, 0, None).mean() * 1e4), "avoid_share": float(np.mean([k == "avoid" for k in kind])),
           "exit_share": float(np.mean([k == "exit" for k in kind])), "follower_loss_rate": float((pf < 0).mean())}
    return out
