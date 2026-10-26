"""Alarm gates at a per-bar false-alarm budget alpha (minimal, self-contained; delayed ACI after Gibbs & Candès 2021 /
El Halabi & Brandt 2026 as used in the CausalGate track). Calibration scores come from the validation fold only."""
from __future__ import annotations
from collections import deque
from typing import Any, Dict, Optional, Tuple
import numpy as np


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    v = np.asarray(values, dtype=np.float64); v = v[np.isfinite(v)]
    if v.size == 0:
        return float("inf")
    n = v.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(v)[k - 1])


class FixedQuantileGate:
    kind = "fixed"

    def __init__(self, alpha: float = 0.01, **kw):
        self.alpha = float(alpha); self.q = float("inf"); self.buf = np.zeros(0)

    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> "FixedQuantileGate":
        s = np.asarray(scores, dtype=np.float64)
        if labels is not None:                                  # budget is a FALSE-alarm rate: calibrate on normal rows
            s = s[np.asarray(labels) == 0]
        self.buf = s[np.isfinite(s)]
        self.q = conformal_quantile(self.buf, self.alpha)
        return self

    def reset_stream(self) -> None:
        pass

    def step(self, s: float, t: int, y: Optional[float] = None) -> Tuple[bool, float]:
        return (bool(np.isfinite(s) and s > self.q), self.q)

    def prob(self, s: float) -> float:
        return float((self.buf < s).mean()) if self.buf.size and np.isfinite(s) else 0.0


class DelayedACIGate(FixedQuantileGate):
    """alpha_t <- alpha_t + gamma * (alpha - err_{t-tau}), err = 1[alarm] on rows whose outcome arrives with delay tau
    (feedback = the row turned out normal => the alarm was false). Quantile over a sliding window of calibration + past
    normal scores. Long-run alarm rate on normal rows converges to alpha under arbitrary drift."""
    kind = "delayed_aci"

    def __init__(self, alpha: float = 0.01, gamma: float = 0.005, window: int = 2000, tau: int = 6, **kw):
        super().__init__(alpha)
        self.gamma, self.window, self.tau = float(gamma), int(window), int(tau)
        self.alpha_t = float(alpha); self._pending: deque = deque(); self._win: deque = deque()

    def calibrate(self, scores, labels=None):
        super().calibrate(scores, labels)
        self._cal = self.buf[-self.window :].copy()
        self.reset_stream()
        return self

    def reset_stream(self) -> None:
        self.alpha_t = self.alpha; self._pending = deque(); self._win = deque(self._cal.tolist()) if hasattr(self, "_cal") else deque()
        self.q = conformal_quantile(np.asarray(self._win), self.alpha_t)

    def step(self, s: float, t: int, y: Optional[float] = None) -> Tuple[bool, float]:
        while self._pending and self._pending[0][0] <= t - self.tau:       # release delayed feedback
            _, alarm_old, y_old, s_old = self._pending.popleft()
            if y_old is None or y_old == 0:                                  # a normal row: alarm = false alarm
                self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - float(alarm_old)), 1e-4, 0.5))
                if np.isfinite(s_old):
                    self._win.append(s_old)
                    while len(self._win) > self.window:
                        self._win.popleft()
                self.q = conformal_quantile(np.asarray(self._win), self.alpha_t)
        alarm = bool(np.isfinite(s) and s > self.q)
        self._pending.append((t, alarm, y, s))
        return alarm, self.q


def make_gate(cfg: Dict[str, Any], horizon: int = 6, latency: int = 1) -> FixedQuantileGate:
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "delayed_aci")
    cfg.setdefault("tau", horizon + latency)
    return DelayedACIGate(**cfg) if kind == "delayed_aci" else FixedQuantileGate(**cfg)
