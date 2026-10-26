"""Cross-platform event streams attached to a Panel (append-only convention: a new attribute `streams`, recorded in
meta['xp_fields']; never required by Panel.validate(), never read by other tracks).

    EventStreams: flat arrays sorted by (asset, time)
        asset (E,) int64 | time (E,) float64 bar units | platform (E,) int8 {0 social, 1 wallet, 2 orderflow}
        mark (E, 4) float32 [magnitude, reach_or_size, text_score, prior_involvement] | avail (E,) float64 >= time
        source (E,) int64 entity id (0 = unknown) | meta dict

Point-in-time rule: an event is usable at decision row t iff avail <= t (decisions are taken at the bar close; an event
whose availability falls strictly inside (t, t+1] becomes usable at row t+1). Row of an event's occurrence = floor(time);
row at which it is first usable = ceil(avail). Bar-derived events (order-flow bursts computed from bar t) carry
time = avail = t + 1, i.e. they are usable from row t + 1 (one bar after the bar they summarise).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from warn2trade.data.schema import Panel

PLATFORMS: Tuple[str, ...] = ("social", "wallet", "orderflow")
MARKS: Tuple[str, ...] = ("magnitude", "reach_or_size", "text_score", "prior_involvement")
P, DM = len(PLATFORMS), len(MARKS)
XP_FIELDS = ("streams",)


@dataclass
class EventStreams:
    asset: np.ndarray
    time: np.ndarray
    platform: np.ndarray
    mark: np.ndarray
    avail: np.ndarray
    source: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def empty(cls, **meta) -> "EventStreams":
        return cls(np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, np.int8), np.zeros((0, DM), np.float32),
                   np.zeros(0, np.float64), np.zeros(0, np.int64), dict(meta))

    @classmethod
    def from_lists(cls, asset: Iterable[int], time: Iterable[float], platform: Iterable[int], mark: Optional[Iterable[Sequence[float]]] = None,
                   avail: Optional[Iterable[float]] = None, source: Optional[Iterable[int]] = None, **meta) -> "EventStreams":
        a = np.asarray(list(asset), dtype=np.int64); t = np.asarray(list(time), dtype=np.float64); p = np.asarray(list(platform), dtype=np.int8)
        E = len(a)
        m = np.full((E, DM), np.nan, dtype=np.float32)
        if mark is not None:
            for k, row in enumerate(mark):
                row = np.asarray(row, dtype=np.float32).ravel()
                m[k, : min(DM, len(row))] = row[:DM]
        av = t.copy() if avail is None else np.asarray(list(avail), dtype=np.float64)
        s = np.zeros(E, np.int64) if source is None else np.asarray(list(source), dtype=np.int64)
        return cls(a, t, p, m, av, s, dict(meta)).sorted()

    @classmethod
    def concat(cls, parts: Sequence["EventStreams"], **meta) -> "EventStreams":
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return cls.empty(**meta)
        m: Dict[str, Any] = {}
        for p in parts:
            m.update(p.meta)
        m.update(meta)
        return cls(np.concatenate([p.asset for p in parts]), np.concatenate([p.time for p in parts]), np.concatenate([p.platform for p in parts]),
                   np.concatenate([p.mark for p in parts]), np.concatenate([p.avail for p in parts]), np.concatenate([p.source for p in parts]), m).sorted()

    def sorted(self) -> "EventStreams":
        order = np.lexsort((self.time, self.asset))
        return EventStreams(self.asset[order], self.time[order], self.platform[order], self.mark[order], self.avail[order], self.source[order], self.meta)

    def __len__(self) -> int:
        return int(len(self.asset))

    # ------------------------------------------------------------------ validation
    def validate(self, T: Optional[int] = None, N: Optional[int] = None) -> None:
        E = len(self)
        assert self.time.shape == (E,) and self.platform.shape == (E,) and self.avail.shape == (E,) and self.source.shape == (E,), "ragged streams"
        assert self.mark.shape == (E, DM), self.mark.shape
        assert (self.avail >= self.time - 1e-9).all(), "availability before the event time"
        assert ((self.platform >= 0) & (self.platform < P)).all(), "unknown platform id"
        if N is not None:
            assert ((self.asset >= 0) & (self.asset < N)).all(), "asset out of range"
        if T is not None and E:
            assert (self.time >= 0).all() and (self.time < T).all(), "event time outside the panel"
        if E > 1:
            key = self.asset.astype(np.float64) * 1e9 + self.time
            assert (np.diff(key) >= 0).all(), "streams must be sorted by (asset, time)"

    # ------------------------------------------------------------------ queries
    def _bounds(self, i: int) -> Tuple[int, int]:
        lo = int(np.searchsorted(self.asset, i, side="left")); hi = int(np.searchsorted(self.asset, i, side="right"))
        return lo, hi

    def for_asset(self, i: int) -> "EventStreams":
        lo, hi = self._bounds(i)
        sl = slice(lo, hi)
        return EventStreams(self.asset[sl], self.time[sl], self.platform[sl], self.mark[sl], self.avail[sl], self.source[sl], self.meta)

    def before(self, i: int, t: float, pit: bool = True) -> "EventStreams":
        """Events of asset i knowable at decision bar t (avail <= t) — or simply with time <= t when pit=False."""
        s = self.for_asset(i)
        keep = (s.avail <= t) if pit else (s.time <= t)
        return EventStreams(s.asset[keep], s.time[keep], s.platform[keep], s.mark[keep], s.avail[keep], s.source[keep], s.meta)

    def subset(self, mask: np.ndarray) -> "EventStreams":
        return EventStreams(self.asset[mask], self.time[mask], self.platform[mask], self.mark[mask], self.avail[mask], self.source[mask], self.meta)

    def platforms_present(self, i: Optional[int] = None) -> List[str]:
        s = self if i is None else self.for_asset(i)
        return [PLATFORMS[p] for p in sorted(set(s.platform.tolist()))]

    def rasterise(self, T: int, N: int, pit: bool = True, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """(T, N, P) per-bar counts (or mark-weighted sums). With pit=True an event is counted at the first row at which
        it is usable (ceil(avail), consistent with `before`: usable at row t iff avail <= t), otherwise at its occurrence row."""
        out = np.zeros((T, N, P), dtype=np.float64)
        if not len(self):
            return out
        row = (np.ceil(self.avail) if pit else np.floor(self.time)).astype(np.int64)
        ok = (row >= 0) & (row < T)
        w = np.ones(len(self)) if weights is None else np.nan_to_num(np.asarray(weights, dtype=np.float64))
        np.add.at(out, (row[ok], self.asset[ok], self.platform[ok].astype(np.int64)), w[ok])
        return out

    def coverage(self, N: int) -> Dict[str, Any]:
        cov = np.zeros((N, P), dtype=bool)
        if len(self):
            cov[self.asset, self.platform.astype(np.int64)] = True
        return {"episodes_with_platform": {PLATFORMS[p]: int(cov[:, p].sum()) for p in range(P)}, "n_events": {PLATFORMS[p]: int((self.platform == p).sum()) for p in range(P)},
                "mask": cov}


# ---------------------------------------------------------------------- attach / get
def attach_streams(panel: Panel, streams: EventStreams) -> Panel:
    streams.validate(panel.T, panel.N)
    setattr(panel, "streams", streams)
    panel.meta["xp_fields"] = sorted(set(panel.meta.get("xp_fields", [])) | {"streams"})
    panel.meta["coverage"] = {k: v for k, v in streams.coverage(panel.N).items() if k != "mask"}
    return panel


def get_streams(panel: Panel) -> Optional[EventStreams]:
    return getattr(panel, "streams", None)


# ---------------------------------------------------------------------- derived streams
def _causal_z(x: np.ndarray, warmup: int, halflife: float) -> np.ndarray:
    """EWMA z-score using statistics of strictly earlier bars (NaN during warm-up / for NaN inputs). x: (T,)."""
    T = len(x)
    z = np.full(T, np.nan)
    lam = 0.5 ** (1.0 / max(halflife, 1.0))
    m = 0.0; v = 0.0; n = 0
    for t in range(T):
        xt = x[t]
        if n >= warmup and np.isfinite(xt) and v > 0:
            z[t] = (xt - m) / np.sqrt(v)
        if np.isfinite(xt):
            if n == 0:
                m, v = float(xt), 0.0
            else:
                d = xt - m
                m = lam * m + (1 - lam) * xt
                v = lam * v + (1 - lam) * d * d
            n += 1
    return z


def derive_orderflow_events(panel: Panel, z_thresh: float = 2.5, warmup: int = 30, halflife: float = 60.0, cols: Optional[Dict[str, int]] = None,
                            min_gap_bars: int = 1) -> EventStreams:
    """Order-flow burst events from the raw bar features, causally: an event at bar t when the EWMA z-score (statistics
    from bars < t) of log_volume or taker_imb exceeds z_thresh. mark = [max z, log_volume z, taker_imb z, NaN]; time = t + 1
    (knowable at the bar close) and avail = time. Used when a loader has no trade-level data."""
    names = list(panel.meta.get("feat_names", []))
    cols = cols or {k: names.index(k) for k in ("log_volume", "taker_imb") if k in names}
    T, N = panel.mid.shape
    a, tm, pl, mk, av = [], [], [], [], []
    for i in range(N):
        zs = {k: _causal_z(panel.price_feat[:, i, c].astype(np.float64), warmup, halflife) for k, c in cols.items()}
        zmax = np.nanmax(np.stack([np.nan_to_num(v, nan=-np.inf) for v in zs.values()], 0), 0) if zs else np.full(T, -np.inf)
        last = -10 ** 9
        for t in np.nonzero(zmax > z_thresh)[0]:
            if t - last < min_gap_bars:
                continue
            last = t
            a.append(i); tm.append(min(t + 1.0, T - 1e-6)); pl.append(2)
            mk.append([zmax[t], zs.get("log_volume", np.full(T, np.nan))[t], zs.get("taker_imb", np.full(T, np.nan))[t], np.nan]); av.append(min(t + 1.0, T - 1e-6))
    es = EventStreams.from_lists(a, tm, pl, mk, av, [0] * len(a), orderflow_rule={"z_thresh": z_thresh, "warmup": warmup, "halflife": halflife, "cols": list(cols)})
    return es


def add_prior_involvement(streams: EventStreams, anchor_order: np.ndarray, is_pump: np.ndarray) -> EventStreams:
    """Fill mark[:, 3] causally: for each event with a known source id, the number of EARLIER pump episodes (by anchor
    order) in which the same source id appears — a channel's or wallet's track record as of that episode (the linked
    graph's memory). anchor_order: (N,) rank of each episode's anchor time; is_pump: (N,) bool."""
    if not len(streams):
        return streams
    m = streams.mark.copy()
    seen: Dict[int, set] = {}
    order = np.argsort(anchor_order)
    rank_of = {int(i): int(r) for r, i in enumerate(order)}
    ev_rank = np.array([rank_of[int(i)] for i in streams.asset])
    idx = np.argsort(ev_rank, kind="stable")
    for j in idx:
        src = int(streams.source[j])
        if src == 0:
            m[j, 3] = np.nan; continue
        prior = seen.get(src, set())
        m[j, 3] = float(len([e for e in prior if int(anchor_order[e]) < int(anchor_order[streams.asset[j]])]))
        if is_pump[streams.asset[j]]:
            seen.setdefault(src, set()).add(int(streams.asset[j]))
    return EventStreams(streams.asset, streams.time, streams.platform, m, streams.avail, streams.source, streams.meta)


def streams_summary(streams: EventStreams, N: int) -> Dict[str, Any]:
    cov = streams.coverage(N)
    return {"n_events": cov["n_events"], "episodes_with_platform": cov["episodes_with_platform"], "lags": streams.meta.get("platform_lags")}
