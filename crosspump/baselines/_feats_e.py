"""Causal feature helpers shared by the group-E baselines B6-B10 (crosspump/baselines/b6_bolz24.py ... b10_lroo26.py).

Every helper is point-in-time by construction: a value indexed by row t of episode i depends only on bars <= t of
that episode and on events whose availability satisfies avail <= t (events are placed at ceil(avail), exactly as
`EventStreams.rasterise(pit=True)` does, so "row r" means "first usable at row r"). Nothing here reads labels.

    causal_ewma_z        (T, N, F) raw bar features -> EWMA z-scores with statistics from bars < t (NaN during warm-up)
    trailing_sum         sum over the last W rows (t - W, t] along axis 0 (cumsum based)
    trailing_max         max over the last W rows along axis 0 (NaN-ignoring; NaN when the whole window is NaN)
    raster_sum/raster_max per-bar rasters of an event subset at ceil(avail) (sum of weights / max of a mark)
    last_event_row       row of the most recent usable event (<= t) per episode, -1 before the first one
    trailing_unique      sliding-window distinct source ids and top-source concentration per episode
    coverage_mask        (N, P) which platforms the dataset indexes for each episode (static dataset metadata; diagnostics only)
    seen_mask            (T, N, P) causal coverage: at least one event of the platform was usable at or before row t
    ca_cfar_causal       one-sided (past-only) CA-CFAR threshold of Moura et al. on a (T, N) count series
    text_window_mean     mean of panel.text embeddings usable at t over the last W rows (optional text modality)
    fit_boosted_classifier  LightGBM if importable else sklearn HistGradientBoosting (records which)
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from ..data.streams import EventStreams, P


# ----------------------------------------------------------------------------------------------- bar features
def causal_ewma_z(x: np.ndarray, warmup: int = 30, halflife: float = 60.0) -> np.ndarray:
    """x (T, N, F) raw features -> z (T, N, F). z[t] = (x[t] - m_{t-1}) / sqrt(v_{t-1}) where m, v are EWMA mean /
    variance of the bars strictly before t (same recursion as crosspump.data.streams._causal_z, vectorised over N, F).
    NaN while fewer than `warmup` finite bars were seen, for NaN inputs and while the variance is zero."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    lam = 0.5 ** (1.0 / max(float(halflife), 1.0))
    z = np.full(x.shape, np.nan)
    m = np.zeros(x.shape[1:]); v = np.zeros(x.shape[1:]); n = np.zeros(x.shape[1:], dtype=np.int64)
    for t in range(T):
        xt = x[t]
        ok = np.isfinite(xt)
        ready = ok & (n >= warmup) & (v > 0)
        if ready.any():
            z[t][ready] = (xt[ready] - m[ready]) / np.sqrt(v[ready])
        first = ok & (n == 0)
        m[first] = xt[first]; v[first] = 0.0
        upd = ok & (n > 0)
        if upd.any():
            d = xt[upd] - m[upd]
            m[upd] = lam * m[upd] + (1 - lam) * xt[upd]
            v[upd] = lam * v[upd] + (1 - lam) * d * d
        n[ok] += 1
    return z


def trailing_sum(a: np.ndarray, W: int) -> np.ndarray:
    """s[t] = sum(a[max(0, t - W + 1) : t + 1]) along axis 0 (the last W rows, t included)."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    W = int(max(W, 1))
    cs = np.cumsum(a, 0)
    out = cs.copy()
    if W < a.shape[0]:
        out[W:] -= cs[:-W]
    return out


def trailing_max(a: np.ndarray, W: int) -> np.ndarray:
    """m[t] = nanmax(a[max(0, t - W + 1) : t + 1]) along axis 0 (NaN only when every value in the window is NaN)."""
    a = np.asarray(a, dtype=np.float64)
    W = int(max(W, 1))
    out = a.copy()
    for w in range(1, min(W, a.shape[0])):
        out[w:] = np.fmax(out[w:], a[:-w])
    return out


# ----------------------------------------------------------------------------------------------- event rasters
def usable_row(streams: EventStreams) -> np.ndarray:
    """Row at which each event becomes usable: ceil(avail) (usable at t iff avail <= t)."""
    return np.ceil(streams.avail).astype(np.int64)


def raster_sum(streams: EventStreams, T: int, N: int, mask: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """(T, N) sum of `weights` (default 1) over the events selected by `mask`, placed at their usable row."""
    out = np.zeros((T, N), dtype=np.float64)
    if not len(streams):
        return out
    sel = np.ones(len(streams), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    row = usable_row(streams)
    sel &= (row >= 0) & (row < T)
    w = np.ones(len(streams)) if weights is None else np.nan_to_num(np.asarray(weights, dtype=np.float64))
    np.add.at(out, (row[sel], streams.asset[sel]), w[sel])
    return out


def raster_max(streams: EventStreams, T: int, N: int, values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(T, N) max of `values` over the selected events at their usable row; NaN where no (finite) event lands."""
    out = np.full((T, N), -np.inf)
    if len(streams):
        sel = np.ones(len(streams), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        row = usable_row(streams)
        vals = np.asarray(values, dtype=np.float64)
        sel &= (row >= 0) & (row < T) & np.isfinite(vals)
        np.maximum.at(out, (row[sel], streams.asset[sel]), vals[sel])
    out[~np.isfinite(out)] = np.nan
    return out


def last_event_row(count_raster: np.ndarray) -> np.ndarray:
    """(T, N) row of the most recent usable event at or before t (-1 before the first one)."""
    T = count_raster.shape[0]
    rows = np.where(count_raster > 0, np.arange(T)[:, None], -1)
    return np.maximum.accumulate(rows, axis=0)


def trailing_unique(streams: EventStreams, T: int, N: int, W: int, mask: Optional[np.ndarray] = None, size: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window (last W rows, usable-row based) statistics of event *sources* per episode:
        uniq (T, N)   number of distinct source ids (source 0 = unknown: every such event counts as its own id)
        conc (T, N)   share of the window's events (or of `size` mass) carried by the single largest source (0 when empty)
        total (T, N)  number of events in the window
    O(E + T) per episode: every event enters the window once (at its usable row) and leaves once (W rows later)."""
    uniq = np.zeros((T, N)); conc = np.zeros((T, N)); total = np.zeros((T, N))
    if not len(streams):
        return uniq, conc, total
    sel = np.ones(len(streams), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    row = usable_row(streams)
    sel &= (row >= 0) & (row < T)
    idx = np.nonzero(sel)[0]
    if idx.size == 0:
        return uniq, conc, total
    src = streams.source[idx].astype(np.int64)
    anon = src == 0
    src = np.where(anon, -(idx + 1), src)                      # unknown sources never collapse into one node
    wgt = np.ones(idx.size) if size is None else np.nan_to_num(np.asarray(size, dtype=np.float64)[idx])
    wgt = np.where(wgt > 0, wgt, 1e-9)
    order = np.lexsort((row[idx], streams.asset[idx]))
    idx, src, wgt, row = idx[order], src[order], wgt[order], row[idx][order]
    asset = streams.asset[idx]
    for i in np.unique(asset):
        lo = int(np.searchsorted(asset, i, "left")); hi = int(np.searchsorted(asset, i, "right"))
        r, s, w = row[lo:hi], src[lo:hi], wgt[lo:hi]
        add_ptr = 0; rem_ptr = 0
        mass: Dict[int, float] = {}; tot = 0.0
        for t in range(T):
            while add_ptr < len(r) and r[add_ptr] <= t:
                mass[int(s[add_ptr])] = mass.get(int(s[add_ptr]), 0.0) + float(w[add_ptr]); tot += float(w[add_ptr]); add_ptr += 1
            while rem_ptr < add_ptr and r[rem_ptr] <= t - W:
                k = int(s[rem_ptr]); mass[k] -= float(w[rem_ptr]); tot -= float(w[rem_ptr])
                if mass[k] <= 1e-12:
                    del mass[k]
                rem_ptr += 1
            if mass:
                uniq[t, i] = len(mass); total[t, i] = add_ptr - rem_ptr; conc[t, i] = max(mass.values()) / max(tot, 1e-12)
    return uniq, conc, total


def coverage_mask(streams: EventStreams, N: int) -> np.ndarray:
    """(N, P) bool: platform p is indexed for episode i (static dataset metadata, see CONVENTIONS.md §2: a missing
    platform is a missing modality). Computed from the whole stream, never from a decision time."""
    if not len(streams):
        return np.zeros((N, P), dtype=bool)
    return streams.coverage(N)["mask"]


def seen_mask(counts: np.ndarray) -> np.ndarray:
    """(T, N, P) bool, CAUSAL coverage: platform p of episode i has delivered at least one event usable at or before
    row t (cumulative availability-based count > 0). Baselines abstain (NaN) on a modality until it has been seen, so
    a score at row t never depends on whether events arrive after t (unlike the static `coverage_mask`)."""
    return np.cumsum(np.asarray(counts, dtype=np.float64), 0) > 0


def cfar_alpha(n_train: int, p_fa: float) -> float:
    """Threshold factor of 02_anomaly_detection.ipynb (cell 17, ca_cfar_1d): alpha = N * (Pfa ** (-1 / N) - 1)."""
    n = int(n_train)
    return float(n * (float(p_fa) ** (-1.0 / n) - 1.0))


def ca_cfar_causal(counts: np.ndarray, cell: int = 1, n_train: int = 5, n_guard: int = 1, p_fa: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided CA-CFAR on a (T, N) per-bar count series (rows = bars, usable-row based).
        cut[t]   = sum of counts over the last `cell` bars (t - cell, t]            (cell under test, ends at t)
        guard    = the `n_guard` cells immediately before the cell under test (skipped)
        train    = the `n_train` cells before the guard cells; noise = mean of the training cells that lie fully
                   inside the series (the notebook's edge rule: a partial window uses the cells that exist)
        thr[t]   = alpha * noise,  alpha = N * (Pfa ** (-1 / N) - 1)
    Returns (cut, thr, n_valid) with thr = NaN where no training cell exists yet (notebook: threshold = inf, never
    detected). Only bars <= t enter row t."""
    c = np.nan_to_num(np.asarray(counts, dtype=np.float64))
    T = c.shape[0]
    C = int(max(cell, 1)); G = int(max(n_guard, 0)); Ntr = int(max(n_train, 1))
    cut = trailing_sum(c, C)
    acc = np.zeros_like(cut); n_valid = np.zeros_like(cut)
    for k in range(1, Ntr + 1):
        sh = (G + k) * C                                   # the k-th training cell ends at row t - sh
        first = sh + C - 1                                 # first row t for which that cell is a full cell (start >= 0)
        if first >= T:
            break
        acc[first:] += cut[C - 1 : T - sh]
        n_valid[first:] += 1.0
    noise = np.where(n_valid > 0, acc / np.maximum(n_valid, 1.0), np.nan)
    return cut, cfar_alpha(Ntr, p_fa) * noise, n_valid


# ----------------------------------------------------------------------------------------------- optional text
def text_window_mean(panel, T: int, N: int, W: int, dims: int = 32) -> Optional[np.ndarray]:
    """(T, N, D') mean embedding of the documents usable at t (avail_text <= t) over the last W rows, zeros when none;
    None when the panel carries no text. Documents are placed at ceil(avail_text) like events."""
    text = getattr(panel, "text", None)
    if text is None:
        return None
    mask = np.asarray(panel.text_mask, dtype=bool)
    avail = getattr(panel, "avail_text", None)
    tt = np.arange(T)[:, None, None]
    row = np.broadcast_to(tt, mask.shape) if avail is None else np.asarray(avail)
    row = np.ceil(np.asarray(row, dtype=np.float64)).astype(np.int64)
    D = min(int(dims), text.shape[-1])
    emb = np.zeros((T, N, D)); cnt = np.zeros((T, N))
    sel = mask & (row >= 0) & (row < T)
    if sel.any():
        r = row[sel]; a = np.broadcast_to(np.arange(N)[None, :, None], mask.shape)[sel]
        np.add.at(emb, (r, a), np.asarray(text, dtype=np.float64)[sel][:, :D])
        np.add.at(cnt, (r, a), 1.0)
    s = trailing_sum(emb, W); c = trailing_sum(cnt, W)
    return s / np.maximum(c, 1.0)[..., None]


# ----------------------------------------------------------------------------------------------- classifiers
def fit_boosted_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6,
                           num_leaves: int = 31, seed: int = 42, prefer: str = "lightgbm") -> Tuple[Any, str, Optional[str]]:
    """Gradient-boosted trees with the PumpSense LightGBM hyper-parameters (lightgbm_classification.py L157-167:
    n_estimators 100, lr 0.1, max_depth 6, num_leaves 31, min_child_samples 20, subsample 0.8, colsample 0.8,
    class_weight balanced). Returns (model, backend, fallback_reason). Falls back to sklearn HistGradientBoosting
    with the matching settings when lightgbm is not importable or fails."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(int)
    reason = None
    if prefer == "lightgbm":
        try:
            import lightgbm as lgb                                                        # optional dependency
            model = lgb.LGBMClassifier(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth), num_leaves=int(num_leaves),
                                       min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
                                       class_weight="balanced", random_state=int(seed), verbose=-1, n_jobs=1)
            model.fit(X, y)
            return model, "lightgbm", None
        except Exception as e:                                                             # ImportError or a runtime failure
            reason = f"{type(e).__name__}: {e}"[:200]
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth),
                                           max_leaf_nodes=int(num_leaves), min_samples_leaf=20, class_weight="balanced", random_state=int(seed),
                                           early_stopping=False)
    model.fit(X, y)
    return model, "sklearn_hgb", reason


def predict_proba1(model: Any, X: np.ndarray) -> np.ndarray:
    """P(y = 1) for any sklearn-style classifier; rows with non-finite features are imputed with 0 for backends
    that cannot handle NaN (LightGBM and HGB both can; the imputation is only a guard for TabPFN-like backends)."""
    X = np.asarray(X, dtype=np.float64)
    try:
        return np.asarray(model.predict_proba(X))[:, 1]
    except ValueError:
        return np.asarray(model.predict_proba(np.nan_to_num(X)))[:, 1]


def feature_names_index(panel, names: Sequence[str], fallback: Dict[str, int]) -> Dict[str, int]:
    """Column indices of the requested raw order-flow features (meta['feat_names'] when present, else the layout of
    CONVENTIONS.md §1: [log_ret, log_volume, hl_range, taker_imb, log_trades])."""
    have = list(panel.meta.get("feat_names", []))
    out = {}
    for n in names:
        if n in have:
            out[n] = have.index(n)
        elif n in fallback and fallback[n] < panel.price_feat.shape[-1]:
            out[n] = fallback[n]
    return out


DEFAULT_FEAT_INDEX = {"log_ret": 0, "log_volume": 1, "hl_range": 2, "taker_imb": 3, "log_trades": 4}
