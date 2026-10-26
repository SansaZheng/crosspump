"""Causal feature helpers shared by the group-D baselines B1-B5 (crosspump/baselines/b1_mirtaheri21.py ... b5_hu23.py).

Every helper here reads, for a decision row t, only
    * events that are USABLE at t: usable row = ceil(avail) <= t (the same rule as EventStreams.before / rasterise(pit=True));
    * bar features of rows <= t (price_feat[t] is knowable at the close of bar t, avail_feat = row);
    * statistics fitted on the TRAIN episodes (chronologically earlier than every validation / test episode).
Nothing here fabricates data: every function transforms the panel / streams it is given.
"""
from __future__ import annotations
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.streams import EventStreams, PLATFORMS, P

SOCIAL, WALLET, ORDERFLOW = 0, 1, 2


# ---------------------------------------------------------------------------------------------- window sums
class CumRaster:
    """Prefix sums of a per-row raster (T, N[, F]) so that sums over rows (t - back, t] cost O(1) per row."""

    def __init__(self, raster: np.ndarray):
        r = np.nan_to_num(np.asarray(raster, dtype=np.float64))
        self.T = int(r.shape[0])
        self.cs = np.concatenate([np.zeros((1,) + r.shape[1:]), np.cumsum(r, 0)], 0)

    def window(self, t: np.ndarray, i: np.ndarray, back: int) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        lo = np.clip(t - int(back), 0, self.T); hi = np.clip(t + 1, 0, self.T)
        return self.cs[hi, i] - self.cs[lo, i]


def usable_row(streams: EventStreams) -> np.ndarray:
    """First row at which each event is usable (ceil(avail)); an event is in F_t iff usable_row <= t."""
    return np.ceil(np.asarray(streams.avail, dtype=np.float64)).astype(np.int64)


def unique_source_raster(streams: EventStreams, T: int, N: int, platform: int, backs: Sequence[int]) -> Dict[int, np.ndarray]:
    """For each `back`: (T, N) number of DISTINCT known source ids (id 0 = unknown is skipped) among events of `platform`
    whose usable row lies in (t - back, t]. Two-pointer sweep per episode: O(T + E) per (episode, window)."""
    out = {int(b): np.zeros((T, N)) for b in backs}
    if not len(streams):
        return out
    sel = (streams.platform == platform) & (streams.source != 0)
    rows = usable_row(streams)[sel]; src = streams.source[sel]; ast = streams.asset[sel]
    order = np.lexsort((rows, ast)); rows, src, ast = rows[order], src[order], ast[order]
    bounds = np.searchsorted(ast, np.arange(N + 1))
    for i in range(N):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue
        r, s = rows[lo:hi], src[lo:hi]
        for b in out:
            cnt: Dict[int, int] = {}; n_distinct = 0; a = 0; z = 0
            arr = out[b][:, i]
            for t in range(T):
                while a < len(r) and r[a] <= t:
                    k = int(s[a]); cnt[k] = cnt.get(k, 0) + 1
                    if cnt[k] == 1:
                        n_distinct += 1
                    a += 1
                while z < a and r[z] <= t - b:
                    k = int(s[z]); cnt[k] -= 1
                    if cnt[k] == 0:
                        n_distinct -= 1
                    z += 1
                arr[t] = n_distinct
    return out


# ---------------------------------------------------------------------------------------------- bar statistics
def rolling(x: np.ndarray, w: int, fn: str = "mean", min_periods: Optional[int] = None) -> np.ndarray:
    """Trailing rolling statistic over rows (t - w, t] (inclusive of t) of a (T, N) array; NaN before min_periods."""
    df = pd.DataFrame(np.asarray(x, dtype=np.float64))
    r = df.rolling(int(w), min_periods=int(min_periods if min_periods is not None else max(1, w // 2)))
    if fn == "std":
        return r.std(ddof=0).to_numpy()
    return getattr(r, fn)().to_numpy()


def expanding_z(x: np.ndarray, warmup: int) -> np.ndarray:
    """z-score of x[t] against the mean / std of the STRICTLY EARLIER finite values of the same column (per episode);
    NaN while fewer than `warmup` earlier values exist or x[t] is NaN."""
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x); xv = np.where(fin, x, 0.0)
    n = np.cumsum(fin, 0).astype(np.float64); s1 = np.cumsum(xv, 0); s2 = np.cumsum(xv * xv, 0)
    shift = lambda a: np.concatenate([np.zeros_like(a[:1]), a[:-1]], 0)
    n, s1, s2 = shift(n), shift(s1), shift(s2)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / np.maximum(n, 1.0); v = s2 / np.maximum(n, 1.0) - m * m
        z = (x - m) / np.sqrt(np.maximum(v, 1e-12))
    z[(n < warmup) | ~fin] = np.nan
    return z


def gather_windows(arr: np.ndarray, t: np.ndarray, i: np.ndarray, L: int) -> np.ndarray:
    """(B, L, F) causal windows arr[t - L + 1 .. t, i]; rows before the panel start are NaN."""
    t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
    idx = t[:, None] - int(L) + 1 + np.arange(int(L))[None, :]
    valid = idx >= 0
    out = arr[np.clip(idx, 0, arr.shape[0] - 1), i[:, None]].astype(np.float64)
    out[~valid] = np.nan
    return out


def clock_features(panel, T: int, N: int) -> np.ndarray:
    """(T, N, 4) [hour_sin, hour_cos, minute_sin, minute_cos] of the bar CLOSE in UTC, with the official formulas of
    SystemsLab-Sapienza/pump-and-dump-dataset features.py L116-119 (sin(2 pi h / 23), sin(2 pi m / 59)); zeros when the
    panel carries no bar_close_ns."""
    bc = panel.meta.get("bar_close_ns")
    if bc is None:
        return np.zeros((T, N, 4))
    ns = np.asarray(bc, dtype=np.int64)
    sec = (ns // 10 ** 9) % 86400
    hour = (sec // 3600).astype(np.float64); minute = ((sec % 3600) // 60).astype(np.float64)
    return np.stack([np.sin(2 * np.pi * hour / 23), np.cos(2 * np.pi * hour / 23), np.sin(2 * np.pi * minute / 59), np.cos(2 * np.pi * minute / 59)], -1)


# ---------------------------------------------------------------------------------------------- coverage
def platform_coverage(panel, streams: EventStreams, N: int, visible: Sequence[str] = PLATFORMS) -> np.ndarray:
    """(N, P) bool: episode i carries platform p. The loader's per-episode flag meta['episodes'][i]['has_<platform>'] wins
    when present (a data-source property); otherwise presence of any event of that platform in the episode's streams.
    Platforms switched off for this run (cfg['platforms']) count as absent."""
    cov = streams.coverage(N)["mask"].copy() if len(streams) else np.zeros((N, P), dtype=bool)
    eps = panel.meta.get("episodes") or []
    for p, name in enumerate(PLATFORMS):
        for i, e in enumerate(eps[:N]):
            flag = e.get(f"has_{name}") if isinstance(e, dict) else None
            if flag is not None:
                cov[i, p] = bool(flag)
        if name not in visible:
            cov[:, p] = False
    return cov


# ---------------------------------------------------------------------------------------------- sklearn
class _ConstantScorer:
    def __init__(self, value: float = 0.5):
        self.value = float(value)

    def predict_proba(self, X):
        return np.stack([np.full(len(X), 1 - self.value), np.full(len(X), self.value)], 1)


def fit_forest(X: np.ndarray, y: np.ndarray, kind: str = "rf", n_estimators: int = 200, max_depth: Optional[int] = 8, class_weight: Optional[str] = None,
               seed: int = 0, lgbm_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """Fit the paper's tree ensemble with graceful fallbacks: lightgbm (optional) -> sklearn RandomForest -> LogisticRegression.
    Returns (model with predict_proba, status string naming what was fitted). Rows with NaN features are zero-filled."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64)); y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return _ConstantScorer(0.5), "constant (single class in training rows)"
    if kind == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
            params = {"n_estimators": n_estimators, "max_depth": -1 if max_depth is None else max_depth, "learning_rate": 0.05, "random_state": seed, "verbose": -1}
            params.update(lgbm_params or {})
            if class_weight:
                params["class_weight"] = "balanced"
            m = lgb.LGBMClassifier(**params); m.fit(X, y)
            return m, "lightgbm"
        except Exception as e:  # noqa: BLE001
            kind = "rf"; fallback = f"lightgbm unavailable ({type(e).__name__}) -> rf"
    else:
        fallback = ""
    if kind == "rf":
        try:
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth, class_weight=class_weight, random_state=int(seed), n_jobs=1)
            m.fit(X, y)
            return m, (fallback or "rf")
        except Exception as e:  # noqa: BLE001
            fallback = (fallback + "; " if fallback else "") + f"rf failed ({type(e).__name__}) -> lr"
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced" if class_weight else None, max_iter=1000, C=1.0))
    m.fit(X, y)
    return m, (fallback or "lr")


def forest_scores(model, X: np.ndarray) -> np.ndarray:
    """P(pump) for each row; rows whose features are entirely NaN abstain (NaN)."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) == 0:
        return np.zeros(0)
    all_nan = ~np.isfinite(X).any(1)
    pr = model.predict_proba(np.nan_to_num(X))
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2:
        col = int(np.nonzero(np.asarray(classes) == 1)[0][0]) if (np.asarray(classes) == 1).any() else 1
    else:
        col = pr.shape[1] - 1
    s = pr[:, col].astype(np.float64)
    s[all_nan] = np.nan
    return s


# ---------------------------------------------------------------------------------------------- torch
def subsample_rows(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Indices keeping every positive row and a random subset of negatives so that at most max_rows rows remain."""
    y = np.asarray(y); n = len(y)
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    pos = np.nonzero(y == 1)[0]; neg = np.nonzero(y != 1)[0]
    k = max(0, max_rows - len(pos))
    keep = np.concatenate([pos, rng.choice(neg, size=min(k, len(neg)), replace=False)])
    return np.sort(keep)


def train_binary(model: nn.Module, inputs: Sequence[torch.Tensor], y: np.ndarray, epochs: int = 10, lr: float = 1e-3, batch: int = 256, seed: int = 0,
                 pos_weight_cap: float = 50.0, weight_decay: float = 0.0) -> Dict[str, Any]:
    """Class-weighted BCE training (pos_weight = n_neg / n_pos, capped) with a seeded shuffling order. Deterministic on CPU."""
    torch.manual_seed(int(seed))
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
    n = int(len(yt)); n_pos = float(yt.sum().item())
    pw = float(np.clip((n - n_pos) / max(n_pos, 1.0), 1.0, pos_weight_cap))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    g = torch.Generator().manual_seed(int(seed))
    last = float("nan")
    model.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=g)
        tot = 0.0; cnt = 0
        for s in range(0, n, int(batch)):
            idx = perm[s : s + int(batch)]
            if len(idx) < 2:
                continue
            logits = model(*[x[idx] for x in inputs])
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss.item()) * len(idx); cnt += len(idx)
        last = tot / max(cnt, 1)
    model.eval()
    return {"epochs": int(epochs), "n_rows": n, "n_pos": int(n_pos), "pos_weight": pw, "final_loss": last}


@torch.no_grad()
def predict_binary(model: nn.Module, inputs: Sequence[torch.Tensor], batch: int = 8192) -> np.ndarray:
    model.eval()
    n = int(inputs[0].shape[0])
    out = np.zeros(n, dtype=np.float64)
    for s in range(0, n, int(batch)):
        out[s : s + batch] = torch.sigmoid(model(*[x[s : s + batch] for x in inputs])).cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------------- ids
def hash_bucket(key: Any, n: int) -> int:
    """Stable bucket in [1, n) for an entity key (0 is reserved for 'unknown')."""
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    return 1 + h % max(1, n - 1)


def id_bucket(ids: np.ndarray, n: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    return np.where(ids > 0, 1 + ids % max(1, n - 1), 0).astype(np.int64)


# ---------------------------------------------------------------------------------------------- bar columns / market features
DEFAULT_COLS = {"log_ret": 0, "log_volume": 1, "hl_range": 2, "taker_imb": 3, "log_trades": 4}


def feat_cols(panel, F: int) -> Dict[str, int]:
    """Column index of each raw bar feature (CONVENTIONS.md section 1), from meta['feat_names'] when present."""
    names = list(panel.meta.get("feat_names") or [])
    out = {k: (names.index(k) if k in names else v) for k, v in DEFAULT_COLS.items()}
    return {k: v for k, v in out.items() if v < F}


def bar_column(feat: np.ndarray, cols: Dict[str, int], key: str) -> np.ndarray:
    """(T, N) float64 column `key` of the raw bar features (all-NaN when the panel has no such column)."""
    if key not in cols:
        return np.full(feat.shape[:2], np.nan)
    return np.asarray(feat[:, :, cols[key]], dtype=np.float64)


def market_features(feat: np.ndarray, cols: Dict[str, int], windows: Sequence[int] = (3, 12, 48), warmup: int = 12) -> Tuple[np.ndarray, List[str]]:
    """(T, N, F) causal market features of the candidate coin at each row: over each trailing window (rows (t - w, t])
    the cumulative log return, the realised volatility, the mean of the expanding z-score (statistics of strictly earlier
    bars of the same episode, warm-up NaN) of log_volume and log_trades, the mean taker imbalance and the mean high-low range.
    Entry [t] depends on bars <= t of its own episode only."""
    ret = bar_column(feat, cols, "log_ret"); out: List[np.ndarray] = []; names: List[str] = []
    zvol = expanding_z(bar_column(feat, cols, "log_volume"), warmup); ztr = expanding_z(bar_column(feat, cols, "log_trades"), warmup)
    imb = bar_column(feat, cols, "taker_imb"); hl = bar_column(feat, cols, "hl_range")
    for w in windows:
        w = int(w)
        out += [rolling(ret, w, "sum", 1), rolling(ret, w, "std", 2) if w > 1 else np.abs(ret), rolling(zvol, w, "mean", 1), rolling(ztr, w, "mean", 1),
                rolling(imb, w, "mean", 1), rolling(hl, w, "mean", 1)]
        names += [f"{k}_{w}" for k in ("cum_ret", "vol", "z_log_volume", "z_log_trades", "taker_imb", "hl_range")]
    return np.stack(out, -1), names


def episode_order_times(panel, T: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """(start, end) wall-clock key of every episode window, used to decide which earlier episodes are realised history:
    bar_close_ns[0] / bar_close_ns[T-1] when the panel carries them, else anchor_ns for both, else the column order."""
    bc = panel.meta.get("bar_close_ns")
    if bc is not None:
        bc = np.asarray(bc, dtype=np.int64)
        return bc[0].astype(np.float64), bc[-1].astype(np.float64)
    an = panel.meta.get("anchor_ns")
    if an is not None:
        an = np.asarray(an, dtype=np.float64)
        return an, an
    k = np.arange(N, dtype=np.float64)
    return k, k


class CausalCoverage:
    """Row-level modality presence that never looks ahead: episode i carries platform p at row t iff the loader's
    per-episode flag meta['episodes'][i]['has_<platform>'] says so (a property of the data source, known up front), or —
    when the loader gives no flag — at least one event of that platform is already usable at t (ceil(avail) <= t).
    Platforms switched off for the run (cfg['platforms']) are absent everywhere."""

    def __init__(self, panel, counts: np.ndarray, visible: Sequence[str] = PLATFORMS):
        T, N = counts.shape[:2]
        self.flag = np.full((N, P), np.nan)
        eps = panel.meta.get("episodes") or []
        for p, name in enumerate(PLATFORMS):
            for i, e in enumerate(eps[:N]):
                f = e.get(f"has_{name}") if isinstance(e, dict) else None
                if f is not None:
                    self.flag[i, p] = float(bool(f))
            if name not in visible:
                self.flag[:, p] = 0.0
        self.seen = np.cumsum(np.nan_to_num(counts), 0) > 0                                  # (T, N, P)

    def at(self, t: np.ndarray, i: np.ndarray, platform: int) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        f = self.flag[i, platform]
        return np.where(np.isnan(f), self.seen[t, i, platform], f == 1.0).astype(bool)

    def table(self, platform: int) -> np.ndarray:
        """(T, N) bool."""
        f = self.flag[None, :, platform]
        return np.where(np.isnan(f), self.seen[:, :, platform], f == 1.0).astype(bool)
