"""Chronological splits over EPISODES (panel columns) for the stacked-episode layout, and the matching dataset view.

EpisodeSplitter: episodes sorted by anchor time -> walk-forward folds over episode indices
    train = first train_frac of the episodes seen so far, val = next val_frac, test = the rest of the fold's block;
    a purge gap of `purge_ns` (>= one window length) is left between the last train/val anchor and the first test anchor so
    that no test window overlaps a training window in wall-clock time (episodes of the same coin can overlap otherwise).
Fold objects reuse warn2trade.data.splits.Fold with asset index arrays in train / val / test.
EpisodeDataset: EventDataset restricted to a set of episodes (all decision rows of those columns).
"""
from __future__ import annotations
from typing import Iterator, List, Optional, Sequence
import numpy as np

from warn2trade.data.base import EventDataset
from warn2trade.data.schema import Panel
from warn2trade.data.splits import Fold


class EpisodeSplitter:
    def __init__(self, anchor_ns: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.15, purge_ns: int = 0, n_folds: int = 1,
                 min_train: int = 8, max_folds: Optional[int] = None):
        self.anchor = np.asarray(anchor_ns, dtype=np.int64)
        self.order = np.argsort(self.anchor, kind="stable")
        self.train_frac, self.val_frac, self.purge_ns = float(train_frac), float(val_frac), int(purge_ns)
        self.n_folds = int(max_folds if max_folds is not None else n_folds)
        self.min_train = int(min_train)

    def __iter__(self) -> Iterator[Fold]:
        N = len(self.order)
        test_frac = 1.0 - self.train_frac - self.val_frac
        assert 0 < test_frac < 1, "train_frac + val_frac must be < 1"
        # fold k: the test block is the k-th of n_folds equal slices of the last test_frac * N episodes (expanding train)
        n_test_total = max(1, int(round(test_frac * N)))
        first_test = N - n_test_total
        bounds = np.linspace(first_test, N, self.n_folds + 1).astype(int)
        for k in range(self.n_folds):
            te_lo, te_hi = int(bounds[k]), int(bounds[k + 1])
            if te_hi <= te_lo:
                continue
            te = self.order[te_lo:te_hi]
            t_test0 = int(self.anchor[te].min())
            pre = self.order[:te_lo]
            pre = pre[self.anchor[pre] <= t_test0 - self.purge_ns]          # purge gap in wall-clock time
            n_pre = len(pre)
            n_val = int(round(self.val_frac / (self.train_frac + self.val_frac) * n_pre))
            n_tr = n_pre - n_val
            if n_tr < self.min_train:
                continue
            yield Fold(k, np.sort(pre[:n_tr]), np.sort(pre[n_tr:]), np.sort(te))

    def __len__(self) -> int:
        return sum(1 for _ in self)


class EpisodeDataset(EventDataset):
    """All decision rows of the given episodes (lookback <= t < T - hmax), nothing else."""

    def __init__(self, panel: Panel, assets: Sequence[int], lookback: int, horizons: Sequence[int], latency: int = 1, exec_max: int = 4,
                 vol_window: int = 60, drop_nan: bool = True, neg_subsample: Optional[float] = None, rng: Optional[np.random.Generator] = None):
        super().__init__(panel, np.arange(panel.T), lookback, horizons, latency, exec_max, vol_window, drop_nan, None, rng)
        keep = np.isin(self.index[:, 1], np.asarray(assets, dtype=int))
        self.index = self.index[keep]
        if neg_subsample is not None and neg_subsample < 1.0 and len(self.index):
            rng = rng or np.random.default_rng(0)
            y = panel.labels[self.index[:, 0], self.index[:, 1]]
            self.index = self.index[(y == 1) | (rng.random(len(self.index)) < neg_subsample)]
        self.assets = np.asarray(assets, dtype=int)
