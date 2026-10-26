"""B7 — Fu, Feng, Wu, Xu, "PERSEUS: Tracing the Masterminds Behind Cryptocurrency Pump-and-Dump Schemes",
arXiv:2503.01686 (Mar 2025). No public code. Registry key: xp_perseus25.

What the paper does. A GNN over temporal attributed graphs of the social diffusion network (who relays the pump
message to whom, direction of information flow, community structure) identifies the masterminds behind
Telegram / X pump-and-dump campaigns (438 found, Feb-Oct 2024); OSN data plus market data serve as validation.

Adaptation (no torch_geometric; plain torch, dense adjacency):
    graph at (i, t)   social events of episode i with avail <= t. Node = source id (channel / account; an unknown
                      source id 0 is its own anonymous node). Directed edge u -> v when v's first post follows u's first
                      post within `gap` bars (information flows from earlier to later posters). The `max_nodes` most
                      recently active nodes are kept (cap 256).
    node features     [log1p(#posts), mean reach (mark[1]), max text_score (mark[2]), max prior involvement (mark[3],
                      causal track record), bars since last post, bars since first post, source known?, #posts / #all]
    model             2-layer GraphSAGE-style mean aggregation with separate in-neighbour (predecessors) and
                      out-neighbour (successors) means, ReLU, mean+max pooling + log1p(#nodes) -> linear head.
    training          BCE (pos_weight = #neg/#pos) on train-episode rows: every positive row plus every k-th negative
                      row; labels only through labels_for(train_assets); torch seeded (deterministic on CPU).
    inference         every k-th bar (t' = k * floor(t / k)) and forward-filled to the bars in between (information up to
                      t' <= t only), cached per (episode, t'); rows whose graph is still empty (no social event usable at t') abstain
                      (NaN) - a causal coverage rule, so a score never depends on events arriving later.
Convention sheet: crosspump/baselines/sheets/perseus25.md.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from .common import XPBaseline, register
from ._feats_e import seen_mask

N_FEAT = 8


class _SAGE(nn.Module):
    """Two mean-aggregation layers over predecessors and successors (dense adjacency), pooled to one logit per graph."""

    def __init__(self, f_in: int, hidden: int):
        super().__init__()
        self.s1, self.i1, self.o1 = nn.Linear(f_in, hidden), nn.Linear(f_in, hidden, bias=False), nn.Linear(f_in, hidden, bias=False)
        self.s2, self.i2, self.o2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(2 * hidden + 1, 1)

    def forward(self, x: torch.Tensor, A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (G, n, F); A (G, n, n) with A[g, u, v] = 1 for an edge u -> v; mask (G, n) real nodes
        A_in = A.transpose(1, 2)                                                   # A_in[v, u] = 1: u is a predecessor of v
        d_in = A_in.sum(-1, keepdim=True).clamp(min=1.0); d_out = A.sum(-1, keepdim=True).clamp(min=1.0)
        m = mask.unsqueeze(-1)
        h = F_.relu(self.s1(x) + self.i1(torch.bmm(A_in, x) / d_in) + self.o1(torch.bmm(A, x) / d_out)) * m
        h = F_.relu(self.s2(h) + self.i2(torch.bmm(A_in, h) / d_in) + self.o2(torch.bmm(A, h) / d_out)) * m
        n = mask.sum(-1, keepdim=True)
        mean = h.sum(1) / n.clamp(min=1.0)
        mx = (h + (m - 1.0) * 1e4).max(1).values
        mx = torch.where(n > 0, mx, torch.zeros_like(mx))
        return self.head(torch.cat([mean, mx, torch.log1p(n)], -1)).squeeze(-1)


@register("xp_perseus25")
class Perseus25Baseline(XPBaseline):
    name = "xp_perseus25"
    paper = "Fu, Feng, Wu, Xu, 'PERSEUS: Tracing the Masterminds Behind Cryptocurrency Pump-and-Dump Schemes', arXiv:2503.01686 (2025)"
    year = 2025
    venue = "arXiv 2503.01686 (Mar 2025)"
    official_repo = ""                     # none public
    status = "reimplemented"
    modalities = {"social", "graph"}
    unspecified = [
        "graph construction from a message stream without explicit forward/reply links: edge u -> v when v's first post follows u's within `gap` bars",
        "node attributes: paper uses account profile + content embeddings; here the 8 stream-derived features listed in the module docstring",
        "GNN architecture (layers, hidden size, aggregation): 2-layer mean aggregation (in/out), hidden 16-32, no torch_geometric",
        "episode-level output: the paper labels accounts (masterminds); here graphs are pooled (mean + max + log1p(#nodes)) to one logit per decision bar",
        "decision cadence: scored every k-th bar and forward-filled (k = model.score_every) to bound run time",
        "node cap for dense adjacency: 256 most recently active sources",
    ]

    # ------------------------------------------------------------------ graphs
    def _prepare_events(self) -> None:
        """Per episode: social events sorted by availability (arrays only; nothing else is read from the streams)."""
        es = self.streams
        soc = es.platform == 0 if len(es) else np.zeros(0, dtype=bool)
        self._ev: Dict[int, Dict[str, np.ndarray]] = {}
        if not soc.any():
            return
        a = es.asset[soc]; av = es.avail[soc]; tm = es.time[soc]; src = es.source[soc].astype(np.int64); mk = es.mark[soc].astype(np.float64)
        order = np.lexsort((av, a))
        a, av, tm, src, mk = a[order], av[order], tm[order], src[order], mk[order]
        for i in np.unique(a):
            lo = int(np.searchsorted(a, i, "left")); hi = int(np.searchsorted(a, i, "right"))
            self._ev[int(i)] = {"avail": av[lo:hi], "time": tm[lo:hi], "src": src[lo:hi], "mark": mk[lo:hi]}

    def _graph(self, i: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Node feature matrix (n, N_FEAT) and adjacency (n, n) of the diffusion graph of episode i as known at bar t."""
        ev = self._ev.get(int(i))
        if ev is None:
            return np.zeros((0, N_FEAT)), np.zeros((0, 0))
        k = int(np.searchsorted(ev["avail"], t + 1e-9, side="right"))              # events with avail <= t
        if k == 0:
            return np.zeros((0, N_FEAT)), np.zeros((0, 0))
        tm, src, mk = ev["time"][:k], ev["src"][:k], ev["mark"][:k]
        key = np.where(src == 0, -(np.arange(k) + 1), src)                              # anonymous posts = singleton nodes
        uniq, inv = np.unique(key, return_inverse=True)
        n = len(uniq)
        cnt = np.bincount(inv, minlength=n).astype(np.float64)
        first = np.full(n, np.inf); np.minimum.at(first, inv, tm)
        last = np.full(n, -np.inf); np.maximum.at(last, inv, tm)
        reach = np.bincount(inv, weights=np.nan_to_num(mk[:, 1]), minlength=n) / cnt
        txt = np.full(n, -np.inf); np.maximum.at(txt, inv, np.nan_to_num(mk[:, 2], nan=-np.inf)); txt[~np.isfinite(txt)] = 0.0
        prior = np.full(n, -np.inf); np.maximum.at(prior, inv, np.nan_to_num(mk[:, 3], nan=-np.inf)); prior[~np.isfinite(prior)] = 0.0
        if n > self.max_nodes:                                                            # keep the most recently active nodes
            keep = np.argsort(-last, kind="stable")[: self.max_nodes]
            keep.sort()
            cnt, first, last, reach, txt, prior, uniq = cnt[keep], first[keep], last[keep], reach[keep], txt[keep], prior[keep], uniq[keep]
            n = len(keep)
        sc = float(self.time_scale)
        x = np.stack([np.log1p(cnt), reach / 10.0, txt, np.log1p(prior), (t - last) / sc, (t - first) / sc, (uniq > 0).astype(np.float64), cnt / float(k)], -1)
        d = first[None, :] - first[:, None]                                               # d[u, v] = first_v - first_u
        A = ((d > 0) & (d <= self.gap)).astype(np.float64)
        return x, A

    def _batch(self, pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graphs = [self._graph(i, t) for i, t in pairs]
        n_max = max(1, max(g[0].shape[0] for g in graphs))
        G = len(graphs)
        X = np.zeros((G, n_max, N_FEAT)); A = np.zeros((G, n_max, n_max)); M = np.zeros((G, n_max))
        for g, (x, a) in enumerate(graphs):
            n = x.shape[0]
            if n:
                X[g, :n] = x; A[g, :n, :n] = a; M[g, :n] = 1.0
        return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(M, dtype=torch.float32)

    def _logits(self, pairs: List[Tuple[int, int]]) -> np.ndarray:
        out = np.zeros(len(pairs))
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(pairs), self.chunk):
                X, A, M = self._batch(pairs[s : s + self.chunk])
                out[s : s + self.chunk] = self.model(X, A, M).numpy()
        return out

    # ------------------------------------------------------------------ fit / score
    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        self.k = int(m.get("score_every", 2)); self.gap = float(m.get("gap_bars", 12)); self.max_nodes = int(m.get("max_nodes", 256))
        hidden = int(m.get("hidden", 16)); epochs = int(m.get("epochs", 30)); lr = float(m.get("lr", 1e-2)); self.chunk = int(m.get("chunk", 128))
        seed = int(m.get("seed", self.cfg.get("seed", 0))); self.time_scale = float(m.get("time_scale", max(self.L, 1)))
        torch.manual_seed(seed); np.random.seed(seed)
        self._prepare_events()
        self.seen_social = seen_mask(self.counts)[:, :, 0]                               # (T, N) causal coverage
        self.model = _SAGE(N_FEAT, hidden)
        self._cache: Dict[Tuple[int, int], float] = {}
        train_assets = np.asarray(train_assets, dtype=int)
        y = self.labels_for(train_assets)
        t, i = self.rows_of(train_assets)
        yy = y[t, i]
        keep = np.isfinite(yy) & self.seen_social[(t // self.k) * self.k, i] & ((yy > 0) | (t % self.k == 0))          # all positives + every k-th negative
        t, i, yy = t[keep], i[keep], yy[keep]
        self.diag.update({"train_graphs": int(len(t)), "train_pos": int(yy.sum()), "score_every": self.k, "gap_bars": self.gap, "hidden": hidden, "epochs": epochs})
        if len(t) == 0 or yy.sum() == 0 or yy.sum() == len(yy):
            self.diag["train_fallback"] = "no two-class training rows: untrained head (constant-ish scores)"
            return
        pairs = [(int(a), int(b)) for a, b in zip(i, t)]
        batches = []
        for s in range(0, len(pairs), self.chunk):
            X, A, M = self._batch(pairs[s : s + self.chunk])
            batches.append((X, A, M, torch.as_tensor(yy[s : s + self.chunk], dtype=torch.float32)))
        pos_w = torch.tensor(float((len(yy) - yy.sum()) / max(yy.sum(), 1.0)))
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()
        g = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            order = torch.randperm(len(batches), generator=g).tolist()
            tot = 0.0
            for b in order:
                X, A, M, yb = batches[b]
                opt.zero_grad()
                loss = F_.binary_cross_entropy_with_logits(self.model(X, A, M), yb, pos_weight=pos_w)
                loss.backward(); opt.step(); tot += float(loss)
            self.diag["last_loss"] = tot / max(len(batches), 1)
        self.model.eval()

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=int); i = np.asarray(i, dtype=int)
        tp = (t // self.k) * self.k                                                        # forward-fill from the last scored bar <= t
        pairs = sorted({(int(a), int(b)) for a, b in zip(i, tp)} - set(self._cache))
        if pairs:
            for p, v in zip(pairs, self._logits(pairs)):
                self._cache[p] = float(v)
        s = np.array([self._cache[(int(a), int(b))] for a, b in zip(i, tp)])
        s[~self.seen_social[tp, i]] = np.nan                                               # empty graph at t' -> abstain
        return s
