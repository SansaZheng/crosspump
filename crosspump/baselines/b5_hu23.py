"""B5 — Hu, Zhang, Lu, He, Li, "Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump", SIGMOD 2023 /
Proc. ACM Manag. Data 1(1), doi:10.1145/3588686, arXiv:2204.12929.
Official code: https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency (TargetCoinPrediction/SeqModel).
Registry key: xp_hu23.

What the official code does (read on c225, data_raw/crosspump/hu_pd/TargetCoinPrediction/SeqModel/modeling.py).
    DNN  (L5-50)    target_features -> Dense 32 relu -> Dense 16 relu -> logit; sigmoid cross-entropy, Adam lr 5e-4.
    SNN  (L52-134)  the channel's sequence of previously pumped coins (max length 50): the 9 numeric statistics of each
                    past coin and its coin embedding are layer-normalised and pooled by POSITIONAL attention (a learned
                    (1, 50, 8) position table -> Dense 8 relu -> Dense 8, masked softmax over positions, L60-125); the
                    pooled sequence is concatenated with the candidate coin's features and coin embedding (L132) and fed
                    to the DNN head. SNNTA (L136-197) is the target-attention variant.
    One sample = (scheduled pump of a channel, candidate coin); negatives = other coins listed on the exchange.

Adaptation to the stacked-episode panel. One row = (bar t, episode i).
    candidate features   causal market features of the episode's coin at row t (cumulative return, volatility, expanding
                         z of volume / trades, taker imbalance, range over {3, 12, 48} bars), standardised with train rows.
    channel              the organiser is not given to the model: it is the known social source id with the largest prior
                         involvement (mark[3]) among the social events of episode i usable at t (avail <= t).
    sequence             the TRAIN pump episodes in which that source id appeared and whose window ended before episode i's
                         window starts (realised history, most recent first, max_seq); element = the market features of that
                         earlier episode at the bar before its label onset + its coin embedding.
    model                positional attention pooling (as L60-125) + channel embedding + candidate coin embedding + scalars
                         [log1p prior involvement, has_channel, log1p sequence length] -> Dense 32 -> Dense 16 -> logit, BCE.
    market branch only   when no known social source is usable at t the sequence, channel and scalars are zero, i.e. the
                         model reduces to the official DNN on the candidate features.
Convention sheet: sheets/hu23.md.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import nn

from .common import XPBaseline, register
from ._feats_d import (market_features, feat_cols, episode_order_times, usable_row, subsample_rows, train_binary, predict_binary, hash_bucket, id_bucket, SOCIAL)


class _SNN(nn.Module):
    def __init__(self, f_x: int, f_seq: int, max_seq: int, heads: int = 8, n_coin: int = 256, d_coin: int = 8, n_chan: int = 64, d_chan: int = 8,
                 hidden1: int = 32, hidden2: int = 16, use_coin: bool = True):
        super().__init__()
        self.use_coin = use_coin
        self.pos = nn.Parameter(torch.randn(1, max_seq, heads) * 0.02)
        self.pos_fx = nn.Sequential(nn.Linear(heads, 8), nn.ReLU(), nn.Linear(8, heads))
        self.norm = nn.LayerNorm(f_seq)
        self.chan = nn.Embedding(n_chan, d_chan, padding_idx=0)
        d_in = f_x + heads * f_seq + d_chan
        if use_coin:
            self.coin = nn.Embedding(n_coin, d_coin, padding_idx=0)
            self.coin_norm = nn.LayerNorm(d_coin)
            d_in += 2 * d_coin
        self.head = nn.Sequential(nn.Linear(d_in, hidden1), nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 1))

    def forward(self, x, seq, mask, chan, coin, seq_coin):
        m = mask.unsqueeze(-1)                                                    # (B, S, 1)
        att = self.pos_fx(self.pos).expand(seq.shape[0], -1, -1)                  # (B, S, H)
        att = torch.softmax(att.masked_fill(~m, -1e9), dim=1) * m                 # rows without history -> all zeros
        pooled = torch.einsum("bsh,bsf->bhf", att, self.norm(seq) * m).flatten(1)
        parts = [x, pooled, self.chan(chan)]
        if self.use_coin:
            parts += [self.coin(coin), torch.einsum("bs,bsd->bd", att.mean(-1), self.coin_norm(self.coin(seq_coin)) * m)]
        return self.head(torch.cat(parts, 1)).squeeze(-1)


@register("xp_hu23")
class Hu23Baseline(XPBaseline):
    name = "xp_hu23"
    paper = ("Hu, Zhang, Lu, He, Li, 'Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump', SIGMOD 2023 / Proc. ACM Manag. Data 1(1), "
             "doi:10.1145/3588686, arXiv:2204.12929")
    year = 2023
    venue = "SIGMOD 2023 / PACMMOD 1(1)"
    official_repo = "https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency"
    status = "reimplemented"
    modalities = {"social", "orderflow", "graph"}
    unspecified = [
        "sample unit: official = (scheduled pump, candidate coin) one hour before the announced time; here every bar of every episode (no announced schedule is given to the model)",
        "channel identity: official = the announcing channel; here the social source id with the largest prior involvement usable at t (ties: longer train history, then first seen); it never expires within the episode",
        "sequence elements: official = 9 statistics of each previously pumped coin + coin embedding; here that train episode's causal market features at the bar before its label onset",
        "history cut-off: an earlier train pump episode enters a sequence only if its window ended before the scored episode's window starts (bar_close_ns; fallback anchor order)",
        "candidate features: official = CoinGecko / Binance statistics over 1 h ... 72 h; here cumulative return, volatility, volume / trade z-scores, taker imbalance, range over {3, 12, 48} bars",
        "coin and channel embeddings are hashed into 256 / 64 buckets (the official vocabulary is the Binance coin list); max_seq 50, heads 8, hidden 32 / 16, Adam 5e-4 as official",
        "class imbalance: BCE with pos_weight = n_neg / n_pos (capped) and negative sub-sampling; the official negatives are the other listed coins of the same pump",
    ]

    # ------------------------------------------------------------------ causal channel table
    def _channel_tables(self, hist_len: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """(T, N) source id of the organiser as known at row t (0 = none yet) and its prior involvement (mark[3])."""
        T, N = self.T, self.N
        ch = np.zeros((T, N), dtype=np.int64); pr = np.zeros((T, N))
        s = self.streams
        if not len(s):
            return ch, pr
        sel = np.nonzero((s.platform == SOCIAL) & (s.source != 0))[0]
        rows = usable_row(s)[sel]; ast = s.asset[sel]; src = s.source[sel]; av = s.avail[sel]
        prior = np.nan_to_num(s.mark[sel, 3].astype(np.float64))
        known = np.array([(prior[k] >= self.min_prior) or (int(src[k]) in hist_len) for k in range(len(sel))], dtype=bool)
        for i in np.unique(ast[known]):
            k = np.nonzero((ast == i) & known)[0]
            k = k[np.lexsort((av[k], rows[k]))]
            r = rows[k]
            best_s = np.zeros(len(k), dtype=np.int64); best_p = np.zeros(len(k)); cur = (-1.0, -1, 0)
            for e, kk in enumerate(k):
                key = (float(prior[kk]), int(hist_len.get(int(src[kk]), 0)))
                if key > cur[:2]:
                    cur = (key[0], key[1], int(src[kk]))
                best_s[e], best_p[e] = cur[2], cur[0]
            pos = np.searchsorted(r, np.arange(T), side="right") - 1
            ok = pos >= 0
            ch[ok, i] = best_s[pos[ok]]; pr[ok, i] = best_p[pos[ok]]
        return ch, pr

    # ------------------------------------------------------------------ model inputs
    def _inputs(self, t: np.ndarray, i: np.ndarray) -> List[torch.Tensor]:
        S = self.max_seq; B = len(t)
        src = self.ch_id[t, i]; slot = self.slot_of(src)                                   # 0 = no train history for this source
        cnt = np.zeros(B, dtype=np.int64)
        for sl in np.unique(slot[slot > 0]):
            mk = slot == sl
            cnt[mk] = np.searchsorted(self.hist_end[sl], self.ep_start[i[mk]], side="left")  # train pump episodes that ended before episode i starts
        idx = cnt[:, None] - 1 - np.arange(S)[None, :]                                     # most recent first
        mask = idx >= 0
        ep = self.hist_eps[slot[:, None], np.clip(idx, 0, self.hist_eps.shape[1] - 1)]
        seq = np.where(mask[..., None], self.seq_feat[ep], 0.0)
        seq_coin = np.where(mask, self.coin_id[ep], 0)
        has = (src != 0).astype(np.float64)
        scal = np.stack([np.log1p(self.ch_prior[t, i]), has, np.log1p(mask.sum(1))], -1)
        x = np.concatenate([self.Xm[t, i], scal], -1)
        f32 = lambda a: torch.as_tensor(np.nan_to_num(a), dtype=torch.float32)
        return [f32(x), f32(seq), torch.as_tensor(mask), torch.as_tensor(id_bucket(src, self.n_chan)), torch.as_tensor(self.coin_id[i]), torch.as_tensor(seq_coin.astype(np.int64))]

    def slot_of(self, src: np.ndarray) -> np.ndarray:
        if not len(self.slot_src):
            return np.zeros(len(src), dtype=np.int64)
        p = np.clip(np.searchsorted(self.slot_src, src), 0, len(self.slot_src) - 1)
        return np.where(self.slot_src[p] == src, p + 1, 0).astype(np.int64)

    # ------------------------------------------------------------------ fit / score
    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        T, N = self.T, self.N; seed = int(m.get("seed", 0))
        self.max_seq = int(m.get("max_seq", 50)); self.n_chan = int(m.get("n_channel_buckets", 64)); n_coin = int(m.get("n_coin_buckets", 256))
        self.min_prior = float(m.get("min_prior", 1.0)); use_coin = bool(m.get("coin_embedding", True))
        tr = np.asarray(train_assets, dtype=int)
        # candidate (market) features, standardised with the train episodes
        cols = feat_cols(panel, self.feat.shape[-1])
        X, self.feat_names = market_features(self.feat, cols, [int(w) for w in m.get("windows", [3, 12, 48])], int(m.get("warmup", 12)))
        if len(tr):
            with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                mu = np.nanmean(X[:, tr].reshape(-1, X.shape[-1]), 0); sd = np.nanstd(X[:, tr].reshape(-1, X.shape[-1]), 0)
        else:
            mu = np.zeros(X.shape[-1]); sd = np.ones(X.shape[-1])
        mu = np.nan_to_num(mu); sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        self.ready = np.isfinite(X).any(-1)                                                # (T, N): something is known about the coin at t
        self.Xm = np.clip(np.nan_to_num((X - mu) / sd), -20, 20)
        # coin ids (hashed symbol)
        eps = panel.meta.get("episodes") or []
        sym = [str(e.get("symbol", k)) if isinstance(e, dict) else str(k) for k, e in enumerate(eps[:N])] + [f"ep{k}" for k in range(len(eps), N)]
        self.coin_id = np.array([hash_bucket(s_, n_coin) for s_ in sym], dtype=np.int64) if use_coin else np.zeros(N, dtype=np.int64)
        # realised history of the train pump episodes (labels only through labels_for)
        y = self.labels_for(tr)
        self.ep_start, ep_end = episode_order_times(panel, T, N)
        pump_tr = [int(j) for j in tr if np.nansum(y[:, j]) > 0]
        onset = {j: int(np.nonzero(y[:, j] == 1)[0][0]) for j in pump_tr}
        self.seq_feat = np.zeros((N, self.Xm.shape[-1]))
        for j in pump_tr:
            self.seq_feat[j] = self.Xm[max(onset[j] - 1, 0), j]
        s = self.streams
        hist: Dict[int, List[int]] = {}
        if len(s) and pump_tr:
            sel = (s.platform == SOCIAL) & (s.source != 0) & np.isin(s.asset, pump_tr)
            for a_, s_ in set(zip(s.asset[sel].tolist(), s.source[sel].tolist())):
                hist.setdefault(int(s_), []).append(int(a_))
        min_hist = int(m.get("min_history", 2))                                            # a source seen in >= 2 train pumps is a recurring organiser
        hist = {k: sorted(v, key=lambda j: ep_end[j]) for k, v in hist.items() if len(v) >= min_hist}
        self.slot_src = np.array(sorted(hist), dtype=np.int64)
        hmax = max([len(v) for v in hist.values()], default=1)
        self.hist_eps = np.zeros((len(hist) + 1, hmax), dtype=np.int64)
        self.hist_end: List[np.ndarray] = [np.zeros(0)]
        for sl, k in enumerate(self.slot_src.tolist(), start=1):
            v = hist[k]; self.hist_eps[sl, : len(v)] = v; self.hist_end.append(np.array([ep_end[j] for j in v], dtype=np.float64))
        self.ch_id, self.ch_prior = self._channel_tables({k: len(v) for k, v in hist.items()})
        # training
        t, i = self.rows_of(tr) if len(tr) else (np.zeros(0, int), np.zeros(0, int))
        ok = (np.isfinite(y[t, i]) & self.ready[t, i]) if len(t) else np.zeros(0, bool)
        t, i = t[ok], i[ok]; yy = y[t, i] if len(t) else np.zeros(0)
        keep = subsample_rows(yy, int(m.get("max_train_rows", 60_000)), seed)
        torch.manual_seed(seed)
        self.net = _SNN(self.Xm.shape[-1] + 3, self.Xm.shape[-1], self.max_seq, int(m.get("heads", 8)), n_coin, int(m.get("d_coin", 8)), self.n_chan, int(m.get("d_channel", 8)),
                        int(m.get("hidden1", 32)), int(m.get("hidden2", 16)), use_coin)
        if len(keep) and 0 < yy[keep].sum() < len(keep):
            info = train_binary(self.net, self._inputs(t[keep], i[keep]), yy[keep], epochs=int(m.get("epochs", 10)), lr=float(m.get("lr", 5e-4)), batch=int(m.get("batch", 256)),
                                seed=seed, pos_weight_cap=float(m.get("pos_weight_cap", 50.0)), weight_decay=float(m.get("weight_decay", 1e-5)))
        else:
            info = {"untrained": "single class in the training rows"}; self.status = "reimplemented-fallback"
        self.net.eval()
        self.diag.update({"train": info, "recurring_sources": int(len(hist)), "train_pump_episodes": int(len(pump_tr)), "max_seq": self.max_seq, "coin_embedding": use_coin,
                          "rows_with_channel_share": float((self.ch_id != 0).mean())})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        out = np.full(len(t), np.nan)
        ok = self.ready[t, i]
        for a in range(0, int(ok.sum()), 8192):
            k = np.nonzero(ok)[0][a : a + 8192]
            out[k] = predict_binary(self.net, self._inputs(t[k], i[k]))
        return out
