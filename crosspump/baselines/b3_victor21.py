"""B3 — Victor & Weintraud, "Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges",
The Web Conference (WWW) 2021, doi:10.1145/3442381.3449824. Registry key: xp_victor21.

What the paper does. On-chain only (EtherDelta / IDEX trade logs). Per token it builds the directed trade graph among
wallet addresses, finds strongly connected components, and inside them flags sets of trades that form closed loops
(self-trades, 2- and 3-account cycles, larger SCCs) in which every participant's net position change within a time window
is (close to) zero — i.e. the traded volumes match up to a small margin; the wash-traded volume share per token is reported.
There is no learning and no labels.

Adaptation to the wallet stream (unsupervised, no fitting). The stream carries one wallet id per transfer (source) and its
log size (mark[0]); there is no counterparty, so cycles are detected on the ordered sequence of wallet ids of an episode
(order = availability, then time). An event k CLOSES a cycle of length c in {1, 2, 3} when the same known wallet id appears
again c events earlier, the intermediate ids are different wallets, the two events are at most `window` bars apart and
their sizes match within `volume_tol` in log units (c = 1 = immediate repeat = self-trade, c = 2 = A B A, c = 3 = A B C A).
Per row t, over the wallet events usable in the last `window` bars:
    wash_share    size of cycle-closing events / total size                        (the paper's wash volume share)
    repeat_share  share of events whose wallet already acted within the window     (back-and-forth without the volume match)
    hhi_excess    Herfindahl concentration of inflow size over wallet ids, in excess of the uniform value 1/n_ids
    score = w_wash * wash_share + w_repeat * repeat_share + w_hhi * hhi_excess     (weights 1, 1, 1; fewer than
            `min_events` events -> 0 = no evidence).
Episodes without a wallet stream abstain (NaN): a missing modality is not "no activity". Everything is precomputed per
episode from events ordered by availability, so entry [t, i] depends only on events with avail <= t. Sheet: sheets/victor21.md.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .common import XPBaseline, register
from ._feats_d import CumRaster, CausalCoverage, usable_row, WALLET


@register("xp_victor21")
class Victor21Baseline(XPBaseline):
    name = "xp_victor21"
    paper = ("Victor & Weintraud, 'Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges', "
             "The Web Conference (WWW) 2021, doi:10.1145/3442381.3449824")
    year = 2021
    venue = "WWW 2021"
    official_repo = ""                      # no official code release located (sheet)
    status = "reimplemented"
    modalities = {"wallet"}
    unspecified = [
        "the paper needs maker AND taker addresses; the wallet stream has one wallet id per transfer, so cycles are detected on the ordered id sequence (A A / A B A / A B C A)",
        "time window: the paper sweeps hour / day / week windows over a token's whole history; here `window` = 48 bars of the episode",
        "volume match margin: the paper uses a small relative margin on net positions (1 %); here |log size difference| <= volume_tol (default 0.25 because sizes are ADV-normalised and coarse)",
        "the paper reports a volume share, not an alarm score: score = wash_share + repeat_share + hhi_excess with unit weights (config model.weights)",
        "Herfindahl concentration is this benchmark's addition requested by the task (wallet-id concentration of recent inflow), not part of the paper",
        "size of an event = exp(mark[0]) (mark[0] is a log size); unknown sizes count with weight 1; wallet id 0 (unknown) never closes a cycle",
    ]

    def _tables(self) -> np.ndarray:
        m = self.m
        T, N = self.T, self.N
        W = int(m.get("window", 48)); tol = float(m.get("volume_tol", 0.25)); min_ev = int(m.get("min_events", 2))
        w_wash, w_rep, w_hhi = [float(x) for x in m.get("weights", [1.0, 1.0, 1.0])]
        s = self.streams
        n_r = np.zeros((T, N)); size_r = np.zeros((T, N)); wash_r = np.zeros((T, N)); rep_r = np.zeros((T, N)); hhi = np.zeros((T, N))
        n_cyc = 0
        if len(s):
            sel = np.nonzero(s.platform == WALLET)[0]
            rows_all = usable_row(s)[sel]; ast = s.asset[sel]; av = s.avail[sel]; tm = s.time[sel]; src = s.source[sel]
            lg = s.mark[sel, 0].astype(np.float64)
            size = np.where(np.isfinite(lg), np.exp(np.clip(lg, -20, 20)), 1.0)
            for i in np.unique(ast):
                k = np.nonzero(ast == i)[0]
                k = k[np.lexsort((tm[k], av[k]))]                                           # order of availability: a later event never reorders earlier ones
                r, sc, sz, l = rows_all[k], src[k], size[k], lg[k]
                inb = r < T
                r, sc, sz, l = r[inb], sc[inb], sz[inb], l[inb]
                E = len(r)
                if not E:
                    continue
                closes = np.zeros(E, dtype=bool); repeat = np.zeros(E, dtype=bool)
                last_row: Dict[int, int] = {}
                for e in range(E):
                    a = int(sc[e])
                    if a == 0:
                        continue
                    if a in last_row and r[e] - last_row[a] < W:
                        repeat[e] = True
                    last_row[a] = int(r[e])
                    for c in (1, 2, 3):
                        j = e - c
                        if j < 0 or int(sc[j]) != a or r[e] - r[j] >= W:
                            continue
                        mid = sc[j + 1 : e]
                        if (mid == a).any() or (mid == 0).any() or len(set(mid.tolist())) != len(mid):
                            continue
                        if np.isfinite(l[e]) and np.isfinite(l[j]) and abs(l[e] - l[j]) > tol:
                            continue
                        closes[e] = True
                        break
                n_cyc += int(closes.sum())
                np.add.at(n_r[:, i], r, 1.0); np.add.at(size_r[:, i], r, sz); np.add.at(wash_r[:, i], r, sz * closes); np.add.at(rep_r[:, i], r, repeat.astype(float))
                # Herfindahl of size over wallet ids in rows (t - W, t]: two-pointer sweep keeping sum_s and sum_s^2
                per: Dict[int, float] = {}; tot = 0.0; sq = 0.0; a_ = 0; z_ = 0; uid = 0
                keys = np.where(sc != 0, sc, -(np.arange(E) + 1))                           # unknown ids are distinct wallets
                for t in range(T):
                    while a_ < E and r[a_] <= t:
                        kk = int(keys[a_]); old = per.get(kk, 0.0); new = old + sz[a_]
                        sq += new * new - old * old; tot += sz[a_]; per[kk] = new; a_ += 1
                    while z_ < a_ and r[z_] <= t - W:
                        kk = int(keys[z_]); old = per[kk]; new = old - sz[z_]
                        sq += new * new - old * old; tot -= sz[z_]
                        if new <= 1e-12:
                            per.pop(kk)
                        else:
                            per[kk] = new
                        z_ += 1
                    n_ids = len(per)
                    if n_ids >= 2 and tot > 0:
                        h = sq / (tot * tot)
                        hhi[t, i] = max(0.0, (h - 1.0 / n_ids) / (1.0 - 1.0 / n_ids))
                    elif n_ids == 1 and (a_ - z_) >= 2:
                        hhi[t, i] = 1.0                                                     # several transfers, a single wallet
        tt, ii = np.meshgrid(np.arange(T), np.arange(N), indexing="ij"); tt, ii = tt.ravel(), ii.ravel()
        n = CumRaster(n_r).window(tt, ii, W); tot = CumRaster(size_r).window(tt, ii, W)
        wash = CumRaster(wash_r).window(tt, ii, W) / np.maximum(tot, 1e-12); rep = CumRaster(rep_r).window(tt, ii, W) / np.maximum(n, 1.0)
        score = w_wash * wash + w_rep * rep + w_hhi * hhi.ravel()
        score = np.where(n >= min_ev, score, 0.0).reshape(T, N)
        self.parts = {"wash_share": np.where(n >= min_ev, wash, 0.0).reshape(T, N), "repeat_share": np.where(n >= min_ev, rep, 0.0).reshape(T, N), "hhi_excess": hhi}
        self.diag.update({"cycle_closing_events": int(n_cyc), "wallet_events": int(n_r.sum()), "window": W, "volume_tol": tol})
        return score

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        self.m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        self.cov = CausalCoverage(panel, self.counts, self.platforms).table(WALLET)        # (T, N)
        self.score = self._tables()                                                         # no labels, nothing fitted
        self.diag["episodes_with_wallet"] = int(self.cov[-1].sum())

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        s = self.score[t, i].astype(np.float64)
        s[~self.cov[t, i]] = np.nan
        return s
