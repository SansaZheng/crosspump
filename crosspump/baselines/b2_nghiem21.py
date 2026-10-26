"""B2 — Nghiem, Muric, Morstatter, Ferrara, "Detecting cryptocurrency pump-and-dump frauds using market and social
signals", Expert Systems with Applications 182:115284, 2021, doi:10.1016/j.eswa.2021.115284.
Code: https://github.com/hnghiem-nlp/Pump_and_Dump_Crypto. Registry key: xp_nghiem21.

What the paper does. For 197 / 55 / 54 (train / val / test) pump events it builds, per candidate coin, hourly sequences of
market OHLCV and of social activity counts before the pump and trains CNN, BLSTM and CLSTM (convolution -> LSTM) networks
for (a) predicting the target coin and (b) regressing the peak price; inputs are normalised per coin window.
The public repository holds the data pipeline (Final_Analysis/Final_Mod_1*/2* notebooks: hourly OHLCV + social download and
normalisation) and a dense reference network (Final_Mod_3a_Conventional_NN_Predictiive.ipynb, create_OHLCV_model: Dense
[200, 200] relu -> sigmoid, Adam 1e-3, binary cross-entropy); the CNN / BLSTM / CLSTM code is NOT in the repository, so
the sequence architectures follow the paper's description.

Adaptation to the stacked-episode panel. One row = (bar t, episode i); input = the causal window of the last L = lookback
bars with 8 channels:
    log_ret / train std | expanding z of log_volume | expanding z of log_trades | hl_range / train mean | taker_imb |
    log1p(social messages usable in the bar) | social text_score mass of the bar | has_social flag
(z statistics = strictly earlier bars of the same episode, warm-up -> 0; social events are counted at ceil(avail)).
Network (model.arch): clstm = Conv1d(k=3) -> ReLU -> MaxPool(2) -> LSTM -> sigmoid (default), blstm, cnn. Trained with
class-weighted BCE on the rows of the train episodes with the episode label; the peak-price regression head is not used
(the benchmark scores alarms). Convention sheet: sheets/nghiem21.md.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch
from torch import nn

from .common import XPBaseline, register
from ._feats_d import (expanding_z, gather_windows, feat_cols, bar_column, CausalCoverage, subsample_rows, train_binary, predict_binary, SOCIAL)


class _SeqNet(nn.Module):
    def __init__(self, c_in: int, arch: str = "clstm", conv: int = 32, hidden: int = 32, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.arch = arch
        if arch in ("clstm", "cnn"):
            self.conv = nn.Sequential(nn.Conv1d(c_in, conv, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout))
            c_in = conv
        if arch in ("clstm", "blstm"):
            self.rnn = nn.LSTM(c_in, hidden, batch_first=True, bidirectional=(arch == "blstm"))
            c_out = hidden * (2 if arch == "blstm" else 1)
        else:
            c_out = c_in
        self.out = nn.Linear(c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                 # x (B, L, C)
        if self.arch in ("clstm", "cnn"):
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.arch in ("clstm", "blstm"):
            h, _ = self.rnn(x)
            z = h[:, -1]
        else:
            z = x.max(1).values
        return self.out(z).squeeze(-1)


@register("xp_nghiem21")
class Nghiem21Baseline(XPBaseline):
    name = "xp_nghiem21"
    paper = ("Nghiem, Muric, Morstatter, Ferrara, 'Detecting cryptocurrency pump-and-dump frauds using market and social signals', "
             "Expert Systems with Applications 182:115284, 2021, doi:10.1016/j.eswa.2021.115284")
    year = 2021
    venue = "Expert Systems with Applications 182"
    official_repo = "https://github.com/hnghiem-nlp/Pump_and_Dump_Crypto"
    status = "reimplemented"
    modalities = {"orderflow", "social"}
    unspecified = [
        "CNN / BLSTM / CLSTM code is not in the public repository (only a Dense [200,200] network): layer sizes fixed to Conv1d 32 (k=3) -> MaxPool 2 -> LSTM 32",
        "input resolution: the paper uses hourly (and daily) sequences; here the panel's bars (5 min) with a window of `lookback` bars",
        "normalisation: the paper normalises each coin window with its own statistics; here causal expanding z-scores (strictly earlier bars) and train-episode scales",
        "task: the paper ranks candidate coins per event and regresses the peak price; here per-row pump probability (the regression head is dropped)",
        "class imbalance handling is not specified: BCE with pos_weight = n_neg / n_pos (capped at 50) and negative sub-sampling to max_train_rows",
        "optimiser: Adam 1e-3 as in the repository's dense network; epochs / batch fixed in the config",
    ]

    def _channels(self, panel, train_assets: np.ndarray) -> np.ndarray:
        cols = feat_cols(panel, self.feat.shape[-1]); wu = int(self.m.get("warmup", 12))
        ret = bar_column(self.feat, cols, "log_ret"); hl = bar_column(self.feat, cols, "hl_range")
        tr = np.asarray(train_assets, dtype=int)
        self.ret_sd = float(np.nanstd(ret[:, tr])) if len(tr) and np.isfinite(ret[:, tr]).any() else 1.0
        self.hl_mu = float(np.nanmean(hl[:, tr])) if len(tr) and np.isfinite(hl[:, tr]).any() else 1.0
        self.ret_sd = self.ret_sd if self.ret_sd > 1e-12 else 1.0; self.hl_mu = self.hl_mu if abs(self.hl_mu) > 1e-12 else 1.0
        cov = CausalCoverage(panel, self.counts, self.platforms).table(SOCIAL).astype(np.float64)
        ch = [np.clip(ret / self.ret_sd, -20, 20), np.clip(expanding_z(bar_column(self.feat, cols, "log_volume"), wu), -20, 20),
              np.clip(expanding_z(bar_column(self.feat, cols, "log_trades"), wu), -20, 20), np.clip(hl / self.hl_mu, 0, 20), bar_column(self.feat, cols, "taker_imb"),
              np.log1p(self.counts[:, :, SOCIAL]), self.mark_sums[:, :, SOCIAL, 2], cov]
        return np.nan_to_num(np.stack(ch, -1)).astype(np.float32)

    def _windows(self, t: np.ndarray, i: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.nan_to_num(gather_windows(self.chan, t, i, self.L)), dtype=torch.float32)

    def _fit(self, panel, train_assets: np.ndarray, val_assets: np.ndarray) -> None:
        self.m: Dict[str, Any] = dict(self.cfg.get("model", {}))
        m = self.m; seed = int(m.get("seed", 0))
        self.chan = self._channels(panel, train_assets)                                    # (T, N, C); entry [t] uses bars / events <= t only
        y = self.labels_for(train_assets)
        t, i = self.rows_of(np.asarray(train_assets, dtype=int), lo=max(self.L - 1, 0)) if len(train_assets) else (np.zeros(0, int), np.zeros(0, int))
        ok = np.isfinite(y[t, i]) if len(t) else np.zeros(0, bool)
        t, i = t[ok], i[ok]; yy = y[t, i] if len(t) else np.zeros(0)
        keep = subsample_rows(yy, int(m.get("max_train_rows", 30_000)), seed)
        torch.manual_seed(seed)
        self.net = _SeqNet(self.chan.shape[-1], str(m.get("arch", "clstm")), int(m.get("conv", 32)), int(m.get("hidden", 32)), int(m.get("kernel", 3)), float(m.get("dropout", 0.1)))
        if len(keep) and 0 < yy[keep].sum() < len(keep):
            info = train_binary(self.net, [self._windows(t[keep], i[keep])], yy[keep], epochs=int(m.get("epochs", 8)), lr=float(m.get("lr", 1e-3)),
                                batch=int(m.get("batch", 256)), seed=seed, pos_weight_cap=float(m.get("pos_weight_cap", 50.0)))
        else:
            info = {"untrained": "single class in the training rows"}
            self.status = "reimplemented-fallback"
        self.net.eval()
        self.diag.update({"arch": str(m.get("arch", "clstm")), "train": info, "channels": int(self.chan.shape[-1])})

    def _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64); i = np.asarray(i, dtype=np.int64)
        out = np.full(len(t), np.nan)
        ok = t >= self.L - 1                                                               # full causal window required (warm-up abstains)
        if ok.any():
            out[ok] = predict_binary(self.net, [self._windows(t[ok], i[ok])])
        return out
