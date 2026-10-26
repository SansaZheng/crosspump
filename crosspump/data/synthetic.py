"""xp_synth: synthetic cross-platform pump testbed with a PLANTED excitation chain social -> wallet -> order flow.

Each episode is a window of `pre_bars + post_bars + 1` bars (bar_minutes each) around an anchor row (= pre_bars).
Pump episodes (is_pump = True)
    accumulation  wallet inflow events from anchor - acc_bars with a rising rate (insiders buying, Hu et al. 2023: price
                  drifts up and trading thickens 57-60 h before the pump); every wallet event excites order-flow burst
                  events after a short delay (wallet -> orderflow kernel); the price drifts up with the order-flow intensity
    teaser        on a `teaser_frac` share of pumps a few pre-announcement social mentions (hint messages) from the organiser
                  channel, which excite wallet events (social -> wallet kernel)
    announcement  a social burst at the anchor (organiser channel + followers), an order-flow burst, a price jump over
                  [anchor, peak] (peak_delay bars) and a partial dump afterwards
Control episodes (is_pump = False): background activity only; a `single_burst_frac` share carries ONE single-platform burst
(social-only hype or order-flow-only volatility) so that cross-platform linkage, not any single burst, is what separates
pumps from controls. Coverage: a `wallet_missing_frac` share of episodes has NO wallet stream (missing modality, as for
coins whose chain is not indexed), recorded in meta['coverage'].

Streams: social sources = channel ids (a few organiser channels recur across pumps -> prior involvement), wallet sources =
wallet ids (repeat-offender wallets recur), order-flow source = 0. Availability: social = time + social_lag bars,
wallet = time + wallet_lag bars (block confirmation), orderflow = bar close. Labels: 1 on [anchor - lead, peak].
Registered as dataset "xp_synth". Everything here is synthetic BY DESIGN (smoke tests and the E0 power study only).
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from warn2trade.utils.registry import DATASETS
from causalgate.data.pit import attach_availability, default_availability
from .streams import EventStreams, attach_streams, add_prior_involvement, PLATFORMS


def _register(name: str):
    def deco(cls):
        try:
            DATASETS.register(name)(cls)
        except KeyError:
            pass
        return cls
    return deco


def _poisson_times(rng: np.random.Generator, rate: np.ndarray, t0: float) -> np.ndarray:
    """Inhomogeneous Poisson times on a per-bar rate array (events uniformly inside each bar)."""
    n = rng.poisson(np.clip(rate, 0, None))
    out = []
    for k, c in enumerate(n):
        if c:
            out.extend((t0 + k + rng.random(c)).tolist())
    return np.asarray(out, dtype=np.float64)


@_register("xp_synth")
class SyntheticCrossPlatformBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "xp_synth", "5min", 365 * 24 * 12, (1, 2, 3, 6, 12, 24)

    def build_panel(self) -> Panel:
        c = self.cfg
        rng = np.random.default_rng(int(c.get("seed", 0)))
        n_pump, n_ctrl = int(c.get("n_pumps", 40)), int(c.get("n_controls", 80))
        pre, post, lead = int(c.get("pre_bars", 144)), int(c.get("post_bars", 48)), int(c.get("lead", 2))
        bar_min = int(c.get("bar_minutes", 5))
        acc_bars, teaser_bars = int(c.get("acc_bars", 100)), int(c.get("teaser_bars", 30))
        peak_delay, dump_bars = int(c.get("peak_delay", 2)), int(c.get("dump_bars", 12))
        chain = float(c.get("chain_strength", 1.0))                   # scales social->wallet and wallet->orderflow excitation
        teaser_frac, single_burst_frac = float(c.get("teaser_frac", 0.6)), float(c.get("single_burst_frac", 0.5))
        wallet_missing_frac = float(c.get("wallet_missing_frac", 0.25))
        soc_lag, wal_lag = float(c.get("social_lag_bars", 0.2)), float(c.get("wallet_lag_bars", 0.5))
        bg = dict(c.get("background", {"social": 0.08, "wallet": 0.03, "orderflow": 0.05}))
        sigma = float(c.get("sigma", 0.004)); jump = float(c.get("pump_jump", 0.35)); revert = float(c.get("dump_revert", 0.7))
        n_channels, n_wallets = int(c.get("n_channels", 6)), int(c.get("n_wallets", 40))
        T, N = pre + post + 1, n_pump + n_ctrl
        hmax = max(self.horizons)
        assert lead + 2 < pre and peak_delay + dump_bars < post

        # ---- episodes in chronological order (pumps and controls interleaved)
        is_pump = np.zeros(N, dtype=bool); is_pump[rng.choice(N, n_pump, replace=False)] = True
        anchor_ns = (np.datetime64("2021-01-01T00:00") + np.arange(N) * np.timedelta64(6, "h")).astype("datetime64[ns]").astype(np.int64)
        anchor_ns = anchor_ns + rng.integers(0, 3600, N) * 10 ** 9
        has_wallet = rng.random(N) >= wallet_missing_frac
        ret = rng.standard_normal((T, N)) * sigma
        base_logvol = 9.0 + 0.4 * rng.standard_normal((T, N))
        taker = 0.08 * rng.standard_normal((T, N))
        trades = np.exp(base_logvol - 5.0)
        of_intensity = np.zeros((T, N))                                    # realised order-flow burst intensity (events per bar)
        labels = np.zeros((T, N), dtype=np.int8); btp = np.full((T, N), np.nan)
        avail_label = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None] + hmax, (T, N)).copy()
        peak_row = np.full(N, -1, dtype=np.int64)
        A, TM, PL, MK, AV, SRC = [], [], [], [], [], []
        episodes: List[Dict[str, Any]] = []
        organiser = rng.integers(1, n_channels + 1, N)                      # channel id per episode (1..n_channels)
        offenders = rng.choice(np.arange(1, n_wallets + 1), size=max(3, n_wallets // 4), replace=False)

        def add(i, times, plat, marks, src, lag):
            for t_, m_, s_ in zip(times, marks, src):
                if 0 <= t_ < T - 1e-6:
                    A.append(i); TM.append(float(t_)); PL.append(plat); MK.append(m_); AV.append(min(float(t_) + lag, T - 1e-6)); SRC.append(int(s_))

        for i in range(N):
            # background activity on every platform (wallet only when the chain is indexed)
            for p, key in enumerate(PLATFORMS):
                if key == "wallet" and not has_wallet[i]:
                    continue
                ts = _poisson_times(rng, np.full(T, bg[key]), 0.0)
                if key == "social":
                    add(i, ts, 0, [[1.0, rng.uniform(2, 6), rng.uniform(0.0, 0.4), np.nan] for _ in ts], rng.integers(100, 10_000, len(ts)), soc_lag)
                elif key == "wallet":
                    add(i, ts, 1, [[rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.5), np.nan, np.nan] for _ in ts], rng.integers(1000, 100_000, len(ts)), wal_lag)
                else:
                    add(i, ts, 2, [[rng.uniform(2.5, 3.5), np.nan, np.nan, np.nan] for _ in ts], [0] * len(ts), 0.0)
            if is_pump[i]:
                pk = pre + peak_delay; peak_row[i] = pk
                end = min(T - 1, pk + dump_bars)
                # accumulation: wallet inflows with a rising rate, each exciting order-flow bursts (wallet -> orderflow)
                wl_rate = np.zeros(T); k = np.arange(acc_bars)
                wl_rate[pre - acc_bars : pre] = 0.05 + 0.35 * (k / acc_bars) ** 2
                # teaser social mentions (social -> wallet): a few hints from the organiser channel
                if rng.random() < teaser_frac:
                    teaser_t = np.sort(rng.uniform(pre - teaser_bars, pre - 2, rng.integers(2, 6)))
                    add(i, teaser_t, 0, [[1.0, 5.0, 0.7, np.nan] for _ in teaser_t], [organiser[i]] * len(teaser_t), soc_lag)
                    for tt in teaser_t:
                        wl_rate[int(tt) : int(tt) + 4] += 0.6 * chain
                wl_times = _poisson_times(rng, wl_rate, 0.0) if has_wallet[i] else np.zeros(0)
                if has_wallet[i]:
                    srcs = [int(rng.choice(offenders)) if rng.random() < 0.5 else int(rng.integers(1000, 100_000)) for _ in wl_times]
                    add(i, wl_times, 1, [[rng.uniform(0.8, 2.0), rng.uniform(1.0, 3.0), np.nan, np.nan] for _ in wl_times], srcs, wal_lag)
                of_rate = np.zeros(T)
                for tw in (wl_times if has_wallet[i] else _poisson_times(rng, wl_rate, 0.0)):   # hidden accumulation still moves order flow
                    j = int(tw)
                    of_rate[j : j + 3] += 0.8 * chain * np.array([0.5, 0.35, 0.15])[: max(0, min(3, T - j))]
                # announcement burst and pump
                of_rate[pre : pk + 1] += 6.0
                of_rate[pk + 1 : end + 1] += np.linspace(3.0, 0.5, end - pk)
                of_times = _poisson_times(rng, of_rate, 0.0)
                add(i, of_times, 2, [[rng.uniform(3.0, 6.0), np.nan, np.nan, np.nan] for _ in of_times], [0] * len(of_times), 0.0)
                ann_times = np.concatenate([[pre + 0.02], np.sort(pre + rng.random(rng.integers(15, 40)) * (peak_delay + 1))])
                src_ann = [organiser[i]] + rng.integers(100, 10_000, len(ann_times) - 1).tolist()
                add(i, ann_times, 0, [[1.0, 6.0 if k == 0 else rng.uniform(2, 5), 0.95 if k == 0 else rng.uniform(0.5, 0.9), np.nan] for k in range(len(ann_times))], src_ann, soc_lag)
                of_intensity[:, i] = of_rate
                # prices: accumulation drift + jump over [pre, pk] + partial dump
                ret[:, i] += 0.004 * np.clip(of_rate, 0, 2) * (np.arange(T) < pre)
                ret[pre : pk + 1, i] += jump / (peak_delay + 1)
                ret[pk + 1 : end + 1, i] -= revert * jump / dump_bars
                taker[pre - acc_bars : pk + 1, i] += 0.15 * np.clip(of_rate[pre - acc_bars : pk + 1], 0, 3) / 3
                taker[pk + 1 : end + 1, i] -= 0.2
                labels[max(0, pre - lead) : pk + 1, i] = 1
                rows = np.arange(max(0, pre - lead), end + 1); btp[rows, i] = pk - rows; avail_label[rows, i] = end
            elif rng.random() < single_burst_frac:
                # one single-platform burst (hype without accumulation, or volatility without social)
                t0 = int(rng.integers(lead + 5, T - dump_bars - 2))
                if rng.random() < 0.5:
                    ts = np.sort(t0 + rng.random(rng.integers(10, 30)) * 3)
                    add(i, ts, 0, [[1.0, rng.uniform(2, 6), rng.uniform(0.4, 0.9), np.nan] for _ in ts], rng.integers(100, 10_000, len(ts)), soc_lag)
                else:
                    r = np.zeros(T); r[t0 : t0 + 4] = 4.0
                    ts = _poisson_times(rng, r, 0.0)
                    add(i, ts, 2, [[rng.uniform(3.0, 5.0), np.nan, np.nan, np.nan] for _ in ts], [0] * len(ts), 0.0)
                    ret[t0 : t0 + 4, i] += rng.choice([-1, 1]) * 0.03
                    of_intensity[:, i] = r
            episodes.append({"key": f"SYN{i:04d}", "symbol": f"SYN{i:04d}", "exchange": "synthetic", "channel": int(organiser[i]), "anchor_ns": int(anchor_ns[i]),
                             "peak_ns": int(anchor_ns[i] + peak_row[i] * bar_min * 60 * 10 ** 9) if is_pump[i] else None, "is_pump": bool(is_pump[i]),
                             "control_of": None, "source": "xp_synth", "has_wallet": bool(has_wallet[i])})

        logvol = base_logvol + 0.5 * np.log1p(of_intensity)
        mid = 1.0 * np.exp(np.cumsum(ret, 0))
        high = mid * np.exp(np.abs(rng.standard_normal((T, N))) * sigma); low = mid * np.exp(-np.abs(rng.standard_normal((T, N))) * sigma)
        feats = np.stack([ret, logvol, (high - low) / mid, np.clip(taker, -0.5, 0.5), np.log1p(trades * (1 + of_intensity))], -1).astype(np.float32)
        half_spread = np.clip(0.002 * (1 + 0.5 * np.log1p(of_intensity)), 1e-4, 0.05)
        adv = np.full((T, N), 0.05)
        liq = np.stack([half_spread, np.ones((T, N)), adv], -1).astype(np.float32)
        bar_close_ns = anchor_ns[None, :] + ((np.arange(T) - pre + 1)[:, None]) * bar_min * 60 * 10 ** 9
        meta = {"freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(self.horizons), "raw_features": True, "ret_col": 0,
                "feat_names": ["log_ret", "log_volume", "hl_range", "taker_imb", "log_trades"], "dataset": "xp_synth", "source": "synthetic (crosspump/data/synthetic.py)",
                "licence": "n/a", "bar_minutes": bar_min, "pre_bars": pre, "post_bars": post, "lead": lead, "anchor_ns": anchor_ns, "peak_row": peak_row, "is_pump": is_pump,
                "episodes": episodes, "split_axis": "assets", "assets_time_aligned": False, "bar_close_ns": bar_close_ns, "n_pumps": int(is_pump.sum()),
                "planted": {"acc_bars": acc_bars, "teaser_bars": teaser_bars, "peak_delay": peak_delay, "dump_bars": dump_bars, "chain_strength": chain,
                            "teaser_frac": teaser_frac, "single_burst_frac": single_burst_frac, "wallet_missing_frac": wallet_missing_frac},
                "label_rule": {"type": "planted", "label_on": "[anchor - lead, peak]"}, "times": "bar index; per-episode close times in meta['bar_close_ns']"}
        panel = Panel(times=np.arange(T, dtype=np.int64), assets=[e["key"] for e in episodes], mid=mid, price_feat=feats, liq=liq, tradable=np.ones((T, N), dtype=bool),
                      labels=labels, bars_to_peak=btp, meta=meta)
        panel.validate()
        attach_availability(panel, avail_label=avail_label, universe=np.ones((T, N), dtype=bool))
        default_availability(panel)
        es = EventStreams.from_lists(A, TM, PL, MK, AV, SRC, platform_lags={"social": soc_lag, "wallet": wal_lag, "orderflow": 0.0},
                                     entities={"channels": {int(k): f"channel_{k}" for k in range(1, n_channels + 1)}}, built="planted (synthetic)")
        es = add_prior_involvement(es, np.argsort(np.argsort(anchor_ns)), is_pump)
        attach_streams(panel, es)
        return panel
