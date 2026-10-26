# B4 — xp_lamorgia23 (La Morgia et al., ACM TOIT 2023)

**Citation.** M. La Morgia, A. Mei, F. Sassi, J. Stefa, "The Doge of Wall Street: Analysis and Detection of Pump and
Dump Cryptocurrency Manipulations", ACM Transactions on Internet Technology 23(1), 2023, doi:10.1145/3561300,
arXiv:2105.00733 (conference version: ICCCN 2020).
Official repository: https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset (clone read on c225 at
`/date/zyy/TradingSignal/data_raw/crosspump/sapienza_pd`).

**Implementation.** `crosspump/baselines/b4_lamorgia23.py` (status `official-features-reimplemented`), config
`configs/crosspump/baselines/lamorgia23.yaml`.

## Code pointers (official repository)
| Official | Lines | Here (`_features`) |
|---|---|---|
| `features.py::std_rush_order_feature` | L9-19 | `std_rush_order_w` |
| `features.py::avg_rush_order_feature` | L22-32 | `avg_rush_order_w` |
| `features.py::std_trades_feature` | L35-41 | `std_trades_w` |
| `features.py::std_volume_feature` | L44-50 | `std_volume_w` |
| `features.py::avg_volume_feature` | L53-59 | `avg_volume_w` |
| `features.py::std_price_feature` | L62-67 | `std_price_w` |
| `features.py::avg_price_feature` | L70-75 | `avg_price_w` |
| `features.py::avg_price_max` | L78-83 | `avg_price_max_w` |
| `features.py::build_features` hour / minute sin / cos | L116-119 | `_feats_d.clock_features` (same formulas, divisors 23 and 59) |
| `classifier.py` feature list | L13-24 | same 12 feature families |
| `classifier.py` RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1) | L29 | `fit_forest(kind="rf", n_estimators=200, max_depth=5, seed=1)` |
| `classifier.py` StratifiedKFold(5) cross_val_predict | L37 | replaced by the chronological episode split |

Each official feature is `pct_change()` of a rolling statistic over `rolling_freq` chunks of 5 / 15 / 25 seconds of
BUY-side trades; a rush order is a group of buy trades with an identical millisecond timestamp (L10-12).

## Data used by the paper
Binance trade-level data around the pumps listed in `pump_telegram.csv` (Telegram groups; exchange == binance),
a few days of trades per event.

## Split and threshold conventions in the paper
Stratified 5-fold cross-validation over all chunks of all pumps (past and future events mixed); the chunk containing the
pump start is the positive; RandomForest hard prediction (probability 0.5), precision / recall / F1.

## What was adapted and why
* Bars instead of trades: rush bar = `taker_imb > 0` and trade count above its rolling mean over the previous 48 bars
  (strictly earlier bars); when the panel has no trade count, volume is used and the fact is recorded in `diag`.
* Chunk sizes become rolling windows {3, 12, 48} bars; each statistic keeps the official `pct_change` form (one-bar
  relative change of the rolling statistic). Prices are log prices rebuilt from `log_ret`, so the two average-price
  features use one-bar differences.
* Chronological train episodes replace the 5-fold CV (the official protocol trains on future pumps).
* Rows whose shortest-window features are undefined (warm-up) abstain (NaN). The alarm threshold is the shared gate.

## Unspecified choices and how they were fixed
1. Rush baseline: 48 bars, at least 12 (`model.rush_base`, `model.rush_min_periods`).
2. Windows: [3, 12, 48] bars (`model.windows`); min_periods = max(2, w // 2).
3. `pct_change` with a zero previous value: denominator |prev| + 1e-9, clipped to +-20 (the official code leaves inf).
4. Buy-side only statistics: not reproducible on bars; total volume / trades are used.
5. Class weighting: none (as official); all positive rows + sub-sampled negatives up to 200 000.
6. Clock features need `meta['bar_close_ns']`; zeros otherwise.
