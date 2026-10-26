# B1 — xp_mirtaheri21 (Mirtaheri et al., IEEE TCSS 2021)

**Citation.** M. Mirtaheri, S. Abu-El-Haija, F. Morstatter, G. Ver Steeg, A. Galstyan, "Identifying and Analyzing
Cryptocurrency Manipulations in Social Media", IEEE Transactions on Computational Social Systems 8(3):607-617, 2021,
doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110.

**Implementation.** `crosspump/baselines/b1_mirtaheri21.py` (status `reimplemented`), config
`configs/crosspump/baselines/mirtaheri21.yaml`, helpers in `crosspump/baselines/_feats_d.py`.

## Code pointers
No official model code was located (the authors released data, not a training pipeline); `official_repo` is empty.
The reimplementation follows the paper text. Pointers inside this repository:

| Paper element | Here |
|---|---|
| tweet volume before the pump | `count_w` — `_features()`, `CumRaster(self.counts[:, :, SOCIAL])` |
| unique users | `unique_w` — `_feats_d.unique_source_raster` |
| audience / followers | `reach_w` — sum of `mark[1]` |
| sentiment | `text_sum_w`, `text_mean_w` — `mark[2]` (loader text_score) |
| Telegram pump-message classifier | `pump_msgs_w` — messages with `text_score >= pump_thr` |
| LR / RF / NN success classifier | `_feats_d.fit_forest` (RandomForest; LogisticRegression fallback) |

## Data used by the paper
Telegram pump-channel messages and Twitter posts about the targeted coins, both collected by the authors, with market
data used only to define the outcome (whether the attempt moved the price). Sizes and dates: see the paper; they are
not re-derived here.

## Split and threshold conventions in the paper
Classification of pump attempts (success vs failure) with standard held-out evaluation; reported with ranking /
classification metrics. No alarm threshold or false-alarm budget is defined: the paper does not run an online detector.

## What was adapted and why
* Unit of prediction: the paper scores a pump attempt once; the benchmark scores every bar of every episode, so the
  features are trailing-window statistics evaluated at each bar and the target is the episode label (1 on
  [anchor - lead, peak]).
* Two platforms (Twitter + Telegram) are one `social` platform in the stream schema; both feed the same features.
* The Telegram message classifier and the sentiment model are replaced by the loader's `text_score` mark, because
  message text is embedded offline by the loaders and only the score is part of the stream contract.
* Point-in-time: an event enters a window at the first row at which it is usable (`ceil(avail) <= t`); nothing else of
  the episode is read. Rows of episodes without a social stream abstain (NaN).
* The alarm threshold comes from the shared gate (false-alarm budget alpha), not from the paper.

## Unspecified choices and how they were fixed
1. Windows: {3, 12, 48, 144} bars (`model.windows`).
2. Pump-message threshold on text_score: 0.5 (`model.pump_thr`).
3. Classifier hyper-parameters (not reported): RandomForest, 200 trees, max_depth 8, class_weight balanced_subsample, seed 0.
4. Bot share and coin descriptors (market cap, exchange listings): not available in the streams, dropped.
5. Negative sub-sampling: all positive rows + random negatives up to `max_train_rows` (200 000).
6. Coverage when the loader gives no `has_social` flag: a row is covered once one social event is usable.
