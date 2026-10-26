# B6 — xp_bolz24 (crosspump/baselines/b6_bolz24.py)

**Citation.** Bolz, Bründler, Kane, Patsias, Tessendorf, Gogol, Kim, Tessone, "Machine Learning-Based Detection of
Pump-and-Dump Schemes in Real-Time", arXiv:2412.18848 (Dec 2024); v2 "Real-Time Machine Learning Detection of
Telegram-Based Pump-and-Dump Schemes", ACM 2025, doi:10.1145/3733815.3764042.

**Code pointers.** No public code was located; nothing on c225. The implementation follows the method as summarised in
the track specification (Z-score model on order-flow features + classifiers ranking candidate coins). No file/line
pointers exist; every numeric setting below is ours.

**Data the paper used.** Live Telegram pump-channel messages plus high-frequency order-book / trade data of the
candidate coins on the announced exchange; reported top-5 hit rate 55.8 %, top-10 74.4 % for the target coin seconds
before the pump.

**Split / threshold conventions in the paper.** Ranking metric (top-k hit rate among candidate coins per announced
pump); the Z-score threshold and the train/test split are not taken from the paper here (see unspecified choices).
In CrossPump the alarm threshold is set by the shared gate (delayed ACI at alpha = 0.01), not by the baseline.

**What was built.**
- Per episode, causal EWMA z-scores (statistics from bars < t; halflife 60 bars; 30-bar warm-up -> NaN) of
  `taker_imb`, `log_volume`, `log_trades`; `z_peak[t]` = max over features and over the last `back` = 3 bars.
- Social / text evidence: sum of `mark[2]` (text_score) and `log1p(count)` of social events usable at t over the last
  `social_back` = 6 bars (availability-based rasters, an event counts from `ceil(avail)`).
- Combination: logistic regression (balanced class weights) on `[z_peak, log1p(social_count), social_text_sum]`
  fitted on train-episode rows only (`labels_for(train_assets)`); score = decision function. `combine: additive`
  gives the untrained `z_peak + beta * social_text_sum`.
- NaN (abstain) during the z-score warm-up or when the bar features are missing.

**Adaptations and why.** Seconds-level order-book stream -> 5-minute bars with taker-buy imbalance / volume / trade
count as order-flow proxies (the only order-flow fields every CrossPump panel has). Cross-sectional ranking of
candidate coins -> per-episode time score (the benchmark's unit is a decision bar of one episode).

**Unspecified choices (and the fix).**
1. Rolling baseline of the Z-score -> EWMA, halflife 60 bars, warm-up 30 bars.
2. Which order-flow features -> taker_imb, log_volume, log_trades.
3. Aggregation of several z-scores -> max over features and over the last 3 bars.
4. How the Telegram signal enters -> text_score sum + log1p(count) over 6 bars.
5. Classifier for the combination -> logistic regression, C = 1 (additive fallback when no two-class train rows).

**Fallbacks.** If sklearn's LogisticRegression fails or the train rows are single-class: additive combination
(`diag['combine'] = 'additive'`, reason in `diag['combine_fallback']`).
