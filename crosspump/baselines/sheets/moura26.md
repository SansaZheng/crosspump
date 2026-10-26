# B9 — xp_moura26 (crosspump/baselines/b9_moura26.py)

**Citation.** Moura, Paoletti, Ferreira, Almeida, "Don't You Know, Pump it Up!", ICWSM 2027 (accepted),
arXiv:2609.01176; code https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027.

**Code pointers** (clone on c225 at `data_raw/crosspump/moura`, notebook `02_anomaly_detection.ipynb`, read over ssh):
- cell 5: `NUM_TRAINING_CELLS_PER_SIDE = 5`, `NUM_GUARD_CELLS_PER_SIDE = 1`, `P_FA = 0.05`, `USE_STL = False`,
  stablecoins USDT / USDC / DAI excluded.
- cell 13: daily series = number of messages per `message_day` with `prediction_id == 1` (RoBERTa crypto-related).
- cell 15: optional STL detrending (period 7, robust), off by default.
- cell 17 `ca_cfar_1d(signal, num_training_cells, num_guard_cells, p_fa)`: `alpha = N * (p_fa ** (-1 / N) - 1)`
  with N = training cells per side (4.1028 for N = 5, Pfa = 0.05); noise = average of the left-window mean and the
  right-window mean; "edge cells with only one valid window use that side's mean alone"; no valid training cell ->
  threshold = inf; detection iff `signal[i] > alpha * noise`.
- cell 25: detections -> `anomaly_time_intervals.json` (coin -> [{date, first_message_unix, last_message_unix}]).
- `01_classify.ipynb`: RoBERTa classifier `filipeasm18/roberta-binary-crypto-related-classifier` (not used here).
- `04_causality.ipynb`: RDD / DiD validation against market data (not part of the detector).

**Data the paper used.** TGDataset (La Morgia, Mei, Mongardini; Zenodo 7640712), last year Jul 2021 - Jul 2022,
14,499 channels, about 20 M messages; 681 coins with bursty days in the released reference data; 47 validated
P&D events (not released).

**Split / threshold conventions in the paper.** Unsupervised; no train/test split. The threshold is the CFAR
threshold `alpha * noise` at Pfa = 0.05 on DAILY counts, two-sided training windows (5 days each side, 1 guard day).

**What was built.** No training. Per episode the availability-based per-bar social count series; cell under test =
trailing `cell_bars` = 3 bars ending at the decision bar; 1 guard cell and 5 training cells on the PAST side only;
noise = mean of the training cells that lie inside the episode window; threshold = alpha * noise with the notebook's
alpha; score = log((count + eps) / (threshold + eps)), eps = 1. The score is > 0 exactly when the notebook rule would
declare a detection. NaN while no training cell exists (first 2 cells) and until the episode's first social event is
usable. With `cell_bars = 1` the threshold equals the notebook's left-window rule exactly (unit-tested against a
literal port in tests/crosspump_parts/test_baselines_e.py).

**Adaptations and why.** (a) Past-only training window: the right window uses future days, which the PIT rule forbids;
the notebook's own edge rule (one-sided mean) is applied at every cell. (b) Cell = 3 five-minute bars instead of one
day: episode windows are 3 days long, so daily cells would give 3 samples. (c) A graded score instead of a 0/1 flag,
because the shared gate needs a ranking.

**Unspecified choices (and the fix).**
1. Cell size on intraday bars -> 3 bars, sliding (evaluated at every bar).
2. One-sided window -> left only, alpha unchanged (N = 5 per side as in the notebook's formula).
3. Score -> log ratio with additive eps = 1 (detection boundary unchanged).
4. Relevance filter -> none inside the baseline (the loader decides which messages are in the social stream).
5. STL -> off (notebook default).

**Fallbacks.** None (numpy only).
