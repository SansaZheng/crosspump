# B2 — xp_nghiem21 (Nghiem et al., ESWA 2021)

**Citation.** H. Nghiem, G. Muric, F. Morstatter, E. Ferrara, "Detecting cryptocurrency pump-and-dump frauds using
market and social signals", Expert Systems with Applications 182:115284, 2021, doi:10.1016/j.eswa.2021.115284.
Code: https://github.com/hnghiem-nlp/Pump_and_Dump_Crypto (clone read on c225 at
`/date/zyy/TradingSignal/data_raw/crosspump/nghiem`).

**Implementation.** `crosspump/baselines/b2_nghiem21.py` (status `reimplemented`), config
`configs/crosspump/baselines/nghiem21.yaml`.

## Code pointers (official repository)
| Notebook | Content |
|---|---|
| `Final_Analysis/Final_Mod_0_Format_Pump_Data.ipynb` | pump event table |
| `Final_Analysis/Final_Mod_1a_OHLCV_Hourly.ipynb`, `Final_Mod_1b/1c_Social_Hourly*.ipynb` | hourly OHLCV and social-count sequences per coin |
| `Final_Analysis/Final_Mod_2a_Normalize_OHLCV_Hourly.ipynb`, `Final_Mod_2b_Normalize_Social_Hourly.ipynb`, `Final_Mod_2c_All_Data.ipynb` | per-window normalisation and merge |
| `Final_Analysis/Final_Mod_3a_Conventional_NN_Predictiive.ipynb`, cell 21 | `create_OHLCV_model`: Dense [200, 200] relu -> Dense 1 sigmoid, Adam(lr 1e-3), binary cross-entropy |
| `Before_Pump_Analysis/RF_Model/Mod_5*.ipynb` | random-forest reference models (social / OHLCV / both) |

The CNN / BLSTM / CLSTM networks reported in the paper are **not** in the public repository (a search of every notebook
for Conv1D / LSTM / Bidirectional found only the import line of `Final_Mod_3a`, cell 2). The sequence architecture here
therefore follows the paper's description, with layer sizes fixed by us.

## Data used by the paper
Pump events announced in online pump groups (collected by the authors), with hourly market OHLCV and social-activity counts before each pump;
197 / 55 / 54 events for train / validation / test.

## Split and threshold conventions in the paper
Event-level train / validation / test split (197 / 55 / 54). Two tasks: target-coin prediction among candidate coins
(classification) and peak-price regression. No false-alarm budget; classification threshold from validation.

## What was adapted and why
* Input: a causal window of the last `lookback` bars (5-minute bars instead of hourly) with five market channels and
  three social channels; social events enter at the bar in which they become usable (`ceil(avail)`).
* Normalisation: the paper normalises a coin's window with statistics of the window; to stay point-in-time the volume
  and trade channels use expanding z-scores from strictly earlier bars, returns and range use train-episode scales.
* Network: CLSTM = Conv1d(k=3) -> ReLU -> MaxPool(2) -> LSTM -> sigmoid; `model.arch` also offers `blstm` and `cnn`.
* Target: per-row pump probability with class-weighted BCE; the peak-price regression head is not part of the alarm
  benchmark and is omitted.
* Rows with fewer than `lookback` bars of history abstain (NaN).

## Unspecified choices and how they were fixed
1. Layer sizes: Conv1d 32 filters (kernel 3), LSTM 32 units, dropout 0.1.
2. Optimiser: Adam 1e-3 (the repository's dense network), 8 epochs, batch 256, seed 0.
3. Class imbalance: pos_weight = n_neg / n_pos capped at 50; negatives sub-sampled to `max_train_rows` = 30 000.
4. Expanding z-score warm-up: 12 bars (earlier values -> 0 in the window).
5. Social channels: log1p(count), text_score mass and a has_social flag per bar.
