# B8 — xp_pumpsense26 (crosspump/baselines/b8_pumpsense26.py)

**Citation.** Mahrous & Di Pietro, "PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on
Telegram", IEEE ICBC 2026, arXiv:2605.09431; data and code Zenodo doi:10.5281/zenodo.19616802 (MIT).

**Code pointers** (archive `icbc_2026_reproducibility-v1.1.zip`, folder `AhmedMahrous00-icbc_2026_reproducibility-2fdffd8/`,
read on c225 with `unzip -p`):
- `lightgbm_classification.py` L45-47: a sample = the messages of one window joined by newlines, label `nearby_pump`.
- `lightgbm_classification.py` L125-127: fixed `train_windows.csv` / `val_windows.csv` / `test_windows.csv`.
- `lightgbm_classification.py` L145: `TfidfVectorizer(ngram_range=(1, 2), max_features=args.features)`.
- `lightgbm_classification.py` L157-167: `LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
  num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', ...)`.
- `reproduce_paper_metrics.py` L374-404 `build_windows(corpus_df, window_size, window_type)`: odd window sizes,
  `centered` (i - half .. i + half) or `causal` (i - (window_size - 1) .. i), per group, sorted by time;
  L1142-1144 defaults `--window_sizes 7,11,15`, `--window_types centered,causal`, `--base_window_size 11`.
- `bge_classification.py`: the BGE-M3 fine-tuning variant (not reimplemented).

**Data the paper used.** 283,017 Telegram messages from 39 pump groups (mid-2017 to mid-2023), 2,246 manually
labelled pump-start announcements, 14 exchanges.

**Split / threshold conventions in the paper.** Pre-built train / val / test window files; main variant = centered
window of 11 messages (the centered window sees messages AFTER the one being classified); a causal variant is
reported as an ablation together with detection delay. Reported F1 0.79 (LightGBM) and 0.83 (BGE-M3).

**What was built.** Window = usable messages (avail <= t) of the last `window_bars` = 12 bars of the episode's social
stream (a CAUSAL window). Features: log1p(#messages) over 12 and 48 bars, short-window excess over the long rate,
sum / mean / max of text_score (mark[2]), mean reach (mark[1]), max prior involvement (mark[3]), bars since the last
message, mean inter-message gap, plus the mean hashed document embedding of `panel.text` over the window when the
panel carries text. Classifier with the paper's LightGBM hyper-parameters; score = P(window contains a pump
announcement). Abstain (NaN) until the episode's first message is usable.

**Adaptations and why.** TF-IDF of raw text is replaced by statistics of the loader's text_score and (optionally) the
loader's offline hashed embeddings, because baselines read the panel, not raw message text. Message-count windows
become time windows in bars because decisions are taken per bar. Only the causal window is admissible under the
benchmark's PIT rule. The LLM / seq2seq target-extraction stage is out of scope (the episode already fixes the coin).

**Unspecified choices (and the fix).**
1. Window length in bars -> 12 (one hour at 5-minute bars), long reference window 48.
2. Feature set replacing TF-IDF -> the ten window statistics above (+ 32 embedding dims when text exists).
3. Positive label -> the panel label (rows [anchor - lead, peak] of pump episodes).
4. Class balance -> `class_weight='balanced'` as in the paper, no subsampling.

**Fallbacks.** `lightgbm` not importable (the case on the local macOS environment) or failing -> sklearn
HistGradientBoostingClassifier (max_iter 100, lr 0.1, max_depth 6, max_leaf_nodes 31, min_samples_leaf 20, balanced);
status becomes `reimplemented-fallback`, `diag['backend'] = 'sklearn_hgb'`, reason in `diag['fallback_reason']`.
The fallback has no row / column subsampling (LightGBM's subsample 0.8 / colsample 0.8).
