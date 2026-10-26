# B10 — xp_lroo26 (crosspump/baselines/b10_lroo26.py)

**Citation.** Shoaei, Pishdar, Bag-Mohammadi, Karami, "LROO Rug Pull Detector: A Leakage-Resistant Framework Based
on On-Chain and OSINT Signals", arXiv:2603.11324 (Mar 2026). Companion dataset: TM-RugPull, arXiv:2602.21529.

**Code pointers.** No LROO training code was located. The companion data-acquisition code is
https://github.com/mohamadpishdar/RugPull-Defender (clone on c225 at `data_raw/crosspump/rugpull_defender`,
`src/DataSet`); it was NOT read for this baseline, so no file/line pointers are claimed. The implementation follows
the method summary of the track specification (tabular on-chain + OSINT features, strictly pre-event, TabPFN).

**Data the paper used.** About 1,000 token projects (TM-RugPull / RugPull1K) with on-chain behavioural metrics and
temporally aligned OSINT signals, features extracted strictly before any liquidity withdrawal; manual rug-pull labels.

**Split / threshold conventions in the paper.** Project-level classification with a temporal cut-off per project
(leakage resistance); the exact split protocol and decision threshold were not taken from the paper. In CrossPump the
alarm threshold is set by the shared gate.

**What was built.** One tabular row per (decision bar, episode), 38 causal features with the default config:
- wallet, per window W in {3, 12, 48}: #inflow events, #distinct wallet ids, sum of mark[0], sum of mark[1]
  (size vs ADV), top-1 wallet share of the window's size, max prior involvement (mark[3]); NaN until the first wallet
  event of the episode is usable;
- order flow, per window: #burst events, sum of burst magnitude; plus causal EWMA z-scores of log_volume and
  taker_imb at t (halflife 60, warm-up 30);
- social, per window: #messages, text_score sum, reach sum;
- flags: wallet seen, social seen (causal).
Classifier: TabPFN on <= 1,000 class-stratified train rows; score = P(label = 1), computed lazily for requested rows.
Abstain (NaN) while no platform has delivered any event.

**Adaptations and why.** Project-level rows become per-bar rows because the benchmark scores decision bars. The
paper's leakage rule (features before the withdrawal) maps onto the PIT rule (events with avail <= t, bar statistics
from bars < t). The paper's exact column list (holder counts, contract metadata, search trends) does not exist in the
event streams; the wallet / order-flow / social statistics above are the available analogues.

**Unspecified choices (and the fix).**
1. Feature windows -> {3, 12, 48} bars.
2. Concentration -> top-1 wallet share of window inflow size.
3. TabPFN context -> all positives up to half of 1,000, negatives sampled uniformly (seed 42).
4. TabPFN version / ensemble size -> package defaults, CPU.
5. Missing wallet platform -> NaN features (not zeros).

**Fallbacks.** `tabpfn` not importable (the case on the local macOS environment), weights unavailable, or any runtime
failure -> sklearn HistGradientBoostingClassifier (max_iter 100, lr 0.1, depth 6, 31 leaves, balanced class weights)
on up to 200,000 train rows; status becomes `reimplemented-fallback`, `diag['backend'] = 'sklearn_hgb'`, reason in
`diag['fallback_reason']`. The TabPFN path itself has not been executed locally.
