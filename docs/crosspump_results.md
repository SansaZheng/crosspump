# CrossPump — results ledger

Only numbers from runs that exist. Real-data cells stay `[pending]` until the run is logged here with its command.

## Synthetic testbed (`xp_synth`, plumbing and power checks, not findings)
Local CPU, 2026-10-03, `python3 tests/test_crosspump.py` and ad-hoc runs through `crosspump/eval/runner.py`.

| Check | Outcome |
|---|---|
| Delayed-ACI gate under a mean shift, budget 0.05 | realised alarm rate 0.0495 |
| Hawkes fitted on pump episodes vs controls | branching social→orderflow 0.25 vs 0.006; wallet→orderflow 0.24 (pump model) |
| Point-in-time tests (streams, Hawkes row state, CrossPump scores) | pass after two leaks were fixed (mark statistics fitted on all episodes; event likelihood using not-yet-available events) |
| CrossPump vs counts rule, seed 1, 20 pumps / 40 controls, 1 fold (before the leak fix) | row AUC-PR 0.49 vs 0.07; 60 % of pumps warned before the announcement; mean lead 96 min; follower loss avoided ≈ 1,100 bps |
| Ablations across seeds (after the leak fix) | see table below |

Ablation across seeds: `scripts/xp_e0_ablation.py` was started on c225 on 2026-10-03 (`results/crosspump/e0_ablation.log`); c225 became
unreachable before the log could be read. **Not yet known.**

Findings that temper the method claim (all synthetic, single seed, smoke sizes):
- After the leak fix, on c225 the full model warned 2 of 7 test pumps before the announcement (29 %), down from 60–70 % before the fix.
- On one seed the counts-only ablation scored a higher row AUC-PR than the full model (0.415 vs 0.375).
- Through the shared runner (seed 42, 1 fold, smoke): `xp_lamorgia23` row AUC-PR 0.643, `xp_lroo26` 0.401, `xp_moura26` 0.122;
  builder tests report `xp_hu23` 0.681, `xp_nghiem21` 0.544, `xp_mirtaheri21` 0.433, `xp_pumpsense26` 0.303, `xp_victor21` 0.059.
  Earlier CrossPump runs on comparable panels gave 0.37–0.55. **CrossPump does not currently beat the strongest baselines on the testbed.**

Build status (2026-10-04): 10 loaders and 10 baselines registered; `tests/test_crosspump.py` and the five files in
`tests/crosspump_parts/` pass locally. Loader tests use synthetic fixtures only; no loader has been run on real files yet.

## Real data
| Dataset | Episodes built | Platform coverage | Status |
|---|---|---|---|
| D1 `xp_hu_pd` … D10 `xp_memechain` | [pending] | [pending] | raw files and 2,368 Binance kline symbol-months on c225; loaders written, not yet run on real files |

## E1 main table
`[pending — scripts/xp_run_all.sh on c225]`
