# Paste-ready prompt — CrossPump track (cross-platform pump precursors)

Copy everything below the line into a fresh Claude Code session opened in
`/Users/zyy/package/anomalydetection/TradingSignal` (or on `c225:/date/zyy/TradingSignal`).

---

You are a senior financial-ML researcher and PyTorch engineer. We are writing a CCF-A paper on **cross-platform manipulation
precursors**: linking social signals, wallet activity and exchange order flow to warn about crypto pump-and-dumps before the
public announcement (Hu et al., SIGMOD 2023: pumped coins rise 9.5 % in the 60 h before the pump and trade more heavily from
57 h ahead).

**Blueprint.** Read `docs/crosspump_research_plan.md` first (audit of the brief, method, ten datasets, ten baselines,
protocol, experiments E0–E8), then `crosspump/CONVENTIONS.md` and `docs/crosspump_results.md` (what has run).

**Repository rules.** Other paper tracks edit this checkout concurrently and there is no git. All code lives in `crosspump/`,
`configs/crosspump/`, `scripts/xp_*.py`, `tests/test_crosspump.py`, `tests/crosspump_parts/`, `results/crosspump/`. New files
only outside that namespace; never rewrite `warn2trade/*`, `causalgate/*`, `configs/default.yaml`.

**Hard constraints.** Cite only papers whose DOI or arXiv id resolves. Never fabricate data: a loader raises with its download
recipe when raw files are missing. Report only numbers from runs that exist. Point-in-time rule: an event is usable at
decision row t iff its availability ≤ t. A missing platform is a missing modality, not zero activity. Chronological episode
splits, seeds 42/2024/2025/2026, costs 5 bps headline and 1/5/10/15/30 bps grid, latency ≥ 1 bar.

**Next steps, in order.**
1. `python3 tests/test_crosspump.py` and the four files in `tests/crosspump_parts/`; fix anything red.
2. On c225 (`/date/zyy/TradingSignal`, `PYTHON=.venv/bin/python`): build each real panel
   (`scripts/xp_prepare_data.py --list`, then `scripts/xp_run.py --dataset <name> --baseline xp_counts_rule --max-folds 1`),
   record episode counts and per-platform coverage in `docs/crosspump_results.md`.
3. Wallet evidence: set `ETHERSCAN_API_KEY`, run `scripts/xp_prepare_data.py --dataset hu_pd --onchain`; report coverage.
4. E1 grid: `PYTHON=.venv/bin/python bash scripts/xp_run_all.sh --datasets all --baselines all`; then `--ablations` (E2, E5).
5. E0 power study on `synth`; E3 coverage; E4 excitation matrices with bootstrap intervals; E6 follower horizons.
6. Fill `docs/crosspump_results.md`, write the paper against the outline in the plan, re-resolve every reference.

**Environment.** Mac: Python 3.9, torch 2.8 CPU, no pytest (plain-script tests), no `timeout`. c225: Python 3.10 venv
`.venv` with torch 2.11 + cu128, lightgbm, xgboost, huggingface_hub; raw data under `data_raw/crosspump/`. Sync with
`rsync -a --exclude runs --exclude data_raw --exclude results --exclude .venv . c225:/date/zyy/TradingSignal/`.

**Reporting.** After each step: what ran, exact command and output, what is still a stub, every guess about a baseline.
