# CrossPump — research plan and paper blueprint
## Cross-platform manipulation precursors: social → wallets → order flow

Codename **CrossPump**. Sibling track in this checkout: package `crosspump/`, configs `configs/crosspump/`, scripts
`scripts/xp_*.py`, tests `tests/test_crosspump.py` (+ `tests/crosspump_parts/`), results `results/crosspump/`.
Written 2026-10-03. Targets: KDD 2027, WWW 2027, NeurIPS 2027 (D&B for the linked benchmark), AAAI 2027.
Experiment node: `c225:/date/zyy/TradingSignal` (venv `.venv`, torch 2.11 + cu128, 2 × RTX 6000 Ada).

---

## 0. Audit of the brief (every claim checked on 2026-10-03)

| Claim in the brief | Status | What the paper may say |
|---|---|---|
| PumpSense reads Telegram messages only | **Verified.** Mahrous & Di Pietro, IEEE ICBC 2026, arXiv:2605.09431: 283,017 messages, 39 groups, 2,246 labelled announcements; LightGBM F1 0.79, BGE-M3 F1 0.83; data + code on Zenodo doi:10.5281/zenodo.19616802 (MIT). | Text-only detector of the coordination message itself; no market or chain data. |
| Moura et al. 2026, 14,499 channels, pump messages read like chatter | **Verified, venue corrected.** Moura, Paoletti, Ferreira, Almeida, arXiv:2609.01176, accepted at **ICWSM 2027**. TGDataset last year (Jul 2021–Jul 2022): 14,499 channels, >20 M messages; 47 P&D events + 73 sustained reactions; "linguistically indistinguishable from organic discussions". | Cite as ICWSM 2027 (arXiv Sep 2026). Their 47-event list is **not released**; only burst intervals for 681 coins. |
| PERSEUS traces organisers through the social network only | **Mostly.** Fu, Feng, Wu, Xu, arXiv:2503.01686: GNN on temporal attributed diffusion graphs; it also "collects real-time data from the OSN and cryptocurrency markets", market data serving validation. No code released. | "Detection on the social graph; market data for validation only." |
| AIMM relies on synthetic social data | **Verified.** Neela, arXiv:2512.16103: "calibrated synthetic social features" because of Reddit API limits; 33 labelled ticker-days, 3 positives. | Preliminary; equities, not crypto. |
| Hu et al.: +9.5 % in the 60 h before, heavier trading from 57 h | **Verified verbatim.** Hu, Zhang, Lu, He, Li, SIGMOD 2023, arXiv:2204.12929: frequent trading starts "about 57 hours before the pump time"; "the highest average return reaches 9.5% when x=60". 709 P&Ds (1,335 samples), Jan 2019–Jan 2022. | Use as the motivating fact. Episode windows must cover ≥ 72 h before the announcement. |
| Hu et al. GitHub, 709 events | **Verified.** `PD_logs_cleaned.txt`: 1,335 rows, 1,003 on Binance, 109 channel ids in the cleaned log (paper: 715 channels explored). | |
| La Morgia et al., about 900 events | **Outdated count.** The public `pump_telegram.csv` holds **1,109** pumps today (Binance 520, YoBit 276, Cryptopia 210, Bittrex 102); MIT licence. | Report the count used after filtering to exchanges with free bars. |
| Losses vs the Twitter followers of Ardia & Bluteau | **Verified.** IRFA 95 (2024) 103479: 1,160 Binance events (public list); Twitter-informed investors sell late and lose. Tweets themselves are not released. | Follower benchmark = buy at the promotion, exit late (§7). |
| "No public method detects pumps from social, on-chain and order-flow data together" | **True as stated, but narrow it.** Closest work: Bolz et al. 2024 (Telegram + order book, no chain); LROO 2026 (on-chain + OSINT, rug pulls, tabular, no order flow, no timing); TM-RugPull 2026 (dataset); Nghiem 2021 (market + social); Karbalaii 2025 (arXiv:2504.15790: 70 % of pre-event volume trades within one hour of the announcement, "pre-accumulation" vs "on-the-spot" archetypes). | Claim: first method that **links** the three with a timestamped excitation model and reports lead time; not "first multimodal". |

Risk the brief does not mention: for exchange-listed coins the wallet evidence is partial. Only ERC-20/BEP-20 tokens have
indexable transfers, exchange-deposit labels are not public, and Karbalaii 2025 suggests part of the accumulation is
"on-the-spot". The paper must report per-platform **coverage** and treat a missing chain as a missing modality.

---

## 1. Title candidates and abstract skeleton
1. **Before the Message: Linking Wallets, Order Flow and Social Signals for Early Pump-and-Dump Warnings**
2. **CrossPump: Cross-Platform Excitation Models for Manipulation Precursors**
3. **Accumulation Leaves a Trace: A Linked Benchmark and Hawkes Model for Pump Precursors**

*Gap.* Each stage of a pump is studied on its own source: messages (PumpSense, Moura), organisers (PERSEUS), market bursts
(La Morgia), yet accumulation is visible 57–60 h earlier (Hu et al.). *Question.* Does linking social, on-chain and
order-flow evidence give earlier and more precise warnings than any single source? *Method.* A linked temporal graph of
channels, wallets and order books; a marked multivariate Hawkes process whose off-diagonal kernels measure cross-platform
excitation; a coordination score; a false-alarm-budgeted alarm gate that stamps every alert. *Evaluation.* 10 datasets, 10
baselines (2021–2026, three from 2026), chronological episode splits, lead time, precision at fixed false-alarm rates, losses
avoided against the Twitter-follower benchmark. *Results.* Placeholders until real runs exist (§10).

---

## 2. Problem formulation
- **Episode.** A window of `pre_bars + post_bars + 1` bars around an anchor: the public coordination message for a pump,
  or a matched time for a control (same coin with no pump within ±24 h; another coin at the same time).
- **Evidence streams.** Timestamped point events on three platforms, `p ∈ {social, wallet, orderflow}`, each with a mark
  (magnitude, reach or size, text score, prior involvement of its source), a source entity (channel, wallet) and an
  **availability time** (message ingestion lag, block confirmation lag, bar close).
- **Decision.** At the close of bar t, with events whose availability ≤ t and bars ≤ t, emit a score and an alarm.
- **Targets.** (i) Early warning: alarm before the anchor (lead time vs announcement and vs peak). (ii) Precision at a
  fixed per-bar false-alarm rate on normal rows. (iii) Follower loss avoided.
- **Linked graph over time.** Nodes: channels/accounts, wallets, coins. Edges: mention (channel → coin), transfer or DEX buy
  (wallet → coin), burst (coin order book). Memory: each source carries its count of earlier pump episodes, computed
  causally (`add_prior_involvement`).

## 3. Method
1. **Cross-platform marked Hawkes** (`crosspump/models/hawkes.py`).
   λ_k(t) = μ_k + Σ_j α_jk Σ_{l: p_l=j, t_l<t} exp(γ_j m̃_l) exp(−β_jk (t − t_l)). Shared parameters, exact MLE with the
   O(E) recursion. n_jk = α_jk/β_jk is the cross-platform excitation matrix (social→wallet, wallet→orderflow are the
   hypothesised precursor paths).
2. **Two-model sequential likelihood ratio.** One Hawkes fitted on training pump episodes, one on training controls;
   LLR_W(t) = Σ over the last W bars of the difference in per-row log-likelihood, scored against the state known before
   each row (strictly point-in-time).
3. **Coordination score.** C(t) = share(t) · log(1 + sync(t)), with share = off-diagonal attribution mass / total excitation
   and sync = geometric mean over present platforms of λ_k/μ_k. High only when platforms excite each other and move together.
4. **Linked-evidence head.** Class-weighted logistic regression on [log λ, lift, attribution shares, C, LLR_W, per-platform
   counts over 3/12/48 bars, source track records, causal z-scores of taker imbalance and volume].
5. **Alarm gate.** Delayed-feedback adaptive conformal threshold at false-alarm budget α (feedback delay = hold + latency);
   every alert carries its row, hence its lead time.
6. **Output.** Avoid (alert before the announcement) or exit (alert after entry) for followers; in the shared backtest an
   exit overlay (w = −w_max on alarm).

Theory obligations (appendix): identifiability of n_jk under partial coverage; stability (spectral radius of n < 1);
alarm-rate control under delay; a bound on lead-time loss from availability lags.

## 4. Datasets (ten; loaders in `crosspump/data/loaders/`)
| # | Registry key | Source (verified) | Episodes | Social | Wallet | Order flow |
|---|---|---|---|---|---|---|
| D1 | `xp_hu_pd` | Hu et al., SIGMOD 2023, arXiv:2204.12929; GitHub Bayi-Hu | 709 pumps / 1,335 samples (1,003 Binance), 2019–2022 | announcements + raw channel messages (516 session files) | ERC-20 transfers (API, partial) | Binance 1-min klines / aggTrades |
| D2 | `xp_sapienza_pd` | La Morgia et al., ACM TOIT 2023, arXiv:2105.00733; MIT | 1,109 pumps (520 Binance), 2017– | announcements + group ids | same | same |
| D3 | `xp_xu_pumpolymp` | Xu & Livshits, USENIX Security 2019, arXiv:1811.10109 | 412 pumps, Jun 2018–Feb 2019 | Telegram/Discord announcements | same | same |
| D4 | `xp_pumpsense` | Mahrous & Di Pietro, ICBC 2026; Zenodo 19616802 (MIT) | 2,246 labelled announcements, 14 exchanges | 283 k messages with text | same | same |
| D5 | `xp_tgdataset_pd` | Moura et al., ICWSM 2027; TGDataset (KDD 2025, Zenodo 7640712, CC-BY-4.0) | bursts for 681 coins, Jul 2021–Jul 2022; **proxy labels** | burst intervals (messages optional, 71 GB) | same | same |
| D6 | `xp_ardia_pd` | Ardia & Bluteau, IRFA 2024; GitHub ArdiaD | 1,160 Binance events with success flag and pump minute | announcements | same | same |
| D7 | `xp_mirtaheri` | Mirtaheri et al., IEEE TCSS 2021, arXiv:1902.03110 | pump attempts Sept 2017–Aug 2018 | Twitter time series + Telegram messages | same | Binance bars where listed |
| D8 | `xp_dex_wash` | Victor & Weintraud, WWW 2021; Zenodo 4540223 (CC-BY-4.0) | EtherDelta/IDEX tokens hit by 2017–18 pumps | announcements (cross-referenced) | **every trade has wallet addresses** | on-chain trades |
| D9 | `xp_rugpull1k` | TM-RugPull, arXiv:2602.21529; HF `fati-80-Sho/RugPull1K` (CC BY 4.0) | 1,000 projects, 5 chains | OSINT signals | on-chain behaviour | volume |
| D10 | `xp_memechain` | MemeChain, arXiv:2601.22185; Zenodo 18246856 (CC-BY-4.0) | 34,988 meme coins, 4 chains | linked accounts (static) | on-chain buys | on-chain trades |

Anchor (not counted): `xp_synth`, a synthetic testbed with a planted social→wallet→orderflow chain, single-platform decoy
bursts and missing wallet coverage (power study E0, smoke tests). Optional extension: Cresci et al. cashtag piggybacking
(ACM TWEB 2019, Zenodo 2686862) for equities.

Caveats to state in the paper: D5 labels are a proxy rule; D6 has no tweets; D7 covers only coins with Binance bars in
2017–18; D9 may be tabular only (then excluded from lead-time tables); D10 social evidence is static; wallet coverage on
D1–D7 depends on an explorer API key and token-contract mapping.

## 5. Baselines (ten, 2021–2026, three from 2026)
| # | Key | Paper (verified) | Modalities |
|---|---|---|---|
| B1 | `xp_mirtaheri21` | Mirtaheri et al., IEEE TCSS 8(3) 2021, doi:10.1109/TCSS.2021.3059286 | social (Twitter + Telegram) |
| B2 | `xp_nghiem21` | Nghiem, Muric, Morstatter, Ferrara, ESWA 182, 2021, doi:10.1016/j.eswa.2021.115284 | market + social |
| B3 | `xp_victor21` | Victor & Weintraud, WWW 2021, doi:10.1145/3442381.3449824 | on-chain |
| B4 | `xp_lamorgia23` | La Morgia et al., ACM TOIT 23(1) 2023, doi:10.1145/3561300 | order flow |
| B5 | `xp_hu23` | Hu et al., SIGMOD 2023, doi:10.1145/3588686 | channel history + market |
| B6 | `xp_bolz24` | Bolz et al., arXiv:2412.18848 (ACM 2025, doi:10.1145/3733815.3764042) | Telegram + order flow |
| B7 | `xp_perseus25` | Fu, Feng, Wu, Xu, arXiv:2503.01686 | social graph |
| B8 | `xp_pumpsense26` | Mahrous & Di Pietro, ICBC **2026**, arXiv:2605.09431 | message text |
| B9 | `xp_moura26` | Moura et al., ICWSM 2027, arXiv:2609.01176 (**2026**) | social bursts (CA-CFAR) |
| B10 | `xp_lroo26` | Shoaei et al., LROO, arXiv:2603.11324 (**2026**) | on-chain + OSINT |

Reference rows (not counted): `xp_counts_rule`; Chadalapaka et al. 2022 (arXiv:2205.04646); Fabre & Muni Toke 2025
Markov-modulated Hawkes (arXiv:2502.04027) as the single-platform Hawkes; Karbalaii 2025 (arXiv:2503.08692); Yu, Park, Chai
2025 (arXiv:2510.00836); Cao et al. 2026 (arXiv:2603.13830); Kamps & Kleinberg 2018. All baselines are re-implementations
adapted to the episode layout; each has a convention sheet in `crosspump/baselines/sheets/`. Say so in the paper.

## 6. Protocol
Chronological episode splits (train < val < test by anchor time, purge ≥ one window and ≥ 96 h), no test episode before a
training one, seeds 42/2024/2025/2026, costs 5 bps headline and 1/5/10/15/30 bps grid, latency ≥ 1 bar, no point-adjust.
Every method goes through the same gate, backtest and metric code (`crosspump/baselines/common.py`, `eval/runner.py`).

## 7. Metrics
- Lead time of the first alarm vs announcement and vs peak (bars, minutes); share of pumps warned before the announcement.
- Precision and recall at per-bar false-alarm rates 0.1 %, 0.5 %, 1 %, 5 % (threshold from validation normal rows).
- Event recall, false alarms per control day, alarm-budget error.
- **Follower loss avoided.** Benchmark follower buys at the announcement + latency and exits `hold_late` bars later;
  with alerts the follower avoids (alert before announcement) or exits at alert + latency. Report mean bps, share of the
  benchmark's losses avoided, forgone gains.
- HR@k for the target-coin task; shared Warn2Trade matrix for the exit overlay.

## 8. Experiments
| Exp | Question | Design |
|---|---|---|
| E0 | Can the method recover a planted chain, and how much evidence does it need? | `xp_synth`: chain strength, wallet coverage and lag sweeps |
| E1 | Main comparison | 10 datasets × (CrossPump + 10 baselines) × 4 seeds |
| E2 | Does linking help? | platform ablations: each single platform, each pair, all three |
| E3 | Coverage | restrict to episodes with wallet evidence; degrade coverage 100 → 0 % |
| E4 | What is the excitation structure? | estimated n_jk per dataset with bootstrap intervals; pump vs control models |
| E5 | Components | counts only, Hawkes only, LLR only, single model, fixed gate |
| E6 | Economic value | follower loss avoided across late-exit horizons 5–60 min and costs 1–30 bps |
| E7 | Transfer | train on CEX Telegram pumps, test on DEX / meme-coin sets and the reverse |
| E8 | Case studies | alert timelines for selected pumps with the three streams overlaid |

## 9. Status on 2026-10-03
Built and tested on the synthetic testbed: streams container with point-in-time rules, synthetic generator, loader helpers,
baseline scaffold, gates, Hawkes model, CrossPump method, metrics, follower overlay, runner, scripts, configs, ablation
configs, `tests/test_crosspump.py`. Loaders D1–D10 and baselines B1–B10 are being written by parallel builders. Raw public
data for all ten sets is on c225 under `data_raw/crosspump/`; Binance 1-minute klines for the Hu, Sapienza and Ardia pumps
are downloading there. No real-data result exists yet. See `docs/crosspump_results.md`.

## 10. Risks and reviewer objections
- *Wallet evidence is thin for exchange-listed coins.* Report coverage; E3; D8–D10 carry full on-chain evidence.
- *Baselines are adaptations.* Convention sheets; official feature lists where code exists (B2, B4, B5, B9).
- *Proxy labels (D5), static social (D10).* Flagged in tables; excluded from headline averages if reviewers object.
- *Hawkes adds little over counts.* A real possibility: on the synthetic testbed the counts-only ablation is competitive
  at small sample sizes. E5 must be reported honestly; the paper's claim may have to rest on linking, not on the kernel.
- *Front-running concern.* The output is an avoid/exit signal; the paper reports losses avoided, not profits from
  trading ahead of pumps.
- *Licences.* MIT (Sapienza, PumpSense), CC-BY-4.0 (Victor, MemeChain, TGDataset, RugPull1K), cite-on-use (Ardia);
  Hu, Xu, Mirtaheri repositories carry no explicit licence: ask the authors before redistribution.

## 11. Reference ledger (resolved 2026-10-03)
arXiv: 2204.12929, 2105.00733, 1811.10109, 2605.09431, 2609.01176, 2503.01686, 2512.16103, 2306.02148, 1902.03110,
2412.18848, 2603.11324, 2602.21529, 2601.22185, 2504.15790, 2502.04027, 2401.09361, 2205.04646, 2510.00836, 2603.13830,
2608.20271. DOIs: 10.1145/3588686, 10.1145/3561300, 10.1016/j.irfa.2024.103479, 10.1109/TCSS.2021.3059286,
10.1016/j.eswa.2021.115284, 10.1145/3442381.3449824, 10.1145/3733815.3764042, 10.5281/zenodo.19616802.
Data records checked: Zenodo 19616802, 4540223, 18246856, 7640712; HF `fati-80-Sho/RugPull1K`; GitHub repositories of Hu,
Sapienza, Xu, Mirtaheri, Nghiem (moved to `hnghiem-nlp`), Moura, Ardia. Re-resolve all at camera-ready.
