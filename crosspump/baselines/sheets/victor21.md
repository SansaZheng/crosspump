# B3 — xp_victor21 (Victor & Weintraud, WWW 2021)

**Citation.** F. Victor, A. M. Weintraud, "Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency
Exchanges", Proceedings of The Web Conference 2021 (WWW '21), doi:10.1145/3442381.3449824.

**Implementation.** `crosspump/baselines/b3_victor21.py` (status `reimplemented`, unsupervised, nothing is fitted),
config `configs/crosspump/baselines/victor21.yaml`.

## Code pointers
No official code release was located; `official_repo` is empty. The method is reimplemented from the paper:

| Paper element | Here (`b3_victor21.py::_tables`) |
|---|---|
| trade graph among addresses, strongly connected components | not available: the wallet stream has one wallet id per transfer; cycles are found on the availability-ordered id sequence |
| self-trades | cycle length 1: the same wallet in two consecutive events |
| 2- and 3-account cycles | patterns A B A and A B C A with distinct intermediate wallets |
| net position change ~ 0 within a window (volume match) | `abs(log size_e - log size_j) <= volume_tol` and both events within `window` bars |
| wash-traded volume share per token | `wash_share` = size of cycle-closing events / total size in the window |

## Data used by the paper
On-chain trade logs of the Ethereum decentralised limit-order-book exchanges EtherDelta and IDEX (every trade with
maker and taker address).

## Split and threshold conventions in the paper
No learning, no train / test split, no labels: rule-based detection over each token's full trade history, with a sweep
over time windows and a small volume-match margin. Output = set of wash trades and their volume share, not an alarm.

## What was adapted and why
* The stream contract carries `source` = wallet id and `mark[0]` = log size, no counterparty, so directed cycles in the
  trade graph cannot be built. The closest causal analogue is the recurrence of wallet ids in the ordered event sequence.
* Events are ordered by availability (then time): deleting later events never changes what an earlier event closes.
* Two extra components requested by the benchmark task: `repeat_share` (wallets acting again within the window, without
  the volume match) and `hhi_excess` (Herfindahl concentration of inflow size over wallet ids, in excess of 1 / n_ids).
* Score = wash_share + repeat_share + hhi_excess; windows with fewer than `min_events` wallet events score 0.
* Episodes without a wallet stream abstain (NaN): loader flag `has_wallet`, or — without a flag — no wallet event usable yet.

## Unspecified choices and how they were fixed
1. Window: 48 bars (`model.window`).
2. Volume tolerance: 0.25 in log units (`model.volume_tol`); the paper's 1 % margin is on exact token amounts, the
   streams carry coarse ADV-normalised sizes. Set `volume_tol: 0.01` to mimic the paper on exact amounts.
3. Component weights: [1, 1, 1] (`model.weights`).
4. Event size = exp(mark[0]); unknown size counts 1; wallet id 0 (unknown) never closes a cycle and is a distinct wallet
   in the concentration index.
5. Cycle length capped at 3 (the paper also reports larger components).
