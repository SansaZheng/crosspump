# B7 — xp_perseus25 (crosspump/baselines/b7_perseus25.py)

**Citation.** Fu, Feng, Wu, Xu, "PERSEUS: Tracing the Masterminds Behind Cryptocurrency Pump-and-Dump Schemes",
arXiv:2503.01686 (Mar 2025).

**Code pointers.** No public code. Architecture details (layer count, hidden size, exact node attributes) are not
copied from any file; the implementation follows the method summary of the track specification.

**Data the paper used.** Online-social-network data (Telegram / X) around pump-and-dump campaigns, Feb-Oct 2024,
market data used for validation; 438 masterminds identified. Not public.

**Split / threshold conventions in the paper.** Node-level task (is an account a mastermind) on temporal attributed
diffusion graphs; no per-bar alarm threshold exists in the paper. In CrossPump the alarm threshold is set by the
shared gate.

**What was built.**
- Graph at (episode i, bar t): social events with `avail <= t`. Node = source id (source 0 = anonymous singleton).
  Directed edge u -> v when v's first post follows u's first post within `gap_bars` = 12 (flow from earlier to later
  posters). At most `max_nodes` = 256 most recently active nodes (dense adjacency).
- Node features (8): log1p(#posts), mean reach (mark[1]), max text_score (mark[2]), log1p(max prior involvement,
  mark[3]), bars since last post, bars since first post, source-known flag, share of the episode's posts.
- Model: 2 mean-aggregation layers with separate predecessor / successor means (GraphSAGE-style, plain torch),
  ReLU, mean + max pooling + log1p(#nodes) -> linear head -> one logit per graph.
- Training: BCE with pos_weight = #neg / #pos on train-episode rows (all positive rows + every k-th negative row),
  Adam lr 1e-2, weight decay 1e-4, 30 epochs (8 in smoke), seeded.
- Inference every k = `score_every` = 2 bars, at t' = k * floor(t / k) <= t, forward-filled to the bars in between
  (so the score at t uses information up to t' only); cached per (episode, t').
- Abstain (NaN) while the graph is empty (no social event usable at t').

**Adaptations and why.** The paper classifies accounts; the benchmark needs a per-bar episode score, so graphs are
pooled. The panel has no explicit forward / reply links, so the diffusion direction is inferred from first-post
order within a time gap. No torch_geometric dependency (dense adjacency is enough at <= 256 nodes).

**Unspecified choices (and the fix).**
1. Edge rule without explicit relay links -> first-post order within 12 bars.
2. Node attributes -> the 8 stream-derived features above (no profile / content embeddings).
3. GNN architecture -> 2 layers, hidden 32 (16 in smoke), in/out mean aggregation.
4. Graph-level read-out -> mean + max + log1p(#nodes).
5. Decision cadence k -> 2 bars, forward-filled (a message that becomes usable at an odd bar is seen one bar late;
   larger k trades run time for staleness: with k = 4 an announcement can be missed for up to 3 bars).
6. Node cap -> 256 most recently active sources.

**Fallbacks.** No two-class training rows -> untrained head (`diag['train_fallback']`).
