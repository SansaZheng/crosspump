# CrossPump track — conventions (read before writing any loader or baseline)

CrossPump = cross-platform pump-and-dump precursor detection: **social -> wallets -> order flow**. It lives in
`crosspump/`, `configs/crosspump/`, `scripts/xp_*.py`, `tests/test_crosspump.py`, `results/crosspump/`, docs in
`docs/crosspump_*.md`. Other tracks edit this checkout concurrently (no git): never rewrite shared `warn2trade/*`,
`causalgate/*`, `configs/default.yaml`; new files only; import shared pieces explicitly.

## 1. Panel layout = stacked episode windows
One column per **episode**: a pump episode (anchored at the public announcement / coordination message) or a
**control window** (same coin at a time with no pump within +-control_gap, or another coin at the same time).
Rows = `pre_bars + post_bars + 1` bars of `bar_minutes` each; row `pre_bars` is the anchor bar. Columns are sorted
chronologically by anchor time; walk-forward splits run over **episodes** (meta['split_axis'] = 'assets').

    mid          close of the bar (NaN where no bar exists)
    price_feat   RAW (un-normalised) order-flow features, meta['feat_names'] =
                 [log_ret, log_volume, hl_range, taker_imb, log_trades]   (taker_imb = taker_buy_vol/vol - 0.5, 0 if unknown;
                 log_trades = log1p(trade count), NaN if unknown); meta['ret_col'] = 0
    liq          [relative half-spread proxy, 1, ADV fraction]   (crosspump.data.loaders.common.liquidity_fields)
    tradable     bar exists and volume > 0
    labels       1 on [anchor - lead, peak] for pump episodes, 0 everywhere else (controls are all-zero)
    bars_to_peak peak_row - row on [anchor - lead, end] (exact, negative after the peak), NaN elsewhere
    text         OPTIONAL message embeddings (T, N, M, D) with text_mask / text_age and avail_text (PIT fields)
    PIT fields   avail_feat (= row), avail_label (= episode end row for event rows, else row + max(horizon)),
                 avail_text (bar containing the message's publication), universe (bar exists)
    meta         bar_minutes, pre_bars, post_bars, lead, anchor_ns (N,), peak_row (N,), is_pump (N,) bool, episodes (list of
                 dicts: key, symbol, exchange, channel, anchor_ns, peak_ns, is_pump, control_of, source), coverage (dict),
                 source, licence, dataset, split_axis='assets', assets_time_aligned=False, bar_close_ns (T, N)

## 2. Event streams (the cross-platform evidence) — `panel.streams`
`crosspump.data.streams.EventStreams`, attached with `attach_streams(panel, streams)`; `get_streams(panel)` reads it.
Flat arrays sorted by (asset, time):
    asset     int64  episode column
    time      float64 in BAR units of that episode (row = floor(time); fraction = position inside the bar)
    platform  int8   0 = social, 1 = wallet (on-chain), 2 = orderflow     (EventStreams.PLATFORMS)
    mark      float32 (E, 4): [magnitude, reach_or_size, text_score, prior_involvement]  (NaN = unknown)
    avail     float64 availability time in bar units (>= time): social = message time + ingestion lag,
              wallet = block time + confirmation lag, orderflow = bar close
    source    int64  entity id (channel / account / wallet address id; 0 = unknown) for the linked graph
    meta      platform_lags, entity tables (id -> name/hash), coverage per platform, how each stream was built
Rules: an event is usable at decision row t iff avail <= t (PIT; decisions are taken at the bar close, so an event whose availability falls inside (t, t+1] is first usable at row t+1 = ceil(avail); `rasterise` and `before` both follow this rule). `derive_orderflow_events` builds the
order-flow stream from `price_feat` causally (streaming z-scores of taker_imb / log_volume, warm-up abstains)
when a loader has no finer trade data. `add_prior_involvement` fills mark[:, 3] causally (how many earlier
episodes the same source id was involved in, as of avail). Loaders must say in meta['coverage'] which episodes have
which platforms; a missing platform is a missing modality, never zeros pretending to be "no activity".

## 3. Loaders (`crosspump/data/loaders/d*_<name>.py`)
BaseDatasetBuilder subclass registered tolerantly as "xp_<name>" (`common.register_dataset`). Raise
FileNotFoundError with the download recipe when raw files are missing (`common.require`); recipes also go into
`scripts/xp_prepare_data.py`. **Never fabricate data** (no synthetic prices, messages, wallets or labels in a real
loader). Market bars for Binance pumps come from data.binance.vision 1-minute klines (taker buy volume + trade
count = order-flow proxies) or daily aggTrades; on-chain transfers come from an explorer API / public dump named in the
recipe; message text is embedded offline (hash-tfidf by default) with a declared cutoff. Document every assumption
in the module docstring and in meta['assumptions'].

## 4. Baselines (`crosspump/baselines/b*_<name>.py`)
Subclass `crosspump.baselines.common.XPBaseline` and implement only
    _fit(self, panel, train_assets, val_assets) -> None
    _score_rows(self, t: np.ndarray, i: np.ndarray) -> np.ndarray      # higher = more pump-like; NaN = abstain
using only information with avail <= t for row t (PIT is the subclass's responsibility; helpers:
`self.counts` (T, N, P) causal event counts per bar, `self.streams_before(i, t)`, `self.panel`). Declare class attrs
name, paper, year, venue, official_repo, status (official | reimplemented | reimplemented-fallback | wrapper-needed),
modalities (subset of {social, wallet, orderflow, text, graph}), unspecified (list of choices the paper leaves open and
how you fixed them). The scaffold adds the alarm gate (fixed quantile / delayed ACI at a false-alarm budget), the
exit-overlay trade (w = -w_max on alarm) and the shared `out` dict, so the backtest and metrics are identical for
every method. Registry key "xp_<name>" (tolerant, first wins). A convention sheet (what the original does per
modality, file/line pointers) goes in `crosspump/baselines/CONVENTIONS_B.md`.

## 5. Protocol
Chronological episode splits (train < val < test by anchor time, purge gap >= window length), seeds 42/2024/2025/2026,
costs 5 bps train / {1,5,10,15,30} eval, latency >= 1 bar, no point-adjust. Metrics: lead time (minutes before
anchor and before peak of the first alarm), precision at fixed per-bar false-alarm rate on control rows, AUC-PR,
event recall, follower loss avoided vs the Twitter-follower benchmark (Ardia & Bluteau 2024), plus the shared
warn2trade trading matrix for the exit overlay.
