# CrossPump group-A loaders (D1, D2, D3, D6) — download recipes and observed raw layouts

All four are CEX (Binance) pump tables joined to **Binance 1-minute klines**. `build_panel()` never downloads; it raises
`FileNotFoundError` with the recipe when a raw file is missing. Downloads happen only through
`python3 scripts/xp_prepare_data.py --dataset <short> [--klines] [--aggtrades] [--onchain]`, which calls the module's
`prepare(root, args)`. Raw root = `cfg['root']` (default `data_raw`).

| key | loader | source | licence | event table (REQUIRED) |
|---|---|---|---|---|
| `xp_hu_pd` | `d1_hu_pd.py` | github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency (Hu et al., SIGMOD 2023, doi:10.1145/3588686) | no licence file (research use, cite) | `crosspump/hu_pd/Data/Telegram/Labeled/PD_logs_cleaned.txt` |
| `xp_sapienza_pd` | `d2_sapienza_pd.py` | github.com/SystemsLab-Sapienza/pump-and-dump-dataset (La Morgia et al., ACM TOIT 2023, doi:10.1145/3561300) | MIT | `crosspump/sapienza_pd/pump_telegram.csv` |
| `xp_xu_pumpolymp` | `d3_xu_pumpolymp.py` | github.com/xujiahuayz/pumpdump (Xu & Livshits, USENIX Security 2019) | no licence file (research use, cite) | `crosspump/xu_pumpdump/code and data/rawpumpolymp.json` |
| `xp_ardia_pd` | `d6_ardia_pd.py` | github.com/ArdiaD/PumpDump, branch master (Ardia & Bluteau, IRFA 2024, doi:10.1016/j.irfa.2024.103479; data DOI 10.5281/zenodo.12019080) | no licence file; README: cite the paper and the data DOI | `crosspump/ardia/data/list_pd_events.csv` |

## 1. Event tables

    git clone --depth 1 https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency data_raw/crosspump/hu_pd
    git clone --depth 1 https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset          data_raw/crosspump/sapienza_pd
    git clone --depth 1 https://github.com/xujiahuayz/pumpdump                                data_raw/crosspump/xu_pumpdump
    git clone --depth 1 --branch master https://github.com/ArdiaD/PumpDump                    data_raw/crosspump/ardia

(`prepare()` runs the clone itself when the event table is missing and the target directory is empty.)

Observed structure (clones on c225, 2026-10-03):

* **D1** `PD_logs_cleaned.txt`: TSV, header `channel_id session_id coin exchange pair timestamp`
  (`1095533634  256  APPC  binance  BTC  2019-06-04 12:01:33+00:00`); 1,335 rows; exchanges binance 1003, yobit 207,
  hotbit 63, kucoin 26, gate.io 16, bittrex 10 and 7 venues with <= 2 rows; pairs BTC 1211 / USDT 122 / ETH 2; 108 channel
  ids; 2019-01-01 .. 2022-01-19. Binance x BTC = 981 rows, 135 coins -> **428 pump episodes** after merging relays
  within 6 h. `Data/Telegram/raw/<channel_id>.pkl` (516 files, 845 MB): plain-python pickles, a list of Telethon
  `Message.to_dict()` dicts (`_`, `id`, `date` tz-aware datetime, `message`, `edit_date`, `views`, `forwards`, ...); no
  telethon needed; read with a restricted unpickler (datetime / timezone / timedelta / bytes only).
  `Labeled/pred_pump_message.csv`: TSV `channel_id sample_id message date pred_label`, 36,696 rows, 213 channels, all
  `pred_label == 1.0` (`sample_id` = Telegram message id). `Labeled/label.txt`: `label channel_id sample_id` (no text,
  unused). `Labeled/<channel_id>_session.pkl`: instances of the repo's own `Session` class (unused).
* **D2** `pump_telegram.csv`: header `symbol,group,date,hour,exchange` (`R,ATW,2018-01-01,16:00,yobit`); 1,110 rows;
  binance 520, yobit 276, cryptopia 210, bittrex 102, kucoin 2; 20 groups; 2017-07-05 .. 2021-01-18. Binance: 85 coins
  -> **318 pump episodes**. `groups.csv`: `group_name,group_code,last_time_checked, telegram_link` (13 rows).
  `labeled_features/features_{5,15,25}S.csv.gz`: `date,pump_index,std_rush_order,avg_rush_order,std_trades,std_volume,
  avg_volume,std_price,avg_price,avg_price_max,hour_sin,hour_cos,minute_sin,minute_cos,symbol,gt`; 25 s file = 482,157
  rows, 327 pump blocks, `gt == 1` on exactly one chunk for 317 pumps (the manually corrected start). 317 of the 318
  Binance episodes get a corrected start within 60 min of the table time (shift: median 0 s, 5-95 % = -50..+50 s,
  range -180..+100 s) — which also confirms that both files use the same clock (UTC). `features/`, `data/` are empty.
* **D3** `rawpumpolymp.json` (25 MB): JSON list of 435 records with keys `currency, exchange, channelTitle, channelLink,
  signalTime, duration, max, priceBeforePump, volume, ourBuyPrice, ourBuyTime, ourProfit, theoreticalBuyPrice,
  theoreticalBuyTime, theoreticalProfit, id, marketHistoryEntries[{rate, amount, timeStamp}]`; exchanges Cryptopia 216,
  Yobit 121, Binance 71, Bittrex 27; 92 channel links; one corrupt record dated 1999; `signalTime` has mixed precision
  (`...:14Z` and `...:00.946Z`), so it is parsed value by value. Binance: 71 records, 38 coins, 2018-09-10 .. 2019-02-26.
* **D6** `list_pd_events.csv`: header `Date,Channel,Currency,Exchange,success,pump_date`
  (`2019-01-07T14:27:13Z,Whales Crypto Guide,LOOM,Binance,1,2019-01-07T14:25:00Z`); 1,160 rows, all Binance, 95 channels,
  221 coins, 2019-01-02 .. 2022-02-02; success 1: 322 / 0: 838; `pump_date` present exactly on the 322 successes;
  `pump_date - Date`: median -0.2 min, quartiles -1.0 / -0.1 min, min -60 min, max +2 min. -> **1,119 pump episodes**
  (the 322 success rows alone merge to 309 clusters).

## 2. Binance market data (shared, `binance_fetch.py`)

    python3 scripts/xp_prepare_data.py --dataset hu_pd,sapienza_pd,xu_pumpolymp,ardia_pd --klines

* 1-minute klines: `https://data.binance.vision/data/spot/monthly/klines/<SYMBOL>BTC/1m/<SYMBOL>BTC-1m-YYYY-MM.zip`
  -> `data_raw/crosspump/binance_klines/<SYMBOL>BTC-1m-YYYY-MM.csv` (12 header-less columns `open_time, open, high, low,
  close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore`; `open_time` in ms,
  microseconds from 2025). Months fetched per pump: every month touched by `[anchor - (72 h + 30 d + 1 d),
  anchor + (8 h + 30 d + 1 d)]` (episode window + control span). Existing files are skipped; 404s (delisted pair, month
  before listing) and network errors are written to `binance_klines/manifest.json` (`missing` / `errors`).
* optional daily aggTrades (`--aggtrades`): `.../daily/aggTrades/<SYMBOL>BTC/<SYMBOL>BTC-aggTrades-YYYY-MM-DD.zip` ->
  `data_raw/crosspump/binance_aggtrades/` (8 columns `agg_id, price, qty, first_id, last_id, time, is_buyer_maker,
  best_match`), fetched for the window days of pumps **and** controls. The loader uses aggTrades bursts only when every
  episode has all its window days (`orderflow_source: auto`); otherwise all episodes use bar-derived bursts, so the
  provenance of the order-flow stream can never separate pumps from controls.

State on c225 (2026-10-03): 2,368 kline files / 235 symbols for D1 + D2 + D6 were fetched by a scratch job (months of
anchor -30 d / anchor / anchor +30 d); 116 symbol-months are 404 and 7 failed with timeouts (ANTBTC 2021-04/10/11,
APPCBTC 2019-03..06) — re-running `--klines` retries exactly those. **D3 was not part of that job**: run
`--dataset xu_pumpolymp --klines` (38 coins, 2018-08 .. 2019-03).

## 3. Optional wallet platform (`--onchain`)

    ETHERSCAN_API_KEY=... python3 scripts/xp_prepare_data.py --dataset hu_pd --onchain

1. `data_raw/crosspump/onchain/token_contracts.csv` (`symbol,chain,contract,coingecko_id,n_candidates`) is built from the
   free CoinGecko `/coins/list?include_platform=true` and `/coins/{id}` endpoints (rate limited). Tickers are ambiguous
   on CoinGecko: `n_candidates > 1` rows should be checked by hand before trusting the transfers.
2. Etherscan V2 `tokentx` (chain ids 1 and 56) for every pump **and control** window ->
   `data_raw/crosspump/onchain/<dataset>_transfers.csv` (`key,time_ns,wallet_id,size_adv,size_raw,to_exchange`);
   `wallet_id` = stable hash of the sender, `size_adv` = transfer size / mean daily Binance volume of the pre-window.
3. Coverage is **partial by construction** (ERC-20 / BEP-20 tokens only; native-chain coins have no transfers) and is
   reported per episode in `meta['coverage']['wallet']`. Exchange-deposit flags need the optional
   `onchain/exchange_addresses.csv` (`address,label`); without it `to_exchange = -1`.
When the file is absent the wallet modality is missing (no zero-filled stream). On c225 there is no Etherscan key, so
this step has not been run.

## 4. Conventions shared by the four loaders

* Episode key `f"{SYMBOL}{quote}@{YYYY-MM-DDTHH:MMZ}"`; control windows `CTRL_<same format>`.
* Layout defaults: 5-minute bars, `pre_bars` 864 (72 h), `post_bars` 96, `lead` 2, `peak_window_bars` 12,
  `end_bars_after_peak` 24, horizons `[1,2,3,6,12,24]`, 2 controls per pump (one same-coin-other-time, one
  other-coin-same-time), `gap_hours` 24, `control_span_days` 30, `merge_hours` 6.
* PIT: an event is usable at decision row `t` iff `avail <= t`; streams are rasterised at `ceil(avail)`; a message is
  placed in `Panel.text` at `ceil(message time + social_lag_bars)` and `avail_text` = that row.
* Every assumption is listed in the module docstring and in `panel.meta['assumptions']`.
