# CrossPump raw-data recipes — group B (D4 PumpSense, D5 Moura/TGDataset bursts, D7 Mirtaheri)

Nothing is downloaded by a loader. `build_panel()` raises `FileNotFoundError` with the recipe below when a raw file is
missing; downloads / extraction happen only through `python3 scripts/xp_prepare_data.py --dataset <name> [...]`
(which calls the module's `prepare(root, args)`). Raw root = `cfg['root']` (default `data_raw`). All three datasets take
their market bars from the shared Binance kline cache `data_raw/crosspump/binance_klines/<SYMBOL><QUOTE>-1m-YYYY-MM.csv`
(data.binance.vision monthly 1-minute klines, 12 header-less columns) and an OPTIONAL wallet table
`data_raw/crosspump/onchain/<dataset>_transfers.csv` (`key,time_ns,wallet_id,size_adv,size_raw`); without it the wallet
modality is *missing* (reported in `meta['coverage']`), never zero-filled.

Structures below were observed on c225 (`/date/zyy/TradingSignal/data_raw/crosspump/`) on 2026-10-03.

## D4 PumpSense — `xp_pumpsense` (`d4_pumpsense.py`, `configs/crosspump/datasets/pumpsense.yaml`)
Mahrous & Di Pietro, *PumpSense: Detection and Target Extraction of Cryptocurrency Pump-and-Dump on Telegram*, IEEE ICBC
2026, arXiv:2605.09431. Zenodo doi:10.5281/zenodo.19616802, MIT.

```bash
mkdir -p data_raw/crosspump/pumpsense
curl -L -o data_raw/crosspump/pumpsense/icbc_2026_reproducibility-v1.1.zip \
  "https://zenodo.org/records/19616802/files/AhmedMahrous00/icbc_2026_reproducibility-v1.1.zip?download=1"   # 1.87 GB
python3 scripts/xp_prepare_data.py --dataset pumpsense            # extracts only the csv tables (optional: the loader reads the zip in place)
python3 scripts/xp_prepare_data.py --dataset pumpsense --klines   # Binance klines of the pumped pairs (+-31 days around each event)
```

Observed layout: `AhmedMahrous00-icbc_2026_reproducibility-2fdffd8/chat_data/<Group title>/result11.csv` (43 group
folders, 25,492 zip members, mostly photos), columns `date,text,Pump,Exchange,Coin,Notes[,photo]`:

* 283,017 messages, 2,246 rows with `Pump == 1.0` (matches the paper); `photo` exists in 14 of the 43 files only.
* `date` = naive **local export time** (`2022-04-26T17:00:31`), not UTC. Calibrated per group with (i) "pump at 2 PM GMT"
  phrases versus the stamp of the labelled message and (ii) the Binance 1-minute volume at `date - k h`:
  **UTC+2** for most groups, **UTC+3** for *Alt the Way*, *Bitcoin Pump Group*, *Cryptic Trading*,
  *Crypto Coin Pump Signals*, *Big Pump Signal*; no daylight-saving pattern. (`export_utc_offset_hours`,
  `group_utc_offset_hours`.)
* `text` = plain string, or the python repr of the Telegram-Desktop entity list
  (`"['Coin: ', {'type': 'bold', 'text': 'XYZ'}]"`, 205,370 rows), or empty (34,518 photo-only rows).
* `Exchange` of labelled rows: binance 1036, yobit 583, cryptopia 194, kucoin 118, bittrex 105, hotbit 81,
  pancakeswap 21, gate.io 10, coinexchange 9, others 5, empty 84. `Coin` = `sym/quote` lower case (btc 1922, usdt 229).
* `Notes` on labelled rows: CANCELLED/Cancelled/cancelled 83, futures 19, "not preannounced" 15, early/late ...
* No sender column. Labelled rows per year: 2017 195, 2018 563, 2019 572, 2020 310, 2021 359, 2022 232, 2023 15.

Expected episodes: 1,036 Binance announcements (1,009 `*/btc`) -> **563 merged market events** (same coin within 1 h,
before the cancelled/futures filters, 136 coins, 35 groups; 2018 112 / 2019 313 / 2020 277 / 2021 293 / 2022 39 rows);
the count that survives depends on the kline cache (delisted pairs return 404).

## D5 Moura et al. bursts over the TGDataset — `xp_tgdataset_pd` (`d5_tgdataset_pd.py`, `tgdataset_pd.yaml`)
Moura, Paoletti, Ferreira, Almeida, *Don't You Know, Pump it Up!*, ICWSM 2027 (accepted), arXiv:2609.01176; repo MIT.
Raw messages: TGDataset (La Morgia, Mei, Mongardini, KDD 2025; Zenodo record 7640712, CC-BY-4.0) — optional.

```bash
git clone --depth 1 https://github.com/Filipey/investigating-cryptocurrency-manipulation-icwsm2027 data_raw/crosspump/moura
python3 scripts/xp_prepare_data.py --dataset tgdataset_pd --klines      # BTC pair, else USDT pair, of the burst coins listed on Binance
# OPTIONAL (71 GB zipped / ~460 GB): unzip the TGDataset channel json files into data_raw/crosspump/tgdataset/ and run
python3 scripts/xp_prepare_data.py --dataset tgdataset_pd --tgdataset   # -> data_raw/crosspump/tgdataset_pd/burst_messages.csv (coin,time_ns,channel_id,text)
# OPTIONAL paper classifier (never required): HuggingFace filipeasm18/roberta-binary-crypto-related-classifier
```

Observed layout (`reference_data/`):

* `anomaly_time_intervals.json`: `{"1INCH": [{"date": "2020-12-25", "first_message_unix": 1608854402,
  "last_message_unix": 1608939178}], ...}` — 681 coins, **810** bursty (coin, day) pairs, dates 2016-08-26 .. 2022-07-31,
  duration 0 .. 24 h (median 14.6 h).
* `coins_exchanges_and_trades_normalized.json`: `coin -> exchange -> [quotes]` (4,364 coins, lower-case exchange keys);
  `binance` lists 61 burst coins = **78 bursts** (56 with a BTC pair, 22 USDT only).
* The 47 validated events, the raw messages and the OHLCV are not in the repository (README: "not included").
* `02_anomaly_detection.ipynb`: CA-CFAR on the daily mention count with 5 training + 1 guard cells **per side**
  (two-sided): a bursty day is only knowable 6 days later. The loader therefore stamps burst-level social events as
  available at day end + 6 days (`burst_avail: cfar`), which is after the window end -> **the burst-level social
  modality is missing by default**; `day_end` / `first_message` are diagnostics flagged in `meta['pit_warning']`.

Labels are a **proxy** (`meta['label_rule']['type'] == 'proxy (Moura et al. event list not released)'`): rise >= 10 %
within 12 bars of the burst start over the pre-burst 1-hour mean close, and at least half of it reverted within
`post_bars`. Expected episodes: at most 78 bursts (those with klines in the burst month; bursts before the pair was
listed drop out) + sampled controls; the pump / no-reaction split is unknown until the klines are fetched.
TGDataset extraction is written against the published layout and is **unverified** (archive not on c225).

## D7 Mirtaheri et al. — `xp_mirtaheri` (`d7_mirtaheri.py`, `mirtaheri.yaml`)
Mirtaheri, Abu-El-Haija, Morstatter, Ver Steeg, Galstyan, IEEE Trans. Computational Social Systems 8(3), 2021,
doi:10.1109/TCSS.2021.3059286, arXiv:1902.03110. No licence file in the repository.

```bash
git clone --depth 1 https://github.com/Mehrnoom/Cryptocurrency-Pump-Dump data_raw/crosspump/mirtaheri     # 7.4 GB with .git
python3 scripts/xp_prepare_data.py --dataset mirtaheri --klines     # <COIN>BTC klines for every coin / month of coin-pump.csv (most return 404)
```

Observed layout:

* `Telegram/classified/coin-pump.csv`: `,Channel ID,Message ID,Date,Time,Coin` — 62,850 rows, 47,992 messages, 209
  channels, 543 coins, 2017-05-18 .. 2018-07-23; `Date Time` are UTC (equal to the `.jl` `$date`). Top "coins":
  BTC 9,027 (the quote of `XYZ/BTC`), HOLD 2,725, CAN 757 ... -> ticker-matching noise, handled by `exclude_coins`
  and `require_ticker_in_text`.
* `Telegram/classified/price_extract.csv`: `,Channel ID,Message ID,Date,Time,Coin,Sell,Buy` (55,751 rows; unused).
* `Telegram/channels/<channel_id>.jl`: 420 files, 1.19 GB (largest 309 MB); one json per line
  `{"_": "Message", "id", "to_id", "date": {"$date": ms}, "message", "views", "from_id": null, ...}`; `MessageService`
  lines carry no text. `views` is a crawl-time total -> not point-in-time -> not used as reach.
* `Twitter/tweets-ts/$COIN.json`, `Twitter/users-ts/$COIN.json` (1,518 files each), `Twitter/sentiment-ts/COIN.json`
  (1,517 files, **no `$` prefix**): `{"data": [[{"$date": ms}, value], ...]}`, **1-minute cadence**, only minutes with
  tweets, unsorted, 2017-09-01 .. 2018-08-25 (e.g. `$BNT`: 158,184 minutes, max 47 tweets per minute).

Expected episodes: 23,981 rows belong to the 126 coins that have *some* cached Binance BTC klines today; only 2,616
rows fall in a month already cached for the group-A datasets. After `--klines` (2017-2018 months) and the default
filters / 6 h merge the count is a few thousand pump attempts (capped by `max_pumps: 2000`); most 2017-18 attempts ran
on YoBit / Cryptopia / Bittrex coins that never traded on Binance and are dropped (counted in `meta['table']`).
