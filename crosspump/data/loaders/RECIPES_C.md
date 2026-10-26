# CrossPump loaders D8 – D10: download recipes and observed raw layouts

All three loaders follow `crosspump/CONVENTIONS.md`. Nothing is downloaded by `build_panel()`; run
`python3 scripts/xp_prepare_data.py --dataset <dex_wash|rugpull1k|memechain>` (it calls the module's `prepare(root, args)`)
or use the manual commands below. Raw root = `cfg['root']` (default `data_raw`). Layouts below were inspected on c225
(`/date/zyy/TradingSignal/data_raw/crosspump/`) on 2026-10-03.

| key | module | registry | panel kind | licence |
|---|---|---|---|---|
| D8 | `d8_dex_wash.py` | `xp_dex_wash` | time series, 15-min bars built from on-chain trades | CC-BY-4.0 |
| D9 | `d9_rugpull1k.py` | `xp_rugpull1k` | **tabular only → degenerate panel**, excluded from lead-time metrics | **CC BY-NC 4.0** |
| D10 | `d10_memechain.py` | `xp_memechain` | **tabular only** with the public release; time series only with a user-supplied trade file | CC-BY-4.0 |

## D8 xp_dex_wash — Victor & Weintraud, WWW 2021 (doi:10.1145/3442381.3449824), Zenodo 4540223

```
mkdir -p data_raw/crosspump/victor_dex && cd data_raw/crosspump/victor_dex
for f in EtherDeltaTrades.csv IDEXTrades.csv EtherDollarPrice.csv token_decimals.json; do
  curl -L -C - -o $f "https://zenodo.org/records/4540223/files/$f?download=1"; done
```
Pump times are read from the D2 / D7 raw files: `data_raw/crosspump/sapienza_pd/pump_telegram.csv` (required) and
`data_raw/crosspump/mirtaheri/Telegram/classified/coin-pump.csv` (optional).

Observed files
- `EtherDeltaTrades.csv` (1.05 GB, 3,576,607 rows, 2017-02-09 .. 2020-05-04): `transaction_hash, block_number, timestamp, tokenGet, amountGet, tokenGive, amountGive, get, give` — amounts are raw integers (up to ~1e30), ETH = zero address.
- `IDEXTrades.csv` (2.1 GB, 5,344,149 rows, 2017-09-27 .. 2020-05-04): `transaction_hash, status, block_number, gas, gas_price, timestamp, amountBuy, amountSell, expires, nonce, amount, tradeNonce, feeMake, feeTake, tokenBuy, tokenSell, maker, taker` — `status` is 1.0 (5,341,213), 0.0 (2,361, dropped) or empty (575, kept).
- `token_decimals.json`: 515 entries `{SYMBOL: {name, decimals, address[, slug]}}` (incl. ETH). Only tokens listed here can be used (decimals are needed).
- `EtherDollarPrice.csv`: `"Date(UTC)","UnixTimeStamp","Value"` (not used: prices stay in ETH).
- `sapienza_pd/pump_telegram.csv`: `symbol,group,date,hour,exchange` (1,109 rows; exchanges binance 520, yobit 276, cryptopia 210, bittrex 102, kucoin 2 — no DEX).
- `mirtaheri/Telegram/classified/coin-pump.csv`: `,Channel ID,Message ID,Date,Time,Coin` (62,850 rows).

Cache: `data_raw/crosspump/victor_dex/cache/xp_dex_wash_<etherdelta|idex>_<selection hash>.csv` (normalised trades of the selected
tokens; written on the first `build_panel()`; `cache_dir` in the config overrides the location).

## D9 xp_rugpull1k — TM-RugPull / RugPull1K, arXiv:2602.21529

```
mkdir -p data_raw/crosspump/rugpull1k
curl -L -o data_raw/crosspump/rugpull1k/dataset.xlsx https://huggingface.co/datasets/fati-80-Sho/RugPull1K/resolve/main/dataset.xlsx
# or: git clone https://github.com/mohamadpishdar/RugPull-Defender data_raw/crosspump/rugpull_defender   (src/DataSet/dataset.xlsx is accepted as a fallback)
```
HuggingFace file list: `README.md`, `dataset.xlsx`, `SOURCE CODE.zip` (contract sources, not used). The HF card and the repository
`Licence` file both say **CC BY-NC 4.0**.

Observed `dataset.xlsx`: one sheet, 1,000 project rows (977 distinct titles), header
`Project Title | MaxPrice (Quarter 1..4) | Blockchain | the number of Transactions | Token concentration ratio per holder | Total Variance | Variance of holders with more than 1% tokens | Token balance | Sign | first deposits | Blockchain Type | Smart Contract (online) | smart Contract (offline) | website | x profile | class | project starting date | project end date | Google results for project {title, website, x profile} x {first day, duration/2}`.
class: scam 599 / normal 401; chains ETH 633, BSC 321, POLYGON 24, ARBI 12, FANTOM 4(+FTM 1), CRONO 2, BASE 2, SNOW 1; dates are Excel
serials; 5 rows have end <= start (dropped). **No per-project time series exist** → degenerate panel (`meta['tabular_only']`).
The xlsx is read with a stdlib reader (`onchain_utils.read_xlsx`); openpyxl is not needed.

## D10 xp_memechain — MemeChain, arXiv:2601.22185, Zenodo 18246856

```
mkdir -p data_raw/crosspump/memechain && cd data_raw/crosspump/memechain
curl -L -C - -o data.zip "https://zenodo.org/records/18246856/files/data.zip?download=1"
curl -L -o README.md "https://zenodo.org/records/18246856/files/README.md?download=1"
```
Observed `data.zip` (read in place): `data/confirmed_meme_coins.csv` (34,988), `data/final_filtered_meme_coins.csv` (31,811;
`symbol,name,address,platform,creation_date`), `data/economic_info_data.csv` (20,692; one crawl-time snapshot per token),
`data/social_info.csv` (31,811; `website,twitter,telegram,discord`), `data/web_page_analysis.csv` (23,837; `chain_patrol_status`:
UNKNOWN 12,682 / ALLOWED 634 / BLOCKED 9 / empty 10,430), `data/html_web_pages_path.csv`, `data/memes_with_image_paths.csv`,
`data/html_web_pages.zip`, `data/logos_images.zip`.

**The release has no transaction table.** The "ceases trading within 24 h" label of the paper cannot be rebuilt from it. Default
mode = tabular (label = ChainPatrol BLOCKED, 9 tokens). For the time-series mode supply
`data_raw/crosspump/memechain/trades.csv` (or `trades/*.csv`) with columns
`address, platform, timestamp, price, amount, side (taker side) | taker_buy, buyer/seller | taker/maker[, tx_hash]`.

## Optional wallet file for exchange-traded sets
`onchain_utils.optional_transfers(root, dataset)` reads `<root>/crosspump/onchain/<dataset>_transfers.csv`
(`key,time_ns,wallet_id,size_adv,size_raw`) and returns `None` when it is absent (the wallet modality is then missing).
D8 – D10 do not need it (their wallet events come from the trades themselves).
