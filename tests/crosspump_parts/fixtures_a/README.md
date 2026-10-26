# SYNTHETIC FIXTURES (group A loaders: D1 hu_pd, D2 sapienza_pd, D3 xu_pumpolymp, D6 ardia_pd)

Everything under `root/` is **synthetic**: fake coins (AAA, BBB, CCC, DDD, ZZZ), fake channels, fake messages, fake wallets
and seeded random-walk "klines" with planted price / volume spikes. Nothing here is real market, Telegram or on-chain data and
none of it may be used for any result. The files only mimic the **column layout** of the real raw files so that the loaders
can be exercised offline by `python3 tests/crosspump_parts/test_loaders_a.py`.

`make_fixtures.py` writes the tree deterministically (seed 7); the test calls it when `root/` is missing.

    root/crosspump/binance_klines/<SYM>BTC-1m-2021-03.csv        12 header-less Binance kline columns, open_time in ms
    root/crosspump/binance_aggtrades/AAABTC-aggTrades-2021-03-03.csv   8 header-less aggTrades columns
    root/crosspump/hu_pd/Data/Telegram/Labeled/PD_logs_cleaned.txt     TSV channel_id session_id coin exchange pair timestamp
    root/crosspump/hu_pd/Data/Telegram/Labeled/pred_pump_message.csv   TSV channel_id sample_id message date pred_label
    root/crosspump/hu_pd/Data/Telegram/raw/111.pkl                     list of Telethon-style message dicts (plain python)
    root/crosspump/sapienza_pd/{pump_telegram.csv,groups.csv,labeled_features/features_25S.csv.gz}
    root/crosspump/xu_pumpdump/code and data/rawpumpolymp.json
    root/crosspump/ardia/data/list_pd_events.csv
    root/crosspump/onchain/hu_pd_transfers.csv                         key,time_ns,wallet_id,size_adv,size_raw (optional wallet file)
