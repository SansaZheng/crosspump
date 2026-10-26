# SYNTHETIC FIXTURES (group loaders_c: D8 dex_wash, D9 rugpull1k, D10 memechain)

Every file under this directory is SYNTHETIC, written by `make_fixtures.py` (seeded). The files only mimic the
column layout of the real raw files observed on c225 (EtherDeltaTrades.csv, IDEXTrades.csv, token_decimals.json,
pump_telegram.csv, coin-pump.csv, RugPull1K dataset.xlsx, MemeChain data.zip) so that
`tests/crosspump_parts/test_loaders_c.py` can run offline. `crosspump/memechain_trades/trades.csv` mimics the OPTIONAL
user-supplied trade export (the MemeChain release has no trade table). Never use these files for results.
