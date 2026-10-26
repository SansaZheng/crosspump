"""SYNTHETIC FIXTURES for tests/crosspump_parts/test_loaders_c.py (D8 dex_wash, D9 rugpull1k, D10 memechain).

Everything written here is made up: it only mimics the COLUMN LAYOUT of the real raw files observed on c225 so the loaders
can be exercised offline. Never use these files for results. Run: python3 tests/crosspump_parts/fixtures_c/make_fixtures.py
"""
from __future__ import annotations
import io
import json
import os
import zipfile
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ETH = "0x0000000000000000000000000000000000000000"
S = 10 ** 9


def ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value // S)


def addr(k: int) -> str:
    return "0x" + f"{k:040x}"


# pump times planted in the fixtures (the test imports these)
D8 = {"AAA": {"address": addr(0xAAA), "decimals": 18, "pump": "2018-03-05 16:00", "peak_offset_min": 40},
      "BBB": {"address": addr(0xBBB), "decimals": 8, "pump": "2018-03-07 18:00", "peak_offset_min": 70}}
D8_START, D8_DAYS = "2018-03-01 00:00", 10
D8_ABSURD_PRICE = 777.0            # only in rows that the loader must drop (failed IDEX tx, token-token trade)


def make_d8(root: str) -> None:
    rng = np.random.default_rng(8)
    vd = os.path.join(root, "crosspump", "victor_dex")
    os.makedirs(vd, exist_ok=True)
    tokens = {"ETH": {"name": "Ether", "decimals": 18, "address": ETH}}
    ed, ix = [], []
    wallets = [addr(0x1000 + k) for k in range(40)]
    n = 0
    for sym, info in D8.items():
        tokens[sym] = {"name": f"Synthetic {sym}", "decimals": info["decimals"], "address": info["address"], "slug": sym.lower()}
        t0, pump = ts(D8_START), ts(info["pump"])
        times = np.sort(t0 + rng.integers(0, D8_DAYS * 86400, size=D8_DAYS * 288))
        extra = pump - rng.integers(0, 6 * 3600, size=150)          # thicker trading before the pump
        after = pump + rng.integers(0, 3 * 3600, size=250)
        times = np.sort(np.concatenate([times, extra, after]))
        base = 0.001 * np.exp(np.cumsum(rng.normal(0, 0.002, len(times))))
        peak_t = pump + info["peak_offset_min"] * 60
        for t, px in zip(times, base):
            if pump <= t <= peak_t + 600:
                px = px * (1.0 + 2.0 * (t - pump) / (peak_t - pump + 1))
            amount = float(rng.uniform(50, 500))
            taker_buy = bool(rng.random() < (0.75 if pump - 6 * 3600 <= t <= peak_t else 0.5))
            maker, taker = rng.choice(wallets, 2, replace=False)
            tok_raw = int(amount * 10 ** info["decimals"]); eth_raw = int(amount * px * 10 ** 18)
            n += 1
            tx = "0x" + f"{n:064x}"
            if rng.random() < 0.5:                                     # EtherDelta row: get = maker, give = taker
                if taker_buy:
                    ed.append((tx, 5000000 + n, int(t), ETH, eth_raw, info["address"], tok_raw, maker, taker))
                else:
                    ed.append((tx, 5000000 + n, int(t), info["address"], tok_raw, ETH, eth_raw, maker, taker))
            else:                                                      # IDEX row: amount in tokenBuy units
                status = "" if n % 97 == 0 else "1.0"
                if taker_buy:                                          # maker buys ETH
                    ix.append((tx, status, 5000000 + n, 290000, 5000000000, int(t), eth_raw * 2, tok_raw * 2, 190000, n, eth_raw, n, 10 ** 15, 10 ** 16, ETH, info["address"], maker, taker))
                else:                                                  # maker buys the token
                    ix.append((tx, status, 5000000 + n, 290000, 5000000000, int(t), tok_raw * 2, eth_raw * 2, 190000, n, tok_raw, n, 10 ** 15, 10 ** 16, info["address"], ETH, maker, taker))
        # the planted peak trade (highest price of the whole token history), a self-trade and a 2-cycle inside the window
        dec = info["decimals"]
        ed.append(("0x" + f"{900000 + len(ed):064x}", 1, peak_t, ETH, int(100 * 0.05 * 10 ** 18), info["address"], 100 * 10 ** dec, wallets[0], wallets[1]))
        ed.append(("0x" + f"{910000 + len(ed):064x}", 1, pump - 7200, info["address"], 200 * 10 ** dec, ETH, int(200 * 0.001 * 10 ** 18), wallets[5], wallets[5]))
        ed.append(("0x" + f"{920000 + len(ed):064x}", 1, pump - 5400, ETH, int(300 * 0.001 * 10 ** 18), info["address"], 300 * 10 ** dec, wallets[6], wallets[7]))
        ed.append(("0x" + f"{930000 + len(ed):064x}", 1, pump - 5300, ETH, int(300 * 0.001 * 10 ** 18), info["address"], 300 * 10 ** dec, wallets[7], wallets[6]))
        # rows the loader must DROP: token-token trade and a failed IDEX transaction, both at an absurd price
        ed.append(("0x" + f"{940000 + len(ed):064x}", 1, pump - 3600, info["address"], 10 ** dec, addr(0xDEAD), 10 ** 18, wallets[2], wallets[3]))
        ix.append(("0x" + f"{950000 + len(ix):064x}", "0.0", 1, 290000, 5000000000, pump - 3000, int(D8_ABSURD_PRICE * 10 ** 18), 10 ** dec, 190000, 1, int(D8_ABSURD_PRICE * 10 ** 18), 1, 0, 0,
                   ETH, info["address"], wallets[2], wallets[3]))
    # a token that is in neither pump table (must not be selected)
    tokens["CCC"] = {"name": "Synthetic CCC", "decimals": 18, "address": addr(0xCCC)}
    ed.append(("0x" + f"{960000:064x}", 1, ts("2018-03-05 12:00"), ETH, 10 ** 18, addr(0xCCC), 10 ** 20, wallets[0], wallets[1]))
    ed_cols = ["transaction_hash", "block_number", "timestamp", "tokenGet", "amountGet", "tokenGive", "amountGive", "get", "give"]
    ix_cols = ["transaction_hash", "status", "block_number", "gas", "gas_price", "timestamp", "amountBuy", "amountSell", "expires", "nonce", "amount", "tradeNonce", "feeMake", "feeTake",
               "tokenBuy", "tokenSell", "maker", "taker"]
    for name, cols, rows in (("EtherDeltaTrades.csv", ed_cols, ed), ("IDEXTrades.csv", ix_cols, ix)):
        rows = sorted(rows, key=lambda r: r[cols.index("timestamp")])
        with open(os.path.join(vd, name), "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(vd, "token_decimals.json"), "w") as f:
        json.dump(tokens, f)
    with open(os.path.join(vd, "EtherDollarPrice.csv"), "w") as f:
        f.write('"Date(UTC)","UnixTimeStamp","Value"\n"3/1/2018","1519862400","870.00"\n')
    sp = os.path.join(root, "crosspump", "sapienza_pd"); os.makedirs(sp, exist_ok=True)
    with open(os.path.join(sp, "pump_telegram.csv"), "w") as f:
        f.write("symbol,group,date,hour,exchange\nAAA,SYNTHGROUP,2018-03-05,16:00,yobit\nBBB,SYNTHGROUP,2018-03-07,18:05,cryptopia\nZZZ,OTHER,2018-03-06,12:00,binance\n")
    mt = os.path.join(root, "crosspump", "mirtaheri", "Telegram", "classified"); os.makedirs(mt, exist_ok=True)
    with open(os.path.join(mt, "coin-pump.csv"), "w") as f:
        f.write(",Channel ID,Message ID,Date,Time,Coin\n0,1111111111,10,2018-03-07,18:00:00,BBB\n1,1111111111,11,2018-03-07,18:10:30,BBB\n2,2222222222,5,2018-03-07,17:30:00,BTC\n")


# ------------------------------------------------------------------------------------------------ D9 (xlsx by hand)
D9_HEADER = ["Project Title", "MaxPrice (Quarter 1) ", "MaxPrice (Quarter 2) ", "MaxPrice (Quarter 3) ", "MaxPrice (Quarter 4) ", "Blockchain", "the number of Transactions",
             "Token concentration ratio per holder", "Total Variance", "Variance of holders with more than 1% tokens", "Token balance", "Sign", "first deposits", "Blockchain Type",
             "Smart Contract (online)", "smart Contract (offline)", "website", "x profile", "class", "project starting date", "project end date",
             "Google results for project title (first day)", "Google results for project title (project duration/2)", "Google results for project website (first day)",
             "Google results for project website (duration/2)", "Google results for project x profile (first days)", "Google results for project x profile (duration/2)"]


def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def write_xlsx(path: str, header, rows) -> None:
    shared, index = [], {}

    def sid(v: str) -> int:
        if v not in index:
            index[v] = len(shared); shared.append(v)
        return index[v]
    lines = []
    for r, row in enumerate([header] + rows, start=1):
        cells = []
        for c, v in enumerate(row):
            if v is None:
                continue
            ref = f"{_col(c)}{r}"
            if isinstance(v, str):
                cells.append(f'<c r="{ref}" t="s"><v>{sid(v)}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{v!r}</v></c>')
        lines.append(f'<row r="{r}">{"".join(cells)}</row>')
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="{ns}"><sheetData>{"".join(lines)}</sheetData></worksheet>'
    sst = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst xmlns="{ns}" count="{len(shared)}" uniqueCount="{len(shared)}">' + "".join(f"<si><t>{escape(v)}</t></si>" for v in shared) + "</sst>"
    wb = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>')
    top = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
           '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
          '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", top); z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rels); z.writestr("xl/worksheets/sheet1.xml", sheet); z.writestr("xl/sharedStrings.xml", sst)


def make_d9(root: str) -> None:
    rng = np.random.default_rng(9)
    d = os.path.join(root, "crosspump", "rugpull1k"); os.makedirs(d, exist_ok=True)
    rows = []
    for k in range(14):
        scam = k % 2 == 0
        start = 44000 + 40 * k
        end = start + (20 if k == 3 else int(rng.integers(120, 600)))     # project 3: start inside the 30-day window
        if k == 11:
            end = start - 5                                               # must be dropped (end <= start)
        p1 = float(rng.uniform(0.001, 2.0)); p2 = p1 * float(rng.uniform(0.5, 3.0))
        ntx = int(rng.integers(100, 2_000_000))
        rows.append([f"Synthetic Project {k} (SP{k})", p1, p2, p2 * (0.01 if scam else 1.1), p2 * (0.0001 if scam else 1.2), "BSC" if k % 3 else "ETH",
                     f" {ntx:,}" if k % 4 == 1 else ntx, int(rng.integers(10, 50000)), float(rng.uniform(0, 5)), float(rng.uniform(0, 5)), float(rng.integers(10 ** 6, 10 ** 12)),
                     "BEP-20 TOKEN*" if k == 5 else f"SP{k}", int(rng.integers(0, 60)), "POSA" if k % 3 else "POS", "sourse code", "CODE",
                     "None" if k == 6 else f"https://synthetic-project-{k}.example/", "None" if k in (6, 7) else f"https://x.com/synthetic{k % 5}", "scam" if scam else "normal",
                     start, end, int(rng.integers(0, 900)), int(rng.integers(0, 900)), "-" if k == 2 else int(rng.integers(0, 900)), "-" if k == 2 else int(rng.integers(0, 900)),
                     int(rng.integers(0, 900)), int(rng.integers(0, 900))])
    write_xlsx(os.path.join(d, "dataset.xlsx"), D9_HEADER, rows)


# ------------------------------------------------------------------------------------------------ D10
D10_T0 = "2024-05-01 00:00:00"


def make_d10(root: str) -> None:
    rng = np.random.default_rng(10)
    toks = []
    for k in range(12):
        plat = ["bsc", "solana", "ethereum", "base"][k % 4]
        a = ("So1" + f"{k:041d}") if plat == "solana" else "0x" + f"{0xABC000 + k:040X}"       # mixed case on purpose (A2)
        toks.append({"symbol": f"mm{k}", "name": f"synthetic meme {k}", "address": a, "platform": plat, "creation_date": float(ts(D10_T0) + 7200 * k)})
    reg = pd.DataFrame(toks)
    soc = reg[["name", "platform"]].copy()
    soc["website"] = [f"https://meme{k}.example/" if k % 2 == 0 else None for k in range(12)]
    soc["twitter"] = [f"https://x.com/synthdev{k % 3}" if k % 3 != 2 else None for k in range(12)]
    soc["telegram"] = [f"https://t.me/meme{k}" if k % 4 == 0 else None for k in range(12)]
    soc["discord"] = None
    soc["symbol"] = reg["symbol"]; soc["address"] = reg["address"]
    web = pd.DataFrame({"name": reg["name"][:8], "address": reg["address"][:8].str.lower(), "platform": reg["platform"][:8], "url": "https://example/",
                        "socket_reachable": True, "request_reachable": True, "status_code": 200.0, "chain_patrol_status_code": 200,
                        "chain_patrol_status": ["BLOCKED" if k in (1, 6) else ("ALLOWED" if k == 2 else "UNKNOWN") for k in range(8)], "chain_patrol_source": "chainpatrol",
                        "chain_patrol_reason": "not-found", "domain_name": "example", "registrar": "SYNTH", "expiration_date": "2026-01-01 00:00:00", "name_servers": "[]",
                        "error_whois": None, "website": "https://example/", "resolved_url": "https://example/"})
    eco = pd.DataFrame({"name": reg["name"], "platform": reg["platform"], "price": 1e-6, "market_cap": 1e4, "volume": None, "symbol": reg["symbol"],
                        "creation_date": reg["creation_date"], "address": reg["address"]})
    conf = reg[["symbol", "name", "platform", "address"]]

    def zip_to(path: str) -> None:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, df in (("confirmed_meme_coins.csv", conf), ("final_filtered_meme_coins.csv", reg[["symbol", "name", "address", "platform", "creation_date"]]),
                             ("economic_info_data.csv", eco), ("social_info.csv", soc), ("web_page_analysis.csv", web)):
                buf = io.StringIO(); df.to_csv(buf, index=False); z.writestr("data/" + name, buf.getvalue())
    for sub in ("memechain", "memechain_trades"):
        d = os.path.join(root, "crosspump", sub); os.makedirs(d, exist_ok=True)
        zip_to(os.path.join(d, "data.zip"))
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("SYNTHETIC FIXTURE mimicking the MemeChain data.zip layout.\n")
    # user-supplied trade file (trades mode): token 0 dies after 6 h (peak at +2 h), token 1 BLOCKED survivor, token 2 survivor,
    # token 3 dies after 1 h (< 10 bars -> dropped), token 4 survivor
    rows = []
    wallets = [f"w{k:03d}" for k in range(30)]
    life_h = {0: 6, 1: 96, 2: 120, 3: 1, 4: 96}
    for k, hours in life_h.items():
        first = ts(D10_T0) + 7200 * k + 600
        times = np.sort(first + rng.integers(1, hours * 3600, size=max(30, hours * 12)))
        times = np.concatenate([[first], times])
        px = 1e-6 * np.exp(np.cumsum(rng.normal(0, 0.01, len(times))))
        for t, p in zip(times, px):
            buy = bool(rng.random() < 0.6)
            rows.append((reg["address"][k], reg["platform"][k], int(t), float(p), float(rng.uniform(1e3, 1e5)), "buy" if buy else "sell", rng.choice(wallets), f"pool{k}", f"tx{len(rows)}"))
        if k == 0:
            rows.append((reg["address"][k], reg["platform"][k], first + 2 * 3600 + 60, 5e-5, 5e4, "buy", wallets[0], f"pool{k}", "txpeak"))
    tr = pd.DataFrame(rows, columns=["address", "platform", "timestamp", "price", "amount", "side", "taker", "maker", "tx_hash"]).sort_values("timestamp")
    tr.to_csv(os.path.join(root, "crosspump", "memechain_trades", "trades.csv"), index=False)


def main(root: str = HERE) -> None:
    make_d8(root); make_d9(root); make_d10(root)
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write("# SYNTHETIC FIXTURES (group loaders_c: D8 dex_wash, D9 rugpull1k, D10 memechain)\n\n"
                "Every file under this directory is SYNTHETIC, written by `make_fixtures.py` (seeded). The files only mimic the\n"
                "column layout of the real raw files observed on c225 (EtherDeltaTrades.csv, IDEXTrades.csv, token_decimals.json,\n"
                "pump_telegram.csv, coin-pump.csv, RugPull1K dataset.xlsx, MemeChain data.zip) so that\n"
                "`tests/crosspump_parts/test_loaders_c.py` can run offline. `crosspump/memechain_trades/trades.csv` mimics the OPTIONAL\n"
                "user-supplied trade export (the MemeChain release has no trade table). Never use these files for results.\n")


if __name__ == "__main__":
    main()
    print("fixtures written under", HERE)
