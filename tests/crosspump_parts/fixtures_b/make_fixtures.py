"""SYNTHETIC FIXTURES for tests/crosspump_parts/test_loaders_b.py (see README.md). Seeded; fake coins, fake groups, fake
messages, random-walk klines with planted spikes. Mimics the real column layouts only -- never use as data."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

MIN_MS = 60_000


def ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 10 ** 6)


def klines(path: str, start: str, days: int, seed: int, pumps=(), p0: float = 1e-4) -> None:
    """12 header-less Binance columns, 1-minute rows; pumps = [(utc time, rise)] plants a 10-minute ramp and a 50-minute decay."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    t0 = ms(start)
    t = t0 + np.arange(n) * MIN_MS
    lr = rng.normal(0, 2e-4, n)
    vol = rng.lognormal(3.0, 0.5, n)
    bump = np.zeros(n)
    for when, rise in pumps:
        k = int((ms(when) - t0) // MIN_MS)
        up = np.log1p(rise)
        bump[k:k + 10] += up * (np.arange(10) + 1) / 10
        bump[k + 10:k + 60] += up * (1 - (np.arange(50) + 1) / 50)
        vol[k:k + 20] *= 40
    close = p0 * np.exp(np.cumsum(lr) + bump)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1e-4, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1e-4, n)))
    tb = vol * np.clip(0.5 + rng.normal(0, 0.05, n), 0.05, 0.95)
    df = pd.DataFrame({0: t, 1: open_, 2: high, 3: low, 4: close, 5: vol, 6: t + MIN_MS - 1, 7: vol * close, 8: rng.integers(1, 50, n), 9: tb, 10: tb * close, 11: 0})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, header=False, index=False, float_format="%.10g")


def build(root: str) -> None:
    X = os.path.join(root, "crosspump")
    K = os.path.join(X, "binance_klines")
    # ------------------------------------------------------------------ klines (UTC)
    klines(os.path.join(K, "AAABTC-1m-2021-03.csv"), "2021-03-01", 8, 1, [("2021-03-03 17:00", 0.6), ("2021-03-06 18:00", 0.4)])
    klines(os.path.join(K, "BBBBTC-1m-2021-03.csv"), "2021-03-01", 8, 2, [("2021-03-04 16:00", 0.5)])
    klines(os.path.join(K, "CCCBTC-1m-2021-03.csv"), "2021-03-01", 8, 3, [])
    klines(os.path.join(K, "MMABTC-1m-2018-03.csv"), "2018-03-01", 8, 4, [("2018-03-03 12:00", 0.3)])
    klines(os.path.join(K, "MMBBTC-1m-2018-03.csv"), "2018-03-01", 8, 5, [("2018-03-05 09:30", 0.25)])
    klines(os.path.join(K, "FUNXBTC-1m-2018-03.csv"), "2018-03-01", 8, 6, [])
    klines(os.path.join(K, "QQABTC-1m-2022-03.csv"), "2022-03-01", 8, 7, [("2022-03-03 06:10", 0.35)])
    klines(os.path.join(K, "QQBUSDT-1m-2022-03.csv"), "2022-03-01", 8, 8, [], p0=2.5)
    # ------------------------------------------------------------------ D4 PumpSense (local export time; group 1/2 = UTC+2, group 3 = UTC+3 via config)
    P = os.path.join(X, "pumpsense", "chat_data")
    g1 = [("2021-03-01T10:00:00", "Welcome to the fixture group", "", "", "", ""),
          ("2021-03-03T15:00:10", "['Next ', {'type': 'bold', 'text': 'pump'}, ' in 4 hours on Binance, get ready']", "", "", "", ""),
          ("2021-03-03T18:30:00", "30 minutes left until the pump! Buy fast and hold", "", "", "", ""),
          ("2021-03-03T18:55:00", "5 min", "", "", "", ""),
          ("2021-03-03T19:00:20", "['The coin is: ', {'type': 'hashtag', 'text': '#AAA'}, ' buy now, target 300%']", 1, "binance", "aaa/btc", ""),
          ("2021-03-03T19:20:00", "Great pump everyone, +60% profit", "", "", "", ""),
          ("2021-03-04T12:00:00", "", "", "", "", ""),
          ("2021-03-04T18:00:05", "", 1, "binance", "bbb/btc", ""),                         # photo-only labelled announcement (UTC 16:00:05)
          ("2021-03-05T19:00:00", "pump cancelled today, sorry", 1, "binance", "ccc/btc", "CANCELLED"),
          ("2021-03-05T20:00:00", "ZZZ on yobit now", 1, "yobit", "zzz/btc", ""),
          ("2021-03-06T20:00:30", "Coin: AAA  buy and hold, target x2", 1, "binance", "aaa/btc", ""),
          ("2021-03-06T21:00:00", "long BBB futures", 1, "binance", "bbb/usdt", "futures")]
    g2 = [("2021-03-03T18:00:00", "Pump in one hour", "", "", "", "", ""),
          ("2021-03-03T19:01:40", "AAA AAA AAA go go", 1, "binance", "aaa/btc", "Copied from VIP", "photos/photo_1@03-03-2021_19-01-40.jpg"),
          ("2021-03-03T19:30:00", "nice", "", "", "", "", "")]
    g3 = [("2021-03-06T20:50:00", "signal incoming, be ready", "", "", "", ""),
          ("2021-03-06T21:00:45", "AAA/BTC signal: buy zone 100 sats", 1, "binance", "aaa/btc", "")]           # UTC+3 group -> 18:00:45 UTC
    for name, rows, cols in (("Fixture Group One", g1, ["date", "text", "Pump", "Exchange", "Coin", "Notes"]),
                             ("Fixture Group Two", g2, ["date", "text", "Pump", "Exchange", "Coin", "Notes", "photo"]),
                             ("Fixture Group Three", g3, ["date", "text", "Pump", "Exchange", "Coin", "Notes"])):
        os.makedirs(os.path.join(P, name), exist_ok=True)
        pd.DataFrame(rows, columns=cols).to_csv(os.path.join(P, name, "result11.csv"), index=False)
    os.makedirs(os.path.join(X, "onchain"), exist_ok=True)
    a = pd.Timestamp("2021-03-03 17:00:20", tz="UTC").value
    pd.DataFrame({"key": ["AAABTC@2021-03-03T17:00Z"] * 3 + ["NOPE@2021-01-01T00:00Z"], "time_ns": [a - 3 * 3600 * 10 ** 9, a - 1800 * 10 ** 9, a - 600 * 10 ** 9, a],
                  "wallet_id": [11, 12, 11, 13], "size_adv": [0.02, 0.05, 0.11, 0.3], "size_raw": [1000.0, 2500.0, 5500.0, 1.0]}
                 ).to_csv(os.path.join(X, "onchain", "pumpsense_transfers.csv"), index=False)
    # ------------------------------------------------------------------ D7 Mirtaheri (UTC)
    M = os.path.join(X, "mirtaheri")
    os.makedirs(os.path.join(M, "Telegram", "classified"), exist_ok=True); os.makedirs(os.path.join(M, "Telegram", "channels"), exist_ok=True)
    rows = [(1001, 50, "2018-03-03", "12:00:30", "MMA"), (1001, 50, "2018-03-03", "12:00:30", "BTC"), (1002, 7, "2018-03-03", "12:20:00", "MMA"),
            (1001, 60, "2018-03-05", "09:30:10", "MMB"), (1002, 9, "2018-03-05", "10:00:00", "HOLD"), (1002, 11, "2018-03-06", "08:00:00", "FUNX"),
            (1002, 12, "2018-03-06", "09:00:00", "ZZZ")]
    pd.DataFrame(rows, columns=["Channel ID", "Message ID", "Date", "Time", "Coin"]).to_csv(os.path.join(M, "Telegram", "classified", "coin-pump.csv"))

    def msg(i, ch, when, text):
        return {"_": "Message", "id": i, "to_id": {"_": "PeerChannel", "channel_id": ch}, "date": {"$date": ms(when)}, "message": text, "out": False, "post": True,
                "from_id": None, "views": 1234 + i, "entities": []}
    ch1 = [{"_": "MessageService", "id": 1, "to_id": {"_": "PeerChannel", "channel_id": 1001}, "date": {"$date": ms("2018-03-01 00:00")}, "action": {"_": "MessageActionChannelCreate", "title": "fixture"}},
           msg(40, 1001, "2018-03-03 09:00:00", "Big signal today, be ready"), msg(49, 1001, "2018-03-03 11:55:00", "5 minutes"),
           msg(50, 1001, "2018-03-03 12:00:30", "Buy $MMA on Binance 100 sats, target 150"), msg(51, 1001, "2018-03-03 12:40:00", "MMA hit target, profit!"),
           msg(55, 1001, "2018-03-04 08:00:00", ""), msg(60, 1001, "2018-03-05 09:30:10", "MMB/BTC buy now on Bittrex"), msg(61, 1001, "2018-03-06 10:00:00", "good morning")]
    ch2 = [msg(7, 1002, "2018-03-03 12:20:00", "MMA pump running, buy"), msg(9, 1002, "2018-03-05 10:00:00", "HOLD your coins"),
           msg(11, 1002, "2018-03-06 08:00:00", "have funx with trading today"), msg(12, 1002, "2018-03-06 09:00:00", "ZZZ on yobit")]
    for ch, lst in ((1001, ch1), (1002, ch2)):
        with open(os.path.join(M, "Telegram", "channels", f"{ch}.jl"), "w") as f:
            for o in lst[::-1]:                                   # deliberately unsorted
                f.write(json.dumps(o) + "\n")
    rng = np.random.default_rng(9)
    for sym, burst in (("MMA", "2018-03-03 11:30"), ("MMB", None)):
        t0 = ms("2018-03-02 00:00")
        minutes = np.sort(rng.choice(6 * 1440, 900, replace=False))
        if burst:
            b = int((ms(burst) - t0) // MIN_MS); minutes = np.unique(np.r_[minutes, np.arange(b, b + 45)])
        n = rng.integers(1, 4, len(minutes)); users = np.maximum(1, n - rng.integers(0, 2, len(minutes))); sent = np.round(rng.uniform(-0.5, 0.9, len(minutes)), 4)
        perm = rng.permutation(len(minutes))
        for sub, name, v in (("tweets-ts", f"${sym}.json", n), ("users-ts", f"${sym}.json", users), ("sentiment-ts", f"{sym}.json", sent)):
            os.makedirs(os.path.join(M, "Twitter", sub), exist_ok=True)
            keep = perm if sub != "sentiment-ts" else perm[: len(perm) - 20]            # sentiment misses some minutes (as in the real files)
            with open(os.path.join(M, "Twitter", sub, name), "w") as f:
                json.dump({"data": [[{"$date": int(t0 + int(minutes[k]) * MIN_MS)}, (int(v[k]) if sub != "sentiment-ts" else float(v[k]))] for k in keep]}, f)
    # ------------------------------------------------------------------ D5 Moura reference data
    R = os.path.join(X, "moura", "reference_data"); os.makedirs(R, exist_ok=True)
    u = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10 ** 9)
    bursts = {"QQA": [{"date": "2022-03-03", "first_message_unix": u("2022-03-03 06:02:10"), "last_message_unix": u("2022-03-03 20:10:00")},
                      {"date": "2022-03-05", "first_message_unix": u("2022-03-05 03:00:00"), "last_message_unix": u("2022-03-05 03:00:00")}],
              "QQB": [{"date": "2022-03-04", "first_message_unix": u("2022-03-04 10:00:00"), "last_message_unix": u("2022-03-04 22:00:00")}],
              "QQC": [{"date": "2022-03-04", "first_message_unix": u("2022-03-04 11:00:00"), "last_message_unix": u("2022-03-04 12:00:00")}],
              "QQD": [{"date": "2022-03-04", "first_message_unix": u("2022-03-04 11:00:00"), "last_message_unix": u("2022-03-04 12:00:00")}]}
    with open(os.path.join(R, "anomaly_time_intervals.json"), "w") as f:
        json.dump(bursts, f, indent=1)
    with open(os.path.join(R, "coins_exchanges_and_trades_normalized.json"), "w") as f:
        json.dump({"QQA": {"binance": ["USDT", "BTC"], "kucoin": ["USDT"]}, "QQB": {"binance": ["USDT"]}, "QQC": {"gateio": ["USDT"]}, "QQD": {"binance": ["BTC"]}}, f, indent=1)
    with open(os.path.join(root, "SYNTHETIC_FIXTURES.txt"), "w") as f:
        f.write("SYNTHETIC FIXTURES generated by tests/crosspump_parts/fixtures_b/make_fixtures.py -- not real data.\n")


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "root"))
    print("fixtures written")
