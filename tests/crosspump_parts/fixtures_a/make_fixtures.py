"""SYNTHETIC FIXTURES for tests/crosspump_parts/test_loaders_a.py -- fake coins, fake channels, seeded random numbers.
Mimics the column layout of the real raw files only; never use for results.  python3 make_fixtures.py [out_root]"""
from __future__ import annotations
import datetime as dt
import gzip
import json
import os
import pickle
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = dt.timezone.utc
T0 = dt.datetime(2021, 3, 2, 0, 0, tzinfo=UTC)          # klines cover 2021-03-02 00:00 .. 2021-03-06 12:00 (6480 minutes)
N_MIN = 6480
# planted spikes: symbol -> (start minute time, minutes to the top)
PUMPS = {"AAA": (dt.datetime(2021, 3, 3, 12, 0, tzinfo=UTC), 7), "BBB": (dt.datetime(2021, 3, 4, 15, 0, tzinfo=UTC), 6), "CCC": (dt.datetime(2021, 3, 5, 9, 0, tzinfo=UTC), 6)}
BASE = {"AAA": 2.0e-5, "BBB": 5.0e-6, "CCC": 1.2e-4, "DDD": 3.3e-5}


def _ms(t: dt.datetime) -> int:
    return int(t.timestamp() * 1000)


def klines(root: str, rng: np.random.Generator) -> None:
    d = os.path.join(root, "crosspump", "binance_klines"); os.makedirs(d, exist_ok=True)
    for sym, p0 in BASE.items():
        r = rng.normal(0, 4e-4, N_MIN); vol = np.exp(rng.normal(5.0, 0.6, N_MIN)); buy = np.clip(rng.normal(0.5, 0.12, N_MIN), 0.02, 0.98)
        if sym in PUMPS:
            start, up = PUMPS[sym]; k = int((start - T0).total_seconds() // 60)
            r[k : k + up + 1] += 0.30 / (up + 1); r[k + up + 1 : k + up + 31] -= 0.25 / 30
            vol[k : k + 40] *= 25; buy[k : k + up + 1] = 0.9
        close = p0 * np.exp(np.cumsum(r)); open_ = np.concatenate([[p0], close[:-1]])
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1e-4, N_MIN))); lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1e-4, N_MIN)))
        cnt = np.maximum(1, (vol / 20).astype(int))
        with open(os.path.join(d, f"{sym}BTC-1m-2021-03.csv"), "w") as f:
            for i in range(N_MIN):
                ot = _ms(T0) + i * 60000
                f.write(f"{ot},{open_[i]:.10f},{hi[i]:.10f},{lo[i]:.10f},{close[i]:.10f},{vol[i]:.2f},{ot + 59999},{vol[i] * close[i]:.8f},{cnt[i]},{vol[i] * buy[i]:.2f},{vol[i] * buy[i] * close[i]:.8f},0\n")


def aggtrades(root: str, rng: np.random.Generator) -> None:
    d = os.path.join(root, "crosspump", "binance_aggtrades"); os.makedirs(d, exist_ok=True)
    day = dt.datetime(2021, 3, 3, tzinfo=UTC); rows = []; aid = 1
    for m in range(0, 1440):
        burst = 715 <= m < 730
        for _ in range(12 if burst else 2):
            t = _ms(day) + m * 60000 + int(rng.integers(0, 60000))
            qty = float(np.exp(rng.normal(3.0, 0.5))) * (20 if burst else 1)
            maker = bool(rng.random() < (0.05 if burst else 0.5))
            rows.append((t, f"{aid},0.00002000,{qty:.2f},{aid},{aid},{t},{'True' if maker else 'False'},True")); aid += 1
    rows.sort()
    with open(os.path.join(d, "AAABTC-aggTrades-2021-03-03.csv"), "w") as f:
        f.write("\n".join(r for _, r in rows) + "\n")


def hu(root: str) -> None:
    lab = os.path.join(root, "crosspump", "hu_pd", "Data", "Telegram", "Labeled"); raw = os.path.join(root, "crosspump", "hu_pd", "Data", "Telegram", "raw")
    os.makedirs(lab, exist_ok=True); os.makedirs(raw, exist_ok=True)
    rows = [("111", "5", "AAA", "binance", "BTC", "2021-03-03 12:01:33+00:00"), ("222", "9", "AAA", "binance", "BTC", "2021-03-03 12:02:10+00:00"),
            ("111", "7", "BBB", "binance", "BTC", "2021-03-04 15:00:02+00:00"), ("222", "12", "CCC", "binance", "BTC", "2021-03-05 09:00:00+00:00"),
            ("333", "1", "DDD", "yobit", "BTC", "2021-03-04 10:00:00+00:00"), ("111", "8", "ZZZ", "binance", "BTC", "2021-03-04 18:00:00+00:00"),
            ("111", "6", "AAA", "binance", "USDT", "2021-03-03 20:00:00+00:00")]
    with open(os.path.join(lab, "PD_logs_cleaned.txt"), "w") as f:
        f.write("channel_id\tsession_id\tcoin\texchange\tpair\ttimestamp\n" + "\n".join("\t".join(r) for r in rows) + "\n")

    def msg(i, t, text, edited=None, kind="Message"):
        return {"_": kind, "id": i, "peer_id": {"_": "PeerChannel", "channel_id": 111}, "date": t, "message": text, "edit_date": edited, "views": 1000 + i, "forwards": 0, "post": True}

    d = lambda *a: dt.datetime(*a, tzinfo=UTC)
    msgs = [msg(9, d(2021, 3, 5, 3, 0, 0), "SYNTHETIC good morning traders"),
            msg(8, d(2021, 3, 4, 15, 0, 2), "SYNTHETIC the coin is BBB buy and hold on binance"),
            msg(7, d(2021, 3, 4, 14, 31, 10), "SYNTHETIC 30 minutes left until the pump signal, be ready"),
            msg(6, d(2021, 3, 4, 12, 0, 0), "SYNTHETIC pump today on binance, btc pair, target 300 profit"),
            msg(5, d(2021, 3, 3, 12, 30, 0), "SYNTHETIC results: great pump, more than 30 profit", edited=d(2021, 3, 3, 14, 0, 0)),
            msg(4, d(2021, 3, 3, 12, 1, 33), "SYNTHETIC the coin is AAA buy and hold on binance"),
            msg(3, d(2021, 3, 3, 11, 32, 7), "SYNTHETIC 30 minutes left until the pump signal, be ready"),
            msg(2, d(2021, 3, 3, 9, 30, 0), "SYNTHETIC next pump signal in 3 hours on binance, btc pair"),
            msg(1, d(2021, 3, 2, 18, 0, 0), None, kind="MessageService"),
            msg(0, d(2021, 3, 2, 10, 0, 0), "SYNTHETIC welcome to the channel")]
    with open(os.path.join(raw, "111.pkl"), "wb") as f:
        pickle.dump(msgs, f, pickle.HIGHEST_PROTOCOL)
    pred = [("222", "40", "synthetic pump signal in 2 hours on binance btc pair be ready", "2021-03-03 10:02:00+00:00", "1.0"),
            ("222", "41", "synthetic the coin is aaa buy now", "2021-03-03 12:02:10+00:00", "1.0"),
            ("222", "55", "synthetic pump in 60 minutes hold your btc", "2021-03-05 08:00:30+00:00", "1.0"),
            ("999", "3", "synthetic unrelated channel message", "2021-03-03 11:00:00+00:00", "1.0")]
    with open(os.path.join(lab, "pred_pump_message.csv"), "w") as f:
        f.write("channel_id\tsample_id\tmessage\tdate\tpred_label\n" + "\n".join("\t".join(r) for r in pred) + "\n")


def sapienza(root: str, rng: np.random.Generator) -> None:
    d = os.path.join(root, "crosspump", "sapienza_pd"); os.makedirs(os.path.join(d, "labeled_features"), exist_ok=True)
    with open(os.path.join(d, "pump_telegram.csv"), "w") as f:
        f.write("symbol,group,date,hour,exchange\nAAA,G1,2021-03-03,12:00,binance\nAAA,G2,2021-03-03,12:00,binance\nBBB,G2,2021-03-04,15:00,binance\n"
                "CCC,G1,2021-03-05,09:00,binance\nDDD,G3,2021-03-04,10:00,yobit\nZZZ,G1,2021-03-04,18:00,binance\n")
    with open(os.path.join(d, "groups.csv"), "w") as f:
        f.write("group_name,group_code,last_time_checked, telegram_link\nSynthetic Group One,G1,2021-03-10,https://t.me/synthetic_one\nSynthetic Group Two,G2,2021-03-10,https://t.me/synthetic_two\n")
    cols = "date,pump_index,std_rush_order,avg_rush_order,std_trades,std_volume,avg_volume,std_price,avg_price,avg_price_max,hour_sin,hour_cos,minute_sin,minute_cos,symbol,gt"
    lines = [cols]
    for pi, (sym, start, gt_at) in enumerate([("AAA", dt.datetime(2021, 3, 3, 11, 50, 0), dt.datetime(2021, 3, 3, 11, 59, 35)),
                                                ("BBB", dt.datetime(2021, 3, 3, 2, 0, 0), dt.datetime(2021, 3, 3, 2, 5, 0))]):      # BBB chunk is > 60 min from its table row
        for k in range(60):
            t = start + dt.timedelta(seconds=25 * k)
            v = rng.normal(0, 0.01, 8)
            lines.append(f"{t:%Y-%m-%d %H:%M:%S},{pi}," + ",".join(f"{x:.3f}" for x in v) + f",0.0,1.0,0.0,1.0,{sym},{int(t == gt_at)}")
    with gzip.open(os.path.join(d, "labeled_features", "features_25S.csv.gz"), "wt") as f:
        f.write("\n".join(lines) + "\n")


def xu(root: str) -> None:
    d = os.path.join(root, "crosspump", "xu_pumpdump", "code and data"); os.makedirs(d, exist_ok=True)

    def rec(cur, ex, title, link, sig, hist, before, mx):
        return {"marketHistoryEntries": hist, "currency": cur, "exchange": ex, "channelTitle": title, "channelLink": link, "max": mx, "priceBeforePump": before,
                "duration": "00:07", "volume": 1.5, "signalTime": sig, "ourBuyPrice": 0.0, "ourBuyTime": "0001-01-01T00:00:00", "ourProfit": 0.0,
                "theoreticalBuyPrice": before, "theoreticalBuyTime": sig, "theoreticalProfit": 10.0, "id": f"synthetic-{cur}-{sig[:10]}"}

    hist = [{"rate": 2.0e-5, "amount": 10.0, "timeStamp": "2021-03-03T11:10:00Z"}, {"rate": 9.9e-5, "amount": 1.0, "timeStamp": "2021-03-03T11:20:00.5Z"},
            {"rate": 2.1e-5, "amount": 12.0, "timeStamp": "2021-03-03T12:00:06.443Z"}, {"rate": 2.6e-5, "amount": 30.0, "timeStamp": "2021-03-03T12:07:30Z"},
            {"rate": 2.3e-5, "amount": 8.0, "timeStamp": "2021-03-03T12:12:00Z"}]
    data = [rec("AAA", "Binance", "Synthetic Pumps", "https://t.me/synthetic_pumps", "2021-03-03T12:00:04.5Z", hist, 2.0e-5, 2.6e-5),
            rec("AAA", "Binance", "Synthetic Relay", "https://t.me/synthetic_relay", "2021-03-03T12:00:40Z", [], 2.0e-5, 2.6e-5),
            rec("BBB", "Binance", "Synthetic Pumps", "https://t.me/synthetic_pumps", "2021-03-04T15:00:01Z", [], 5.0e-6, 6.0e-6),
            rec("CCC", "Cryptopia", "Synthetic Other", "https://t.me/synthetic_other", "2021-03-05T09:00:00Z", [], 1.2e-4, 1.5e-4),
            rec("BBB", "Binance", "Synthetic Pumps", "https://t.me/synthetic_pumps", "1999-01-14T17:00:37Z", [], 5.0e-6, 6.0e-6)]
    with open(os.path.join(d, "rawpumpolymp.json"), "w") as f:
        json.dump(data, f)


def ardia(root: str) -> None:
    d = os.path.join(root, "crosspump", "ardia", "data"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "list_pd_events.csv"), "w") as f:
        f.write("Date,Channel,Currency,Exchange,success,pump_date\n"
                "2021-03-03T12:01:00Z,Synthetic Chan A,AAA,Binance,1,2021-03-03T12:07:00Z\n"
                "2021-03-03T12:03:00Z,Synthetic Chan B,AAA,Binance,1,2021-03-03T12:07:00Z\n"
                "2021-03-04T15:02:13Z,Synthetic Chan A,BBB,Binance,1,2021-03-04T14:58:00Z\n"
                "2021-03-05T09:00:00Z,Synthetic Chan C,CCC,Binance,0,\n")


def onchain(root: str) -> None:
    d = os.path.join(root, "crosspump", "onchain"); os.makedirs(d, exist_ok=True)
    ns = lambda *a: int(dt.datetime(*a, tzinfo=UTC).timestamp()) * 10 ** 9
    with open(os.path.join(d, "hu_pd_transfers.csv"), "w") as f:
        f.write("key,time_ns,wallet_id,size_adv,size_raw\n"
                f"AAABTC@2021-03-03T12:01Z,{ns(2021, 3, 3, 10, 15, 20)},9001,0.02,5000\n"
                f"AAABTC@2021-03-03T12:01Z,{ns(2021, 3, 3, 11, 40, 0)},9002,0.10,25000\n"
                f"NOPEBTC@2021-03-03T12:01Z,{ns(2021, 3, 3, 11, 40, 0)},9003,0.10,25000\n")


def build(root: str) -> str:
    rng = np.random.default_rng(7)
    klines(root, rng); aggtrades(root, rng); hu(root); sapienza(root, rng); xu(root); ardia(root); onchain(root)
    with open(os.path.join(root, "SYNTHETIC_FIXTURES.txt"), "w") as f:
        f.write("SYNTHETIC FIXTURES written by tests/crosspump_parts/fixtures_a/make_fixtures.py (seed 7). Not real data.\n")
    return root


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "root")))
