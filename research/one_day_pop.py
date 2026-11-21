"""
Step A of the "1-day pop" idea: hold one day, skip Technology.

This rule was proposed AFTER seeing reports 01-02 (90% of the excess arrives on day 1), so on
this sample it is a hypothesis, not a result. The script reports:

  1. The rule on the tracked trades: full sample, before/after 2026-01-09 (the first row with
     a Date Added), by month, after assumed costs.
  2. Whether dropping Technology is special: the 1-day excess per sector, and the result of
     dropping each sector in turn. If dropping Tech is no better than dropping a random sector,
     the exclusion is noise.
  3. When to sell on day 1: hourly exits at the 10:30, 12:30 and 15:30 bars vs the day-1 close,
     each net of the sector ETF over the same span (hourly ETF prices from yfinance).

The clean test is step B (oos_rebuild.py), on earnings seasons no rule has seen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
CACHE = HERE / "cache"
SPLIT = pd.Timestamp("2026-01-09")
EXCLUDE = "Technology"
BOOT_N = 10_000


def wstats(x: pd.Series, entry: pd.Series, rng=None) -> dict:
    d = pd.DataFrame({"x": x.to_numpy(), "e": pd.to_datetime(entry).to_numpy()}).dropna()
    if len(d) < 5:
        return {"n": len(d)}
    wk = d["e"].dt.to_period("W-FRI")
    w = d.groupby(wk)["x"].mean()
    out = {"n": len(d), "weeks": len(w), "mean": d["x"].mean(), "median": d["x"].median(),
           "hit": (d["x"] > 0).mean(),
           "t_weekly": w.mean() / (w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 2 else np.nan}
    if rng is not None and len(w) > 2:
        groups = [g.to_numpy() for _, g in d.groupby(wk)["x"]]
        b = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(BOOT_N)]
        out["ci_lo"], out["ci_hi"] = np.percentile(b, [2.5, 97.5])
    return out


def hourly_exits(trades: pd.DataFrame) -> pd.DataFrame:
    """Day-1 exits at given hourly bars, net of the sector ETF over the same span."""
    from excess_returns import SECTOR_ETF
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    h = pd.read_csv(ROOT / "hourly_prices.csv", parse_dates=["Datetime"])
    h["Earnings Date"] = pd.to_datetime(h["Earnings Date"]).dt.date

    path = CACHE / "etf_hourly.parquet"
    etfs = sorted(set(SECTOR_ETF.values()))
    if path.exists():
        eh = pd.read_parquet(path)
    else:
        raw = yf.download(etfs, period="730d", interval="1h", auto_adjust=True, progress=False,
                          group_by="column")
        eh = raw["Close"]
        eh.index = pd.to_datetime(eh.index).tz_convert("America/New_York").tz_localize(None)
        eh.to_parquet(path)

    rows = []
    for t in trades.itertuples(index=False):
        g = h[(h["Ticker"] == t.Ticker) & (h["Earnings Date"] == t._1)].sort_values("Datetime")
        etf = SECTOR_ETF.get(t.Sector)
        if g.empty or etf not in eh:
            continue
        base_px = g["Base Price"].iloc[0]
        # ETF base: last hourly bar of the entry day (the 15:30 bar closes at 16:00).
        e = eh[etf].dropna()
        e_base = e[e.index.normalize() == pd.Timestamp(t._5)]
        d1 = g[g["Trading Day"] == 1]
        if e_base.empty or d1.empty:
            continue
        e0 = e_base.iloc[-1]
        row = {"Ticker": t.Ticker, "Entry Date": t._5, "Sector": t.Sector}
        for bar in ["09:30", "11:30", "14:30", "15:30"]:
            b = d1[d1["Time"] == bar]
            if b.empty:
                continue
            ts = b["Datetime"].iloc[0]
            eb = e[e.index == ts]
            if eb.empty:
                continue
            row[f"xs_{bar}"] = (b["Close"].iloc[0] / base_px - 1) - (eb.iloc[0] / e0 - 1)
            row[f"raw_{bar}"] = b["Close"].iloc[0] / base_px - 1
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    import sys
    sys.path.insert(0, str(HERE))
    rng = np.random.default_rng(3)
    tr = pd.read_csv(OUT / "excess_returns_trades.csv", parse_dates=["Entry Date"])
    tr["Earnings Date"] = pd.to_datetime(tr["Earnings Date"]).dt.date
    tr["net_1d"] = tr["xs_sector_1d"] - tr["cost"]
    rule = tr[tr["Sector"] != EXCLUDE]

    rows = []
    for label, d in [("all sectors", tr), (f"ex-{EXCLUDE}", rule)]:
        for per, sub in [("full sample", d), ("before 2026-01-09", d[d["Entry Date"] < SPLIT]),
                         ("from 2026-01-09", d[d["Entry Date"] >= SPLIT])]:
            for col in ["xs_sector_1d", "net_1d"]:
                s = wstats(sub[col], sub["Entry Date"], rng)
                s.update({"universe": label, "period": per, "measure": col})
                rows.append(s)
    res = pd.DataFrame(rows)

    sec = []
    late = tr[tr["Entry Date"] >= SPLIT]
    for name, g in tr.groupby("Sector"):
        a, b = wstats(g["xs_sector_1d"], g["Entry Date"]), wstats(
            late.loc[late["Sector"] == name, "xs_sector_1d"], late.loc[late["Sector"] == name, "Entry Date"])
        drop_all = tr[tr["Sector"] != name]["xs_sector_1d"].mean()
        drop_late = late[late["Sector"] != name]["xs_sector_1d"].mean()
        sec.append({"sector": name, "n": a["n"], "mean_1d": a.get("mean"), "n_late": b["n"],
                    "mean_1d_late": b.get("mean"), "mean_if_dropped": drop_all,
                    "mean_if_dropped_late": drop_late})
    sec = pd.DataFrame(sec).sort_values("mean_1d")
    sec["rank_drop_late"] = sec["mean_if_dropped_late"].rank(ascending=False)

    hx = hourly_exits(rule)
    hrows = []
    for bar, label in [("09:30", "10:30 (first hourly bar)"), ("11:30", "12:30"),
                       ("14:30", "15:30"), ("15:30", "16:00 close")]:
        for per, sub in [("full sample", hx), ("from 2026-01-09", hx[pd.to_datetime(hx["Entry Date"]) >= SPLIT])]:
            s = wstats(sub[f"xs_{bar}"], sub["Entry Date"])
            s.update({"exit": label, "period": per})
            hrows.append(s)
    hres = pd.DataFrame(hrows)

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "pop_rule.csv", index=False)
    sec.to_csv(OUT / "pop_by_sector.csv", index=False)
    hres.to_csv(OUT / "pop_intraday_exit.csv", index=False)

    pd.set_option("display.width", 220)
    f4 = "{:.4f}".format
    print("=== 1-day sector excess: all sectors vs ex-Tech ===")
    print(res.to_string(index=False, float_format=f4))
    print("\n=== By sector (1-day sector excess); 'late' = from 2026-01-09 ===")
    print(sec.to_string(index=False, float_format=f4))
    print(f"\n=== Day-1 exit time, ex-{EXCLUDE} (sector-ETF adjusted) ===")
    print(hres.to_string(index=False, float_format=f4))


if __name__ == "__main__":
    main()
