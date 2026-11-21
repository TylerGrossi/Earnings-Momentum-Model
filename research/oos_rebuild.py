"""
Step B: out-of-sample test on earnings seasons the strategy never saw, plus a control group.

The live scraper stopped adding trades around 2026-05-01. This script rebuilds the screen
point-in-time for the same company universe, so it can be tested on May-Nov 2026 earnings.

PRE-REGISTERED (written before any out-of-sample number was looked at, 2026-11-20):
  Primary  H1: screened trades, all sectors, 1-day hold: mean return in excess of the sector
               ETF > 0, tested on weekly-portfolio returns (one-sided, 5%).
  Primary  H2: screened trades beat the control group (same universe, same seasons, screen did
               not fire) on 1-day sector excess, tested on the weekly difference.
  Secondary: ex-Technology; 5-day hold; after assumed costs. Reported, not used to pick a rule.
  Nothing else (sector picks, hold lengths, thresholds) is tuned on the out-of-sample window.

Rebuilt screen (calibrated on the in-sample trades with a known Date Added, see report 04):
  * earnings date and BMO/AMC time from yfinance (ET hour < 12 = BMO, >= 16 = AMC, else dropped)
  * at the close of the trading day BEFORE the entry day: SMA20 > SMA50 and SMA20 crossed above
    SMA50 within the last 10 trading days (Finviz ta_sma20_cross50a; 91% of known-time
    tracked trades had crossed within 10 days)
  * the Barchart "Buy" opinion CANNOT be rebuilt (no history), so this tests the SMA screen
    alone. That is a limitation: the live strategy also required the Barchart signal.

Universe: every ticker that ever appeared in returns_tracker.csv (names that passed the live
screen at least once). The control group comes from the same universe, so the comparison is
like for like, but neither is a random sample of the market.

Run after excess_returns.py:  python research/oos_rebuild.py   (first run ~10 min to fetch dates)
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE, OUT = HERE / "cache", HERE / "output"
sys.path.insert(0, str(HERE))
from excess_returns import COST_BUCKETS, SECTOR_ETF, cost_bps, load_prices  # noqa: E402

IS_WINDOW = ("2025-11-01", "2026-05-01")
OOS_WINDOW = ("2026-05-04", "2026-11-18")
CROSS_WITHIN = 10
HOLDS = [1, 5]
BOOT_N = 10_000


def universe() -> pd.DataFrame:
    t = pd.read_csv(ROOT / "returns_tracker.csv")
    t["yf"] = t["Ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
    return t.groupby("yf").agg(Sector=("Sector", "last"), mcap=("Market Cap", "last")).reset_index()


def fetch_dates(tickers: list[str]) -> pd.DataFrame:
    path = CACHE / "earnings_dates.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    def one(t):
        for attempt in range(3):
            try:
                d = yf.Ticker(t).get_earnings_dates(limit=12)
                if d is None or d.empty:
                    return None
                d = d.reset_index()
                d.columns = ["ts", "eps_est", "eps_rep", "surprise"][: len(d.columns)]
                d["yf"] = t
                return d
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        parts = [p for p in ex.map(one, tickers) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
    df.to_parquet(path)
    print(f"Fetched earnings dates for {df['yf'].nunique()} of {len(tickers)} tickers")
    return df


def build_events(dates: pd.DataFrame, uni: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    cal = px["SPY"].dropna().index
    px = px.reindex(cal)
    s20, s50 = px.rolling(20).mean(), px.rolling(50).mean()
    above = (s20 > s50) & s50.notna()
    # Trading days since the latest False->True flip, as of each close.
    flip = above & ~above.shift(1, fill_value=False)
    last_flip = pd.DataFrame(np.where(flip, np.arange(len(cal))[:, None], np.nan),
                             index=cal, columns=px.columns).ffill()
    since = np.arange(len(cal))[:, None] - last_flip

    d = dates.merge(uni, on="yf", how="inner")
    d["hour"] = d["ts"].dt.hour
    d["Timing"] = np.where(d["hour"] < 12, "BMO", np.where(d["hour"] >= 16, "AMC", None))
    d["earn_date"] = pd.to_datetime(d["ts"].dt.date)
    d = d[d["Timing"].notna()].drop_duplicates(["yf", "earn_date"])

    spy = px["SPY"].to_numpy(float)
    rows = []
    for r in d.itertuples(index=False):
        if r.yf not in px:
            continue
        if r.Timing == "BMO":
            i0 = cal.searchsorted(r.earn_date, side="left") - 1
        else:
            i0 = cal.searchsorted(r.earn_date, side="right") - 1
        if i0 < 60 or cal[i0] > pd.Timestamp.today():
            continue
        sig_i = i0 - 1  # decide on the close before the entry day
        a = bool(above[r.yf].iloc[sig_i])
        k = since[r.yf].iloc[sig_i]
        screened = a and np.isfinite(k) and k < CROSS_WITHIN
        s = px[r.yf].to_numpy(float)
        etf = SECTOR_ETF.get(r.Sector)
        b = px[etf].to_numpy(float) if etf in px else None
        row = {"yf": r.yf, "Sector": r.Sector, "mcap": r.mcap, "Timing": r.Timing,
               "earn_date": r.earn_date, "Entry Date": cal[i0], "screened": screened,
               "sma_above": a, "surprise": r.surprise}
        for h in HOLDS:
            if i0 + h >= len(cal) or not (np.isfinite(s[i0]) and np.isfinite(s[i0 + h])):
                row[f"xs_{h}d"] = np.nan
                continue
            raw = s[i0 + h] / s[i0] - 1
            row[f"raw_{h}d"] = raw
            row[f"xs_{h}d"] = raw - (b[i0 + h] / b[i0] - 1) if b is not None else np.nan
            row[f"xs_spy_{h}d"] = raw - (spy[i0 + h] / spy[i0] - 1)
        rows.append(row)
    ev = pd.DataFrame(rows)
    ev["cost"] = ev["mcap"].map(cost_bps) / 1e4
    ev["week"] = ev["Entry Date"].dt.to_period("W-FRI")
    return ev


def wtest(x: pd.Series, week: pd.Series, rng) -> dict:
    d = pd.DataFrame({"x": x.to_numpy(), "w": week.to_numpy()}).dropna()
    if len(d) < 5:
        return {"n": len(d)}
    w = d.groupby("w")["x"].mean()
    groups = [g.to_numpy() for _, g in d.groupby("w")["x"]]
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(BOOT_N)]
    t = w.mean() / (w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 2 else np.nan
    from scipy import stats
    return {"n": len(d), "weeks": len(w), "mean": d["x"].mean(), "median": d["x"].median(),
            "hit": (d["x"] > 0).mean(), "t_weekly": t,
            "p_one_sided": 1 - stats.t.cdf(t, len(w) - 1) if np.isfinite(t) else np.nan,
            "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5)}


def diff_test(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    """Weekly mean(screened) - mean(control), over weeks that have both."""
    from scipy import stats
    wa, wb = a.groupby("week")[col].mean(), b.groupby("week")[col].mean()
    dd = (wa - wb).dropna()
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 2 else np.nan
    return {"weeks": len(dd), "mean_weekly_diff": dd.mean(), "t": t,
            "p_one_sided": 1 - stats.t.cdf(t, len(dd) - 1) if np.isfinite(t) else np.nan}


def main() -> None:
    rng = np.random.default_rng(5)
    uni = universe()
    tickers = sorted(set(uni["yf"]))
    px = load_prices(sorted(set(tickers) | {"SPY"} | set(SECTOR_ETF.values())), "2024-09-01", False)
    px = px.loc[:, px.notna().any()]
    dates = fetch_dates([t for t in tickers if t in px])
    ev = build_events(dates, uni, px)
    ev.to_csv(OUT / "oos_events.csv", index=False)

    rows, diffs = [], []
    for wname, (lo, hi) in [("in-sample (Nov 2025-Apr 2026)", IS_WINDOW),
                            ("OUT-OF-SAMPLE (May-Nov 2026)", OOS_WINDOW)]:
        w = ev[(ev["earn_date"] >= lo) & (ev["earn_date"] <= hi)]
        for uname, u in [("all sectors", w), ("ex-Technology", w[w["Sector"] != "Technology"])]:
            for grp, g in [("screened", u[u["screened"]]), ("control", u[~u["screened"]])]:
                for col in ["xs_1d", "xs_5d"]:
                    for net in [False, True]:
                        x = g[col] - (g["cost"] if net else 0)
                        s = wtest(x, g["week"], rng)
                        s.update({"window": wname, "universe": uname, "group": grp,
                                  "measure": col + (" net" if net else "")})
                        rows.append(s)
            for col in ["xs_1d", "xs_5d"]:
                s = diff_test(u[u["screened"]], u[~u["screened"]], col)
                s.update({"window": wname, "universe": uname, "measure": col})
                diffs.append(s)
    res, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    res.to_csv(OUT / "oos_results.csv", index=False)
    diffs.to_csv(OUT / "oos_screen_vs_control.csv", index=False)

    # How well does the rebuilt screen match the live one in-sample?
    tr = pd.read_csv(OUT / "excess_returns_trades.csv", parse_dates=["Entry Date"])
    tr["yf"] = tr["Ticker"].astype(str).str.replace(".", "-", regex=False)
    m = tr.merge(ev, on=["yf", "Entry Date"], how="left")
    match = {"tracked trades": len(tr), "found in rebuilt events": int(m["screened"].notna().sum()),
             "rebuilt screen fired": int(m["screened"].fillna(False).sum())}

    pd.set_option("display.width", 240)
    f4 = "{:.4f}".format
    print("Rebuild vs live (in-sample):", match)
    print(ev.groupby([ev["earn_date"].dt.to_period("M"), "screened"]).size().unstack(fill_value=0).to_string())
    print("\n=== Results ===")
    cols = ["window", "universe", "group", "measure", "n", "weeks", "mean", "median", "hit",
            "t_weekly", "p_one_sided", "ci_lo", "ci_hi"]
    print(res[cols].to_string(index=False, float_format=f4))
    print("\n=== Screened minus control (weekly difference) ===")
    print(diffs.to_string(index=False, float_format=f4))


if __name__ == "__main__":
    main()
