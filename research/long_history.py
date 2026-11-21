"""
Report 05: the 1-day earnings pop with the SMA screen alone, over 2010-2026.

Rule under test (frozen in report 04; no Barchart signal):
  * a company reports earnings; time of day from Yahoo (before 09:30 ET = BMO, from 16:00 = AMC,
    anything in between is dropped)
  * at the close BEFORE the entry day: SMA20 > SMA50, and SMA20 crossed above SMA50 within the
    last 10 trading days
  * buy at the close before the announcement (BMO: prior day; AMC: earnings day),
    sell at the next session's close

PRE-REGISTERED (written 2026-11-20 before running; the 2010-01-01 to 2025-10-31 window has
never been looked at by any rule in this research):
  H1: screened events, 1-day return in excess of the sector ETF, mean > 0
      (t-test on weekly equal-weight returns, one-sided, 5%).
  H2: screened minus control (same universe, same weeks, screen did not fire) > 0
      (t-test on the weekly difference, one-sided, 5%).
  Secondary (reported, not used to choose anything): every year separately; size (S&P 500 /
  400 / 600); BMO vs AMC; cross window 5 / 10 / 20 days; "uptrend but no recent cross" group;
  5-day hold; SPY and market-model benchmarks; assumed costs; the 2025-11 to 2026-11 window
  already seen in reports 01-04, shown for reference only.

Universe: current S&P 500, 400 and 600 members (Wikipedia), with their GICS sectors.
Survivorship bias: companies that left these indexes (often after bad news) are missing,
which can flatter both groups. H2 compares groups drawn from the same companies, so it is the
more robust test.

Sector ETF benchmark: the GICS sector's SPDR ETF; before XLRE (Oct 2015) and XLC (Jun 2018)
existed, SPY is used for those sectors.

Data check: if the earnings dates and times are right, the absolute abnormal move should spike
on the assumed reaction day compared with the days around it. Wrong times would shift the
spike and bias the test toward zero, not create a false positive.

Run:  python research/long_history.py   (first run downloads ~1,500 tickers; ~15 minutes)
"""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

HERE = Path(__file__).resolve().parent
CACHE, OUT = HERE / "cache", HERE / "output"
START = "2009-06-01"           # price history (SMA50 warm-up before 2010)
TEST = ("2010-01-01", "2025-10-31")
SEEN = ("2025-11-01", "2026-11-18")
CROSS_WINDOWS = [5, 10, 20]
PRIMARY_CROSS = 10
GICS_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Industrials": "XLI", "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC", "Materials": "XLB",
}
INDEX_COST_BPS = {"S&P 500": 10, "S&P 400": 20, "S&P 600": 40}  # assumed round trip
BOOT_N = 5000
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------------- data

def universe() -> pd.DataFrame:
    path = CACHE / "sp1500.csv"
    if path.exists():
        return pd.read_csv(path)
    H = {"User-Agent": "Mozilla/5.0 research-script"}
    parts = []
    for idx, slug in [("S&P 500", "List_of_S%26P_500_companies"),
                      ("S&P 400", "List_of_S%26P_400_companies"),
                      ("S&P 600", "List_of_S%26P_600_companies")]:
        html = requests.get(f"https://en.wikipedia.org/wiki/{slug}", headers=H, timeout=30).text
        tables = pd.read_html(io.StringIO(html))
        t = next(t for t in tables if "Symbol" in t.columns and "GICS Sector" in t.columns)
        parts.append(pd.DataFrame({"yf": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
                                   "Sector": t["GICS Sector"], "index": idx}))
    u = pd.concat(parts).drop_duplicates("yf")
    CACHE.mkdir(exist_ok=True)
    u.to_csv(path, index=False)
    return u


def prices(tickers: list[str]) -> pd.DataFrame:
    path = CACHE / "daily_close_long.parquet"
    px = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    import yfinance as yf
    for attempt in range(3):
        missing = [t for t in tickers if t not in px.columns or not px[t].notna().any()]
        if not missing:
            break
        print(f"Price download pass {attempt + 1}: {len(missing)} tickers")
        for i in range(0, len(missing), 200):
            chunk = missing[i:i + 200]
            raw = yf.download(chunk, start=START, auto_adjust=True, progress=False,
                              group_by="column", threads=True)
            if raw.empty:
                continue
            c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
                columns={"Close": chunk[0]})
            c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
            c = c.loc[:, c.notna().any()]
            px = c if px.empty else px.drop(columns=[x for x in c if x in px]).join(c, how="outer")
        px = px.sort_index()
        px.to_parquet(path)
    return px


def earnings_dates(tickers: list[str]) -> pd.DataFrame:
    path = CACHE / "earnings_dates_long.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf

    def one(t):
        for attempt in range(3):
            try:
                d = yf.Ticker(t).get_earnings_dates(limit=100)
                if d is None or d.empty:
                    return None
                d = d.reset_index().iloc[:, :4]
                d.columns = ["ts", "eps_est", "eps_rep", "surprise"]
                d["yf"] = t
                return d
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        parts = [p for p in ex.map(one, tickers) if p is not None]
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"], utc=True).dt.tz_convert("America/New_York")
    d.to_parquet(path)
    print(f"Earnings dates for {d['yf'].nunique()} of {len(tickers)} tickers")
    return d


# ---------------------------------------------------------------------------------- events

def build_events(u: pd.DataFrame, px: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    cal = px["SPY"].dropna().index
    px = px.reindex(cal)
    cols = {c: i for i, c in enumerate(px.columns)}
    P = px.to_numpy(float)
    s20 = px.rolling(20).mean().to_numpy()
    s50 = px.rolling(50).mean().to_numpy()
    above = (s20 > s50) & np.isfinite(s50)
    flip = above & ~np.vstack([np.zeros((1, P.shape[1]), bool), above[:-1]])
    idx = np.where(flip, np.arange(len(cal))[:, None], np.nan)
    last_flip = pd.DataFrame(idx).ffill().to_numpy()
    since = np.arange(len(cal))[:, None] - last_flip
    spy = P[:, cols["SPY"]]
    # Market-model betas: rolling 250-day beta to SPY, lagged 21 days.
    rets = px.pct_change()
    beta = (rets.rolling(250, min_periods=120).cov(rets["SPY"])
            .div(rets["SPY"].rolling(250, min_periods=120).var(), axis=0)).shift(21).to_numpy()

    d = dates.merge(u, on="yf", how="inner")
    mins = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d["Timing"] = np.where(mins < 9 * 60 + 30, "BMO", np.where(mins >= 16 * 60, "AMC", None))
    d["earn_date"] = pd.to_datetime(d["ts"].dt.date)
    d = d[d["Timing"].notna()].drop_duplicates(["yf", "earn_date"])
    d = d[(d["earn_date"] >= "2010-01-01") & (d["earn_date"] <= SEEN[1])]

    def etf_col(sector, when):
        e = GICS_ETF.get(sector)
        if e is None or e not in cols or not np.isfinite(P[when, cols[e]]):
            return cols["SPY"]
        return cols[e]

    rows = []
    for r in d.itertuples(index=False):
        j = cols.get(r.yf)
        if j is None:
            continue
        side = "left" if r.Timing == "BMO" else "right"
        i0 = int(cal.searchsorted(r.earn_date, side=side)) - 1
        if i0 < 60 or i0 + 6 >= len(cal):
            continue
        s = P[:, j]
        if not np.isfinite(s[i0]) or not np.isfinite(s[i0 - 1]):
            continue
        sig = i0 - 1
        k = since[sig, j]
        e = etf_col(r.Sector, i0)
        b = P[:, e]
        row = {"yf": r.yf, "Sector": r.Sector, "index": r.index, "Timing": r.Timing,
               "earn_date": r.earn_date, "Entry Date": cal[i0], "above": bool(above[sig, j]),
               "since_cross": k, "surprise": r.surprise, "bench": px.columns[e]}
        for h in (1, 5):
            if not np.isfinite(s[i0 + h]):
                continue
            raw = s[i0 + h] / s[i0] - 1
            row[f"raw_{h}d"] = raw
            row[f"xs_{h}d"] = raw - (b[i0 + h] / b[i0] - 1)
            row[f"xs_spy_{h}d"] = raw - (spy[i0 + h] / spy[i0] - 1)
            bt = beta[i0, j]
            if np.isfinite(bt):
                row[f"mm_{h}d"] = raw - bt * (spy[i0 + h] / spy[i0] - 1)
        # Reaction-day data check: |sector-excess| on the day before, of, and after day 1.
        for lag, name in [(-1, "abs_d0"), (0, "abs_d1"), (1, "abs_d2")]:
            a, c = i0 + lag, i0 + lag + 1
            if np.isfinite(s[a]) and np.isfinite(s[c]):
                row[name] = abs((s[c] / s[a] - 1) - (b[c] / b[a] - 1))
        rows.append(row)
    ev = pd.DataFrame(rows)
    ev["week"] = ev["Entry Date"].dt.to_period("W-FRI")
    ev["cost"] = ev["index"].map(INDEX_COST_BPS) / 1e4
    for w in CROSS_WINDOWS:
        ev[f"scr_{w}"] = ev["above"] & (ev["since_cross"] < w)
    ev["screened"] = ev[f"scr_{PRIMARY_CROSS}"]
    ev["uptrend_old_cross"] = ev["above"] & ~ev["screened"]
    return ev


# ---------------------------------------------------------------------------------- stats

def wtest(x: pd.Series, week: pd.Series, rng=None) -> dict:
    d = pd.DataFrame({"x": x.to_numpy(), "w": week.to_numpy()}).dropna()
    if len(d) < 10:
        return {"n": len(d)}
    w = d.groupby("w")["x"].mean()
    t = w.mean() / (w.std(ddof=1) / np.sqrt(len(w)))
    out = {"n": len(d), "weeks": len(w), "mean": d["x"].mean(), "median": d["x"].median(),
           "hit": (d["x"] > 0).mean(), "weekly_mean": w.mean(), "t": t,
           "p_one_sided": 1 - stats.t.cdf(t, len(w) - 1),
           "ann_sharpe_weekly": w.mean() / w.std(ddof=1) * np.sqrt(52)}
    if rng is not None:
        g = [v.to_numpy() for _, v in d.groupby("w")["x"]]
        b = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(BOOT_N)]
        out["ci_lo"], out["ci_hi"] = np.percentile(b, [2.5, 97.5])
    return out


def diff(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    dd = (a.groupby("week")[col].mean() - b.groupby("week")[col].mean()).dropna()
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    return {"weeks": len(dd), "mean_weekly_diff": dd.mean(), "t": t,
            "p_one_sided": 1 - stats.t.cdf(t, len(dd) - 1)}


def main() -> None:
    rng = np.random.default_rng(2010)
    u = universe()
    tick = sorted(set(u["yf"]))
    px = prices(tick + ["SPY"] + sorted(set(GICS_ETF.values())))
    dates = earnings_dates([t for t in tick if t in px.columns])
    ev = build_events(u, px, dates)
    ev.to_parquet(OUT / "long_events.parquet")
    print(f"Universe {len(u)}; with prices {sum(t in px for t in tick)}; events {len(ev)};"
          f" screened {int(ev['screened'].sum())}")

    test = ev[(ev["earn_date"] >= TEST[0]) & (ev["earn_date"] <= TEST[1])]
    seen = ev[(ev["earn_date"] >= SEEN[0]) & (ev["earn_date"] <= SEEN[1])]
    scr, ctl = test[test["screened"]], test[~test["screened"]]

    out = {}
    # Data check.
    out["reaction_day_check"] = test[["abs_d0", "abs_d1", "abs_d2"]].median().to_frame("median |excess|").T

    # Primary.
    prim = []
    for name, g in [("screened", scr), ("control (screen did not fire)", ctl),
                    ("uptrend, cross older than 10d", test[test["uptrend_old_cross"]]),
                    ("below SMA50 trend", test[~test["above"]])]:
        for col in ["xs_1d"]:
            s = wtest(g[col], g["week"], rng)
            s.update({"group": name, "measure": col})
            prim.append(s)
    out["primary"] = pd.DataFrame(prim)
    out["primary_diff"] = pd.DataFrame([dict(diff(scr, ctl, "xs_1d"), comparison="screened - control"),
                                        dict(diff(scr, test[test["uptrend_old_cross"]], "xs_1d"),
                                             comparison="screened - uptrend old cross")])

    # Secondary.
    sec = []
    for label, g in [("screened, after assumed costs", scr.assign(v=scr["xs_1d"] - scr["cost"])),
                     ("screened, SPY benchmark", scr.assign(v=scr["xs_spy_1d"])),
                     ("screened, market model", scr.assign(v=scr["mm_1d"])),
                     ("screened, raw return", scr.assign(v=scr["raw_1d"])),
                     ("screened, 5-day hold", scr.assign(v=scr["xs_5d"])),
                     ("control, 5-day hold", ctl.assign(v=ctl["xs_5d"]))]:
        s = wtest(g["v"], g["week"])
        s["variant"] = label
        sec.append(s)
    for w in CROSS_WINDOWS:
        g = test[test[f"scr_{w}"]]
        s = wtest(g["xs_1d"], g["week"])
        s["variant"] = f"cross within {w} days"
        sec.append(s)
    for key in ["index", "Timing", "Sector"]:
        for v, g in scr.groupby(key):
            s = wtest(g["xs_1d"], g["week"])
            s["variant"] = f"{key} = {v}"
            sec.append(s)
    out["secondary"] = pd.DataFrame(sec)

    yr = []
    for y, g in test.groupby(test["earn_date"].dt.year):
        a, b = g[g["screened"]], g[~g["screened"]]
        s = wtest(a["xs_1d"], a["week"])
        yr.append({"year": y, "n_screened": s["n"], "mean_screened": s.get("mean"),
                   "median_screened": s.get("median"), "t_screened": s.get("t"),
                   "mean_control": b["xs_1d"].mean(), "diff": s.get("mean", np.nan) - b["xs_1d"].mean()})
    out["by_year"] = pd.DataFrame(yr)

    sn = []
    for name, g in [("screened", seen[seen["screened"]]), ("control", seen[~seen["screened"]])]:
        s = wtest(g["xs_1d"], g["week"])
        s["group"] = name
        sn.append(s)
    out["seen_window_reference"] = pd.DataFrame(sn)

    # Weekly long-screened / short-sector-ETF return series, for an equity curve.
    weekly = scr.groupby("week")["xs_1d"].mean()
    weekly.to_frame("weekly_mean_xs_1d").to_csv(OUT / "long_weekly_series.csv")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    for k, v in out.items():
        v.to_csv(OUT / f"long_{k}.csv", index=False)
        print(f"\n=== {k} ===")
        print(v.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
