"""
Report 06: can a different, historically trackable buy signal replace the Barchart opinion?

PRE-REGISTERED (written 2026-11-20 before running). Four signals, each computed only from data
available before the entry day (analyst actions dated before the entry day; prices up to the
close before entry):

  S1 prior beat        the previous earnings report beat the EPS estimate (Yahoo surprise > 0)
  S2 analyst momentum  more upgrades than downgrades (yfinance upgrades_downgrades) in the 90
                       calendar days before the entry day
  S3 technical opinion a Barchart-style composite of 10 trend indicators, each +1/-1:
                       price vs SMA20/50/100, SMA20>50, SMA20>100, SMA50>100, SMA50>150,
                       SMA50>200, CCI(40)>0, CCI(60)>0; "Buy" = average > 0.6 (8+ of 10 bullish)
  S4 pre-event strength stock beat its sector ETF over the 20 trading days before entry

Trade: buy at the close before the announcement, sell at the next close (report 04 rule).
Outcome: 1-day return in excess of the sector ETF. Universe and events: report 05 (S&P 1500).

Split: development 2012-01-01..2018-12-31, holdout 2019-01-01..2025-10-31.
A signal PASSES only if, in BOTH periods, signal-on events beat signal-off events (weekly
difference > 0), and in the holdout that difference is significant after a Holm correction for
4 tests (one-sided, family 5%). Secondary, reported only: each signal combined with the SMA
cross screen; signal-on mean vs zero; after assumed costs; the Nov 2025-Nov 2026 window.

Run after long_history.py.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CACHE, OUT = HERE / "cache", HERE / "output"
DEV = ("2012-01-01", "2018-12-31")
HOLD = ("2019-01-01", "2025-10-31")
SEEN = ("2025-11-01", "2026-11-18")
SIGNALS = {"S1 prior beat": "s1", "S2 analyst momentum": "s2",
           "S3 technical opinion": "s3", "S4 pre-event strength": "s4"}
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def upgrades(tickers: list[str]) -> pd.DataFrame:
    path = CACHE / "upgrades_downgrades.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf

    def one(t):
        for attempt in range(3):
            try:
                u = yf.Ticker(t).upgrades_downgrades
                if u is None or u.empty:
                    return None
                u = u.reset_index()[["GradeDate", "Action"]]
                u["yf"] = t
                return u
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        parts = [p for p in ex.map(one, tickers) if p is not None]
    u = pd.concat(parts, ignore_index=True)
    u["GradeDate"] = pd.to_datetime(u["GradeDate"]).dt.tz_localize(None)
    u.to_parquet(path)
    print(f"Analyst actions for {u['yf'].nunique()} of {len(tickers)} tickers")
    return u


def technicals(px: pd.DataFrame) -> tuple[dict, pd.DatetimeIndex]:
    cal = px["SPY"].dropna().index
    px = px.reindex(cal)
    sma = {n: px.rolling(n).mean() for n in (20, 50, 100, 150, 200)}
    votes = [np.sign(px - sma[20]), np.sign(px - sma[50]), np.sign(px - sma[100]),
             np.sign(sma[20] - sma[50]), np.sign(sma[20] - sma[100]), np.sign(sma[50] - sma[100]),
             np.sign(sma[50] - sma[150]), np.sign(sma[50] - sma[200])]
    for n in (40, 60):  # CCI on closes (no high/low in the cache)
        m = px.rolling(n).mean()
        md = px.rolling(n).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
        votes.append(np.sign((px - m) / (0.015 * md)))
    comp = sum(votes) / len(votes)
    comp = comp.where(sma[200].notna())
    rel20 = px.pct_change(20)
    return {"comp": comp, "ret20": rel20, "px": px}, cal


def main() -> None:
    ev = pd.read_parquet(OUT / "long_events.parquet")
    ev = ev[ev["earn_date"] >= "2012-01-01"].copy()
    px = pd.read_parquet(CACHE / "daily_close_long.parquet")
    dates = pd.read_parquet(CACHE / "earnings_dates_long.parquet")
    up = upgrades(sorted(ev["yf"].unique()))

    tech, cal = technicals(px)
    pos = {d: i for i, d in enumerate(cal)}
    from long_history import GICS_ETF

    # S1: previous report's surprise, known before this event.
    d = dates.dropna(subset=["surprise"]).copy()
    d["earn_date"] = pd.to_datetime(d["ts"].dt.date)
    d = d.sort_values("earn_date")  # merge_asof needs the "on" key sorted globally
    ev = ev.sort_values(["Entry Date"])
    prev = pd.merge_asof(ev[["yf", "Entry Date"]].reset_index(),
                         d[["yf", "earn_date", "surprise"]].rename(columns={"earn_date": "prev_date",
                                                                            "surprise": "prev_surprise"}),
                         left_on="Entry Date", right_on="prev_date", by="yf",
                         allow_exact_matches=False, direction="backward").set_index("index")
    ev["prev_surprise"] = prev["prev_surprise"]
    # The previous report must be a different (earlier) quarter: at least 30 days before entry.
    ev.loc[(ev["Entry Date"] - prev["prev_date"]).dt.days < 30, "prev_surprise"] = np.nan
    ev["s1"] = np.where(ev["prev_surprise"].notna(), ev["prev_surprise"] > 0, np.nan)

    # S2: net upgrades in the 90 days before the entry day.
    up["net"] = up["Action"].map({"up": 1, "down": -1}).fillna(0)
    up["day"] = up["GradeDate"].dt.normalize()
    g = up.groupby(["yf", "day"])["net"].sum().reset_index()
    covered = set(up["yf"])
    s2 = []
    byt = {t: x.set_index("day")["net"] for t, x in g.groupby("yf")}
    first = up.groupby("yf")["day"].min()
    for r in ev[["yf", "Entry Date"]].itertuples():
        if r.yf not in covered or r._2 - pd.Timedelta(days=90) < first[r.yf]:
            s2.append(np.nan)
            continue
        s = byt[r.yf]
        w = s[(s.index >= r._2 - pd.Timedelta(days=90)) & (s.index < r._2)]
        s2.append(float(w.sum() > 0))
    ev["s2"] = s2

    # S3, S4: technicals at the close before entry.
    comp, ret20, P = tech["comp"], tech["ret20"], tech["px"]
    s3, s4 = [], []
    for r in ev[["yf", "Entry Date", "Sector"]].itertuples():
        i = pos.get(r._2)
        if i is None or i < 1 or r.yf not in comp:
            s3.append(np.nan); s4.append(np.nan); continue
        c = comp[r.yf].iloc[i - 1]
        s3.append(float(c > 0.6) if np.isfinite(c) else np.nan)
        etf = GICS_ETF.get(r.Sector)
        b = ret20[etf].iloc[i - 1] if etf in ret20 and np.isfinite(ret20[etf].iloc[i - 1]) else ret20["SPY"].iloc[i - 1]
        a = ret20[r.yf].iloc[i - 1]
        s4.append(float(a - b > 0) if np.isfinite(a) and np.isfinite(b) else np.nan)
    ev["s3"], ev["s4"] = s3, s4
    ev.to_parquet(OUT / "alt_signal_events.parquet")

    def weekly_diff(on, off):
        dd = (on.groupby("week")["xs_1d"].mean() - off.groupby("week")["xs_1d"].mean()).dropna()
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        return dd.mean(), t, 1 - stats.t.cdf(t, len(dd) - 1), len(dd)

    def wmean(x):
        w = x.groupby("week")["xs_1d"].mean()
        t = w.mean() / (w.std(ddof=1) / np.sqrt(len(w)))
        return x["xs_1d"].mean(), x["xs_1d"].median(), t, len(x)

    rows = []
    for name, col in SIGNALS.items():
        for per, (lo, hi) in [("development 2012-2018", DEV), ("holdout 2019-2025", HOLD),
                              ("seen Nov 2025-Nov 2026", SEEN)]:
            p = ev[(ev["earn_date"] >= lo) & (ev["earn_date"] <= hi) & ev[col].notna()]
            on, off = p[p[col] == 1], p[p[col] == 0]
            m_on, med_on, t_on, n_on = wmean(on)
            m_off, _, t_off, n_off = wmean(off)
            dm, dt, dp, dw = weekly_diff(on, off)
            onc = on.assign(xs_1d=on["xs_1d"] - on["cost"])
            m_net, _, t_net, _ = wmean(onc)
            scr = p[p["screened"]]
            s_on, s_off = scr[scr[col] == 1], scr[scr[col] == 0]
            m_son = s_on["xs_1d"].mean() if len(s_on) else np.nan
            rows.append({"signal": name, "period": per, "coverage": len(p), "n_on": n_on,
                         "share_on": n_on / len(p), "mean_on": m_on, "median_on": med_on,
                         "t_on_vs_0": t_on, "mean_off": m_off, "weekly_diff": dm, "t_diff": dt,
                         "p_diff": dp, "weeks": dw, "mean_on_net": m_net, "t_on_net": t_net,
                         "with_sma_cross_n": len(s_on), "with_sma_cross_mean": m_son,
                         "sma_cross_without_signal_mean": s_off["xs_1d"].mean() if len(s_off) else np.nan})
    res = pd.DataFrame(rows)

    # Holm correction on the 4 holdout difference tests.
    h = res[res["period"] == "holdout 2019-2025"].sort_values("p_diff").copy()
    m = len(h)
    adj, running = [], 0.0
    for k, p in enumerate(h["p_diff"]):
        running = max(running, min(1.0, (m - k) * p))
        adj.append(running)
    h["p_holm"] = adj
    res = res.merge(h[["signal", "period", "p_holm"]], on=["signal", "period"], how="left")
    dev_pos = res[res["period"].str.startswith("development")].set_index("signal")["weekly_diff"] > 0
    hold = res[res["period"].str.startswith("holdout")].set_index("signal")
    verdict = pd.DataFrame({"dev_diff_positive": dev_pos, "holdout_diff_positive": hold["weekly_diff"] > 0,
                            "holdout_p_holm": hold["p_holm"]})
    verdict["PASS"] = verdict["dev_diff_positive"] & verdict["holdout_diff_positive"] & (verdict["holdout_p_holm"] < 0.05)

    res.to_csv(OUT / "alt_signals_results.csv", index=False)
    verdict.to_csv(OUT / "alt_signals_verdict.csv")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(res.drop(columns=["coverage"]).to_string(index=False, float_format="{:.4f}".format))
    print("\n=== Verdict (pre-registered pass rule) ===")
    print(verdict.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
