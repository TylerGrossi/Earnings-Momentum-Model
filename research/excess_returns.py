"""
Session 1 of the research upgrade: are the earnings-week returns real excess returns, and do
they survive trading costs?

Reads returns_tracker.csv (never writes to it), re-derives every trade's return from daily
adjusted closes with one explicit timing rule, and compares each trade with three benchmarks
over the same holding window:

  * SPY                      - did the stock beat the market?
  * the sector SPDR ETF      - did it beat its own sector?
  * a market model (beta)    - did it beat what its beta to SPY predicts?

Timing rule (matches the README strategy):
  * BMO: buy at the close of the last trading day before the earnings date; day 1 = earnings day.
  * AMC: buy at the close of the earnings day; day 1 = the next trading day.
  * Exit at the close of trading day h (h = 1, 3, 5, 10).

Note: the tracker's own "nD Return" columns start counting after the earnings timestamp, so for
BMO trades day 1 is the day AFTER earnings (an h+1 day hold). The reconciliation section below
measures that.

Statistics are reported two ways. Event-level t-stats treat every trade as independent, which
they are not: trades in the same week share market moves. The calendar-time version averages
all trades entered in the same week into one weekly return and tests across weeks, with a
week-clustered bootstrap for the confidence interval. Treat the weekly numbers as the honest ones.

Costs are an assumption, not a measurement (no quote data): round-trip cost by market-cap
bucket, plus a break-even cost (the round-trip cost at which the mean excess return hits zero).

Run:  python research/excess_returns.py            (uses cached prices if present)
      python research/excess_returns.py --refresh  (re-downloads prices)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "output"

HOLDS = [1, 3, 5, 10]
EST_WINDOW = (260, 21)  # market-model beta estimated from entry-260 to entry-21 trading days
MIN_EST_DAYS = 120
BOOT_N = 10_000
SEED = 7

SECTOR_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
}

# Assumed round-trip cost (bps) by market cap. Entry and exit are both at the close, where
# spreads are tightest, so these lean conservative for large caps and are rough for small ones.
COST_BUCKETS = [  # (min market cap, label, round-trip bps)
    (10e9, "large (>=10B)", 10),
    (2e9, "mid (2-10B)", 20),
    (300e6, "small (300M-2B)", 40),
    (0, "micro (<300M)", 100),
]


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------

def load_events() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "returns_tracker.csv")
    n_all = len(df)
    df = df[df["Date Check"].eq("OK") & df["Earnings Timing"].isin(["BMO", "AMC"])].copy()
    # The calendar date as the scraper records it (string prefix; avoids UTC/ET shifts).
    df["earn_date"] = pd.to_datetime(df["Earnings Date"].astype(str).str[:10], errors="coerce")
    df = df[df["earn_date"].notna()]
    df["yf_ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
    df["sector_etf"] = df["Sector"].map(SECTOR_ETF)
    df["date_added_known"] = df["Date Added"].notna()
    print(f"Events: {n_all} in tracker, {len(df)} with Date Check OK and BMO/AMC timing")
    return df.reset_index(drop=True)


def load_prices(tickers: list[str], start: str, refresh: bool) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "daily_close.parquet"
    px = pd.read_parquet(path) if path.exists() and not refresh else pd.DataFrame()

    import logging
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    def got(t):  # a column of all NaN is a failed download, not data
        return t in px.columns and px[t].notna().any()

    # Yahoo drops some symbols at random under load, so retry the failures a few times,
    # one at a time on the last pass.
    for attempt in range(3):
        missing = [t for t in tickers if not got(t)]
        if not missing:
            break
        print(f"Downloading {len(missing)} tickers (pass {attempt + 1})")
        raw = yf.download(missing, start=start, auto_adjust=True, progress=False,
                          group_by="column", threads=attempt < 2)
        if raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
            columns={"Close": missing[0]})
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close.loc[:, close.notna().any()]
        px = close if px.empty else px.drop(columns=[c for c in close if c in px]).join(close, how="outer")
    px = px.sort_index()
    px.to_parquet(path)
    return px


# ----------------------------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------------------------

def entry_index(cal: pd.DatetimeIndex, earn_date: pd.Timestamp, timing: str) -> int | None:
    """Position in the trading calendar of the entry close."""
    if timing == "BMO":
        pos = cal.searchsorted(earn_date, side="left") - 1  # last day strictly before
    else:
        pos = cal.searchsorted(earn_date, side="right") - 1  # earnings day, or last day before
    return int(pos) if pos >= 0 else None


def window_ret(series: np.ndarray, i0: int, i1: int) -> float:
    if i1 >= len(series):
        return np.nan
    a, b = series[i0], series[i1]
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return np.nan
    return b / a - 1


def market_model_car(stock: np.ndarray, mkt: np.ndarray, i0: int, h: int) -> tuple[float, float]:
    """Beta from the estimation window; CAR = sum of daily (r_i - beta * r_m) over days 1..h."""
    lo, hi = i0 - EST_WINDOW[0], i0 - EST_WINDOW[1]
    if lo < 1 or i0 + h >= len(stock):
        return np.nan, np.nan
    rs = stock[lo:hi + 1] / stock[lo - 1:hi] - 1
    rm = mkt[lo:hi + 1] / mkt[lo - 1:hi] - 1
    ok = np.isfinite(rs) & np.isfinite(rm)
    if ok.sum() < MIN_EST_DAYS:
        return np.nan, np.nan
    beta = np.cov(rs[ok], rm[ok])[0, 1] / np.var(rm[ok], ddof=1)
    ev_s = stock[i0 + 1:i0 + h + 1] / stock[i0:i0 + h] - 1
    ev_m = mkt[i0 + 1:i0 + h + 1] / mkt[i0:i0 + h] - 1
    ar = ev_s - beta * ev_m
    return (float(np.sum(ar)) if np.all(np.isfinite(ar)) else np.nan), float(beta)


def build_trades(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    cal = px["SPY"].dropna().index
    px = px.reindex(cal)
    spy = px["SPY"].to_numpy(float)
    rows = []
    for r in ev.itertuples(index=False):
        if r.yf_ticker not in px.columns:
            continue
        i0 = entry_index(cal, r.earn_date, r.timing)
        if i0 is None:
            continue
        s = px[r.yf_ticker].to_numpy(float)
        sec = px[r.sector_etf].to_numpy(float) if isinstance(r.sector_etf, str) and r.sector_etf in px else None
        row = {
            "Ticker": r.Ticker, "Earnings Date": r.earn_date.date(), "Timing": r.timing,
            "Sector": r.Sector, "Market Cap": r.mcap, "Entry Date": cal[i0].date(),
            "Date Added Known": r.date_added_known,
        }
        for h in HOLDS:
            raw = window_ret(s, i0, i0 + h)
            row[f"ret_{h}d"] = raw
            row[f"xs_spy_{h}d"] = raw - window_ret(spy, i0, i0 + h)
            row[f"xs_sector_{h}d"] = (raw - window_ret(sec, i0, i0 + h)) if sec is not None else np.nan
            car, beta = market_model_car(s, spy, i0, h)
            row[f"car_mm_{h}d"] = car
            if h == HOLDS[0]:
                row["beta_est"] = beta
        # For the reconciliation with the tracker's BMO convention (h+1 day hold).
        row["ret_6d"] = window_ret(s, i0, i0 + 6)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------

def cost_bps(mcap: float) -> float:
    if not np.isfinite(mcap):
        return COST_BUCKETS[-1][2]
    for floor, _, bps in COST_BUCKETS:
        if mcap >= floor:
            return bps
    return COST_BUCKETS[-1][2]


def summarize(tr: pd.DataFrame, col: str, rng: np.random.Generator) -> dict:
    d = tr[[col, "Entry Date", "cost"]].dropna()
    x = d[col].to_numpy()
    n = len(x)
    out = {"measure": col, "n_trades": n}
    if n < 10:
        return out
    out["mean"] = x.mean()
    out["median"] = np.median(x)
    out["hit_rate"] = (x > 0).mean()
    out["t_event"] = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    net = x - d["cost"].to_numpy()
    out["mean_net_of_cost"] = net.mean()
    # Calendar-time: one equal-weight return per entry week, then test across weeks.
    wk = pd.to_datetime(d["Entry Date"]).dt.to_period("W-FRI")
    weekly = pd.Series(x).groupby(wk.to_numpy()).mean()
    weekly_net = pd.Series(net).groupby(wk.to_numpy()).mean()
    w = weekly.to_numpy()
    out["n_weeks"] = len(w)
    out["weekly_mean"] = w.mean()
    out["t_weekly"] = w.mean() / (w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 2 else np.nan
    out["weekly_mean_net"] = weekly_net.mean()
    out["weeks_positive"] = (w > 0).mean()
    # Week-clustered bootstrap CI for the trade-level mean.
    groups = [g.to_numpy() for _, g in pd.Series(x).groupby(wk.to_numpy())]
    k = len(groups)
    boots = np.empty(BOOT_N)
    for b in range(BOOT_N):
        pick = rng.integers(0, k, k)
        sample = np.concatenate([groups[i] for i in pick])
        boots[b] = sample.mean()
    out["ci95_lo"], out["ci95_hi"] = np.percentile(boots, [2.5, 97.5])
    out["p_mean_le_0"] = (boots <= 0).mean()
    # Break-even round-trip cost (bps) that takes the mean to zero.
    out["breakeven_bps"] = x.mean() * 1e4
    # Outlier sensitivity: the mean is skewed by a few big winners.
    lo, hi = np.percentile(x, [1, 99])
    out["mean_winsor_1_99"] = np.clip(x, lo, hi).mean()
    out["mean_ex_top5"] = np.sort(x)[:-5].mean()
    return out


def reconcile(tr: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    m = tr.merge(ev.assign(**{"Earnings Date": ev["earn_date"].dt.date})[
        ["Ticker", "Earnings Date", "5D Return"]], on=["Ticker", "Earnings Date"], how="inner")
    rows = []
    for timing, g in m.groupby("Timing"):
        for col, label in [("ret_5d", "recomputed 5-day"), ("ret_6d", "recomputed 6-day")]:
            gg = g[[col, "5D Return"]].dropna()
            diff = (gg[col] - gg["5D Return"]).abs()
            rows.append({"Timing": timing, "compare tracker 5D Return to": label, "n": len(gg),
                         "corr": gg[col].corr(gg["5D Return"]),
                         "share within 0.5pt": (diff <= 0.005).mean(),
                         "median abs diff": diff.median()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ev = load_events().rename(columns={"Earnings Timing": "timing", "Market Cap": "mcap"})
    tickers = sorted(set(ev["yf_ticker"]) | {"SPY"} | set(SECTOR_ETF.values()))
    start = (ev["earn_date"].min() - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    px = load_prices(tickers, start, args.refresh)

    have = px.columns[px.notna().any()]
    lost = sorted(set(ev["yf_ticker"]) - set(have))
    if lost:
        print(f"No price history for {len(lost)} tickers (delisted/renamed?): {', '.join(lost[:20])}"
              + (" ..." if len(lost) > 20 else ""))

    tr = build_trades(ev, px[have])
    tr["cost"] = tr["Market Cap"].map(cost_bps) / 1e4
    tr["cost_bucket"] = tr["Market Cap"].map(
        lambda m: next((lab for f, lab, _ in COST_BUCKETS if np.isfinite(m) and m >= f), COST_BUCKETS[-1][1]))
    print(f"Trades with prices: {len(tr)}  (window {tr['Entry Date'].min()} to {tr['Entry Date'].max()})")

    rng = np.random.default_rng(SEED)
    summ = []
    for h in HOLDS:
        for kind in ["ret", "xs_spy", "xs_sector", "car_mm"]:
            s = summarize(tr, f"{kind}_{h}d", rng)
            s["hold_days"] = h
            s["benchmark"] = {"ret": "none (raw)", "xs_spy": "SPY", "xs_sector": "sector ETF",
                              "car_mm": "market model (beta)"}[kind]
            summ.append(s)
    summ = pd.DataFrame(summ)

    # Splits for the 5-day sector-excess return: timing, size bucket, signal-time known.
    splits = []
    for by in ["Timing", "cost_bucket", "Date Added Known"]:
        for key, g in tr.groupby(by):
            s = summarize(g, "xs_sector_5d", rng)
            s.update({"split": by, "group": key})
            splits.append(s)
    splits = pd.DataFrame(splits)

    rec = reconcile(tr, ev)

    OUT.mkdir(exist_ok=True)
    tr.to_csv(OUT / "excess_returns_trades.csv", index=False)
    summ.to_csv(OUT / "excess_returns_summary.csv", index=False)
    splits.to_csv(OUT / "excess_returns_splits.csv", index=False)
    rec.to_csv(OUT / "tracker_reconciliation.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    fmt = {c: "{:.4f}".format for c in ["mean", "median", "mean_net_of_cost", "weekly_mean",
                                         "weekly_mean_net", "ci95_lo", "ci95_hi"]}
    cols = ["hold_days", "benchmark", "n_trades", "mean", "median", "hit_rate", "t_event",
            "n_weeks", "weekly_mean", "t_weekly", "ci95_lo", "ci95_hi", "p_mean_le_0",
            "mean_net_of_cost", "breakeven_bps", "mean_winsor_1_99", "mean_ex_top5"]
    fmt.update({c: "{:.4f}".format for c in ["mean_winsor_1_99", "mean_ex_top5"]})
    print("\n=== Summary ===")
    print(summ[cols].to_string(index=False, formatters=fmt, float_format="{:.3f}".format))
    print("\n=== 5-day sector-excess return by split ===")
    print(splits[["split", "group", "n_trades", "mean", "hit_rate", "t_weekly", "ci95_lo", "ci95_hi",
                  "mean_net_of_cost"]].to_string(index=False, formatters=fmt, float_format="{:.3f}".format))
    print("\n=== Tracker reconciliation (5D Return) ===")
    print(rec.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    sys.exit(main())
