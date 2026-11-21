"""Price-based signals, all formed at month end t from data up to and including t.

Each function takes the monthly panel (month_end, ticker, adj_close, ret) and returns a Series
aligned to it. Nothing here looks past month t; the forward return is added separately in
evaluate.py.
"""
import numpy as np
import pandas as pd


def _wide(monthly: pd.DataFrame, col: str) -> pd.DataFrame:
    return monthly.pivot(index="month_end", columns="ticker", values=col).sort_index().asfreq("ME")


def _long(wide: pd.DataFrame, name: str) -> pd.Series:
    s = wide.stack(future_stack=True).rename(name)
    s.index.names = ["month_end", "ticker"]
    return s.dropna()


def momentum_12_1(monthly: pd.DataFrame) -> pd.Series:
    """Return from t-12 to t-1, skipping the latest month (Jegadeesh-Titman / Carhart style).

    Uses compounded monthly returns and requires all 11 of them, so a data gap can't fake a
    momentum value.
    """
    r = _wide(monthly, "ret")
    lagged = (1 + r.shift(1)).rolling(11, min_periods=11).apply(lambda x: x.prod(), raw=True) - 1
    return _long(lagged, "mom_12_1")


def reversal_1m(monthly: pd.DataFrame) -> pd.Series:
    """Last month's return. Expected to predict negatively (short-term reversal)."""
    return _long(_wide(monthly, "ret"), "rev_1m")


def split_adjusted_shares(shares: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Restate each reported share count on today's split basis, so it can be multiplied by
    Yahoo's split-adjusted close. A 2009 Apple count of 895.8M becomes about 25.1B after the 7:1
    (2014) and 4:1 (2020) splits."""
    sp = daily.loc[daily["split"] > 0, ["ticker", "date", "split"]].sort_values("date")
    out = []
    for t, g in shares.groupby("ticker"):
        s = sp[sp["ticker"] == t]
        g = g.copy()
        if s.empty:
            g["shares_adj"] = g["shares"]
        else:
            cum = s["split"].cumprod().to_numpy()
            dates = s["date"].to_numpy()
            # Product of all splits strictly after the share-count date.
            k = np.searchsorted(dates, g["share_date"].to_numpy(), side="right")
            before = np.where(k > 0, cum[np.maximum(k - 1, 0)], 1.0)
            g["shares_adj"] = g["shares"] * cum[-1] / before
        out.append(g)
    return pd.concat(out, ignore_index=True)


def drop_scale_errors(df: pd.DataFrame, col: str, by: str, order: str, factor: float) -> pd.Series:
    """NaN out values more than `factor` times away from the company's own centered rolling
    median (5 observations). Catches XBRL unit mistakes, e.g. a share count reported in
    thousands in one filing, without touching normal growth."""
    s = df.sort_values(order)
    med = s.groupby(by)[col].transform(lambda x: x.abs().rolling(5, center=True, min_periods=3).median())
    ratio = s[col].abs() / med
    bad = (ratio > factor) | (ratio < 1 / factor)
    return s[col].where(~bad).reindex(df.index)


def fundamental_panel(monthly: pd.DataFrame, daily: pd.DataFrame, annual: pd.DataFrame,
                      shares: pd.DataFrame) -> pd.DataFrame:
    """Month-end market cap and fundamental ratios, each using only filings made by month end.

    Annual data counts as current for 18 months after fiscal year end; share counts for 9 months
    after filing. Older than that is treated as missing rather than carried forward.
    """
    base = monthly[["month_end", "ticker", "close"]].dropna().sort_values("month_end")
    sh = split_adjusted_shares(shares, daily)
    sh["shares_adj"] = drop_scale_errors(sh, "shares_adj", "ticker", "share_date", 4)
    sh = sh.dropna(subset=["shares_adj"]).sort_values("filed")
    annual = annual.copy()
    for c in ("assets", "assets_prev", "revenue", "gross_profit"):
        annual[c] = drop_scale_errors(annual, c, "ticker", "fy_end", 20)
    p = pd.merge_asof(base, sh[["ticker", "filed", "shares_adj"]], left_on="month_end",
                      right_on="filed", by="ticker", direction="backward")
    p.loc[(p["month_end"] - p["filed"]).dt.days > 270, "shares_adj"] = np.nan
    p["mcap"] = p["close"] * p["shares_adj"]
    # An S&P 500 member below $0.5B or above $10T is a data error, not a real company size.
    p.loc[~p["mcap"].between(5e8, 1e13), "mcap"] = np.nan
    p = p.drop(columns=["filed"])

    a = annual.sort_values("filed")
    p = pd.merge_asof(p, a.drop(columns=["cik"]), left_on="month_end", right_on="filed",
                      by="ticker", direction="backward")
    stale = (p["month_end"] - p["fy_end"]).dt.days > 548
    fcols = [c for c in a.columns if c not in ("ticker", "cik", "filed", "fy_end")]
    p.loc[stale, fcols] = np.nan

    pos_eq = p["equity"].where(p["equity"] > 0)
    p["size"] = np.log(p["mcap"])
    p["book_to_market"] = pos_eq / p["mcap"]
    p["earnings_yield"] = p["net_income"] / p["mcap"]
    p["gross_prof"] = p["gross_profit"] / p["assets"]
    p["op_prof"] = p["op_income"] / pos_eq
    p["roe"] = p["net_income"] / pos_eq
    p["asset_growth"] = p["assets"] / p["assets_prev"] - 1
    keep = ["mcap", "size", "book_to_market", "earnings_yield", "gross_prof", "op_prof", "roe",
            "asset_growth"]
    return p.set_index(["month_end", "ticker"])[keep].replace([np.inf, -np.inf], np.nan)


def sue(earnings_q: pd.DataFrame, month_ends: pd.DatetimeIndex, fresh_days: int = 92) -> pd.Series:
    """Standardized unexpected earnings (seasonal random walk), as pre-registered in Test A.

    earnings_q: ticker, q_end, ni, filed (first filing date of that quarter's value).
    SUE_q = (NI_q - NI_{q-4}) / std of the previous 8 such changes (at least 6 present).
    At month end t, a stock's value is the SUE of its latest quarter filed in the last
    `fresh_days` days; older surprises are treated as missing.
    """
    rows = []
    for t, g in earnings_q.sort_values("q_end").groupby("ticker"):
        g = g.set_index("q_end")
        # Seasonal lag by date (about one year earlier), not by row, so a missing quarter can't
        # pair the wrong quarters.
        lag = g["ni"].reindex(g.index - pd.DateOffset(months=12), method="nearest",
                              tolerance=pd.Timedelta(days=20)).to_numpy()
        chg = pd.Series(g["ni"].to_numpy() - lag, index=g.index)
        scale = chg.shift(1).rolling(8, min_periods=6).std()
        s = (chg / scale).replace([np.inf, -np.inf], np.nan)
        rows.append(pd.DataFrame({"ticker": t, "filed": g["filed"].to_numpy(), "sue": s.to_numpy()}))
    ev = pd.concat(rows).dropna().sort_values("filed")
    grid = pd.DataFrame([(m, t) for m in month_ends for t in ev["ticker"].unique()],
                        columns=["month_end", "ticker"]).sort_values("month_end")
    out = pd.merge_asof(grid, ev, left_on="month_end", right_on="filed", by="ticker",
                        direction="backward")
    out = out[(out["month_end"] - out["filed"]).dt.days <= fresh_days]
    return out.set_index(["month_end", "ticker"])["sue"].dropna()


def volatility_12m(monthly: pd.DataFrame) -> pd.Series:
    """Std of the last 12 monthly returns (at least 10 present). Low-vol anomaly: predicts negatively."""
    r = _wide(monthly, "ret")
    return _long(r.rolling(12, min_periods=10).std(), "vol_12m")
