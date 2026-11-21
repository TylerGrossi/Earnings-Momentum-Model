"""Signal evaluation: information coefficient, quantile portfolios, long-short performance.

The forward return for a signal formed at month end t is the stock's return over month t+1.
Only stocks in the index at t are used. A stock with no t+1 return (delisted, data gap) drops out
of that month, which is a known survivorship issue with free data; see data_coverage.md.
"""
import numpy as np
import pandas as pd


def attach_forward(signal: pd.Series, monthly: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    # On a full month-by-ticker grid, shift(-1) is always the next calendar month (NaN if missing).
    wide = monthly.pivot(index="month_end", columns="ticker", values="ret").sort_index().asfreq("ME")
    fwd = wide.shift(-1).stack(future_stack=True).rename("fwd_ret")
    fwd.index.names = ["month_end", "ticker"]
    df = pd.DataFrame({"signal": signal}).join(fwd, how="inner")
    members = pd.MultiIndex.from_frame(universe[["month_end", "ticker"]])
    df = df[df.index.isin(members)]
    return df.dropna()


def rank_ic(df: pd.DataFrame, min_names: int = 50) -> pd.Series:
    """Monthly Spearman correlation between signal and next-month return."""
    def one(g):
        if len(g) < min_names:
            return np.nan
        return g["signal"].rank().corr(g["fwd_ret"].rank())
    return df.groupby(level="month_end").apply(one).dropna().rename("rank_ic")


def quantile_returns(df: pd.DataFrame, n: int = 5, min_names: int = 50) -> pd.DataFrame:
    """Equal-weighted next-month return of each signal quantile (1 = lowest signal)."""
    def one(g):
        if len(g) < min_names:
            return None
        q = pd.qcut(g["signal"].rank(method="first"), n, labels=range(1, n + 1))
        return g.groupby(q, observed=True)["fwd_ret"].mean()
    out = df.groupby(level="month_end").apply(one)
    if isinstance(out, pd.Series):  # groupby-apply returned stacked
        out = out.unstack()
    # Label with the month the return is earned (t+1), so it lines up with benchmark returns.
    out.index = out.index + pd.offsets.MonthEnd(1)
    return out


def perf_stats(r: pd.Series, periods: int = 12) -> dict:
    r = r.dropna()
    mean, sd = r.mean(), r.std()
    wealth = (1 + r).cumprod()
    return {
        "months": len(r),
        "ann_return": (1 + r).prod() ** (periods / len(r)) - 1,
        "ann_vol": sd * np.sqrt(periods),
        "sharpe": mean / sd * np.sqrt(periods) if sd > 0 else np.nan,
        "t_stat": mean / sd * np.sqrt(len(r)) if sd > 0 else np.nan,
        "max_drawdown": (wealth / wealth.cummax() - 1).min(),
        "hit_rate": (r > 0).mean(),
    }


def ic_stats(ic: pd.Series) -> dict:
    return {
        "months": len(ic),
        "mean_ic": ic.mean(),
        "ic_sd": ic.std(),
        "ic_ir_ann": ic.mean() / ic.std() * np.sqrt(12),
        "t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "pct_positive": (ic > 0).mean(),
    }
