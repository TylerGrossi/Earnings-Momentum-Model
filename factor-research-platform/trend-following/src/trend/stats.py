"""Performance statistics on monthly returns."""
import numpy as np
import pandas as pd


def newey_west_t(x: pd.Series, lags: int = 6) -> float:
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(var / n)


def summary(total: pd.Series, excess: pd.Series) -> dict:
    total, excess = total.dropna(), excess.reindex(total.index)
    wealth = (1 + total).cumprod()
    yearly = (1 + total).groupby(total.index.year).prod() - 1
    full_years = total.groupby(total.index.year).size()
    yearly = yearly[full_years == 12]
    return {
        "start": f"{total.index.min():%Y-%m}", "end": f"{total.index.max():%Y-%m}",
        "months": len(total),
        "cagr": wealth.iloc[-1] ** (12 / len(total)) - 1,
        "vol": excess.std() * np.sqrt(12),
        "sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "nw_t": newey_west_t(excess),
        "max_dd": (wealth / wealth.cummax() - 1).min(),
        "worst_year": yearly.min() if len(yearly) else np.nan,
        "worst_year_label": int(yearly.idxmin()) if len(yearly) else None,
        "best_year": yearly.max() if len(yearly) else np.nan,
    }


def calendar_years(series: dict) -> pd.DataFrame:
    out = {}
    for k, r in series.items():
        n = r.groupby(r.index.year).size()
        y = (1 + r).groupby(r.index.year).prod() - 1
        out[k] = y.where(n == 12)
    return pd.DataFrame(out)


def window_return(r: pd.Series, start: str, end: str) -> float:
    w = r[(r.index >= start) & (r.index <= end)]
    return float((1 + w).prod() - 1) if len(w) else np.nan
