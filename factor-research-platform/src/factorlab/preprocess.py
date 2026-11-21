"""Assemble all signals into one panel and clean them cross-sectionally, month by month.

Per month: restrict to index members, flip signs so that higher always means "expected to
outperform", winsorize at the 1st/99th percentile, z-score, subtract the sector mean and
re-standardize. All operations use only that month's cross-section, so nothing leaks over time.
"""
import numpy as np
import pandas as pd

from . import paths, signals

FUND_SIGNALS = paths.DATA / "fundamental_signals.parquet"

# signal -> sign (so that high = expected to outperform, per the published anomaly)
SIGNS = {
    "mom_12_1": +1,
    "rev_1m": -1,
    "vol_12m": -1,
    "book_to_market": +1,
    "earnings_yield": +1,
    "op_prof": +1,
    "gross_prof": +1,
    "asset_growth": -1,
    "size": -1,
}


def raw_panel(monthly: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    fund = pd.read_parquet(FUND_SIGNALS)
    price = pd.concat([signals.momentum_12_1(monthly), signals.reversal_1m(monthly),
                       signals.volatility_12m(monthly)], axis=1)
    panel = price.join(fund[[c for c in SIGNS if c in fund]], how="outer")
    members = pd.MultiIndex.from_frame(universe[["month_end", "ticker"]])
    return panel[panel.index.isin(members)][list(SIGNS)]


def _clean_month(g: pd.DataFrame, sector: pd.Series, neutralize: bool) -> pd.DataFrame:
    out = {}
    sec = sector.reindex(g.index.get_level_values("ticker")).fillna("Unknown").to_numpy()
    for c in g.columns:
        x = g[c] * SIGNS[c]
        if x.notna().sum() < 50:
            out[c] = x * np.nan
            continue
        lo, hi = x.quantile([0.01, 0.99])
        x = x.clip(lo, hi)
        z = (x - x.mean()) / x.std()
        if neutralize:
            z = z - z.groupby(sec).transform("mean")
            z = z / z.std()
        out[c] = z
    return pd.DataFrame(out, index=g.index)


def clean_panel(raw: pd.DataFrame, sectors: pd.DataFrame, neutralize: bool = True) -> pd.DataFrame:
    sector = sectors.set_index("ticker")["sector"]
    parts = [_clean_month(g, sector, neutralize) for _, g in raw.groupby(level="month_end")]
    return pd.concat(parts).sort_index()


def forward_returns(monthly: pd.DataFrame, horizons=range(1, 13)) -> pd.DataFrame:
    """fwd_h = the stock's return in month t+h (single month, not cumulative), for IC decay.
    fwd_1 is the return the portfolio actually earns."""
    wide = monthly.pivot(index="month_end", columns="ticker", values="ret").sort_index().asfreq("ME")
    cols = {}
    for h in horizons:
        s = wide.shift(-h).stack(future_stack=True)
        s.index.names = ["month_end", "ticker"]
        cols[f"fwd_{h}"] = s
    return pd.DataFrame(cols)
