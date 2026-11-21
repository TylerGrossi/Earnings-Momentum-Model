"""Daily prices from yfinance, cached to parquet, and monthly returns built from them.

Known limits of this free source (measured in reports/data_coverage.md, not assumed away):
- Yahoo drops many delisted stocks, so some former members have no prices. Their months are
  missing, which biases results toward survivors.
- A stock's last month before delisting (often a large loss) is usually missing too.
- Reused tickers can map to the wrong company. We only use prices inside each ticker's
  membership window, which limits but doesn't remove this.
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

from . import paths
from .universe import to_yahoo


# Tickers whose Yahoo history is not the company that was in the index (ticker reuse or a
# merger rewriting the series). Found from the extreme-return list in data_coverage.md.
BAD_TICKERS = {
    "MCIC": "1998 MCI series: -99% then +15,898% in consecutive months",
    "EP": "El Paso: Yahoo series is a different security (+400% in a month in 2003)",
    "CPWR": "Compuware: repeated +100% to +300% months 2010-11, not the real stock",
    "PARA": "Paramount: -82% in Dec 2023 and +328% in Jun 2025, broken by the Skydance merger",
}


def download_daily(tickers: list[str], start: str = "1995-01-01", chunk: int = 80) -> pd.DataFrame:
    """Adjusted daily close and volume, long format. Cached per chunk so a rerun resumes."""
    cache_dir = paths.RAW / "prices"
    cache_dir.mkdir(exist_ok=True)
    tickers = sorted(set(tickers))
    frames, missing = [], []
    for k in range(0, len(tickers), chunk):
        part = tickers[k : k + chunk]
        f = cache_dir / f"chunk_{k:05d}.parquet"
        if f.exists():
            got = pd.read_parquet(f)
        else:
            ymap = {to_yahoo(t): t for t in part}
            raw = None
            for attempt in range(3):
                try:
                    # auto_adjust=False keeps both Close (split-adjusted only) and Adj Close
                    # (split + dividend); actions=True adds the split history, which market cap
                    # needs to line up filing share counts with prices.
                    raw = yf.download(list(ymap), start=start, auto_adjust=False, actions=True,
                                      group_by="ticker", threads=True, progress=False)
                    break
                except Exception as e:  # network hiccup or rate limit
                    print(f"  chunk {k} attempt {attempt + 1} failed: {e}")
                    time.sleep(10 * (attempt + 1))
            got = _to_long(raw, ymap) if raw is not None else pd.DataFrame()
            got.to_parquet(f, index=False)
            print(f"  chunk {k // chunk + 1}/{-(-len(tickers) // chunk)}: "
                  f"{got['ticker'].nunique() if len(got) else 0}/{len(part)} tickers with data")
        frames.append(got)
        have = set(got["ticker"]) if len(got) else set()
        missing += [t for t in part if t not in have]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    pd.Series(missing, name="ticker").to_csv(paths.PRICE_MISSING, index=False)
    return out


def _to_long(raw: pd.DataFrame, ymap: dict[str, str]) -> pd.DataFrame:
    cols = ["date", "ticker", "adj_close", "close", "volume", "split"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for ysym, t in ymap.items():
        if ysym not in raw.columns.get_level_values(0):
            continue
        sub = raw[ysym]
        sub = sub[[c for c in ["Adj Close", "Close", "Volume", "Stock Splits"] if c in sub]]
        sub = sub.dropna(subset=["Adj Close"])
        if sub.empty:
            continue
        sub = sub.rename(columns={"Adj Close": "adj_close", "Close": "close", "Volume": "volume",
                                  "Stock Splits": "split"})
        if "split" not in sub:
            sub["split"] = 0.0
        sub["ticker"] = t
        out.append(sub.reset_index().rename(columns={"Date": "date"}))
    if not out:
        return pd.DataFrame(columns=cols)
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["split"] = df["split"].fillna(0.0)
    return df[cols]


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted close, monthly return and average daily dollar volume.

    A return is only computed when the previous calendar month also has a price, so a gap in
    the data never turns into a multi-month return labelled as one month.
    """
    d = daily.dropna(subset=["adj_close"])
    d = d[~d["ticker"].isin(BAD_TICKERS)].copy()
    d["month_end"] = d["date"] + pd.offsets.MonthEnd(0)
    d["dollar_vol"] = d["close"] * d["volume"]
    g = d.sort_values("date").groupby(["ticker", "month_end"])
    m = g.agg(adj_close=("adj_close", "last"), close=("close", "last"),
              dollar_vol=("dollar_vol", "mean"), n_days=("adj_close", "size")).reset_index()
    m = m.sort_values(["ticker", "month_end"])
    prev_px = m.groupby("ticker")["adj_close"].shift(1)
    prev_me = m.groupby("ticker")["month_end"].shift(1)
    contiguous = prev_me == (m["month_end"] - pd.offsets.MonthEnd(1))
    m["ret"] = np.where(contiguous, m["adj_close"] / prev_px - 1, np.nan)
    return m
