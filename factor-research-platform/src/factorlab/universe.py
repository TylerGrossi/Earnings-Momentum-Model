"""Point-in-time S&P 500 universe.

Source: github.com/fja05680/sp500, a CSV of index membership snapshots (one row per date the
membership changed, with the full comma-separated ticker list). For each month end we take the
latest snapshot on or before that date, so a stock is only in the universe while it was actually
in the index. That is what keeps later-delisted and later-removed stocks in the sample.
"""
import pandas as pd
import requests

from . import paths

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def download_membership(force: bool = False) -> None:
    if paths.MEMBERSHIP_CSV.exists() and not force:
        return
    r = requests.get(MEMBERSHIP_URL, timeout=60)
    r.raise_for_status()
    paths.MEMBERSHIP_CSV.write_bytes(r.content)


def load_snapshots() -> pd.DataFrame:
    snaps = pd.read_csv(paths.MEMBERSHIP_CSV, parse_dates=["date"]).sort_values("date")
    snaps["tickers"] = snaps["tickers"].str.split(",")
    return snaps


def monthly_universe(start: str = "1996-01-31", end: str | None = None) -> pd.DataFrame:
    """One row per (month_end, ticker) for stocks in the index at that month end."""
    snaps = load_snapshots()
    end = end or snaps["date"].max()
    month_ends = pd.date_range(start, end, freq="ME")
    # Latest snapshot on or before each month end.
    idx = snaps["date"].searchsorted(month_ends, side="right") - 1
    rows = []
    for m, i in zip(month_ends, idx):
        if i < 0:
            continue
        for t in snaps["tickers"].iloc[i]:
            rows.append((m, t.strip()))
    return pd.DataFrame(rows, columns=["month_end", "ticker"])


def to_yahoo(ticker: str) -> str:
    """Index tickers use dots for share classes (BRK.B); Yahoo uses dashes (BRK-B)."""
    return ticker.replace(".", "-")
