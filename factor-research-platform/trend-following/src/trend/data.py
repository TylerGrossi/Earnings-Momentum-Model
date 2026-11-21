"""Price data from yfinance (dividend-adjusted closes), cached to data/prices.parquet."""
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
PRICES = DATA / "prices.parquet"

UNIVERSE = {
    "Equities": ["SPY", "IWM", "EFA", "EEM", "EWJ"],
    "Bonds": ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG"],
    "Commodities": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "Currencies": ["UUP", "FXE", "FXY"],
    "Real estate": ["VNQ"],
}
ASSETS = [t for ts in UNIVERSE.values() for t in ts]
ASSET_CLASS = {t: c for c, ts in UNIVERSE.items() for t in ts}
FUNDS = {"AQMIX": "AQR Managed Futures (fund)", "DBMF": "iMGP DBi Managed Futures ETF",
         "KMLM": "KraneShares Mount Lucas Managed Futures ETF"}
RATE = "^IRX"  # 13-week T-bill yield, percent


def download(force: bool = False) -> pd.DataFrame:
    if PRICES.exists() and not force:
        return pd.read_parquet(PRICES)
    DATA.mkdir(exist_ok=True)
    tickers = ASSETS + list(FUNDS) + [RATE]
    raw = yf.download(tickers, start="1993-01-01", auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    px = raw["Close"].copy()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    missing = [t for t in tickers if t not in px or px[t].dropna().empty]
    if missing:
        raise RuntimeError(f"no data for {missing}")
    px.to_parquet(PRICES)
    return px


def daily_rf(px: pd.DataFrame) -> pd.Series:
    """Daily T-bill return from the ^IRX yield (percent, annualized), per trading day."""
    return (px[RATE].ffill() / 100.0 / 252.0).rename("rf")
