"""Where data and reports live. Everything under data/ is rebuildable and gitignored.

FACTORLAB_SOURCE selects the data source: "free" (default; yfinance + SEC) or "crsp"
(CRSP/Compustat via WRDS). Both write the same file names, to data/ and data/crsp/, and
reports to reports/ and reports/crsp/, so every research script runs on either.
"""
import os
from pathlib import Path

SOURCE = os.environ.get("FACTORLAB_SOURCE", "free").lower()
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / ("crsp" if SOURCE == "crsp" else "")
REPORTS = ROOT / "reports" / ("crsp" if SOURCE == "crsp" else "")

MEMBERSHIP_CSV = RAW / "sp500_membership.csv"      # fja05680/sp500 snapshots
UNIVERSE = DATA / "universe_monthly.parquet"        # month_end, ticker
PRICES_DAILY = DATA / "prices_daily.parquet"        # date, ticker, adj_close, volume
PRICE_MISSING = DATA / "prices_missing.csv"         # tickers yfinance returned nothing for
MONTHLY = DATA / "monthly.parquet"                  # month_end, ticker, adj_close, ret, dollar_vol
FRENCH = DATA / "french_monthly.parquet"            # Ken French factors, decimal returns

for p in (DATA, RAW, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
