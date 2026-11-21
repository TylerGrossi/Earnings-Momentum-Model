"""Industry for each ticker: SEC SIC code mapped to the Fama-French 12 industries.

SIC comes from SEC's submissions API (current SIC, which rarely changes for large companies).
Tickers SEC can't map (mostly former members) get "Unknown" and are neutralized as their own group.
"""
import io
import re
import time
import zipfile

import pandas as pd
import requests

from . import paths
from .fundamentals import CIK_MAP, UA, _get

SECTORS = paths.DATA / "sectors.csv"
FF12_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip"


def ff12_ranges() -> list[tuple[int, int, str]]:
    r = requests.get(FF12_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    out, current = [], None
    for line in text.splitlines():
        m = re.match(r"\s*\d+\s+(\w+)\s", line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r"\s*(\d{4})-(\d{4})", line)
        if m and current:
            out.append((int(m.group(1)), int(m.group(2)), current))
    return out


def sic_to_ff12(sic: int | None, ranges) -> str:
    if sic is None or pd.isna(sic):
        return "Unknown"
    for lo, hi, name in ranges:
        if lo <= sic <= hi:
            return name
    return "Other"  # FF12 industry 12 is "everything else"


def build(tickers: list[str]) -> pd.DataFrame:
    cmap = pd.read_csv(CIK_MAP).set_index("ticker")["cik"]
    ranges = ff12_ranges()
    cache = paths.RAW / "sec_sic.csv"
    known = pd.read_csv(cache).set_index("cik")["sic"].to_dict() if cache.exists() else {}
    rows = []
    for t in sorted(set(tickers)):
        cik = cmap.get(t)
        sic = None
        if cik is not None:
            cik = int(cik)
            if cik not in known:
                r = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
                time.sleep(0.15)
                s = r.json().get("sic") if r.status_code == 200 else None
                known[cik] = int(s) if s not in (None, "") else None
            sic = known[cik]
        rows.append({"ticker": t, "cik": cik, "sic": sic, "sector": sic_to_ff12(sic, ranges)})
    pd.DataFrame([{"cik": k, "sic": v} for k, v in known.items()]).to_csv(cache, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(SECTORS, index=False)
    return df
