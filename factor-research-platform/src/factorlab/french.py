"""Ken French Data Library factors (monthly), used as the benchmark to validate our factors.

Files are zipped CSVs with a text header, a monthly block, then an annual block. Values are in
percent; we convert to decimals and index by month end.
"""
import io
import zipfile

import pandas as pd
import requests

from . import paths

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",           # Mkt-RF, SMB, HML, RF
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",     # + RMW, CMA
    "mom": "F-F_Momentum_Factor_CSV.zip",                 # Mom (UMD)
    "strev": "F-F_ST_Reversal_Factor_CSV.zip",            # ST_Rev
}


def _monthly_block(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    # Header row is the first line that starts with a comma (blank first column name).
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(","))
    body = []
    for l in lines[start + 1 :]:
        first = l.split(",")[0].strip()
        if not (first.isdigit() and len(first) == 6):  # YYYYMM; annual block uses YYYY
            break
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join([lines[start]] + body)))
    df = df.rename(columns={df.columns[0]: "yyyymm"})
    df.columns = [c.strip() for c in df.columns]
    df["month_end"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.drop(columns="yyyymm").set_index("month_end").astype(float) / 100.0


def download_all() -> pd.DataFrame:
    parts = []
    for key, fname in FILES.items():
        r = requests.get(BASE + fname, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        df = _monthly_block(text)
        if key == "ff5":
            df = df[["RMW", "CMA"]]
        parts.append(df)
    out = pd.concat(parts, axis=1)
    out = out.rename(columns={"Mom": "UMD", "ST_Rev": "STREV"})
    out.to_parquet(paths.FRENCH)
    return out
