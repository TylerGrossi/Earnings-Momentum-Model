"""Point-in-time annual fundamentals from SEC EDGAR XBRL (companyfacts API).

Every XBRL fact carries the date it was filed. We keep, for each fiscal period, the value from
the FIRST filing that reported it (later filings repeat prior years as comparatives, sometimes
restated). A fundamental is usable from its filing date on, never from its fiscal period end.
That is the lookahead guard.

Coverage limits (measured in reports/fundamentals_coverage.md):
- XBRL starts with 2009 filings, so fundamental signals start around 2010.
- SEC's ticker-to-CIK file only maps CURRENT tickers. Former members that were acquired or
  delisted are mostly unmapped, the same survivorship gap as the free prices.

SEC asks automated clients to identify themselves in the User-Agent. Set SEC_USER_AGENT to
"<name> <email>" to use your own; the default is a generic label.
"""
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from . import paths

UA = os.environ.get("SEC_USER_AGENT", "factorlab research contact@example.com")
SEC_DIR = paths.RAW / "sec"
FUND_ANNUAL = paths.DATA / "fundamentals_annual.parquet"
SHARES = paths.DATA / "shares_outstanding.parquet"
CIK_MAP = paths.DATA / "ticker_cik.csv"

# Item -> concepts in priority order. Companies switch concepts over time (revenue moved to
# RevenueFromContractWithCustomer... under ASC 606 in 2018), so for each period we take the
# highest-priority concept that has a value for that period.
CONCEPTS = {
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "op_income": ["OperatingIncomeLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
}
INSTANT = {"equity", "assets"}
FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "20-F", "40-F")


def _get(url: str) -> requests.Response:
    for attempt in range(4):
        r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
        if r.status_code == 200 or r.status_code == 404:
            return r
        time.sleep(2 ** attempt)  # 429 / 5xx: back off
    r.raise_for_status()
    return r


def ticker_cik_map() -> pd.DataFrame:
    r = _get("https://www.sec.gov/files/company_tickers.json")
    df = pd.DataFrame(r.json().values()).rename(columns={"cik_str": "cik", "title": "name"})
    df["ticker"] = df["ticker"].str.replace("-", ".", regex=False)  # BRK-B -> BRK.B (index style)
    df = df.drop_duplicates("ticker")
    df.to_csv(CIK_MAP, index=False)
    return df


def fetch_companyfacts(cik: int) -> dict | None:
    """Raw companyfacts JSON, cached gzipped. Polite to SEC's 10 requests/second limit."""
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    f = SEC_DIR / f"CIK{cik:010d}.json.gz"
    if f.exists():
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    r = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    time.sleep(0.15)
    if r.status_code == 404:
        return None
    with gzip.open(f, "wt", encoding="utf-8") as fh:
        fh.write(r.text)
    return r.json()


def _rows(facts: dict, ns: str, concept: str, unit: str) -> list[dict]:
    try:
        return facts["facts"][ns][concept]["units"][unit]
    except KeyError:
        return []


def extract(facts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annual table (one row per fiscal year end) and share-count table for one company."""
    recs = []
    for item, concepts in CONCEPTS.items():
        for rank, c in enumerate(concepts):
            for r in _rows(facts, "us-gaap", c, "USD"):
                if r.get("form") not in FORMS:
                    continue
                recs.append({"item": item, "rank": rank, "start": r.get("start"), "end": r["end"],
                             "val": r["val"], "filed": r["filed"], "form": r["form"]})
    if not recs:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(recs)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])

    # Annual flows: duration of roughly one year.
    dur = (df["end"] - df["start"]).dt.days
    flows = df[~df["item"].isin(INSTANT) & dur.between(350, 380)]
    inst = df[df["item"].isin(INSTANT)]

    def first_report(d: pd.DataFrame) -> pd.DataFrame:
        # Best concept for the period, then the earliest filing of it.
        d = d.sort_values(["item", "end", "rank", "filed"])
        best = d.groupby(["item", "end"])["rank"].transform("min")
        d = d[d["rank"] == best]
        return d.groupby(["item", "end"], as_index=False).first()

    flows, inst = first_report(flows), first_report(inst)
    # Fiscal year ends come from annual net income or revenue; filed = first filing of that year.
    fy = flows[flows["item"].isin(["net_income", "revenue"])].groupby("end")["filed"].min()
    if fy.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide_f = flows.pivot(index="end", columns="item", values="val")
    wide_i = inst.pivot(index="end", columns="item", values="val")
    annual = pd.DataFrame(index=fy.index).join(wide_f).join(wide_i)
    annual["filed"] = fy
    if "gross_profit" not in annual or annual["gross_profit"].isna().all():
        annual["gross_profit"] = np.nan
    if {"revenue", "cogs"} <= set(annual):
        annual["gross_profit"] = annual["gross_profit"].fillna(annual["revenue"] - annual["cogs"])
    annual = annual.sort_index()
    annual["assets_prev"] = annual["assets"].shift(1).where(
        (annual.index.to_series().diff().dt.days.between(330, 400)).values)
    annual = annual.reset_index().rename(columns={"end": "fy_end"})

    # Shares outstanding from the cover page (dei), summed across share classes per filing.
    sh = pd.DataFrame(_rows(facts, "dei", "EntityCommonStockSharesOutstanding", "shares"))
    if len(sh):
        sh["end"] = pd.to_datetime(sh["end"])
        sh["filed"] = pd.to_datetime(sh["filed"])
        sh = sh.groupby(["accn", "end", "filed"], as_index=False)["val"].sum()
        sh = sh.sort_values("filed").groupby("end", as_index=False).first()[["end", "filed", "val"]]
        sh = sh.rename(columns={"end": "share_date", "val": "shares"})
    return annual, sh


EARNINGS_Q = paths.DATA / "earnings_quarterly.parquet"


def extract_quarterly_ni(facts: dict) -> pd.DataFrame:
    """Quarterly net income with the date each quarter was first filed.

    Q1-Q3 come from 3-month facts in 10-Qs. Q4 is rarely reported as a 3-month fact, so it is
    fiscal-year net income minus the year's three reported quarters, dated by the 10-K filing.
    """
    recs = []
    for rank, c in enumerate(CONCEPTS["net_income"]):
        for r in _rows(facts, "us-gaap", c, "USD"):
            if r.get("form") in FORMS and r.get("start"):
                recs.append({"rank": rank, "start": r["start"], "end": r["end"], "val": r["val"],
                             "filed": r["filed"]})
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    dur = (df["end"] - df["start"]).dt.days

    def first(d):
        d = d.sort_values(["end", "rank", "filed"])
        d = d[d["rank"] == d.groupby("end")["rank"].transform("min")]
        return d.groupby("end", as_index=False).first()[["end", "val", "filed"]]

    q = first(df[dur.between(80, 100)])
    a = first(df[dur.between(350, 380)])
    q4 = []
    for r in a.itertuples():
        inside = q[(q["end"] > r.end - pd.Timedelta(days=365)) & (q["end"] < r.end - pd.Timedelta(days=20))]
        if len(inside) == 3 and not (q["end"] == r.end).any():
            q4.append({"end": r.end, "val": r.val - inside["val"].sum(), "filed": r.filed})
    out = pd.concat([q, pd.DataFrame(q4)], ignore_index=True).sort_values("end")
    return out.drop_duplicates("end").rename(columns={"end": "q_end", "val": "ni"})


def build_quarterly(tickers: list[str]) -> pd.DataFrame:
    """Quarterly net income for every mapped ticker, read from the companyfacts cache."""
    cmap = pd.read_csv(CIK_MAP).set_index("ticker")["cik"]
    parts = []
    for t in sorted(set(tickers)):
        if t not in cmap.index:
            continue
        facts = fetch_companyfacts(int(cmap[t]))
        q = extract_quarterly_ni(facts) if facts else pd.DataFrame()
        if len(q):
            parts.append(q.assign(ticker=t))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(EARNINGS_Q, index=False)
    return out


def build(tickers: list[str]) -> dict:
    cmap = ticker_cik_map().set_index("ticker")["cik"]
    annual, shares, unmapped, empty = [], [], [], []
    todo = [t for t in sorted(set(tickers)) if t in cmap.index]
    unmapped = sorted(set(tickers) - set(todo))
    for i, t in enumerate(todo, 1):
        facts = fetch_companyfacts(int(cmap[t]))
        a, s = extract(facts) if facts else (pd.DataFrame(), pd.DataFrame())
        if a.empty:
            empty.append(t)
        else:
            annual.append(a.assign(ticker=t, cik=int(cmap[t])))
        if len(s):
            shares.append(s.assign(ticker=t))
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} companies")
    pd.concat(annual, ignore_index=True).to_parquet(FUND_ANNUAL, index=False)
    pd.concat(shares, ignore_index=True).to_parquet(SHARES, index=False)
    return {"requested": len(set(tickers)), "mapped": len(todo), "unmapped": unmapped, "empty": empty}
