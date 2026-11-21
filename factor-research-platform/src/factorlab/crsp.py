"""CRSP / Compustat data source via WRDS (Test C in PREREGISTRATION.md).

Written 2026-11-20 before WRDS access was approved, against the documented table layouts, so it
is UNTESTED. Run `python scripts/build_crsp.py --probe` first: it lists the tables your account
can see and checks units on Apple before anything is built.

Writes the same files as the free source (universe_monthly, monthly, fundamental_signals,
earnings_quarterly, sectors.csv, french_monthly) into data/crsp/, with `ticker` = CRSP PERMNO as
a string, so every research script runs unchanged with FACTORLAB_SOURCE=crsp.

CRSP comes in two formats. The current CIZ format (crsp.msf_v2) builds delisting returns into
the monthly return. The legacy SIZ format (crsp.msf, frozen after 2024) needs them merged from
crsp.msedelist. Both are supported; CIZ is preferred.
"""
import numpy as np
import pandas as pd

from . import french, paths, sectors

APPLE_PERMNO = 14593


def connect(username: str | None = None):
    import wrds  # pip install wrds; first login prompts for a password and can save a .pgpass

    return wrds.Connection(wrds_username=username)


def probe(db) -> dict:
    found = {}
    for lib, tables in {"crsp": ["msf_v2", "dsf_v2", "msf", "msenames", "msedelist", "dsf", "dsi",
                                 "ccmxpf_lnkhist"],
                        "comp": ["funda", "fundq"]}.items():
        have = set(db.list_tables(library=lib))
        for t in tables:
            found[f"{lib}.{t}"] = t in have
    for k, v in found.items():
        print(f"  {k:22s} {'yes' if v else 'NO'}")
    return found


# ---------------------------------------------------------------- CRSP monthly

def crsp_monthly(db, start: str = "1985-01-01", ciz: bool = True) -> pd.DataFrame:
    if ciz:
        # CIZ equivalents of share code 10/11 and NYSE/AMEX/NASDAQ (WRDS SIZ-to-CIZ mapping).
        df = db.raw_sql(f"""
            select permno, mthcaldt as date, mthret as ret, mthprc as prc, shrout, siccd,
                   primaryexch, ticker
            from crsp.msf_v2
            where mthcaldt >= '{start}'
              and sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM'
              and usincflg = 'Y' and issuertype in ('ACOR', 'CORP')
              and primaryexch in ('N', 'A', 'Q')
        """, date_cols=["date"])
    else:
        df = db.raw_sql(f"""
            select a.permno, a.date, a.ret, a.prc, a.shrout, b.siccd, b.exchcd, b.ticker
            from crsp.msf a join crsp.msenames b
              on a.permno = b.permno and a.date between b.namedt and b.nameendt
            where a.date >= '{start}' and b.shrcd in (10, 11) and b.exchcd in (1, 2, 3)
        """, date_cols=["date"])
        df["primaryexch"] = df["exchcd"].map({1: "N", 2: "A", 3: "Q"})
        dl = db.raw_sql("select permno, dlstdt, dlret, dlstcd from crsp.msedelist",
                        date_cols=["dlstdt"])
        # Shumway (1997): missing performance-delisting returns (codes 500-599) set to -30%.
        perf = dl["dlstcd"].between(500, 599)
        dl.loc[dl["dlret"].isna() & perf, "dlret"] = -0.30
        dl["month_end"] = dl["dlstdt"] + pd.offsets.MonthEnd(0)
        df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
        df = df.merge(dl[["permno", "month_end", "dlret"]], on=["permno", "month_end"], how="left")
        has = df["dlret"].notna()
        df.loc[has, "ret"] = (1 + df.loc[has, "ret"].fillna(0)) * (1 + df.loc[has, "dlret"]) - 1
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
    df["permno"] = df["permno"].astype(int)
    df["close"] = df["prc"].abs()  # negative CRSP price = bid/ask midpoint
    return df


def shares_scale(df: pd.DataFrame) -> float:
    """CRSP reports shares outstanding in thousands. Confirm on Apple (Dec 2020 market cap
    about $2.26T) instead of assuming, in case a format reports raw shares."""
    a = df[(df["permno"] == APPLE_PERMNO) & (df["month_end"] == "2020-12-31")]
    if a.empty:
        raise RuntimeError("Apple Dec 2020 not found; can't verify share units")
    raw = float(a["close"].iloc[0] * a["shrout"].iloc[0])
    for scale in (1000.0, 1.0):
        if 1.5e12 <= raw * scale <= 3.0e12:
            return scale
    raise RuntimeError(f"Apple Dec 2020 close x shrout = {raw:,.0f}; units not recognized")


def build_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered universe: market cap above the NYSE 20th percentile each month."""
    d = df.dropna(subset=["mcap"])
    bp = d[d["primaryexch"] == "N"].groupby("month_end")["mcap"].quantile(0.2).rename("nyse_p20")
    d = d.join(bp, on="month_end")
    return d.loc[d["mcap"] >= d["nyse_p20"], ["month_end", "ticker"]].reset_index(drop=True)


# ---------------------------------------------------------------- Compustat

def ccm_links(db) -> pd.DataFrame:
    ln = db.raw_sql("""
        select gvkey, lpermno as permno, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LU', 'LC') and linkprim in ('P', 'C')
    """, date_cols=["linkdt", "linkenddt"])
    ln["linkenddt"] = ln["linkenddt"].fillna(pd.Timestamp("2100-12-31"))
    ln["permno"] = ln["permno"].astype(int)
    return ln


def _link(df: pd.DataFrame, links: pd.DataFrame, on_date: str) -> pd.DataFrame:
    m = df.merge(links, on="gvkey", how="inner")
    return m[(m[on_date] >= m["linkdt"]) & (m[on_date] <= m["linkenddt"])].drop(columns=["linkdt", "linkenddt"])


def compustat_annual(db, links: pd.DataFrame, start: str = "1980-01-01") -> pd.DataFrame:
    f = db.raw_sql(f"""
        select gvkey, datadate, at, lt, seq, ceq, pstk, pstkl, pstkrv, txditc, revt, cogs, xsga,
               xint, ib
        from comp.funda
        where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
          and datadate >= '{start}'
    """, date_cols=["datadate"])
    # Fama-French book equity.
    se = f["seq"].fillna(f["ceq"] + f["pstk"].fillna(0)).fillna(f["at"] - f["lt"])
    ps = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0)
    f["be"] = se + f["txditc"].fillna(0) - ps
    costs = f[["cogs", "xsga", "xint"]]
    f["op"] = (f["revt"] - costs.fillna(0).sum(axis=1)).where(costs.notna().any(axis=1))
    f = f.sort_values(["gvkey", "datadate"])
    prev_at = f.groupby("gvkey")["at"].shift(1)
    prev_dt = f.groupby("gvkey")["datadate"].shift(1)
    f["assets_prev"] = prev_at.where((f["datadate"] - prev_dt).dt.days.between(330, 400))
    # Pre-registered availability lag: 6 months after fiscal year end.
    f["avail"] = f["datadate"] + pd.offsets.MonthEnd(6)
    return _link(f, links, "avail")


def compustat_quarterly(db, links: pd.DataFrame, start: str = "1985-01-01") -> pd.DataFrame:
    q = db.raw_sql(f"""
        select gvkey, datadate, rdq, ibq
        from comp.fundq
        where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
          and datadate >= '{start}' and rdq is not null
    """, date_cols=["datadate", "rdq"])
    q = _link(q, links, "rdq")
    out = q.rename(columns={"datadate": "q_end", "ibq": "ni", "rdq": "filed"})
    out["ticker"] = out["permno"].astype(str)
    return out[["ticker", "q_end", "ni", "filed"]].dropna().drop_duplicates(["ticker", "q_end"])


def fundamental_panel(monthly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    base = monthly[["month_end", "permno", "mcap"]].dropna().sort_values("month_end")
    a = annual.sort_values("avail")
    p = pd.merge_asof(base, a, left_on="month_end", right_on="avail", by="permno", direction="backward")
    stale = (p["month_end"] - p["avail"]).dt.days > 365
    for c in ("be", "op", "revt", "cogs", "at", "assets_prev", "ib"):
        p.loc[stale, c] = np.nan
    pos_be = p["be"].where(p["be"] > 0)
    p["size"] = np.log(p["mcap"])
    p["book_to_market"] = pos_be / p["mcap"]
    p["earnings_yield"] = p["ib"] / p["mcap"]
    p["gross_prof"] = (p["revt"] - p["cogs"]) / p["at"]
    p["op_prof"] = p["op"] / pos_be
    p["roe"] = p["ib"] / pos_be
    p["asset_growth"] = p["at"] / p["assets_prev"] - 1
    p["ticker"] = p["permno"].astype(str)
    keep = ["mcap", "size", "book_to_market", "earnings_yield", "gross_prof", "op_prof", "roe", "asset_growth"]
    return p.set_index(["month_end", "ticker"])[keep].replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- daily (Test C2)

def crsp_daily(db, permnos: list[int], start: int, end: int, ciz: bool = True) -> pd.DataFrame:
    """Daily returns for the given PERMNOs, one year per query to keep requests small."""
    tbl, dcol, rcol = ("crsp.dsf_v2", "dlycaldt", "dlyret") if ciz else ("crsp.dsf", "date", "ret")
    ids = ",".join(str(p) for p in sorted(set(permnos)))
    parts = []
    for y in range(start, end + 1):
        parts.append(db.raw_sql(f"""
            select permno, {dcol} as date, {rcol} as ret from {tbl}
            where {dcol} between '{y}-01-01' and '{y}-12-31' and permno in ({ids})
        """, date_cols=["date"]))
        print(f"  daily {y}")
    return pd.concat(parts, ignore_index=True)


def market_daily(db, start: str = "1985-01-01") -> pd.Series:
    m = db.raw_sql(f"select date, vwretd from crsp.dsi where date >= '{start}'", date_cols=["date"])
    return m.set_index("date")["vwretd"]


# ---------------------------------------------------------------- build all

def build(db, ciz: bool = True) -> None:
    paths.DATA.mkdir(parents=True, exist_ok=True)
    print("CRSP monthly")
    df = crsp_monthly(db, ciz=ciz)
    scale = shares_scale(df)
    df["mcap"] = df["close"] * df["shrout"] * scale
    df["ticker"] = df["permno"].astype(str)
    uni = build_universe(df)
    uni.to_parquet(paths.UNIVERSE, index=False)
    monthly = df[["month_end", "ticker", "permno", "ret", "close", "mcap", "primaryexch"]]
    monthly = monthly.drop_duplicates(["month_end", "ticker"])
    monthly.to_parquet(paths.MONTHLY, index=False)
    print(f"  {len(monthly):,} stock-months; universe {uni.groupby('month_end').size().median():.0f} "
          f"stocks per month (median); shares scale x{scale:g}")

    print("sectors (CRSP SIC -> FF12)")
    ranges = sectors.ff12_ranges()
    last = df.sort_values("month_end").groupby("ticker")["siccd"].last()
    sec = pd.DataFrame({"ticker": last.index, "sic": last.to_numpy()})
    sec["sector"] = [sectors.sic_to_ff12(s, ranges) for s in sec["sic"]]
    sec.to_csv(sectors.SECTORS, index=False)

    print("Compustat annual + quarterly")
    links = ccm_links(db)
    annual = compustat_annual(db, links)
    fund = fundamental_panel(monthly, annual)
    fund.to_parquet(paths.DATA / "fundamental_signals.parquet")
    compustat_quarterly(db, links).to_parquet(paths.DATA / "earnings_quarterly.parquet", index=False)

    print("Ken French factors")
    french.download_all()
