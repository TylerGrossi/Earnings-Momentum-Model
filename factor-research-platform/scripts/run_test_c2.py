"""Test C2 (PREREGISTRATION.md change log): SUE event-time drift on CRSP/Compustat.

Events: Compustat announcements (rdq) for stocks in the Test C universe that month. Each calendar
quarter, events are sorted into SUE deciles; outcome = market-adjusted (CRSP value-weighted)
return from trading day +2 to +61 after rdq. Statistic: quarterly top-minus-bottom decile mean,
t-stat across quarters.

    FACTORLAB_SOURCE=crsp python scripts/run_test_c2.py             # exploration, 1990-2017
    FACTORLAB_SOURCE=crsp python scripts/run_test_c2.py --holdout   # holdout, 2018 on (run once)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("FACTORLAB_SOURCE", "crsp")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorlab import crsp, paths  # noqa: E402

HOLDOUT = "--holdout" in sys.argv
WINDOW = ("2018-01-01", "2100-01-01") if HOLDOUT else ("1990-01-01", "2017-12-31")


def event_sue(eq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in eq.sort_values("q_end").groupby("ticker"):
        g = g.set_index("q_end")
        lag = g["ni"].reindex(g.index - pd.DateOffset(months=12), method="nearest",
                              tolerance=pd.Timedelta(days=20)).to_numpy()
        chg = pd.Series(g["ni"].to_numpy() - lag, index=g.index)
        s = chg / chg.shift(1).rolling(8, min_periods=6).std()
        rows.append(pd.DataFrame({"ticker": t, "rdq": g["filed"].to_numpy(), "sue": s.to_numpy()}))
    return pd.concat(rows).replace([np.inf, -np.inf], np.nan).dropna()


def main() -> None:
    assert paths.SOURCE == "crsp", "Test C2 runs on CRSP data only"
    eq = pd.read_parquet(paths.DATA / "earnings_quarterly.parquet")
    uni = pd.read_parquet(paths.UNIVERSE)
    ev = event_sue(eq)
    ev = ev[(ev["rdq"] >= WINDOW[0]) & (ev["rdq"] <= WINDOW[1])]
    # Universe membership at the month end before the announcement (known at the time).
    ev["month_end"] = ev["rdq"] - pd.offsets.MonthEnd(1)
    ev = ev.merge(uni, on=["month_end", "ticker"])
    print(f"{len(ev):,} events in window {WINDOW}")

    db = crsp.connect()
    years = (pd.Timestamp(WINDOW[0]).year, min(pd.Timestamp(WINDOW[1]).year, pd.Timestamp.today().year))
    daily = crsp.crsp_daily(db, ev["ticker"].astype(int).unique().tolist(), years[0], years[1] + 1)
    mkt = crsp.market_daily(db, f"{years[0]}-01-01")
    days = mkt.index
    R = daily.pivot(index="date", columns="permno", values="ret").reindex(days)
    logex = np.log1p(R.sub(mkt, axis=0))  # market-adjusted, summed in logs then converted

    def car(t: int, d: pd.Timestamp) -> float:
        i = days.searchsorted(d)
        if t not in logex.columns or i + 61 >= len(days):
            return np.nan
        w = logex[t].iloc[i + 2: i + 62]
        return float(np.expm1(w.sum())) if w.notna().sum() >= 50 else np.nan

    ev["car"] = [car(int(t), d) for t, d in zip(ev["ticker"], ev["rdq"])]
    ev = ev.dropna(subset=["car"])
    ev["q"] = ev["rdq"].dt.to_period("Q")
    ev["decile"] = ev.groupby("q")["sue"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    spread = ev.groupby("q").apply(lambda g: g.loc[g["decile"] == 10, "car"].mean() - g.loc[g["decile"] == 1, "car"].mean())
    t = spread.mean() / spread.std() * np.sqrt(len(spread))
    by_dec = ev.groupby("decile")["car"].mean()
    label = "holdout 2018 on" if HOLDOUT else "exploration 1990-2017"
    L = [f"# Test C2: SUE event-time drift ({label})", "",
         f"{len(ev):,} announcements, {len(spread)} quarters. Market-adjusted return, trading days +2 to +61 after rdq.", "",
         "| Decile | " + " | ".join(str(i) for i in by_dec.index) + " |",
         "|---" * (len(by_dec) + 1) + "|",
         "| Mean CAR | " + " | ".join(f"{v:+.2%}" for v in by_dec) + " |", "",
         f"**Top minus bottom decile: {spread.mean():+.2%} per event window, t {t:.2f}.**", "",
         "Criterion: exploration t >= 2 and holdout mean > 0."]
    out = paths.REPORTS / ("test_c2_holdout.md" if HOLDOUT else "test_c2_exploration.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
