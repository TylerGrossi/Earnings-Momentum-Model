"""
Session 2: do the stop-loss level and exit day hold up out of sample?

The README rule (-10% stop, exit at the day-5 close) was picked by looking at the data. This
script asks how much of a chosen rule's backtest is real and how much is selection.

Grid: stop in {none, -2%, -3%, ..., -10%, -12%, ..., -20%} x exit day in {1..5} = 75 rules.

Stop fills (same convention as the app's stop_loss_analysis.py):
  * Stops are checked on hourly bar closes (the data has no intraday lows).
  * If the FIRST bar after entry is already below the stop, the trade takes that bar's loss
    (a gap). Otherwise it fills exactly at the stop.

Three tests:
  1. In-sample: every rule on the full sample, and the best one. This is the optimistic number.
  2. Walk-forward: for each entry week after a warm-up, pick the rule with the best mean on all
     EARLIER weeks, and apply it to this week. The realized returns are out of sample.
  3. Probability of backtest overfitting (Bailey, Borwein, Lopez de Prado & Zhu, 2017, CSCV):
     split the weeks into 8 blocks; for every way of choosing 4 blocks as in-sample (70 splits),
     find the in-sample best rule and see where it ranks out of sample. PBO = share of splits
     where it lands in the bottom half. Near 0 means selection carries over; 0.5 means it is
     as good as picking at random.

Returns are reported raw and in excess of the sector ETF over the same days. For a stopped
trade the benchmark runs to the close of the stop day (an approximation; the stop fill is
intraday). Assumed costs are the session-1 market-cap tiers.

Run after excess_returns.py (it reuses that script's price cache and trade list).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

STOPS = [None] + [-s / 100 for s in range(2, 11)] + [-s / 100 for s in (12, 14, 16, 18, 20)]
EXITS = [1, 2, 3, 4, 5]
WARMUP_WEEKS = 8
PBO_BLOCKS = 8
README_RULE = (-0.10, 5)


def rule_name(stop, exit_day):
    return f"{'no stop' if stop is None else f'SL {round(stop * 100)}%'} / day {exit_day}"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(ROOT / "hourly_prices.csv", parse_dates=["Datetime"])
    h["Earnings Date"] = pd.to_datetime(h["Earnings Date"]).dt.date
    h["r"] = h["Return From Earnings (%)"] / 100
    h = h.sort_values(["Ticker", "Earnings Date", "Datetime"])
    trades = pd.read_csv(OUT / "excess_returns_trades.csv", parse_dates=["Entry Date"])
    trades["Earnings Date"] = pd.to_datetime(trades["Earnings Date"]).dt.date
    px = pd.read_parquet(HERE / "cache" / "daily_close.parquet")
    return h, trades, px


def simulate(path: pd.DataFrame) -> dict:
    """Return {(stop, exit): (return, exit_trading_day)} for one trade's hourly path."""
    r = path["r"].to_numpy()
    day = path["Trading Day"].to_numpy()
    out = {}
    for exit_day in EXITS:
        upto = day <= exit_day
        if not upto.any() or day.max() < exit_day:
            continue
        rr, dd = r[upto], day[upto]
        close_ret = rr[-1]
        for stop in STOPS:
            if stop is None:
                out[(stop, exit_day)] = (close_ret, exit_day)
                continue
            hit = np.nonzero(rr <= stop)[0]
            if hit.size == 0:
                out[(stop, exit_day)] = (close_ret, exit_day)
            else:
                i = hit[0]
                fill = rr[i] if i == 0 else stop  # first bar below stop = gap, take it
                out[(stop, exit_day)] = (fill, int(dd[i]))
    return out


def build_panel(h, trades, px) -> pd.DataFrame:
    from excess_returns import SECTOR_ETF  # same mapping as session 1

    cal = px["SPY"].dropna().index
    rows = []
    paths = dict(tuple(h.groupby(["Ticker", "Earnings Date"])))
    for t in trades.itertuples(index=False):
        path = paths.get((t.Ticker, t._1))  # _1 = "Earnings Date"
        if path is None:
            continue
        i0 = cal.searchsorted(t._5)  # _5 = "Entry Date"
        etf = SECTOR_ETF.get(t.Sector)
        bench = px[etf].reindex(cal).to_numpy(float) if etf in px else None
        for (stop, exit_day), (ret, exit_td) in simulate(path).items():
            b = np.nan
            if bench is not None and i0 + exit_td < len(cal):
                b = bench[i0 + exit_td] / bench[i0] - 1
            rows.append({"Ticker": t.Ticker, "Entry Date": t._5, "cost": t.cost,
                         "rule": rule_name(stop, exit_day), "stop": stop, "exit": exit_day,
                         "ret": ret, "xs": ret - b})
    p = pd.DataFrame(rows)
    p["week"] = p["Entry Date"].dt.to_period("W-FRI")
    return p


def weekly_t(x: pd.Series, week: pd.Series) -> tuple[float, float, int]:
    w = x.groupby(week.to_numpy()).mean()
    return w.mean(), w.mean() / (w.std(ddof=1) / np.sqrt(len(w))), len(w)


def in_sample(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, g in p.groupby("rule"):
        m, t, n = weekly_t(g["xs"], g["week"])
        rows.append({"rule": rule, "stop": g["stop"].iloc[0], "exit": g["exit"].iloc[0],
                     "n": len(g), "mean_ret": g["ret"].mean(), "mean_xs": g["xs"].mean(),
                     "mean_xs_net": (g["xs"] - g["cost"]).mean(), "hit_rate": (g["ret"] > 0).mean(),
                     "weekly_mean_xs": m, "t_weekly_xs": t, "stopped_share": np.nan})
    return pd.DataFrame(rows).sort_values("mean_xs", ascending=False).reset_index(drop=True)


def walk_forward(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = sorted(p["week"].unique())
    picks, realized = [], []
    for k, wk in enumerate(weeks):
        if k < WARMUP_WEEKS:
            continue
        past = p[p["week"] < wk]
        best = past.groupby("rule")["xs"].mean().idxmax()
        picks.append({"week": str(wk), "rule": best,
                      "train_mean_xs": past[past["rule"] == best]["xs"].mean()})
        cur = p[(p["week"] == wk)]
        realized.append(cur[cur["rule"] == best].assign(chosen=best))
    return pd.DataFrame(picks), pd.concat(realized, ignore_index=True)


def pbo(p: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    # Rules x weeks matrix of mean excess returns.
    m = p.pivot_table(index="week", columns="rule", values="xs", aggfunc="mean").sort_index()
    blocks = np.array_split(np.arange(len(m)), PBO_BLOCKS)
    logits, rows = [], []
    for ins in itertools.combinations(range(PBO_BLOCKS), PBO_BLOCKS // 2):
        is_idx = np.concatenate([blocks[i] for i in ins])
        os_idx = np.concatenate([blocks[i] for i in range(PBO_BLOCKS) if i not in ins])
        is_perf, os_perf = m.iloc[is_idx].mean(), m.iloc[os_idx].mean()
        best = is_perf.idxmax()
        rank = os_perf.rank(pct=True)[best]  # 1.0 = best out of sample
        w = np.clip(rank, 1e-6, 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
        rows.append({"is_best": best, "is_mean": is_perf[best], "os_mean": os_perf[best],
                     "os_rank_pct": rank, "os_median_rule": os_perf.median()})
    return float(np.mean(np.array(logits) <= 0)), pd.DataFrame(rows)


def main() -> None:
    import sys
    sys.path.insert(0, str(HERE))
    h, trades, px = load()
    p = build_panel(h, trades, px)
    p = p.dropna(subset=["xs"])
    n_trades = p.groupby(["Ticker", "Entry Date"]).ngroups
    print(f"Trades with hourly paths: {n_trades}; weeks: {p['week'].nunique()}; rules: {p['rule'].nunique()}")

    ins = in_sample(p)
    readme = rule_name(*README_RULE)
    base = rule_name(None, 5)

    picks, wf = walk_forward(p)
    wf_m, wf_t, wf_n = weekly_t(wf["xs"], wf["week"])
    # Same test weeks for the fixed rules, so the comparison is like for like.
    test_weeks = set(wf["week"])
    def fixed(rule):
        g = p[(p["rule"] == rule) & p["week"].isin(test_weeks)]
        m, t, n = weekly_t(g["xs"], g["week"])
        return {"rule": rule, "n": len(g), "mean_ret": g["ret"].mean(), "mean_xs": g["xs"].mean(),
                "mean_xs_net": (g["xs"] - g["cost"]).mean(), "weekly_t_xs": t}
    best_is = ins.iloc[0]["rule"]
    comp = pd.DataFrame([
        {"rule": "walk-forward pick", "n": len(wf), "mean_ret": wf["ret"].mean(), "mean_xs": wf["xs"].mean(),
         "mean_xs_net": (wf["xs"] - wf["cost"]).mean(), "weekly_t_xs": wf_t},
        fixed(readme), fixed(base), fixed(best_is),
    ])
    comp.loc[comp["rule"] == best_is, "rule"] = f"{best_is} (full-sample best, hindsight)"

    prob, pbo_rows = pbo(p)

    OUT.mkdir(exist_ok=True)
    ins.to_csv(OUT / "rules_in_sample.csv", index=False)
    picks.to_csv(OUT / "walk_forward_picks.csv", index=False)
    comp.to_csv(OUT / "walk_forward_comparison.csv", index=False)
    pbo_rows.to_csv(OUT / "pbo_splits.csv", index=False)

    pd.set_option("display.width", 200)
    f4 = "{:.4f}".format
    print("\n=== In-sample: top 10 rules by mean sector-excess return ===")
    print(ins.head(10).drop(columns=["stop", "exit", "stopped_share"]).to_string(index=False, float_format=f4))
    for r in (readme, base):
        row = ins[ins["rule"] == r].iloc[0]
        print(f"{r}: rank {ins.index[ins['rule'] == r][0] + 1} of {len(ins)}, mean xs {row['mean_xs']:.4f}")
    print(f"Worst rule: {ins.iloc[-1]['rule']}  mean xs {ins.iloc[-1]['mean_xs']:.4f}")
    print(f"Spread of means across all {len(ins)} rules: {ins['mean_xs'].min():.4f} to {ins['mean_xs'].max():.4f}")

    print(f"\n=== Walk-forward (warm-up {WARMUP_WEEKS} weeks, {wf_n} test weeks) ===")
    print(picks["rule"].value_counts().to_string())
    print(comp.to_string(index=False, float_format=f4))

    # Is the edge stable over time? Baseline rule (no stop, day 5) by entry month and by half.
    g = p[p["rule"] == base].copy()
    g["month"] = g["Entry Date"].dt.to_period("M")
    by_month = g.groupby("month").agg(n=("xs", "size"), mean_xs=("xs", "mean"),
                                      median_xs=("xs", "median"), hit=("xs", lambda s: (s > 0).mean()))
    first = sorted(g["week"].unique())[:WARMUP_WEEKS]
    # The tracker's first commit is 2025-12-24 and no row before 2026-01-09 has a Date Added,
    # so earlier rows can't be shown to have been recorded before the trade.
    known = trades.loc[trades["Date Added Known"], "Entry Date"].min()
    halves = []
    for label, sub in [(f"first {WARMUP_WEEKS} weeks", g[g["week"].isin(first)]),
                       ("later weeks", g[~g["week"].isin(first)]),
                       (f"before first Date Added ({known.date()})", g[g["Entry Date"] < known]),
                       (f"from first Date Added ({known.date()})", g[g["Entry Date"] >= known])]:
        m, t, n = weekly_t(sub["xs"], sub["week"])
        halves.append({"period": label, "weeks": n, "trades": len(sub), "mean_xs": sub["xs"].mean(),
                       "median_xs": sub["xs"].median(), "weekly_t": t})
    halves = pd.DataFrame(halves)
    by_month.to_csv(OUT / "baseline_by_month.csv")
    halves.to_csv(OUT / "baseline_by_period.csv", index=False)
    print(f"\n=== Stability: {base}, sector excess ===")
    print(by_month.to_string(float_format=f4))
    print(halves.to_string(index=False, float_format=f4))

    print(f"\n=== Probability of backtest overfitting (CSCV, {PBO_BLOCKS} blocks, {len(pbo_rows)} splits) ===")
    print(f"PBO = {prob:.3f}")
    print(f"In-sample-best rule's mean OOS rank percentile: {pbo_rows['os_rank_pct'].mean():.3f}")
    print(f"Mean IS excess of chosen rule {pbo_rows['is_mean'].mean():.4f} vs its OOS {pbo_rows['os_mean'].mean():.4f}"
          f" (median rule OOS {pbo_rows['os_median_rule'].mean():.4f})")


if __name__ == "__main__":
    main()
