"""Research layer: single-signal tests, IC decay, Fama-MacBeth, walk-forward combinations,
long-short portfolio with costs, and attribution against FF5 + momentum.

Run after build_data.py and build_fundamentals.py:
    python scripts/run_research.py
Writes reports/research_results.md and reports/research_*.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from factorlab import paths, preprocess, research, sectors

CRSP = paths.SOURCE == "crsp"
HOLDOUT = "--holdout" in sys.argv  # CRSP only: score Jan 2018 on, once (PREREGISTRATION.md)
# Free data: fundamentals (XBRL) usable through 2010. CRSP: pre-registered exploration sample.
START = "1990-01-31" if CRSP else "2011-01-31"
OOS_START = "1995-01-31" if CRSP else "2014-01-31"  # first month scored by a fitted model
EXPLORE_END = "2017-11-30"  # last formation month whose return (Dec 2017) is in exploration
EVAL_START = "2017-12-31" if (CRSP and HOLDOUT) else None
COST_BPS = 10               # one-way cost per unit traded; S&P 500 large caps
COSTS = (0, 5, 10, 20)

LABELS = {
    "mom_12_1": "Momentum 12-1", "rev_1m": "1-month reversal", "vol_12m": "Low volatility",
    "book_to_market": "Book-to-market", "earnings_yield": "Earnings yield",
    "op_prof": "Operating profitability", "gross_prof": "Gross profitability",
    "asset_growth": "Low asset growth", "size": "Small size",
}


def main() -> None:
    uni = pd.read_parquet(paths.UNIVERSE)
    monthly = pd.read_parquet(paths.MONTHLY)
    ff = pd.read_parquet(paths.FRENCH)

    print("1/6 sectors (SIC -> Fama-French 12)")
    if CRSP:
        sec = pd.read_csv(sectors.SECTORS, dtype={"ticker": str})
    else:
        sec = sectors.build(uni.loc[uni["month_end"] >= "2009-01-31", "ticker"].unique().tolist())
    print("  " + ", ".join(f"{k} {v}" for k, v in sec["sector"].value_counts().items()))

    print("2/6 signal panel")
    raw = preprocess.raw_panel(monthly, uni)
    raw = raw[raw.index.get_level_values("month_end") >= START]
    if CRSP and not HOLDOUT:
        # Exploration never sees holdout data, not even as training rows.
        raw = raw[raw.index.get_level_values("month_end") <= EXPLORE_END]
    panel = preprocess.clean_panel(raw, sec, neutralize=True)
    panel_raw = preprocess.clean_panel(raw, sec, neutralize=False)
    fwd = preprocess.forward_returns(monthly).reindex(panel.index)
    f1 = fwd["fwd_1"]

    def window(x, default_start):
        start = EVAL_START or default_start
        return x[x.index.get_level_values("month_end") >= pd.Timestamp(start)]

    print("3/6 single signals, IC decay, Fama-MacBeth")
    single = []
    for c in panel.columns:
        sig = window(panel[c], START)
        ic = research.monthly_rank_ic(sig, f1)
        ic_mean, ic_t = research.newey_west_t(ic)
        ls = research.long_short(sig, f1, cost_bps=COST_BPS)
        g, n = research.perf(ls["gross"]), research.perf(ls["net"])
        single.append({"signal": LABELS[c], "mean_ic": ic_mean, "ic_nw_t": ic_t,
                       "gross_ann": g["ann_return"], "gross_t": g["nw_t"],
                       "net_ann": n["ann_return"], "net_sharpe": n["sharpe"],
                       "turnover": ls["turnover"].mean() / 2})
    single = pd.DataFrame(single)
    decay = research.ic_decay(window(panel, START), fwd).rename(index=LABELS)
    fm = research.fama_macbeth(window(panel, START), f1).rename(index=LABELS)
    print(single.round(3).to_string(index=False))
    print(fm.round(3).to_string())

    print("4/6 combinations (walk-forward)")
    combos = {
        "Equal weight": research.combo_equal(panel),
        "IC-weighted (36m trailing)": research.combo_ic_weighted(panel, f1),
        "Ridge (expanding, refit yearly)": research.combo_model(
            panel, f1, lambda: Ridge(alpha=100.0), "ridge", OOS_START),
        "LightGBM (expanding, refit yearly)": research.combo_model(
            panel, f1, lambda: LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15,
                                            min_child_samples=200, subsample=0.8, subsample_freq=1,
                                            colsample_bytree=0.8, random_state=0, verbose=-1),
            "lgbm", OOS_START),
        "Equal weight, not sector-neutral": research.combo_equal(panel_raw),
    }

    print("5/6 portfolios, costs, attribution")
    combo_rows, cost_rows, attr_rows, curves = [], [], [], {}
    for name, score in combos.items():
        score = window(score, OOS_START)
        ic = research.monthly_rank_ic(score, f1)
        ic_mean, ic_t = research.newey_west_t(ic)
        ls = research.long_short(score, f1, cost_bps=COST_BPS)
        g, n = research.perf(ls["gross"]), research.perf(ls["net"])
        combo_rows.append({"method": name, "mean_ic": ic_mean, "ic_nw_t": ic_t,
                           "gross_ann": g["ann_return"], "net_ann": n["ann_return"],
                           "net_sharpe": n["sharpe"], "net_nw_t": n["nw_t"], "max_dd": n["max_dd"],
                           "turnover": ls["turnover"].mean() / 2, "months": n["months"]})
        cost_rows.append({"method": name, **{
            f"{c} bps": research.perf(ls["gross"] - ls["turnover"] * c / 1e4)["sharpe"] for c in COSTS}})
        attr_rows.append({"method": name, **research.attribution(ls["net"], ff)})
        curves[name] = ls["net"]
    combo_df, cost_df, attr_df = map(pd.DataFrame, (combo_rows, cost_rows, attr_rows))
    print(combo_df.round(3).to_string(index=False))
    print(attr_df.round(3).to_string(index=False))

    print("6/6 report")
    chart(curves, decay)
    report(single, decay, fm, combo_df, cost_df, attr_df, sec)


def chart(curves, decay) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, r in curves.items():
        (1 + r).cumprod().plot(ax=axes[0], label=name)
    axes[0].axhline(1, color="grey", lw=0.8)
    axes[0].set_title(f"Long-short top vs bottom quintile, net of {COST_BPS} bps (out-of-sample)")
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("")
    im = axes[1].imshow(decay.to_numpy(), cmap="RdBu", vmin=-0.04, vmax=0.04, aspect="auto")
    axes[1].set_xticks(range(decay.shape[1]), decay.columns, fontsize=8)
    axes[1].set_yticks(range(decay.shape[0]), decay.index, fontsize=8)
    for i in range(decay.shape[0]):
        for j in range(decay.shape[1]):
            axes[1].text(j, i, f"{decay.iat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    axes[1].set_title("Mean rank IC by horizon (return in month t+h)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.REPORTS / ("research_results_holdout.png" if EVAL_START else "research_results.png"), dpi=120)


def md(df: pd.DataFrame, fmt: dict) -> list[str]:
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return L


def report(single, decay, fm, combo_df, cost_df, attr_df, sec) -> None:
    pct, f2, f3 = "{:+.1%}", "{:.2f}", "{:.3f}"
    n_variants = len(single) + len(combo_df)
    L = [
        "# Research results",
        "",
        "Generated by `scripts/run_research.py`. Universe: point-in-time S&P 500 members. Signals are "
        "winsorized (1/99%), z-scored and sector-neutralized (Fama-French 12 industries from SEC SIC "
        "codes) each month. Portfolios: equal-weighted long top quintile, short bottom quintile, "
        f"monthly rebalance, dollar neutral. Costs: {COST_BPS} bps per unit traded unless stated. "
        "t-stats are Newey-West (6 lags).",
        "",
        "**Caveats.** Free data (yfinance + SEC XBRL) misses many stocks that were later delisted or "
        "acquired, so results lean toward survivors (see data_coverage.md, fundamentals_coverage.md). "
        f"Variants tested in this report: {n_variants} (9 single signals, {len(combo_df)} combinations); "
        "none were dropped. Model hyperparameters were fixed in advance, not tuned on this data.",
        "",
        f"## Single signals, {START[:7]} on",
        "",
    ]
    L += md(single, {"mean_ic": f3, "ic_nw_t": f2, "gross_ann": pct, "gross_t": f2, "net_ann": pct,
                     "net_sharpe": f2, "turnover": "{:.0%}"})
    L += ["", "Turnover = average monthly share of the book replaced.", "",
          "## IC decay", "", "Mean monthly rank IC between the signal at t and the return in month t+h.", ""]
    d = decay.reset_index().rename(columns={"index": "signal"})
    L += md(d, {c: f3 for c in decay.columns})
    L += ["", "## Fama-MacBeth (all signals together)", "",
          "Monthly cross-sectional regression of next-month return on all nine cleaned signals; "
          "slope = return in basis points per one standard deviation of the signal, holding the others fixed.", ""]
    f = fm.reset_index().rename(columns={"index": "signal"})[["signal", "mean_slope_bp", "nw_t", "months"]]
    L += md(f, {"mean_slope_bp": "{:+.1f}", "nw_t": f2, "months": "{:.0f}"})
    L += ["", f"## Combined signals, out-of-sample {OOS_START[:7]} on", "",
          "Ridge and LightGBM are refit every January on all earlier data whose outcome was already "
          "known, then score the following 12 months. IC weights use only past ICs. Equal weight has "
          "no fitted parameters.", ""]
    L += md(combo_df, {"mean_ic": f3, "ic_nw_t": f2, "gross_ann": pct, "net_ann": pct,
                       "net_sharpe": f2, "net_nw_t": f2, "max_dd": pct, "turnover": "{:.0%}",
                       "months": "{:.0f}"})
    L += ["", "### Net Sharpe by cost assumption", ""]
    L += md(cost_df, {c: f2 for c in cost_df.columns if c != "method"})
    L += ["", "### Attribution: net returns on Fama-French 5 + momentum", ""]
    a = attr_df[["method", "alpha_ann", "alpha_nw_t", "r2", "b_Mkt-RF", "b_SMB", "b_HML", "b_RMW",
                 "b_CMA", "b_UMD"]]
    L += md(a, {"alpha_ann": pct, "alpha_nw_t": f2, "r2": f2, **{c: f2 for c in a.columns if c.startswith("b_")}})
    L += ["", "## Sector counts (tickers in the index since 2009)", "",
          ", ".join(f"{k}: {v}" for k, v in sec["sector"].value_counts().items()), "",
          "![results](research_results.png)", ""]
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    name = "research_results_holdout.md" if EVAL_START else "research_results.md"
    (paths.REPORTS / name).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
