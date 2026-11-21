"""
Session 3: does the win-probability classifier predict anything out of sample?

The production model (win_probability_predictor.py) is a random forest on P/E, beta, market cap,
sector and earnings timing, with target "tracker 3D Return > 0". Every nightly run retrains it
on ALL rows and rewrites the Win Probability column for every row, including each row's own
outcome. So the stored column is a set of in-sample fitted values, not predictions. This script:

  1. Measures the stored column anyway, to show how much in-sample fitting inflates it.
  2. Rebuilds the model walk-forward: for each entry week after an 8-week warm-up, train only on
     trades whose outcome was known before that week (entry at least 8 calendar days earlier,
     so a 5-day hold has closed), then predict that week. Production features and
     hyperparameters, plus gradient boosting (the app's other option) and a logistic regression
     baseline.
  3. Scores the out-of-sample predictions: AUC, Brier score vs the training win rate, a
     calibration table, and whether the top half by predicted probability beats the bottom
     half on the strategy's real outcome (5-day return in excess of the sector ETF).

Targets: "prod" = production target (tracker 3D Return > 0; for BMO trades that's a 4-day hold,
see report 01) and "xs5" = recomputed 5-day sector-excess return > 0.

Run after excess_returns.py.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from win_probability_predictor import (  # noqa: E402  production feature helpers
    categorize_beta, categorize_market_cap, categorize_pe, parse_market_cap)

WARMUP_WEEKS = 8
PURGE_DAYS = 8
BOOT_N = 5000
SEED = 11
NUM = ["pe", "beta", "mcap"]
CAT = ["pe_cat", "beta_cat", "cap_cat", "Sector", "Timing"]


def features() -> pd.DataFrame:
    """Production features, built row-for-row like prepare_features() but keeping dates."""
    tr = pd.read_csv(ROOT / "returns_tracker.csv")
    tr = tr[tr["Date Check"] != "DATE PASSED"].copy()
    tr["Earnings Date"] = pd.to_datetime(tr["Earnings Date"].astype(str).str[:10]).dt.date
    pe_col = "Forward P/E"
    f = pd.DataFrame({
        "Ticker": tr["Ticker"], "Earnings Date": tr["Earnings Date"],
        "pe": pd.to_numeric(tr[pe_col], errors="coerce"),
        "beta": pd.to_numeric(tr["Beta"], errors="coerce"),
        "mcap": tr["Market Cap"].apply(parse_market_cap),
        "pe_cat": tr[pe_col].apply(categorize_pe),
        "beta_cat": tr["Beta"].apply(categorize_beta),
        "cap_cat": tr["Market Cap"].apply(categorize_market_cap),
        "Sector_f": tr["Sector"].fillna("Unknown"),
        "ret3_tracker": tr["3D Return"],
        "stored_prob": tr["Win Probability"],
    })
    f.loc[f["pe"].isna() | (f["pe"] < 0), "pe"] = -1
    return f


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Integer codes, as production uses (category list only; no outcome information)."""
    out = df.copy()
    for c in CAT:
        out[c + "_code"] = out[c].astype("category").cat.codes
    return out


def models():
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    lr = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), [0, 1, 2]),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), [3, 4])]),
        LogisticRegression(C=0.5, max_iter=2000))
    return {"random forest (production)": ("codes", rf), "gradient boosting": ("codes", gb),
            "logistic regression": ("raw", lr)}


def design(df: pd.DataFrame, kind: str):
    if kind == "codes":
        return df[NUM + [c + "_code" for c in CAT]].to_numpy(float)
    return df[["pe", "beta", "mcap_log", "Sector", "Timing"]].to_numpy(object)


def week_bootstrap_auc(df: pd.DataFrame, prob: str, y: str, rng) -> tuple[float, float]:
    groups = [g for _, g in df.groupby("week")]
    vals = []
    for _ in range(BOOT_N):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if s[y].nunique() == 2:
            vals.append(roc_auc_score(s[y], s[prob]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main() -> None:
    trades = pd.read_csv(OUT / "excess_returns_trades.csv", parse_dates=["Entry Date"])
    trades["Earnings Date"] = pd.to_datetime(trades["Earnings Date"]).dt.date
    f = features().rename(columns={"Sector_f": "Sector_tr"})
    df = trades.merge(f, on=["Ticker", "Earnings Date"], how="inner")
    df["Sector"] = df["Sector_tr"]
    df = df.dropna(subset=["beta", "mcap", "xs_sector_5d", "ret3_tracker"]).copy()
    df["mcap_log"] = np.log(df["mcap"])
    df["y_prod"] = (df["ret3_tracker"] > 0).astype(int)
    df["y_xs5"] = (df["xs_sector_5d"] > 0).astype(int)
    df["week"] = df["Entry Date"].dt.to_period("W-FRI")
    df = label_encode(df).sort_values("Entry Date").reset_index(drop=True)
    print(f"Rows with features and outcomes: {len(df)}; weeks: {df['week'].nunique()}")

    rng = np.random.default_rng(SEED)
    rows = []

    # 1. The stored column (in-sample fitted values).
    for y in ["y_prod", "y_xs5"]:
        d = df.dropna(subset=["stored_prob"])
        rows.append({"model": "stored Win Probability (in-sample, leaky)", "target": y, "n": len(d),
                     "auc": roc_auc_score(d[y], d["stored_prob"]),
                     "brier": brier_score_loss(d[y], d["stored_prob"]),
                     "brier_base": brier_score_loss(d[y], np.full(len(d), d[y].mean()))})

    # 2. Walk-forward.
    weeks = sorted(df["week"].unique())
    preds = []
    for k, wk in enumerate(weeks):
        if k < WARMUP_WEEKS:
            continue
        test = df[df["week"] == wk]
        cutoff = test["Entry Date"].min() - pd.Timedelta(days=PURGE_DAYS)
        train = df[df["Entry Date"] <= cutoff]
        p = test[["Ticker", "Entry Date", "week", "y_prod", "y_xs5", "xs_sector_5d", "cost"]].copy()
        for y in ["y_prod", "y_xs5"]:
            p[f"base_{y}"] = train[y].mean()
            for name, (kind, m) in models().items():
                m.fit(design(train, kind), train[y])
                p[f"{name}|{y}"] = m.predict_proba(design(test, kind))[:, 1]
        preds.append(p)
    pr = pd.concat(preds, ignore_index=True)

    spreads = []
    for y in ["y_prod", "y_xs5"]:
        for name in models():
            col = f"{name}|{y}"
            lo, hi = week_bootstrap_auc(pr, col, y, rng)
            rows.append({"model": name + " (walk-forward)", "target": y, "n": len(pr),
                         "auc": roc_auc_score(pr[y], pr[col]), "auc_ci_lo": lo, "auc_ci_hi": hi,
                         "brier": brier_score_loss(pr[y], pr[col]),
                         "brier_base": brier_score_loss(pr[y], pr[f"base_{y}"])})
            # Does ranking by the prediction pick better trades? Split each week at its median.
            med = pr.groupby("week")[col].transform("median")
            top = pr[col] > med
            wk_spread = (pr[top].groupby("week")["xs_sector_5d"].mean()
                         - pr[~top].groupby("week")["xs_sector_5d"].mean()).dropna()
            spreads.append({"model": name, "target": y,
                            "top_half_mean_xs": pr.loc[top, "xs_sector_5d"].mean(),
                            "bottom_half_mean_xs": pr.loc[~top, "xs_sector_5d"].mean(),
                            "weekly_spread_mean": wk_spread.mean(),
                            "weekly_spread_t": wk_spread.mean() / (wk_spread.std(ddof=1) / np.sqrt(len(wk_spread))),
                            "weeks": len(wk_spread),
                            "spearman_prob_vs_xs": pr[col].corr(pr["xs_sector_5d"], method="spearman")})

    res = pd.DataFrame(rows)
    res["brier_skill"] = 1 - res["brier"] / res["brier_base"]
    spreads = pd.DataFrame(spreads)

    # Calibration of the production model on the production target.
    col = "random forest (production)|y_prod"
    pr["bin"] = pd.cut(pr[col], [0, .4, .5, .6, .7, 1.0])
    calib = pr.groupby("bin", observed=True).agg(n=("y_prod", "size"), mean_pred=(col, "mean"),
                                                 actual=("y_prod", "mean"))

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "classifier_metrics.csv", index=False)
    spreads.to_csv(OUT / "classifier_ranking.csv", index=False)
    calib.to_csv(OUT / "classifier_calibration.csv")
    pr.drop(columns=["bin"]).to_csv(OUT / "classifier_walk_forward_predictions.csv", index=False)

    pd.set_option("display.width", 220)
    f3 = "{:.3f}".format
    print(f"\nTest weeks: {pr['week'].nunique()}, test trades: {len(pr)}")
    print("\n=== Discrimination and calibration ===")
    print(res.to_string(index=False, float_format=f3))
    print("\n=== Ranking: top vs bottom half by predicted probability (5-day sector excess) ===")
    print(spreads.to_string(index=False, float_format="{:.4f}".format))
    print("\n=== Calibration: production RF, production target ===")
    print(calib.to_string(float_format=f3))


if __name__ == "__main__":
    main()
