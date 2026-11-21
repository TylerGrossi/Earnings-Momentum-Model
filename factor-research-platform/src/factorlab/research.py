"""Research layer: IC decay, Fama-MacBeth, walk-forward signal combination, portfolio with
costs, and factor attribution.

Timing convention: signals are formed at month end t; fwd_1 is the return over month t+1.
A model used at month end t may only be trained on rows s with s + 1 <= t (their fwd_1 is known
by t). Every combination method below follows that rule.
"""
import numpy as np
import pandas as pd


# ---------- statistics ----------

def newey_west_t(x: pd.Series, lags: int = 6) -> tuple[float, float]:
    """Mean and Newey-West (Bartlett kernel) t-stat of a time series."""
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    e = x - x.mean()
    var = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean(), x.mean() / np.sqrt(var / n)


def monthly_rank_ic(score: pd.Series, fwd: pd.Series, min_names: int = 50) -> pd.Series:
    df = pd.concat([score.rename("s"), fwd.rename("r")], axis=1).dropna()
    def one(g):
        return g["s"].rank().corr(g["r"].rank()) if len(g) >= min_names else np.nan
    return df.groupby(level="month_end").apply(one).dropna()


def ic_decay(panel: pd.DataFrame, fwd: pd.DataFrame, horizons=(1, 2, 3, 6, 9, 12)) -> pd.DataFrame:
    rows = {}
    for c in panel.columns:
        rows[c] = {h: monthly_rank_ic(panel[c], fwd[f"fwd_{h}"]).mean() for h in horizons}
    return pd.DataFrame(rows).T.rename(columns=lambda h: f"IC t+{h}")


def fama_macbeth(panel: pd.DataFrame, fwd1: pd.Series, min_names: int = 100) -> pd.DataFrame:
    """Monthly cross-sectional OLS of next-month return on all cleaned signals at once (missing
    signals set to 0, the sector-neutral mean). Reports the time-series mean slope and its
    Newey-West t-stat: does each signal add information given the others?"""
    X_all = panel.fillna(0.0)
    df = X_all.join(fwd1.rename("y"), how="inner").dropna(subset=["y"])
    slopes = []
    for m, g in df.groupby(level="month_end"):
        if len(g) < min_names:
            continue
        X = np.column_stack([np.ones(len(g)), g[panel.columns].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
        slopes.append(pd.Series(beta[1:], index=panel.columns, name=m))
    S = pd.DataFrame(slopes)
    out = {c: dict(zip(["mean_slope", "nw_t"], newey_west_t(S[c]))) for c in S.columns}
    res = pd.DataFrame(out).T
    res["mean_slope_bp"] = res["mean_slope"] * 1e4
    res["months"] = len(S)
    return res


# ---------- signal combination (walk-forward) ----------

def combo_equal(panel: pd.DataFrame, min_signals: int = 5) -> pd.Series:
    """Average of available cleaned signals. Nothing is fitted, so it can't overfit."""
    n = panel.notna().sum(axis=1)
    return panel.mean(axis=1).where(n >= min_signals).rename("equal_weight")


def combo_ic_weighted(panel: pd.DataFrame, fwd1: pd.Series, window: int = 36,
                      min_signals: int = 5) -> pd.Series:
    """Weights = trailing mean IC of each signal over the last `window` months whose outcome is
    known at t (ICs formed at t-1 and earlier), normalized by the sum of absolute weights."""
    ics = pd.DataFrame({c: monthly_rank_ic(panel[c], fwd1) for c in panel.columns})
    ics = ics.asfreq("ME")
    # IC formed at s is known at s+1, so at t use ICs up to t-1: shift by one month.
    w = ics.shift(1).rolling(window, min_periods=24).mean()
    w = w.div(w.abs().sum(axis=1), axis=0)
    months = panel.index.get_level_values("month_end")
    W = w.reindex(months).to_numpy()
    filled = panel.fillna(0.0).to_numpy()
    score = np.nansum(filled * W, axis=1)
    valid = ~np.isnan(W).all(axis=1) & (panel.notna().sum(axis=1).to_numpy() >= min_signals)
    return pd.Series(np.where(valid, score, np.nan), index=panel.index, name="ic_weighted")


def _rank_target(fwd1: pd.Series) -> pd.Series:
    """Cross-sectional rank of next-month return, scaled to [-0.5, 0.5]. Less noisy than raw
    returns and immune to single-stock outliers."""
    return fwd1.groupby(level="month_end").rank(pct=True) - 0.5


def combo_model(panel: pd.DataFrame, fwd1: pd.Series, model_factory, name: str,
                first_test: str = "2014-01-31", min_signals: int = 5) -> pd.Series:
    """Expanding-window model refit every January. For test year Y, trains on all rows with
    month_end <= Dec 31 of Y-1 minus one month (so every training label is known before the
    first prediction), then scores every month of Y."""
    X = panel.fillna(0.0)
    y = _rank_target(fwd1).reindex(panel.index)
    ok = panel.notna().sum(axis=1) >= min_signals
    months = panel.index.get_level_values("month_end")
    preds = []
    for year in range(pd.Timestamp(first_test).year, months.max().year + 1):
        test_start = pd.Timestamp(f"{year}-01-31")
        train_end = test_start - pd.offsets.MonthEnd(2)  # label for train_end is month t-1 return
        tr = (months <= train_end) & ok & y.notna()
        te = (months >= test_start) & (months <= pd.Timestamp(f"{year}-12-31")) & ok
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        m = model_factory()
        m.fit(X[tr].to_numpy(), y[tr].to_numpy())
        preds.append(pd.Series(m.predict(X[te].to_numpy()), index=panel.index[te]))
    return pd.concat(preds).rename(name)


# ---------- portfolio ----------

def long_short(score: pd.Series, fwd1: pd.Series, q: float = 0.2,
               cost_bps: float = 10.0) -> pd.DataFrame:
    """Equal-weighted long top q / short bottom q of the score, rebalanced monthly, dollar
    neutral (+1 long, -1 short). Turnover = sum |w_t - w_{t-1}|; cost = turnover x cost_bps per
    unit traded. Returns are labelled with the month they are earned (t+1)."""
    df = pd.concat([score.rename("s"), fwd1.rename("r")], axis=1).dropna(subset=["s"])
    prev_w = pd.Series(dtype=float)
    rows = []
    for m, g in df.groupby(level="month_end"):
        g = g.droplevel("month_end")
        g = g[g["r"].notna()]  # stocks without a next-month return can't be held (data gap)
        if len(g) < 50:
            continue
        lo, hi = g["s"].quantile([q, 1 - q])
        longs, shorts = g.index[g["s"] >= hi], g.index[g["s"] <= lo]
        w = pd.concat([pd.Series(1 / len(longs), longs), pd.Series(-1 / len(shorts), shorts)])
        gross = float((w * g["r"].reindex(w.index)).sum())
        turnover = float(w.sub(prev_w, fill_value=0).abs().sum())
        rows.append({"month_end": m + pd.offsets.MonthEnd(1), "gross": gross, "turnover": turnover,
                     "net": gross - turnover * cost_bps / 1e4,
                     "long": float(g["r"].reindex(longs).mean()),
                     "short": float(g["r"].reindex(shorts).mean())})
        prev_w = w
    out = pd.DataFrame(rows).set_index("month_end")
    out.iloc[0, out.columns.get_loc("net")] = out["gross"].iloc[0]  # no prior book to trade out of
    return out


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    wealth = (1 + r).cumprod()
    mean, t = newey_west_t(r)
    return {"months": len(r), "ann_return": (1 + r).prod() ** (12 / len(r)) - 1,
            "ann_vol": r.std() * np.sqrt(12), "sharpe": r.mean() / r.std() * np.sqrt(12),
            "nw_t": t, "max_dd": (wealth / wealth.cummax() - 1).min(), "hit_rate": (r > 0).mean()}


def attribution(r: pd.Series, ff: pd.DataFrame,
                factors=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD")) -> dict:
    """Regress long-short returns on FF5 + momentum. A dollar-neutral portfolio needs no
    risk-free adjustment. Alpha t-stat uses Newey-West on the residual-plus-alpha series."""
    j = pd.concat([r.rename("r"), ff[list(factors)]], axis=1).dropna()
    X = np.column_stack([np.ones(len(j)), j[list(factors)].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, j["r"].to_numpy(), rcond=None)
    resid = j["r"].to_numpy() - X @ beta
    _, t_alpha = newey_west_t(pd.Series(resid + beta[0]))
    r2 = 1 - resid.var() / j["r"].var()
    return {"alpha_ann": beta[0] * 12, "alpha_nw_t": t_alpha, "r2": r2,
            **{f"b_{f}": b for f, b in zip(factors, beta[1:])}, "months": len(j)}
