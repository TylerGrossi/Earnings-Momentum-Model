"""Time-series momentum, exactly as pre-specified in README.md. Nothing here is tuned.

Timing: signals, volatilities and weights are formed with data up to the last trading day of
month t and earn the asset returns of month t+1.
"""
import numpy as np
import pandas as pd

from .data import ASSET_CLASS, ASSETS

ASSET_VOL_TARGET = 0.40
PORT_VOL_TARGET = 0.10
EWMA_COM = 60
VOL_ANNUAL_DAYS = 261
MIN_VOL_DAYS = 120
PORT_VOL_LOOKBACK = 252
ASSET_CAP = 1.0
GROSS_CAP = 4.0
COST_BPS = 10.0
BORROW_FEE = 0.005


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Compounded calendar-month returns; NaN for a month with any missing day."""
    grp = daily.groupby(daily.index.to_period("M"))
    out = grp.apply(lambda g: (1 + g).prod(min_count=1) - 1)
    full = grp.apply(lambda g: g.notna().all())
    out = out.where(full)
    out.index = out.index.to_timestamp("M")
    # Drop the current month if the data stops before its last business day (month-to-date
    # is not a monthly return).
    last_day = daily.index.max()
    if last_day < last_day + pd.offsets.BMonthEnd(0):
        out = out[out.index < last_day + pd.offsets.MonthEnd(0)]
    return out


def build_inputs(px: pd.DataFrame, rf_d: pd.Series) -> dict:
    r = px[ASSETS].pct_change(fill_method=None)
    r = r[r.index >= "1993-02-01"]
    rf_d = rf_d.reindex(r.index).ffill()
    ex_d = r.sub(rf_d, axis=0)
    R = monthly_returns(r)
    RF = monthly_returns(rf_d.to_frame())["rf"]
    EX = R.sub(RF, axis=0)
    var = (ex_d ** 2).ewm(com=EWMA_COM, min_periods=MIN_VOL_DAYS).mean()
    sigma_d = np.sqrt(var * VOL_ANNUAL_DAYS)
    month_last_day = r.groupby(r.index.to_period("M")).apply(lambda g: g.index.max())
    sigma = sigma_d.loc[month_last_day.values]
    sigma.index = month_last_day.index.to_timestamp("M")
    return {"ex_d": ex_d, "R": R, "RF": RF, "EX": EX, "sigma": sigma,
            "month_last_day": pd.Series(month_last_day.values, index=sigma.index)}


def _cum(EX: pd.DataFrame, k: int) -> pd.DataFrame:
    return (1 + EX).rolling(k, min_periods=k).apply(np.prod, raw=True) - 1


def signals(EX: pd.DataFrame) -> dict:
    s12 = np.sign(_cum(EX, 12))
    blend = (np.sign(_cum(EX, 1)) + np.sign(_cum(EX, 3)) + s12) / 3
    return {"TSMOM 12m (primary)": s12, "TSMOM 1/3/12m blend (secondary)": blend}


def weights(signal: pd.DataFrame, inp: dict) -> pd.DataFrame:
    sigma, ex_d, last = inp["sigma"], inp["ex_d"], inp["month_last_day"]
    rows = {}
    for m in signal.index:
        s, v = signal.loc[m], sigma.loc[m] if m in sigma.index else None
        if v is None:
            continue
        active = s.notna() & v.notna() & (v > 0)
        if active.sum() == 0:
            continue
        w = (s[active] * ASSET_VOL_TARGET / v[active]) / active.sum()
        hist = ex_d.loc[:last[m]].iloc[-PORT_VOL_LOOKBACK:][w.index].fillna(0.0)
        port_vol = float((hist @ w).std() * np.sqrt(252))
        if not np.isfinite(port_vol) or port_vol <= 0:
            continue
        w = (w * PORT_VOL_TARGET / port_vol).clip(-ASSET_CAP, ASSET_CAP)
        gross = w.abs().sum()
        if gross > GROSS_CAP:
            w = w * GROSS_CAP / gross
        rows[m] = w
    return pd.DataFrame(rows).T.reindex(columns=ASSETS).fillna(0.0)


def backtest(W: pd.DataFrame, inp: dict, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Monthly returns labelled by the month earned."""
    EX, RF = inp["EX"], inp["RF"]
    prev = pd.Series(0.0, index=W.columns)
    rows = []
    for m in W.index:
        nxt = m + pd.offsets.MonthEnd(1)
        if nxt not in EX.index:
            continue
        w = W.loc[m]
        ex = EX.loc[nxt].reindex(W.columns)
        contrib = (w * ex.fillna(0.0))
        turnover = float((w - prev).abs().sum())
        cost = turnover * cost_bps / 1e4
        borrow = float(w.clip(upper=0).abs().sum()) * BORROW_FEE / 12
        excess = float(contrib.sum()) - cost - borrow
        row = {"month": nxt, "excess": excess, "total": excess + float(RF.loc[nxt]),
               "rf": float(RF.loc[nxt]), "turnover": turnover, "cost": cost, "borrow": borrow,
               "gross": float(w.abs().sum()), "net_long": float(w.sum()),
               "n_assets": int((w != 0).sum())}
        for c in set(ASSET_CLASS.values()):
            row[f"cls_{c}"] = float(contrib[[t for t in W.columns if ASSET_CLASS[t] == c]].sum())
        rows.append(row)
        prev = w
    return pd.DataFrame(rows).set_index("month")
