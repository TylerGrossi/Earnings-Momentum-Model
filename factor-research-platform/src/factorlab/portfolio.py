"""Portfolio construction rules for Test B (pre-registered): plain quintiles, 3-month overlapping
holdings, and banding. Each builder returns {formation month: weight Series}; `run` turns weights
into returns with turnover and costs.

Weights are +1 long / -1 short (dollar neutral). At each month, stocks without a next-month
return are dropped and each side is renormalized to +1 / -1, so a missing price never counts as
a zero return.
"""
import pandas as pd


def _books(score: pd.Series, q: float = 0.2) -> dict:
    books = {}
    for m, g in score.dropna().groupby(level="month_end"):
        g = g.droplevel("month_end")
        if len(g) < 50:
            continue
        lo, hi = g.quantile([q, 1 - q])
        L, S = g.index[g >= hi], g.index[g <= lo]
        books[m] = pd.concat([pd.Series(1 / len(L), L), pd.Series(-1 / len(S), S)])
    return books


def quintile(score: pd.Series) -> dict:
    return _books(score)


def overlapping(score: pd.Series, hold: int = 3) -> dict:
    """Average of the last `hold` monthly books; each book is held `hold` months."""
    books = _books(score)
    months = sorted(books)
    out = {}
    for i, m in enumerate(months):
        recent = [books[x] for x in months[max(0, i - hold + 1): i + 1]]
        out[m] = pd.concat(recent, axis=1).fillna(0).sum(axis=1) / len(recent)
    return out


def banded(score: pd.Series, enter: float = 0.2, exit_: float = 0.4) -> dict:
    """Enter the long side in the top `enter` share; stay until falling out of the top `exit_`.
    Mirror rule for shorts. Sides are equal-weighted."""
    out, longs, shorts = {}, set(), set()
    for m, g in score.dropna().groupby(level="month_end"):
        g = g.droplevel("month_end")
        if len(g) < 50:
            continue
        pct = g.rank(pct=True)
        longs = {t for t in longs if t in pct.index and pct[t] >= 1 - exit_} | set(pct.index[pct >= 1 - enter])
        shorts = {t for t in shorts if t in pct.index and pct[t] <= exit_} | set(pct.index[pct <= enter])
        L, S = sorted(longs), sorted(shorts)
        out[m] = pd.concat([pd.Series(1 / len(L), L), pd.Series(-1 / len(S), S)])
    return out


def run(weights: dict, fwd1: pd.Series, cost_bps: float = 10.0) -> pd.DataFrame:
    r_all = fwd1.dropna()
    prev = pd.Series(dtype=float)
    rows = []
    for m in sorted(weights):
        w = weights[m]
        try:
            r = r_all.xs(m, level="month_end")
        except KeyError:
            continue
        w = w[w.index.isin(r.index)]
        pos, neg = w[w > 0], w[w < 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        w = pd.concat([pos / pos.sum(), -neg / neg.sum()])
        gross = float((w * r.reindex(w.index)).sum())
        turnover = float(w.sub(prev, fill_value=0).abs().sum())
        rows.append({"month_end": m + pd.offsets.MonthEnd(1), "gross": gross, "turnover": turnover,
                     "net": gross - (turnover * cost_bps / 1e4 if len(prev) else 0.0)})
        prev = w
    return pd.DataFrame(rows).set_index("month_end")
