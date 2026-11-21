# 03: Out-of-sample test of the win-probability classifier

*Run 2026-11-20 with `python research/classifier_eval.py` (after `excess_returns.py`). Outputs are
in `research/output/`.*

## Question

The app shows a "Win Probability" for each trade. Does the classifier behind it predict
outcomes it hasn't seen?

## The production setup, and why the stored number can't answer this

- **The model** (`win_probability_predictor.py`): a random forest (100 trees, max depth 10).
  Features are forward P/E, beta and market cap (each raw and bucketed), sector, and
  BMO/AMC timing. The target is the tracker's `3D Return > 0`.
- **The leak:** every nightly run of `Earnings Momentum Model.py` retrains the model on *all*
  rows and rewrites `Win Probability` for every row. Each stored value therefore comes from a
  model that was trained on that trade's own outcome. These are in-sample fitted values, not
  predictions.
- **The target is skewed by the timing bug** (report 01). For BMO trades the tracker's 3-day
  return is really a 4-day hold.
- **The app's own accuracy figure** comes from a random 80/20 split. That mixes future trades
  into training, so it isn't a time-ordered test either.

## Method

- **Walk-forward:** for each entry week after an 8-week warm-up (16 test weeks, 382 trades),
  train only on trades entered at least 8 calendar days before the week starts, so their
  holding period has closed. Then predict the week's trades.
- **Models:** the production random forest with its exact features and hyperparameters, the
  app's gradient-boosting option, and a regularized logistic regression as a simple baseline.
- **Targets:** the production target, and the strategy's real outcome (5-day return in
  excess of the sector ETF > 0).
- **Scoring:**
  - AUC, with a 95% CI from a bootstrap that resamples whole weeks.
  - Brier score against a baseline that always predicts the training win rate.
  - Calibration (whether, say, trades given 70% actually win about 70% of the time).
  - Whether the top half of each week by predicted probability beats the bottom half.

## Results

| Model | Target | AUC | 95% CI | Brier skill vs base rate |
|---|---|---|---|---|
| Stored `Win Probability` (in-sample) | production | **0.937** | – | +0.387 |
| Stored `Win Probability` (in-sample) | 5-day sector excess | 0.809 | – | +0.235 |
| Random forest (production), walk-forward | production | **0.462** | 0.401–0.511 | −0.103 |
| Gradient boosting, walk-forward | production | 0.444 | 0.377–0.501 | −0.438 |
| Logistic regression, walk-forward | production | 0.534 | 0.475–0.589 | −0.029 |
| Random forest (production), walk-forward | 5-day sector excess | 0.495 | 0.438–0.551 | −0.077 |

**Ranking trades (5-day sector excess, top half vs bottom half of each week):**

| Model | Top half | Bottom half | Weekly spread t |
|---|---|---|---|
| Random forest (production) | +0.04% | +2.17% | −0.25 |
| Gradient boosting | −0.73% | +2.91% | −1.73 |
| Logistic regression | +0.55% | +1.69% | −1.06 |

**Calibration** (production random forest, production target):

| Predicted probability | Trades | Mean predicted | Actual win rate |
|---|---|---|---|
| 0–40% | 23 | 31.7% | 73.9% |
| 40–50% | 58 | 46.0% | 58.6% |
| 50–60% | 100 | 55.8% | 57.0% |
| 60–70% | 112 | 65.5% | 48.2% |
| 70–100% | 89 | 76.0% | 58.4% |

## Conclusion

- **The classifier has no out-of-sample skill.**
  - Walk-forward AUC is 0.46–0.53 for every model, and every confidence interval includes 0.5.
  - Every model's Brier score is worse than always predicting the base rate.
  - Ranking by predicted probability did *not* pick better trades. If anything the top half
    did worse, though not significantly.
  - Calibration doesn't follow the predictions: trades rated under 40% won 74% of the time.
- **The stored 0.937 AUC is an artifact of training on the answers.** The model memorizes
  each row's outcome. The gap between 0.937 and 0.462 is a measure of how much the in-sample
  number is inflated.
- **The features don't carry the signal either.** The logistic baseline, which can't
  memorize, sits at 0.53 with an interval spanning 0.5. Valuation, beta, size and sector
  don't predict which screened trades beat their sector over 5 days in this sample.

## What would fix it (not done; the production model is unchanged)

1. Predict each new row with a model trained only on rows whose outcomes were known at the
   time, and never overwrite stored predictions. That makes the stored column a genuine
   forecast record.
2. Point the target at the strategy's actual outcome (sector-excess return) with the
   corrected timing.
3. Use features more likely to matter for an announcement-day reaction. Examples are
   pre-announcement run-up, implied volatility or the expected move, and the stock's own
   surprise history. Then test the model the way this report does.
