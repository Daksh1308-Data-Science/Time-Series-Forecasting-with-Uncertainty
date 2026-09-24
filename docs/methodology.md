# Methodology

How the models are built, how the intervals are produced, and how to read the
evaluation numbers.

## The series

Weekly retail demand on Fridays. The real series is the Kaggle Walmart
competition data collapsed to one total per week (`ds`, `y`, `IsHoliday`); the
demo series is generated with the same structure. Both are 3-year, 52-week
seasonal retail series with four large markdown weeks (Super Bowl, Labor Day,
Thanksgiving, Christmas).

## Models

All three produce the same table: `ds, yhat, lower_80, upper_80, lower_95,
upper_95`.

### 1. Seasonal-naive + residual bands (frequentist reference)

Point forecast: last season's value for the same week (`y[t-52]`). Interval:
the empirical quantiles of the in-sample seasonal differences
`e_t = y_t - y_{t-52}`, added to the point forecast. No sampling, no priors —
just the observed error distribution. Retail hard to beat, and a fair
frequentist yardstick for "how wide do bands have to be to be honest?".

### 2. Prophet

Piecewise-linear trend + yearly Fourier seasonality + the four Walmart holidays
as regressors (`changepoint_prior_scale=0.05`, `seasonality_prior_scale=10`,
`uncertainty_samples=1000`). Prophet's bands come from its MAP fit plus
simulated trend variation — a pragmatic, frequentist-flavoured construction.
It is the practical baseline a data scientist would ship first.

### 3. Bayesian structural model (this project's core)

```
log y_t  ~  Normal(mu_t, sigma)
mu_t = const + beta_trend * t
     + sum_k [ a_k cos(2 pi k t / 52) + b_k sin(2 pi k t / 52) ]
     + sum_h gamma_h * 1[week t is holiday h]
```

- Fourier terms capture the smooth yearly cycle; holiday indicators capture
  the markdown spikes, which are additive jumps no smooth seasonal term
  represents.
- The point forecast is the **posterior predictive median** (sales are
  right-skewed, so the median beats the mean/MAP for a central estimate).
- Intervals are **posterior predictive**: parameter uncertainty (posterior
  spread of beta and sigma) *and* observation noise (the predictive draw for a
  new week) are both included. This is the substantive difference from
  Prophet's simulation-based bands.

The Gaussian sits on `log(demand)` (`log_target=True`, the default in
`configs/default.toml`) because retail demand is multiplicative and right-skewed.
Predictive draws are exponentiated back to units; a monotone transform leaves
the median and every quantile **exact**, so no smearing correction is needed.

Measured effect of the log specification on the real Walmart series:

| Bayesian variant | MAE | Winkler @95% ↓ | 95% width |
|---|---:|---:|---:|
| level-space (`--no-log-target`) | 2,095,763 | 16,317,831 | 15,984,013 |
| **log-space (default)** | **1,960,355** | **14,423,970** | **12,992,137** |

**Known weakness, stated plainly:** this model has a single global linear trend.
Prophet fits a *piecewise* trend with automatic changepoints, which absorbs
Walmart's 2011 level shift. On the real data that is the main reason Prophet
achieves a 45% lower MAE. Trend flexibility is the first thing to add — not a
tuning knob.

Two interchangeable engines:

| Engine | What it does | When |
|---|---|---|
| `pymc` | NUTS sampling of the full posterior (hyperparameters learned by MCMC) | when `pymc` is importable and you want the sampling-based posterior |
| `closed_form` | exact conjugate posterior under the reference prior `p(beta, sigma^2) ∝ 1/sigma^2`: `sigma^2 | y ~ InvGamma(nu/2, nu*s^2/2)`, `beta | sigma^2, y ~ N(beta_hat, sigma^2 (X'X)^-1)`, posterior predictive `~ t_nu(x* beta_hat, s^2(1 + x*'(X'X)^-1 x*))` | always available; exact, instant, deterministic |

`engine="auto"` uses PyMC when importable and falls back to `closed_form`
otherwise. The closed form is not a shortcut around uncertainty — it is the
exact posterior for this model class.

### The two engines do not use the same prior (and that is fine)

| | prior on beta | prior on sigma |
|---|---|---|
| `closed_form` | reference prior `p ∝ 1` (flat) | reference `p ∝ 1/sigma^2` → InvGamma posterior |
| `pymc` | `N(0, 1)` on the **standardised** response `z = (y - loc)/scale` | `HalfNormal(1)` on `z`, mapped back to units |

Same likelihood, different priors, so *term-level* estimates are not
guaranteed to match — and for the holiday dummies they are the most likely to
diverge, because each dummy is supported by only 3–4 event weeks and is
therefore strongly prior-sensitive. `scripts/run_evaluation.py` measures this
rather than hiding it (`reports/engine_crosscheck_*.csv`). On the real Walmart
series:

| Cross-check | Measured |
|---|---|
| Structural terms (intercept, trend, Fourier) | agree within **0.65 posterior SD** (max) |
| Holiday dummies | agree within 1.16 posterior SD (max) |
| Terms within 2 SD of each other | **18 / 18** |
| Predictive medians | agree to **1.7%** (max relative difference) |
| Predictive 95% bands | overlap **96.4%** (mean overlap fraction) |

The predictive agreement is the number that matters — it is what the project
ships. The term-level gaps are reported because a 3-week event effect *is*
prior-sensitive, and that is a finding about the data, not a bug.

> Standardising the response in the PyMC engine is not cosmetic. An
> unstandardised `N(0, std(y))` prior on the intercept is harmless on the level
> scale but, in log space, places the intercept ~170 prior SDs away from its
> posterior mass and stalls NUTS (the cross-check reported 1212-SD "gaps" and
> −334% band overlaps before this was fixed).

If you want the two engines to be term-comparable, fit PyMC with the reference
prior (or add a matching normal prior to the closed form); the project instead
reports the gap and explains it.

## Evaluation: walk-forward, pooled

A forecast interval can only be *calibrated* against unseen data, and a single
holdout block is far too small to say anything about a 95% coverage target. So
`forecast/validation.py` rolls an expanding training window forward:

- train on the first `min_train` (104) weeks,
- forecast the next `horizon` (6) weeks,
- roll forward by `horizon` weeks, repeat (up to `max_folds`),
- **pool** all out-of-sample predictions and compute metrics once.

Nothing is ever evaluated on data the model trained on.

## How to read the numbers

| Metric | Definition | Want |
|---|---|---|
| MAE | mean absolute error (units) | low |
| RMSE | root mean squared error (units) | low |
| PICP | prediction interval coverage probability: share of actuals inside the band | ≈ nominal level (0.80 / 0.95) |
| MPIW | mean prediction interval width (units) | small |
| Winkler / interval score | `(u-l) + (2/α)(l-y)_+ + (2/α)(y-u)_+` averaged | low |

PICP alone is gameable: an absurdly wide band has perfect coverage and is
useless. Always read PICP together with MPIW and the Winkler score — the score
rewards coverage and punishes width in one number, which is why it is the
column to use when choosing between models.

Coverage on a few dozen pooled weeks is itself noisy: a "95%" band that covers
32 of 36 weeks (89%) is not evidence of miscalibration. The walk-forward
design exists to make the estimate as honest as the sample size allows, and
the README reports the raw counts.

## Scenarios

`apply_scenario` multiplies the **entire** forecast table (point and both
bands) by a factor, optionally restricted to selected weeks. A +20% demand
scenario therefore returns a +20% *distribution*, not a +20% line. These are
model-based counterfactuals, never observations.
