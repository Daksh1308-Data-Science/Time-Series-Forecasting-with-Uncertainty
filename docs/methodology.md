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
y_t  ~  Normal(mu_t, sigma)
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
| `pymc` | weakly informative `N(0, 0.5 sd(y))` | `HalfNormal(0.25 sd(y))` |

Same likelihood, different priors, so *term-level* estimates are not
guaranteed to match — and for the holiday dummies they visibly do not, because
each dummy is supported by only 3–4 event weeks and is therefore strongly
prior-sensitive. `scripts/run_evaluation.py` measures this rather than hiding
it (`reports/engine_crosscheck_*.csv`):

- well-identified structural terms (intercept, trend, Fourier) should agree to
  a fraction of a posterior standard deviation;
- holiday dummies may disagree — and the size of that disagreement *is* the
  honest uncertainty on a 3-week event effect;
- the **predictive** distributions agree closely, because the term-level
  disagreement largely cancels in `mu_t`. The predictive interval is what the
  project ships, so the predictive comparison is the one that matters.

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
