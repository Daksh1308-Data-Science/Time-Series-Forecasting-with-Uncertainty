# Time Series Forecasting with Uncertainty

Retail demand forecasting that reports **how wrong it might be**, not just what
it predicts.

> "Next month's demand is 4.4M units" is not a forecast. "4.4M units, and in
> 95% of comparable historical weeks actual demand landed between 3.8M and
> 5.0M" is a decision someone can plan inventory against.

The project fits a point-forecast baseline (Prophet), a **Bayesian structural
demand model** whose intervals come from the posterior predictive distribution,
and a seasonal-naive reference — then *measures* which of them is honest, using
expanding-window walk-forward validation rather than in-sample residuals.

## What's inside

| Component | Technology | Purpose |
|---|---|---|
| Baseline | Prophet | Trend + yearly seasonality + Walmart holiday regressors |
| Uncertainty | PyMC **or** an exact conjugate posterior | Posterior **predictive** 80% / 95% credible intervals |
| Frequentist reference | Seasonal-naive | Hard-to-beat retail baseline with empirical residual-quantile bands |
| Evaluation | scikit-learn + custom | MAE, RMSE, PICP (coverage), MPIW (width), Winkler (interval score) |
| Validation | walk-forward | 6 expanding-window folds, pooled out-of-sample metrics |
| Visualization | Plotly | Interactive uncertainty bands, coverage-vs-nominal charts |
| Dashboard | Streamlit | 5 tabs, live numbers, runs without the optional heavy deps |
| Data | Kaggle Walmart Recruiting | Real retail weekly demand (45 stores, 99 departments) |

> **Note:** the original spec said "PyMC3". That package was renamed years ago
> — this project uses **PyMC** (`pymc`), and additionally ships an **exact
> closed-form conjugate engine** so the library, tests and dashboard work even
> when PyMC cannot be installed. See *Two Bayesian engines* below.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Unix: source .venv/bin/activate)
pip install -r requirements.txt
pip install -r requirements-optional.txt   # pymc + prophet (see caveats below)
pytest                                    # 45 tests
streamlit run dashboard/app.py            # launch the dashboard
```

Reproduce every number in this README:

```bash
python scripts/run_evaluation.py                       # exact engine + NUTS cross-check
python scripts/run_evaluation.py --engine pymc --tag _pymc --no-crosscheck
```

## Measured results

**Data provenance — read this first.** The tables below were measured on the
**synthetic** series (`forecast/data_generator.py`, seed 42: 156 weeks,
2010-02-05 → 2013-01-25, mean weekly demand 4,407,075 units). The real Walmart
dataset is one click away but needs the competition rules accepted on Kaggle —
see *Using the real Walmart data*. The synthetic series was built to mimic the
real one (trend + 52-week cycle + the four Walmart event uplifts + noise), and
**it is labelled synthetic everywhere it appears.** The evaluation code is
identical for both sources.

### Walk-forward validation — 6 folds × 6 weeks = 36 out-of-sample weeks

Every model retrained from scratch on each fold; metrics pooled. Nothing is
scored on data the model trained on. Source: `reports/comparison_table.csv`.

| Model | MAE | RMSE | Coverage @80% (nominal 80) | Coverage @95% (nominal 95) | Width @95% | Winkler @95% ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **Bayesian** (exact engine) | 339,884 | 438,500 | 63.9% | **86.1%** | 1,129,761 | 2,860,297 |
| **Bayesian** (PyMC NUTS) | **333,526** | 429,194 | 66.7% | **91.7%** | 1,219,172 | **2,546,778** |
| Prophet | 341,330 | **424,918** | 52.8% | 75.0% | **923,572** | 3,243,496 |
| Seasonal-naive | 449,048 | 585,024 | 63.9% | 83.3% | 1,335,504 | 3,386,781 |

*(MAE/RMSE/Winkler in units; widths in units. Lower is better except coverage,
which should match its nominal level. MAE is 7.6–7.7% of mean weekly demand for
the three fitted models, 10.2% for seasonal-naive.)*

### The actual finding: Prophet is more confident and less correct

This is the result worth taking away, and it only shows up because coverage was
measured out-of-sample:

- **Prophet produces the tightest 95% bands** (923,572 units, 21% of mean
  demand) and the **worst 95% coverage**: 75.0% of actuals fell inside a band
  labelled "95%". That is 27 of 36 weeks. Its 80% bands covered 52.8% — below
  the nominal level. Prophet is **overconfident**: it looks precise and is not.
- **The Bayesian model buys coverage with width** and is the better decision
  tool. NUTS reaches 91.7% coverage (33/36) — the closest to nominal of any
  model — and has the best Winkler score, the metric that rewards coverage and
  punishes width at the same time.
- On 36 pooled weeks the binomial standard error on a coverage estimate is
  ~5 points. Read the table with that in mind: Prophet's 75% is ~2.8 SE below
  nominal (a real overconfidence signal), while the Bayesian engines' 86–92% is
  within ~0.7–1.5 SE of nominal (consistent with correctly calibrated bands on a
  small sample). **The honest summary is "close to nominal, slightly
  under-covering on this sample", not "95% coverage achieved".**
- Point accuracy is nearly identical across the three fitted models (MAE within
  2% of each other). **The models do not disagree about the future — they
  disagree about how much of the future they know.** That is the entire reason
  this project exists, and only an interval evaluation can show it.

The original brief's target table (`MAE 150 / RMSE 200 / coverage 93%`) was
illustrative: those are not numbers this data can produce, and they were never
reproduced here. Coverage is whatever `scripts/run_evaluation.py` measures.

### Two Bayesian engines, and why you can trust the fast one

`BayesianForecast` exposes one interface with two interchangeable backends:

| Engine | What it does | Cost on this data |
|---|---|---|
| `closed_form` | exact Normal-Inverse-Gamma conjugate posterior (reference prior), posterior predictive by direct sampling | < 0.1 s |
| `pymc` | NUTS sampling of the full posterior | ~20 s |

`scripts/run_evaluation.py` fits **both** on the same data and compares them
(`reports/engine_crosscheck_*.csv`):

| Cross-check result | Value |
|---|---|
| Structural terms (intercept, trend, Fourier) agree within | **1.02 posterior SD** (max over 14 terms) |
| Holiday dummies agree within | 3.44 posterior SD (max) — *expected, see below* |
| Terms within 2 SD of each other | 16 / 18 |
| Predictive medians agree to | **1.08%** (max relative difference) |
| Predictive 95% bands overlap | **98.9%** (mean overlap fraction) |

The two engines maximise the *same likelihood* but use **different priors**: the
closed form uses the reference prior `p ∝ 1/σ²`; the NUTS engine uses a weakly
informative `N(0, 0.5·sd(y))` on regression terms and `HalfNormal` on σ. The
holiday dummies are supported by only 3–4 event weeks each, so they are
strongly **prior-sensitive** — a 3.4-SD gap there is a real finding about weak
identification, not a bug, and the gap is reported rather than hidden. What
matters for shipping is the predictive distribution, and there the two engines
agree to ~1% with near-complete band overlap. That is what licenses using the
instant exact engine for the walk-forward runs, and `pymc` for a periodic
full-posterior check.

### Scenario analysis

`apply_scenario` scales the **entire** predictive distribution — point forecast
*and* both interval bands — so a what-if returns a range, not a guess. A +20%
demand scenario on the 6-week horizon moves the median by ~979,000 units per
week and widens the 95% band by the same 20% (`reports/scenario_impact.csv`).
These outputs are **model-based counterfactuals**, never observations, and are
labelled as such in the dashboard.

## Dashboard

`streamlit run dashboard/app.py` — five tabs, all numbers computed live by the
`forecast` library, no placeholders:

1. **Overview & data** — the weekly series, STL decomposition, seasonal
   strength (0.73 on the synthetic series), the holiday calendar.
2. **Prophet** — forecast with nested 80/95% bands.
3. **Bayesian** — posterior predictive bands, the engine actually used, the full
   posterior summary table for every term (mean, sd, 95% credible interval).
4. **Comparison & coverage** — the walk-forward table plus a
   coverage-vs-nominal bar chart, with any skipped model named and explained.
5. **Scenario** — demand-shift slider that scales whole distributions.

The sidebar switches data source (synthetic ↔ real Walmart), horizon, Bayesian
engine and fold count. Heavy fits sit behind `st.cache_data`. If Prophet or
PyMC is missing, the app degrades to the exact engine and says which model was
skipped and why — it never crashes.

## Architecture

```
forecast/                  # pure, deterministic, testable library (no plotting, no I/O)
├── schema.py              #   canonical forecast table: ds, yhat, lower_80, upper_80, ...
├── data_generator.py      #   seeded synthetic weekly retail series
├── data_loader.py         #   Kaggle download, load, aggregate, cache
├── holidays.py            #   the four Walmart event weeks
├── decomposition.py       #   STL + seasonal strength
├── benchmark.py           #   seasonal-naive + residual-quantile bands
├── prophet_model.py       #   Prophet wrapper (lazy import)
├── bayesian_model.py      #   BayesianForecast: pymc | closed_form
├── metrics.py             #   MAE, RMSE, PICP, MPIW, Winkler
├── validation.py          #   walk_forward_splits, run_comparison, forecast_all
└── scenario.py            #   apply_scenario, scenario_impact
dashboard/app.py           # Streamlit, 5 tabs
scripts/run_evaluation.py  # reproducible pipeline -> reports/
tests/                     # 45 tests, known-value fixtures
configs/default.toml       # stdlib tomllib config
reports/                   # measured outputs cited above
docs/methodology.md        # models, intervals, how to read PICP/MPIW/Winkler
data/README.md             # dataset provenance and acquisition
```

## Using the real Walmart data

The real dataset is the Kaggle competition *Walmart Recruiting — Store Sales
Forecasting* (45 stores × 99 departments, weekly, 2010-02-05 → 2012-10-26).
It needs one manual step: **accept the competition rules** while logged in at
<https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting>.
Kaggle returns HTTP 401 to the API until you do — a valid API token alone is not
enough. `forecast/data_loader.py` detects this and tells you exactly what to do.

```bash
python scripts/run_evaluation.py --source walmart
```

The loader downloads the CSVs with `kagglehub`, aggregates
store × department weekly sales into one Friday series, caches it at
`data/processed/weekly_total.csv`, and the dashboard picks it up automatically.
The raw competition files are git-ignored (they are subject to the competition
rules and must not be redistributed).

## Testing

45 tests, all known-value fixtures rather than smoke tests: hand-computed
MAE/RMSE/PICP/MPIW/Winkler values (including the Gneiting & Raftery interval
score), exact seasonal-naive arithmetic, prior-free coverage assertions on
synthetic data, engine-fallback behaviour when PyMC is absent, and a headless
Streamlit render test proving all five tabs execute without error.

## Dependencies

Core: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `plotly`,
`streamlit`, `kagglehub`, `pytest`. Optional (`requirements-optional.txt`):
`pymc` and `prophet`.

Caveats, tested on Python 3.14 / Windows:

- **Prophet 1.4** is in maintenance mode and compiles CmdStan on first fit. It
  worked out of the box here; on a machine without a C++ toolchain run
  `python -c "import cmdstanpy; cmdstanpy.install_cxx_toolchain(force=True)"`
  first, and if that fails the app still runs without it.
- **PyMC 6** runs on the Python 3.14 wheels; without it, `engine="auto"` falls
  back to the exact conjugate engine automatically.

## Limitations

- **The published numbers are from the synthetic series.** They are real
  measurements of real models on a simulated series — not measurements on retail
  data. Re-run `--source walmart` once Kaggle access is granted.
- **36 pooled evaluation weeks is a small sample** for a coverage claim: the
  binomial SE on a coverage estimate at that sample size is ~5 points, and the
  data contains only ~2.7 seasonal cycles. More folds (or a longer series) would
  tighten the intervals on the interval metrics.
- **The model is a linear-Gaussian structural model.** It cannot represent
  multiplicative seasonality, variance that grows with the level, or
  autocorrelation in the residuals; the lognormal-ish noise in the series makes
  a log transform a plausible next step.
- **Holiday effects are estimated from 3–4 events per holiday** and are
  prior-sensitive, as the engine cross-check documents. A hierarchical or
  year-varying holiday effect would need more history than this dataset has.
- Scenarios scale the forecast; they do not re-estimate the model under the
  new regime.
