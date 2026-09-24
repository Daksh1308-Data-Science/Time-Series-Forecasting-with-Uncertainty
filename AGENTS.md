# AGENTS.md — Ground rules for working in this repo

This file is the contract for any AI agent (or human) changing this codebase.
**Read it fully before writing or modifying any file.** If a task conflicts with
it, stop and ask the user.

## 1. Mission

Build **retail demand forecasting with honest uncertainty**:

- Prophet baseline (trend + seasonality + holiday regressors)
- A Bayesian structural model (PyMC or the exact conjugate posterior) whose
  intervals are posterior *predictive*
- Walk-forward evaluation of **measured** coverage (PICP), sharpness (MPIW) and
  the Winkler interval score
- A Streamlit dashboard showing the same numbers interactively
- Scenario analysis that scales whole predictive distributions

## 2. Non-negotiable decisions (confirmed with the user)

1. **`pymc3` does not exist** — the package was renamed to `pymc`. Never install
   or import `pymc3`.
2. **Headline numbers are measured, never hardcoded.** Every metric in the
   README must be reproducible via `python scripts/run_evaluation.py` and live
   in `reports/`. The original spec's "MAE 150 / coverage 93%" figures are
   illustrative targets, not expected results.
3. **Real vs synthetic is never blurred.** The real series comes from
   `forecast/data_loader.py` (Kaggle Walmart competition) and must be labelled
   "Walmart / real". `forecast/data_generator.py` output must be labelled
   "synthetic". No fabricated series may be presented as observed demand.
4. **Coverage claims require out-of-sample evidence.** Any statement about
   interval quality must come from `forecast/validation.py` walk-forward folds
   (expanding window), never from in-sample residuals or a single holdout.
5. **Optional dependencies must never break anything.** PyMC and Prophet are
   optional; `engine="auto"` falls back to the closed-form conjugate posterior,
   and `run_comparison` reports a failed model in `skipped` with the reason
   instead of crashing.
6. **Scenario results are counterfactual.** Anything from `forecast/scenario.py`
   is a model-based what-if, labelled as such in UI and docs.

## 3. Repo layout (add to it, don't reinvent)

```
Time Series Forecasting with Uncertainty/
├── forecast/               # core library: pure, deterministic, testable
│   ├── __init__.py
│   ├── schema.py           # canonical forecast table + band-column conventions
│   ├── data_generator.py   # seeded synthetic weekly retail series
│   ├── data_loader.py      # Kaggle Walmart download, load, aggregate, cache
│   ├── holidays.py         # the four Walmart event weeks (2010-2013)
│   ├── decomposition.py    # STL + seasonal strength
│   ├── benchmark.py        # seasonal-naive point + empirical residual bands
│   ├── prophet_model.py    # Prophet wrapper -> point + interval forecasts
│   ├── bayesian_model.py   # BayesianForecast (pymc | closed_form) -> posterior predictive
│   ├── metrics.py          # MAE, RMSE, PICP, MPIW, Winkler
│   ├── validation.py       # walk_forward_splits, run_comparison, forecast_all
│   └── scenario.py         # apply_scenario, scenario_impact
├── dashboard/app.py        # Streamlit app, exactly 5 tabs
├── scripts/run_evaluation.py  # reproducible pipeline -> reports/
├── tests/                  # pytest, known-value fixtures
├── configs/default.toml    # stdlib tomllib config (no YAML dependency)
├── data/                   # processed series committed; raw Kaggle files ignored
├── reports/                # measured outputs cited by the README
├── docs/methodology.md
├── notebooks/README.md
├── requirements.txt / requirements-optional.txt
├── README.md
└── AGENTS.md
```

## 4. Module contracts

### `forecast/schema.py`
- `level_tag(level: float) -> str` — 0.95 -> "95".
- `band_columns(levels) -> list[str]` — ordered `lower_*, upper_*` column names.
- `future_dates(last, periods) -> pd.Series` — next weekly dates after `last`.
- `standardize_forecast(df, levels) -> pd.DataFrame` — validate/reorder the
  canonical table `ds, yhat, lower_80, upper_80, lower_95, upper_95`.

### `forecast/data_loader.py`
- `kaggle_credentials_present() -> bool`
- `download_walmart(raw_dir=None, force=False) -> Path` — copies `train.csv`,
  `features.csv`, `stores.csv` into `data/raw`. `FileNotFoundError` without a
  token; `RuntimeError` with the rules URL when Kaggle refuses (401).
- `load_raw(data_dir=None) -> dict[str, DataFrame]`
- `aggregate_weekly_sales(train_df, features_df=None) -> DataFrame` — sums
  store x department sales per Friday week, attaches `IsHoliday`.
- `load_cached_series(cache_path=None)` / `prepare_walmart_series(...)` —
  the single entry point the dashboard and scripts use.

### `forecast/bayesian_model.py`
- `BayesianForecast(df, period=52, order=6, holiday_dates=None, engine="auto",
  seed=42, draws=1500, tune=1500, chains=2, levels=(0.80, 0.95))`
  - `.fit() -> self`; `engine` resolves to `"pymc"` or `"closed_form"`.
  - `.sample_predictive(periods, n_samples=2000) -> (dates, samples)`
  - `.predict(periods=6, levels=None, n_samples=2000) -> DataFrame` (median +
    central bands; lower bound clipped at zero).
  - `.parameter_summary() -> DataFrame` (term, mean, sd, lower_95, upper_95).
- `build_design(t, denom, period, order, holidays) -> (X, names)` — intercept,
  scaled trend, Fourier pairs, holiday dummies.
- `resolve_engine(engine) -> "pymc" | "closed_form"`.
- `bayesian_forecast(train_df, periods=6, levels=..., **kwargs) -> DataFrame`.

The closed-form engine is the exact Normal-Inverse-Gamma posterior
(`sigma^2 ~ InvGamma(nu/2, nu*s^2/2)`, `beta | sigma^2 ~ N(beta_hat, sigma^2 (X'X)^-1)`).
It is a real Bayesian engine, not a shortcut around uncertainty.

### `forecast/validation.py`
- `walk_forward_splits(df, horizon=6, min_train=104, step=6, max_folds=None)
  -> list[(train, test)]` — expanding window, chronological, no overlap.
- `run_comparison(df, models=("naive","bayesian","prophet"), levels=..., ...) -> dict`
  with keys `table, folds, ran, skipped, info, descriptions`. A model that
  raises is recorded in `skipped` with `"TypeName: message"`.
- `forecast_all(train_df, periods=6, models=..., ...) -> {"forecasts", "skipped"}`.

### `forecast/scenario.py`
- `apply_scenario(forecast_df, factor=1.2, dates=None) -> DataFrame` — scales
  `yhat` and every band column (reject non-positive factors).
- `scenario_impact(base_df, scenario_df, level=0.80) -> DataFrame` — per-week deltas.

### `dashboard/app.py`
- Exactly **5 tabs, fixed order/names**: `Overview & data`, `Prophet`,
  `Bayesian`, `Comparison & coverage`, `Scenario`.
- Sidebar: data source, horizon, Bayesian engine, walk-forward folds.
- Heavy work behind `st.cache_data`; missing optional deps render a message,
  never a crash.

## 5. Build order & Definition of Done per milestone

### M0 — Scaffold (DONE)
- `git init`, venv, requirements, `forecast/` skeleton, `tests/`, `configs/`.
- DoD: `python -c "import forecast"` works; `pytest` green.

### M1 — Data & baseline (DONE)
- `data_generator.py`, `data_loader.py`, `holidays.py`, `decomposition.py`,
  `benchmark.py`, `metrics.py`; Prophet wrapper behind a lazy import.
- DoD: STL decomposition runs on the weekly series; seasonal-naive bands are
  built from empirical seasonal-difference quantiles; `run_evaluation.py`
  writes `reports/`.

### M2 — Bayesian uncertainty (DONE)
- `bayesian_model.py` with both engines; `validation.py` walk-forward.
- DoD: `run_comparison` returns measured PICP/MPIW/Winkler for the models that
  ran; 95% bands cover at least as many actuals as 80% bands; tests assert
  predictive coverage is near nominal on synthetic data.

### M3 — Dashboard & docs (DONE)
- `dashboard/app.py`, `README.md`, `docs/methodology.md`, `AGENTS.md`,
  `data/README.md`, `notebooks/README.md`.
- DoD: `streamlit run dashboard/app.py` serves all 5 tabs with live numbers
  (verified headlessly by `tests/test_dashboard.py`); README tables cite
  `reports/` outputs; real-vs-synthetic labels present.

### M4 — Real data (BLOCKED on the user, not on code)
- The loader, aggregation, cache and `--source walmart` path are implemented and
  tested. Kaggle returns HTTP 401 until the competition rules are accepted in a
  browser with that account; `download_walmart()` raises with the URL and the
  dashboard renders the same instruction. Once accepted:
  `python scripts/run_evaluation.py --source walmart`, then update the README
  tables and delete the "numbers are synthetic" caveat.

## 6. Conventions

- **Pure functions in `forecast/`:** no plotting, no file writes. Plotting lives
  in `dashboard/`; artifact writing in `scripts/`.
- **Determinism:** every stochastic path takes a `seed`; generators and
  Bayesian sampling use `np.random.default_rng`.
- **Tests are known-value fixtures,** not smoke tests: hand-computed
  MAE/RMSE/PICP/MPIW/Winkler values, exact seasonal-naive arithmetic, coverage
  assertions on synthetic data.
- **Forecast tables use the `schema.py` layout** for every model.
- **Dependencies:** no new dependency without a reason and a note. `pymc` and
  `prophet` live in `requirements-optional.txt`; config uses stdlib `tomllib`.

## 7. Pitfalls (recurring agent mistakes)

- **`pymc3` does not exist** — use `pymc` or the closed-form engine.
- **Don't hardcode metrics** — regenerate `reports/` and quote those numbers.
- **Don't present synthetic data as real** (or vice versa).
- **Don't compute "coverage" on training data** — that is calibration theatre.
- **Windows + Prophet:** the first fit compiles CmdStan. Try
  `cmdstanpy.install_cxx_toolchain(force=True)`; if it still fails, the app
  must keep working and report Prophet as skipped.
- **Don't scale only `yhat` in scenarios** — the bands are the point of the
  project.
- **Don't widen an interval to "fix" coverage** and report it as a win; report
  PICP together with MPIW/Winkler so width is visible.

## 8. Definition of done (whole project)

1. `pytest` green.
2. `streamlit run dashboard/app.py` runs with all 5 tabs populated by real
   library calls.
3. Every metric in the README is reproducible from `scripts/run_evaluation.py`
   and present in `reports/`.
4. Real data is used if Kaggle access is available; otherwise the README says
   so explicitly and shows the exact command to reproduce it with real data.
