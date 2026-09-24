# Data

## Real data: Walmart Recruiting — Store Sales Forecasting

| Item | Value |
|---|---|
| Source | Kaggle competition *Walmart Recruiting - Store Sales Forecasting* |
| URL | https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting |
| Grain | one row per (store, department, week) |
| Coverage | 45 stores, 99 departments, 2010-02-05 → 2012-10-26 (weekly, Fridays) |
| Files used | `train.csv` (Weekly_Sales, IsHoliday), `features.csv` (IsHoliday), `stores.csv` (not modelled) |

### Getting it

The processed series is **already committed** at
`data/processed/weekly_total.csv`, so the dashboard and tests run without any
credentials. To rebuild it from Kaggle:

1. Log in to Kaggle and **accept the competition rules** (competition data is
   gated separately from authentication).
2. Create an API token: Kaggle → Settings → API → *Create New Token*. Kaggle now
   issues `KGAT_`-style tokens — save it as `~/.kaggle/access_token`
   (Windows: `C:\Users\<you>\.kaggle\access_token`, one line, no quotes) or
   export `KAGGLE_API_TOKEN`.

   > The legacy `~/.kaggle/kaggle.json` (username + key) is **no longer honoured
   > by the Kaggle API**. A stale one produces `401` on *every* call, including
   > endpoints that need no competition access — so check authentication before
   > assuming the rules gate is the problem.
3. Fetch and cache the processed series:

```bash
python scripts/run_evaluation.py --source walmart
```

or in Python:

```python
from forecast.data_loader import prepare_walmart_series
series = prepare_walmart_series(download=True)   # -> ds, y, IsHoliday
```

Kaggle serves `train.csv.zip`, `features.csv.zip`, `test.csv.zip` and
`stores.csv`; the loader reads the zipped CSVs directly (pandas infers the
compression), so nothing needs extracting first.

### What is committed here

- `data/processed/weekly_total.csv` — the aggregated weekly series (`ds`, `y`,
  `IsHoliday`). Small, derived, and committed so the project runs offline.
- `data/raw/` — the original competition CSVs are **git-ignored**: they are
  subject to the competition rules and must not be redistributed.

The raw files are never modified; `aggregate_weekly_sales()` sums
`Weekly_Sales` across stores and departments per Friday week and carries the
`IsHoliday` flag.

## Synthetic demo series

`forecast/data_generator.py` generates a seeded weekly retail series
(trend + 52-week Fourier seasonality + the four Walmart event uplifts +
lognormal noise). It exists so the tests, the dashboard demo and CI work
without Kaggle access. **It is synthetic and is always labelled as such** — it
is never presented as observed demand. The event dates match the real Walmart
calendar, so the holiday machinery is exercised identically.
