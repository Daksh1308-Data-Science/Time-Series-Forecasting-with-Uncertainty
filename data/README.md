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

1. Log in to Kaggle and **accept the competition rules** (the API returns 401
   until you do — the token alone is not enough).
2. Create an API token: Kaggle → Settings → API → *Create New Token* → save as
   `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
3. Fetch and cache the processed series:

```bash
python scripts/run_evaluation.py --source walmart
```

or in Python:

```python
from forecast.data_loader import prepare_walmart_series
series = prepare_walmart_series(download=True)   # -> ds, y, IsHoliday
```

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
