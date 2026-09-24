"""Reproducible end-to-end evaluation. Writes reports/*, the source of every
headline number in the README.

    python scripts/run_evaluation.py                      # synthetic demo series
    python scripts/run_evaluation.py --source walmart     # real Kaggle data
    python scripts/run_evaluation.py --engine pymc         # NUTS instead of closed form
    python scripts/run_evaluation.py --no-crosscheck       # skip the closed-form vs NUTS check

Defaults come from configs/default.toml; CLI flags win.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/run_evaluation.py`
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from forecast import __version__
from forecast.bayesian_model import BayesianForecast, resolve_engine
from forecast.data_generator import generate_weekly_sales
from forecast.data_loader import prepare_walmart_series
from forecast.decomposition import seasonal_strength, stl_decompose
from forecast.scenario import apply_scenario, scenario_impact
from forecast.validation import forecast_all, run_comparison

REPORTS = ROOT / "reports"


def load_config() -> dict:
    with open(ROOT / "configs" / "default.toml", "rb") as handle:
        return tomllib.load(handle)


def load_series(source: str) -> tuple[pd.DataFrame, str]:
    if source == "walmart":
        series = prepare_walmart_series(download=True)
        return series, "Walmart Recruiting - Store Sales Forecasting (real, Kaggle)"
    series = generate_weekly_sales(n_weeks=156, seed=42)
    return series, "Synthetic weekly retail demand (seed 42) - NOT real data"


def engine_crosscheck(
    series: pd.DataFrame,
    order: int,
    seed: int,
    horizon: int,
    draws: int,
    tune: int,
    chains: int,
    log_target: bool,
) -> dict | None:
    """Check the exact closed-form posterior against a full NUTS fit.

    Both engines maximise the *same likelihood* but do not use the same prior:
    the closed form uses the reference prior p(beta, sigma^2) ~ 1/sigma^2, the
    NUTS engine a weakly-informative N(0, 0.5 sd(y)) on the regression terms and
    HalfNormal on sigma. So exact equality is not expected term-by-term. What
    must hold:

    1. Well-identified structural terms (intercept, trend, Fourier) agree to
       within a fraction of a posterior standard deviation.
    2. The weakly-identified holiday dummies may differ (they are prior
       sensitive - each has only 3-4 event weeks), and the report says so.
    3. The *predictive* distributions - the thing the project actually ships -
       agree closely, because the disagreement partially cancels in mu_t.
    """
    if resolve_engine("auto") != "pymc":
        print("cross-check skipped: pymc is not importable")
        return None

    closed = BayesianForecast(
        series, order=order, engine="closed_form", seed=seed, log_target=log_target
    ).fit()
    sampled = BayesianForecast(
        series,
        order=order,
        engine="pymc",
        seed=seed,
        draws=draws,
        tune=tune,
        chains=chains,
        log_target=log_target,
    ).fit()

    params = closed.parameter_summary().merge(
        sampled.parameter_summary(), on="term", suffixes=("_closed_form", "_pymc")
    )
    params["mean_abs_diff"] = (params["mean_closed_form"] - params["mean_pymc"]).abs()
    params["gap_in_closed_form_sd"] = params["mean_abs_diff"] / params["sd_closed_form"]
    params["group"] = params["term"].str.startswith("holiday_").map(
        {True: "holiday (weakly identified)", False: "structural"}
    )
    params.to_csv(REPORTS / "engine_crosscheck_parameters.csv", index=False)

    structural = params[params["group"] == "structural"]
    holidays = params[params["group"] != "structural"]
    within_2sd = int((params["gap_in_closed_form_sd"] <= 2).sum())

    forecast = closed.predict(horizon, n_samples=4000).merge(
        sampled.predict(horizon, n_samples=4000), on="ds", suffixes=("_closed_form", "_pymc")
    )
    forecast["median_rel_diff"] = (
        forecast["yhat_pymc"] - forecast["yhat_closed_form"]
    ).abs() / forecast["yhat_closed_form"]
    overlap = np.minimum(forecast["upper_95_pymc"], forecast["upper_95_closed_form"]) - np.maximum(
        forecast["lower_95_pymc"], forecast["lower_95_closed_form"]
    )
    forecast["band95_overlap_fraction"] = overlap / (
        forecast["upper_95_closed_form"] - forecast["lower_95_closed_form"]
    )
    forecast.to_csv(REPORTS / "engine_crosscheck_forecast.csv", index=False)

    print(
        f"engine cross-check: structural terms agree within "
        f"{structural['gap_in_closed_form_sd'].max():.2f} posterior sd (max); "
        f"holiday dummies within {holidays['gap_in_closed_form_sd'].max():.2f} sd (prior-sensitive); "
        f"{within_2sd}/{len(params)} terms within 2 sd"
    )
    print(
        f"engine cross-check: predictive medians agree to "
        f"{forecast['median_rel_diff'].max():.2%} (max), 95% bands overlap "
        f"{forecast['band95_overlap_fraction'].mean():.0%} (mean)"
    )
    return {
        "structural_max_gap_sd": float(structural["gap_in_closed_form_sd"].max()),
        "holiday_max_gap_sd": float(holidays["gap_in_closed_form_sd"].max()),
        "terms_within_2sd": within_2sd,
        "terms_total": int(len(params)),
        "predictive_median_max_rel_diff": float(forecast["median_rel_diff"].max()),
        "predictive_band95_mean_overlap": float(forecast["band95_overlap_fraction"].mean()),
        "note": (
            "Same likelihood, different (documented) priors: closed_form uses the "
            "reference prior, pymc a weakly-informative N(0, 0.5 sd(y))/HalfNormal. "
            "Holiday dummies have only 3-4 event weeks and are prior-sensitive; the "
            "predictive distributions agree closely regardless."
        ),
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "walmart"], default="synthetic")
    parser.add_argument("--horizon", type=int, default=cfg["evaluation"]["horizon"])
    parser.add_argument("--min-train", type=int, default=cfg["evaluation"]["min_train"])
    parser.add_argument("--max-folds", type=int, default=cfg["evaluation"]["max_folds"])
    parser.add_argument(
        "--engine", choices=["auto", "pymc", "closed_form"], default=cfg["bayesian"]["engine"]
    )
    parser.add_argument("--fourier-order", type=int, default=cfg["bayesian"]["fourier_order"])
    parser.add_argument("--draws", type=int, default=cfg["bayesian"]["draws"])
    parser.add_argument("--tune", type=int, default=cfg["bayesian"]["tune"])
    parser.add_argument("--chains", type=int, default=cfg["bayesian"]["chains"])
    parser.add_argument("--seed", type=int, default=cfg["bayesian"]["seed"])
    parser.add_argument(
        "--log-target",
        action=argparse.BooleanOptionalAction,
        default=cfg["bayesian"]["log_target"],
        help="model log(demand) instead of raw demand (recommended for retail)",
    )
    parser.add_argument("--scenario-factor", type=float, default=1.2)
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the reports/ filenames, e.g. --tag _pymc writes comparison_table_pymc.csv",
    )
    parser.add_argument(
        "--no-crosscheck", action="store_true", help="skip the closed-form vs NUTS check"
    )
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    series, label = load_series(args.source)
    horizon = min(args.horizon, max(1, len(series) // 4))
    min_train = min(args.min_train, len(series) - horizon)
    tag = args.tag
    table_path = REPORTS / f"comparison_table{tag}.csv"
    scenario_path = REPORTS / f"scenario_impact{tag}.csv"
    params_path = REPORTS / f"bayesian_parameters{tag}.csv"
    metrics_path = REPORTS / f"metrics{tag}.json"

    print(f"source: {label}")
    print(f"rows: {len(series)} weeks, {series['ds'].min().date()} .. {series['ds'].max().date()}")
    print(f"bayesian engine requested: {args.engine} -> {resolve_engine(args.engine)}")

    # --- Week 1: decomposition / seasonality -----------------------------
    decomposition = stl_decompose(series.set_index("ds")["y"], period=52)
    strength = seasonal_strength(decomposition["seasonal"], decomposition["resid"])
    print(f"seasonal strength (STL, 52-week): {strength:.3f}")

    # --- Week 2: walk-forward coverage ------------------------------------
    model_options = {
        "bayesian": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "log_target": args.log_target,
        }
    }
    comparison = run_comparison(
        series,
        horizon=horizon,
        min_train=min_train,
        step=horizon,
        max_folds=args.max_folds,
        engine=args.engine,
        fourier_order=args.fourier_order,
        seed=args.seed,
        model_options=model_options,
    )
    table = comparison["table"]
    print(f"\nwalk-forward: {comparison['folds']} folds x {horizon} weeks")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    for name, reason in comparison["skipped"].items():
        print(f"  [skipped] {name}: {reason}")
    table.to_csv(table_path, index=False)

    # --- Week 2/3: a full-history fit + scenario --------------------------
    full = forecast_all(
        series, periods=horizon, engine=args.engine, seed=args.seed, model_options=model_options
    )
    base = full["forecasts"].get("bayesian")
    if base is None:
        print("bayesian model unavailable; skipping scenario analysis")
    else:
        scenario = apply_scenario(base, factor=args.scenario_factor)
        impact = scenario_impact(base, scenario)
        impact.to_csv(scenario_path, index=False)
        print(f"\nscenario +{int((args.scenario_factor - 1) * 100)}% demand:")
        print(impact[["ds", "yhat_base", "yhat_scenario", "delta_yhat"]].to_string(index=False))

    posterior = BayesianForecast(
        series,
        order=args.fourier_order,
        engine=args.engine,
        seed=args.seed,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        log_target=args.log_target,
    ).fit()
    posterior.parameter_summary().to_csv(params_path, index=False)

    # --- Bayesian engine cross-check --------------------------------------
    crosscheck = None
    if args.no_crosscheck:
        print("engine cross-check skipped (--no-crosscheck)")
    else:
        crosscheck = engine_crosscheck(
            series,
            args.fourier_order,
            args.seed,
            horizon,
            args.draws,
            args.tune,
            args.chains,
            args.log_target,
        )

    metrics = {
        "library_version": __version__,
        "source": args.source,
        "source_label": label,
        "rows": int(len(series)),
        "first_week": str(series["ds"].min().date()),
        "last_week": str(series["ds"].max().date()),
        "decomposition": {"seasonal_strength": round(strength, 4)},
        "walk_forward": {
            "folds": comparison["folds"],
            "horizon": horizon,
            "min_train": min_train,
            "engine_requested": args.engine,
            "engines_used": {k: v.get("engine") for k, v in comparison["info"].items()},
            "skipped": comparison["skipped"],
            "table": json.loads(table.to_json(orient="records")),
        },
        "scenario": {"factor": args.scenario_factor},
        "bayesian_log_target": args.log_target,
        "engine_crosscheck": crosscheck,
    }
    (REPORTS / f"metrics{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {table_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
