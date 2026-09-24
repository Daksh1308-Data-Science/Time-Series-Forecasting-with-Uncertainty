"""Streamlit dashboard: forecast exploration with honest uncertainty.

    streamlit run dashboard/app.py

Five tabs: Overview (data + decomposition) | Prophet | Bayesian | Comparison &
Coverage | Scenario. Numbers are computed live by the ``forecast`` library — no
placeholders — except the Comparison tab's default view, which shows the
committed measured results from ``reports/`` (Prophet included) so a deployment
without Prophet still shows the full comparison. Switch it to the live re-run
for an in-session refit. Heavy work is cached; missing optional dependencies
degrade with an explanation instead of an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast.bayesian_model import BayesianForecast, resolve_engine
from forecast.data_generator import generate_weekly_sales
from forecast.data_loader import kaggle_credentials_present, prepare_walmart_series
from forecast.decomposition import seasonal_strength, stl_decompose
from forecast.holidays import walmart_holidays
from forecast.prophet_model import PROPHET_MISSING_MESSAGE, prophet_available
from forecast.scenario import apply_scenario, scenario_impact
from forecast.validation import (
    forecast_all,
    load_measured_table,
    measured_config_mismatches,
    run_comparison,
)

st.set_page_config(page_title="Retail Demand Forecasting with Uncertainty", layout="wide")

LEVELS = (0.80, 0.95)


# ---------------------------------------------------------------- app settings
@st.cache_data(show_spinner=False)
def app_config() -> dict:
    """Single source of truth for settings the app shares with the pipeline.

    Without this the dashboard's live re-run would silently use the library
    defaults (e.g. a level-space Gaussian) while the committed measured results
    and the README use the configured log-space model — a 7-23% discrepancy a
    visitor would rightly distrust.
    """
    import tomllib

    with open(ROOT / "configs" / "default.toml", "rb") as handle:
        return tomllib.load(handle)


CONFIG = app_config()
LOG_TARGET = bool(CONFIG["bayesian"].get("log_target", True))
EVAL_DEFAULTS = CONFIG["evaluation"]
BAYES_OPTIONS = {"log_target": LOG_TARGET}


# --------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def load_series(source: str, n_weeks: int = 156):
    if source == "walmart":
        return prepare_walmart_series(download=False)
    return generate_weekly_sales(n_weeks=n_weeks, seed=42)


@st.cache_data(show_spinner="Fitting models on the full history...")
def all_forecasts(source: str, horizon: int, engine: str):
    series = load_series(source)
    return forecast_all(
        series,
        periods=horizon,
        engine=engine,
        levels=LEVELS,
        model_options={"bayesian": dict(BAYES_OPTIONS)},
    )


@st.cache_data(show_spinner="Running walk-forward validation (this takes a moment)...")
def comparison(source: str, horizon: int, engine: str, min_train: int, max_folds: int):
    series = load_series(source)
    return run_comparison(
        series,
        levels=LEVELS,
        horizon=horizon,
        min_train=min_train,
        step=horizon,
        max_folds=max_folds,
        engine=engine,
        model_options={"bayesian": dict(BAYES_OPTIONS)},
    )


@st.cache_data(show_spinner="Running walk-forward validation (this takes a moment)...")
def comparison_prophet(source: str, horizon: int, min_train: int, max_folds: int):
    series = load_series(source)
    return run_comparison(
        series,
        models=("naive", "bayesian", "prophet"),
        levels=LEVELS,
        horizon=horizon,
        min_train=min_train,
        step=horizon,
        max_folds=max_folds,
        engine="closed_form",  # Prophet folds are fast; keep the Bayesian side cheap
        model_options={"bayesian": dict(BAYES_OPTIONS)},
    )


# ------------------------------------------------------------------ plotting
def band_figure(forecast: pd.DataFrame, history: pd.DataFrame | None, title: str) -> go.Figure:
    """Actual line + nested 80/95% bands + median forecast."""
    fig = go.Figure()
    if history is not None and not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["ds"], y=history["y"], name="actual", mode="lines",
                line=dict(color="#444", width=1.5),
            )
        )
    # Wider (95%) band first, narrower (80%) drawn on top of it.
    for tag, alpha in (("95", 0.12), ("80", 0.28)):
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast[f"upper_{tag}"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast[f"lower_{tag}"], name=f"{tag}% interval",
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=f"rgba(31,119,180,{alpha})",
        ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], name="forecast (posterior median)",
        mode="lines", line=dict(color="#1f77b4", width=2.5),
    ))
    fig.update_layout(
        title=title, template="plotly_white", height=430,
        yaxis_title="weekly demand (units)", xaxis_title="week",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def decomposition_figure(parts: dict) -> go.Figure:
    fig = go.Figure()
    for name, color in (("trend", "#1f77b4"), ("seasonal", "#ff7f0e"), ("resid", "#7f7f7f")):
        fig.add_trace(go.Scatter(
            x=parts[name].index, y=parts[name].to_numpy(), name=name, mode="lines",
            line=dict(color=color, width=1.5),
        ))
    fig.update_layout(
        title="STL decomposition (52-week period)", template="plotly_white", height=380,
        yaxis_title="units", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Demand forecasting")
source_label = st.sidebar.selectbox(
    "Data source",
    ["Walmart — real Kaggle data", "Synthetic demo series (seed 42)"],
)
source = "walmart" if source_label.startswith("Walmart") else "synthetic"
horizon = st.sidebar.slider("Forecast horizon (weeks)", 1, 12, 6)
engine = st.sidebar.selectbox(
    "Bayesian engine",
    [
        "closed_form (exact, instant)",
        "auto (PyMC if available)",
        "pymc (NUTS sampler — local only)",
    ],
)
engine_key = {
    "closed_form (exact, instant)": "closed_form",
    "auto (PyMC if available)": "auto",
    "pymc (NUTS sampler — local only)": "pymc",
}[engine]
# Defaults come from configs/default.toml so the default view of the Comparison
# tab needs no mismatch warning and the live re-run uses the configured model.
max_folds = st.sidebar.slider(
    "Walk-forward folds", 1, 8, int(EVAL_DEFAULTS.get("max_folds", 6))
)
st.sidebar.caption(
    f"pymc importable: **{resolve_engine('auto') == 'pymc'}** · "
    f"chosen engine: **{engine_key}**"
)

def rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns to whole units (dates are left alone)."""
    return df.round({column: 0 for column in df.select_dtypes("number").columns})


PRETTY_COLUMNS = {
    "model": "model", "mae": "MAE", "rmse": "RMSE", "n_eval": "n (weeks)",
    "picp_80": "coverage 80%", "picp_95": "coverage 95%",
    "mpiw_80": "width 80%", "mpiw_95": "width 95%",
    "winkler_80": "Winkler 80%", "winkler_95": "Winkler 95%",
}


def render_comparison_outputs(table: pd.DataFrame, skipped: dict | None = None) -> None:
    """Shared by both Comparison-tab modes: table + coverage-vs-nominal chart."""
    if table.empty:
        st.warning("No model produced results.")
        return
    st.dataframe(
        table.rename(columns=PRETTY_COLUMNS).round(3), width="stretch", hide_index=True
    )
    for name, reason in (skipped or {}).items():
        st.caption(f"Skipped `{name}`: {reason}")

    melted = table.melt(
        id_vars="model", value_vars=["picp_80", "picp_95"],
        var_name="interval", value_name="coverage",
    )
    melted["nominal"] = melted["interval"].map({"picp_80": 0.80, "picp_95": 0.95})
    melted["interval"] = melted["interval"].str.replace("picp_", "% interval", regex=False)
    bar = go.Figure()
    for model_name, group in melted.groupby("model"):
        bar.add_trace(go.Bar(
            x=group["interval"], y=group["coverage"], name=model_name,
            text=[f"{v:.0%}" for v in group["coverage"]], textposition="outside",
        ))
    bar.add_trace(go.Scatter(
        x=melted["interval"].unique(),
        y=melted["nominal"].groupby(melted["interval"]).first().values,
        mode="markers", name="nominal level",
        marker=dict(symbol="line-ew-open", size=22, color="black"),
    ))
    bar.update_layout(
        barmode="group", template="plotly_white", height=380,
        title="Measured coverage vs nominal level (pooled out-of-sample)",
        yaxis=dict(title="coverage (PICP)", range=[0, 1.05]), margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(bar, width="stretch")


# ------------------------------------------------------------------ load data
try:
    series = load_series(source)
except FileNotFoundError as exc:
    st.error(f"Real Walmart data is not available locally: {exc}")
    st.info(
        "Accept the competition rules at "
        "https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting, "
        "ensure `~/.kaggle/kaggle.json` exists, then run:\n\n"
        "```\npython scripts/run_evaluation.py --source walmart\n```"
    )
    st.stop()

if source == "synthetic":
    st.warning(
        "**Synthetic demo series** — generated by `forecast.data_generator` "
        "(seed 42), not real retail data. Switch the data source in the sidebar to "
        "Walmart for the real dataset."
    )
else:
    st.caption(
        f"Real data: Walmart Recruiting — Store Sales Forecasting · "
        f"{len(series)} weeks ({series['ds'].min().date()} → {series['ds'].max().date()})"
    )

min_train = min(104, max(26, len(series) - horizon))
overview, prophet_tab, bayesian_tab, compare_tab, scenario_tab = st.tabs(
    ["Overview & data", "Prophet", "Bayesian", "Comparison & coverage", "Scenario"]
)

# ------------------------------------------------------------------- overview
with overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weeks of history", f"{len(series)}")
    c2.metric("Mean weekly demand", f"{series['y'].mean():,.0f}")
    c3.metric("Holiday weeks", f"{int(series['IsHoliday'].sum())}")
    with c4:
        parts = stl_decompose(series.set_index("ds")["y"], period=52)
        c4.metric("Seasonal strength", f"{seasonal_strength(parts['seasonal'], parts['resid']):.2f}")
    st.plotly_chart(
        go.Figure(go.Scatter(x=series["ds"], y=series["y"], name="weekly demand", mode="lines",
                             line=dict(color="#1f77b4", width=1.5))),
        width="stretch",
    )
    st.plotly_chart(decomposition_figure(parts), width="stretch")
    with st.expander("Holiday / event calendar used by both models"):
        st.dataframe(walmart_holidays(), width="stretch", hide_index=True)

# -------------------------------------------------------------------- prophet
with prophet_tab:
    result = all_forecasts(source, horizon, engine_key)
    prophet = result["forecasts"].get("prophet")
    if prophet is None:
        # One block, platform-neutral: a Cloud visitor must not be told to run a
        # Windows CmdStan command.
        st.info(PROPHET_MISSING_MESSAGE, icon="ℹ️")
    else:
        st.caption("Prophet: piecewise-linear trend + yearly seasonality + Walmart holiday regressors.")
        st.plotly_chart(
            band_figure(prophet, series.tail(52), f"Prophet forecast, next {horizon} weeks"),
            width="stretch",
        )
        st.dataframe(
            rounded(prophet.rename(columns={"ds": "week", "yhat": "forecast"})),
            width="stretch", hide_index=True,
        )

# ------------------------------------------------------------------- bayesian
with bayesian_tab:
    result = all_forecasts(source, horizon, engine_key)
    bayes = result["forecasts"].get("bayesian")
    if bayes is None:
        st.warning(f"Bayesian model unavailable — {result['skipped'].get('bayesian')}")
    else:
        model = BayesianForecast(
            series, engine=engine_key, seed=42, **BAYES_OPTIONS
        ).fit()
        scale_note = "log(demand)" if LOG_TARGET else "raw demand"
        engine_note = (
            f"Engine: **{model.engine}** · modelled on {scale_note} · posterior "
            "predictive intervals (parameter uncertainty + observation noise)."
        )
        if model.engine != model.engine_requested:
            engine_note += (
                f" You requested `{model.engine_requested}`, which is not importable "
                f"here, so the exact conjugate engine is used instead."
            )
        st.caption(engine_note)
        st.plotly_chart(
            band_figure(bayes, series.tail(52), f"Bayesian forecast, next {horizon} weeks"),
            width="stretch",
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Posterior summary**")
            st.dataframe(
                model.parameter_summary().round(2), width="stretch", hide_index=True
            )
        with right:
            st.markdown("**Forecast**")
            st.dataframe(
                rounded(bayes.rename(columns={"ds": "week", "yhat": "median"})),
                width="stretch", hide_index=True,
            )
        st.info(
            "The point forecast is the posterior *median* (robust to the right skew of "
            "sales), not the mode of a Gaussian fit."
        )

# ------------------------------------------------------ comparison & coverage
with compare_tab:
    st.markdown(
        "**Expanding-window walk-forward validation.** Each fold trains on everything "
        "before it, forecasts the next block of weeks, and the out-of-sample predictions "
        "are pooled. PICP = share of actuals inside the interval (target: the nominal "
        "level). MPIW = average interval width (smaller is better). Winkler = interval "
        "score (lower is better; punishes width and misses)."
    )
    mode = st.radio(
        "Results",
        ["Measured results (from reports/)", "Live re-run in this environment"],
        horizontal=True,
        help=(
            "Measured = the committed walk-forward results, including Prophet, computed "
            "offline by scripts/run_evaluation.py. Live = refit here for the models "
            "installed in this environment."
        ),
    )

    if mode.startswith("Measured"):
        try:
            measured = load_measured_table()
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Committed measured results unavailable: {exc}")
        else:
            meta = measured["meta"]
            diffs = measured_config_mismatches(
                meta,
                horizon=horizon,
                max_folds=max_folds,
                engine=engine_key,
                min_train=min_train,
            )
            if diffs:
                st.warning(
                    "Your sidebar settings differ from how this table was measured "
                    f"({'; '.join(diffs)}). The numbers below are the **committed run**, "
                    "not a re-run under the current settings — switch to "
                    "'Live re-run in this environment' to compare."
                )
            engines = ", ".join(
                f"{name}: {value}" for name, value in (meta.get("engines_used") or {}).items()
                if value
            )
            measured_scale = (
                "log(demand)" if meta.get("log_target") else "raw demand"
            ) if meta.get("log_target") is not None else "unknown scale"
            prophet_note = (
                ""
                if prophet_available()
                else " Prophet is included because it was fitted offline; it is not "
                "installed in this environment."
            )
            st.caption(
                f"{meta.get('source_label', 'committed series')} · "
                f"{meta.get('folds', '?')} folds × {meta.get('horizon', '?')} weeks = "
                f"{meta.get('n_eval', '?')} out-of-sample weeks · "
                f"engine {meta.get('engine_requested', '?')} on {measured_scale}"
                + (f" ({engines})" if engines else "")
                + prophet_note
            )
            render_comparison_outputs(measured["table"])
    else:
        if prophet_available():
            use_prophet = st.checkbox("Include Prophet (slower: one fit per fold)", value=False)
            result = (
                comparison_prophet(source, horizon, min_train, max_folds)
                if use_prophet
                else comparison(source, horizon, engine_key, min_train, max_folds)
            )
        else:
            st.caption(
                "Prophet is not installed in this environment, so it cannot be re-run "
                "here — its measured results are in the 'Measured results' view above."
            )
            result = comparison(source, horizon, engine_key, min_train, max_folds)
        render_comparison_outputs(result["table"], result.get("skipped"))

# -------------------------------------------------------------------- scenario
with scenario_tab:
    st.markdown(
        "**What-if demand scenarios** scale the entire predictive distribution — point "
        "*and* both bands — so a scenario is a range, not a guess. Counterfactual: "
        "model-based, not observed."
    )
    result = all_forecasts(source, horizon, engine_key)
    base = result["forecasts"].get("bayesian")
    if base is None:
        st.warning("Bayesian model unavailable — cannot build scenarios.")
    else:
        uplift = st.slider("Demand change (%)", -50, 100, 20)
        only_next = st.checkbox("Apply only to the first 4 forecast weeks", value=False)
        factor = 1.0 + uplift / 100.0
        dates = base["ds"].iloc[:4] if only_next else None
        scenario = apply_scenario(base, factor=factor, dates=dates)
        scope = f"weeks 1–4 only" if only_next else f"all {horizon} weeks"
        st.plotly_chart(
            band_figure(
                scenario, series.tail(26),
                f"Scenario: demand {uplift:+d}% ({scope}) — counterfactual band",
            ),
            width="stretch",
        )
        impact = scenario_impact(base, scenario, level=0.95)
        st.markdown("**Impact vs baseline**")
        st.dataframe(rounded(impact), width="stretch", hide_index=True)
        st.caption(
            "Positive delta = extra units demanded under the scenario. Because the bands "
            "scale too, the interval shows how much the range of outcomes moves, not "
            "just its midpoint."
        )
