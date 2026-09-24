"""Streamlit dashboard: forecast exploration with honest uncertainty.

    streamlit run dashboard/app.py

Five tabs: Overview (data + decomposition) | Prophet | Bayesian | Comparison &
Coverage | Scenario. Every number is computed live by the ``forecast`` library —
no placeholders. Heavy work is cached; the app runs even if PyMC and Prophet
are not installed (each model is reported as skipped, with the reason).
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
from forecast.scenario import apply_scenario, scenario_impact
from forecast.validation import forecast_all, run_comparison

st.set_page_config(page_title="Retail Demand Forecasting with Uncertainty", layout="wide")

LEVELS = (0.80, 0.95)


# --------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def load_series(source: str, n_weeks: int = 156):
    if source == "walmart":
        return prepare_walmart_series(download=False)
    return generate_weekly_sales(n_weeks=n_weeks, seed=42)


@st.cache_data(show_spinner="Fitting models on the full history...")
def all_forecasts(source: str, horizon: int, engine: str):
    series = load_series(source)
    return forecast_all(series, periods=horizon, engine=engine, levels=LEVELS)


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
    ["Synthetic demo series (seed 42)", "Walmart — real Kaggle data"],
)
source = "walmart" if source_label.startswith("Walmart") else "synthetic"
horizon = st.sidebar.slider("Forecast horizon (weeks)", 1, 12, 6)
engine = st.sidebar.selectbox(
    "Bayesian engine",
    ["closed_form (exact, instant)", "auto (PyMC if available)", "pymc (NUTS sampler)"],
)
engine_key = {"closed_form (exact, instant)": "closed_form", "auto (PyMC if available)": "auto", "pymc (NUTS sampler)": "pymc"}[engine]
max_folds = st.sidebar.slider("Walk-forward folds", 1, 8, 4)
st.sidebar.caption(
    f"pymc importable: **{resolve_engine('auto') == 'pymc'}** · "
    f"chosen engine: **{engine_key}**"
)

def rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns to whole units (dates are left alone)."""
    return df.round({column: 0 for column in df.select_dtypes("number").columns})


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
        use_container_width=True,
    )
    st.plotly_chart(decomposition_figure(parts), use_container_width=True)
    with st.expander("Holiday / event calendar used by both models"):
        st.dataframe(walmart_holidays(), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------- prophet
with prophet_tab:
    result = all_forecasts(source, horizon, engine_key)
    prophet = result["forecasts"].get("prophet")
    if prophet is None:
        st.warning(f"Prophet unavailable — {result['skipped'].get('prophet', 'unknown error')}")
        st.markdown(
            "Install it with `pip install -r requirements-optional.txt`. On Windows the "
            "first fit compiles CmdStan: "
            "`python -c \"import cmdstanpy; cmdstanpy.install_cxx_toolchain(force=True)\"`."
        )
    else:
        st.caption("Prophet: piecewise-linear trend + yearly seasonality + Walmart holiday regressors.")
        st.plotly_chart(
            band_figure(prophet, series.tail(52), f"Prophet forecast, next {horizon} weeks"),
            use_container_width=True,
        )
        st.dataframe(
            rounded(prophet.rename(columns={"ds": "week", "yhat": "forecast"})),
            use_container_width=True, hide_index=True,
        )

# ------------------------------------------------------------------- bayesian
with bayesian_tab:
    result = all_forecasts(source, horizon, engine_key)
    bayes = result["forecasts"].get("bayesian")
    if bayes is None:
        st.warning(f"Bayesian model unavailable — {result['skipped'].get('bayesian')}")
    else:
        model = BayesianForecast(series, engine=engine_key, seed=42).fit()
        st.caption(
            f"Engine: **{model.engine}** · posterior predictive intervals "
            "(parameter uncertainty + observation noise)."
        )
        st.plotly_chart(
            band_figure(bayes, series.tail(52), f"Bayesian forecast, next {horizon} weeks"),
            use_container_width=True,
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Posterior summary**")
            st.dataframe(
                model.parameter_summary().round(2), use_container_width=True, hide_index=True
            )
        with right:
            st.markdown("**Forecast**")
            st.dataframe(
                rounded(bayes.rename(columns={"ds": "week", "yhat": "median"})),
                use_container_width=True, hide_index=True,
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
    use_prophet = st.checkbox("Include Prophet (slower: one fit per fold)", value=False)
    result = (
        comparison_prophet(source, horizon, min_train, max_folds)
        if use_prophet
        else comparison(source, horizon, engine_key, min_train, max_folds)
    )
    table = result["table"]
    if table.empty:
        st.warning("No model produced results.")
    else:
        pretty = table.rename(columns={
            "model": "model", "mae": "MAE", "rmse": "RMSE", "n_eval": "n (weeks)",
            "picp_80": "coverage 80%", "picp_95": "coverage 95%",
            "mpiw_80": "width 80%", "mpiw_95": "width 95%",
            "winkler_80": "Winkler 80%", "winkler_95": "Winkler 95%",
        })
        st.dataframe(pretty.round(3), use_container_width=True, hide_index=True)
        for name, reason in result["skipped"].items():
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
            x=melted["interval"].unique(), y=melted["nominal"].groupby(melted["interval"]).first().values,
            mode="markers", name="nominal level", marker=dict(symbol="line-ew-open", size=22, color="black"),
        ))
        bar.update_layout(
            barmode="group", template="plotly_white", height=380,
            title="Measured coverage vs nominal level (pooled out-of-sample)",
            yaxis=dict(title="coverage (PICP)", range=[0, 1.05]), margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(bar, use_container_width=True)

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
            use_container_width=True,
        )
        impact = scenario_impact(base, scenario, level=0.95)
        st.markdown("**Impact vs baseline**")
        st.dataframe(rounded(impact), use_container_width=True, hide_index=True)
        st.caption(
            "Positive delta = extra units demanded under the scenario. Because the bands "
            "scale too, the interval shows how much the range of outcomes moves, not "
            "just its midpoint."
        )
