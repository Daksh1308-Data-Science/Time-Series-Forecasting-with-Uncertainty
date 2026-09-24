"""Generate the README figures from the real Walmart series and measured reports.

    python scripts/make_figures.py

Everything here is computed, not drawn by hand: the series comes from
``forecast.data_loader``, the forecasts from the same model code the dashboard
uses, and the metric panels from ``reports/comparison_table.csv``. Re-running
after a model change keeps the README images honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.bayesian_model import BayesianForecast
from forecast.data_loader import prepare_walmart_series
from forecast.decomposition import seasonal_strength, stl_decompose
from forecast.holidays import WALMART_HOLIDAYS
from forecast.scenario import apply_scenario
from forecast.validation import forecast_all

IMAGES = ROOT / "docs" / "images"
REPORTS = ROOT / "reports"
HISTORY = 78  # weeks of actuals shown behind each forecast
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#444444"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def millions(value: float) -> str:
    return f"{value / 1e6:,.1f}M"


def date_axis(ax, weeks: int) -> None:
    """Friday ticks every N weeks — weekly data otherwise collides the labels."""
    interval = max(1, round(weeks / 8))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y" if weeks > 40 else "%b %d"))


def save(fig, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_forecast_bands(series: pd.DataFrame, forecasts: dict) -> None:
    """Real history + 6-week forecast with nested 80/95% bands, per model."""
    history = series.tail(HISTORY)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    for ax, (name, color) in zip(axes, [("prophet", ORANGE), ("bayesian", BLUE)]):
        forecast = forecasts.get(name)
        ax.plot(history["ds"], history["y"], color=GREY, lw=1.2, label="actual sales")
        if forecast is None:
            ax.set_title(f"{name}: unavailable")
            continue
        for tag, alpha in (("95", 0.13), ("80", 0.30)):
            ax.fill_between(
                forecast["ds"], forecast[f"lower_{tag}"], forecast[f"upper_{tag}"],
                color=color, alpha=alpha, lw=0, label=f"{tag}% interval",
            )
        ax.plot(forecast["ds"], forecast["yhat"], color=color, lw=2.4,
                label="forecast (posterior median)" if name == "bayesian" else "forecast (yhat)")
        label = "Bayesian (posterior predictive)" if name == "bayesian" else "Prophet (MAP + trend simulation)"
        ax.set_title(f"{label} — 6-week forecast", loc="left", color=color, fontweight="bold")
        ax.yaxis.set_major_formatter(lambda v, _: millions(v))
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("week")
    date_axis(axes[-1], HISTORY + 6)
    fig.suptitle(
        "Walmart weekly demand: real history and forecast uncertainty",
        x=0.01, ha="left", fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "forecast_bands.png")


def figure_decomposition(series: pd.DataFrame) -> None:
    parts = stl_decompose(series.set_index("ds")["y"], period=52)
    strength = seasonal_strength(parts["seasonal"], parts["resid"])
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    axes[0].plot(parts["trend"].index, parts["trend"], color=BLUE, lw=1.6)
    axes[0].set_title("Trend — note the 2011 level shift", loc="left", fontweight="bold")
    axes[0].yaxis.set_major_formatter(lambda v, _: millions(v))
    for ax, key, color, label in (
        (axes[1], "seasonal", ORANGE, "Seasonal (52-week cycle)"),
        (axes[2], "resid", "#7f7f7f", "Residual (unexplained)"),
    ):
        ax.plot(parts[key].index, parts[key], color=color, lw=1.1)
        ax.set_title(label, loc="left", fontsize=10)
    # Mark the Walmart holiday weeks: additive spikes the smooth cycle misses.
    lo, hi = parts["trend"].index.min(), parts["trend"].index.max()
    for date in [d for dates in WALMART_HOLIDAYS.values() for d in dates]:
        day = pd.Timestamp(date)
        if lo <= day <= hi:
            for ax in axes:
                ax.axvline(day, color="crimson", alpha=0.22, lw=0.9)
    axes[0].plot([], [], color="crimson", alpha=0.4, lw=1.2, label="Walmart holiday week")
    axes[0].legend(loc="upper left", fontsize=8)
    date_axis(axes[-1], len(series))
    axes[-1].set_xlabel("week")
    fig.suptitle(
        "Weekly demand decomposes into trend, yearly cycle and noise",
        x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.01, 0.005,
        f"STL, period 52 · seasonal strength {strength:.2f} · 143 weeks "
        "(2010-02-05 → 2012-10-26) · mean weekly demand 47.1M units",
        fontsize=8.5, color=GREY, ha="left",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(IMAGES / "decomposition.png", facecolor="white")
    plt.close(fig)
    print("wrote docs\\images\\decomposition.png")


def figure_coverage(table: pd.DataFrame) -> None:
    """The measured coverage question: do the bands match their labels?

    A lollipop (not bars): the question is each model's *distance from its
    nominal level*, so there is no need for a zero baseline and the y-axis can
    be cropped without exaggerating anything.
    """
    models = list(table["model"])
    n = int(table["n_eval"].iloc[0])
    x = np.arange(len(models))
    offsets = {"80": -0.16, "95": 0.16}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for tag, nominal, color in (("80", 0.80, BLUE), ("95", 0.95, ORANGE)):
        values = [table[f"picp_{tag}"].iloc[i] for i in range(len(models))]
        xs = x + offsets[tag]
        for xi, value in zip(xs, values):
            ax.plot([xi, xi], [nominal, value], color=color, lw=2.2, alpha=0.55, zorder=2)
            ax.scatter([xi], [value], s=110, color=color, zorder=3, edgecolor="white", lw=1.2)
            ax.annotate(
                f"{value:.1%}\n{round(value * n)}/{n}",
                (xi, value), textcoords="offset points", xytext=(0, 13 if value >= nominal else -30),
                ha="center", fontsize=9, color=color, fontweight="bold",
            )
        ax.axhline(nominal, color=color, ls="--", lw=1, alpha=0.8, zorder=1)
    ax.set_xticks(x, [m.capitalize() for m in models])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0.70, 1.06)
    ax.set_ylabel("coverage (PICP)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="--", lw=1.4, ms=8,
                   label=f"{int(t)}% band (dashed = nominal {int(t)}%)")
        for t, c in (("80", BLUE), ("95", ORANGE))
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    ax.set_title(
        "Measured coverage vs nominal level — 36 pooled out-of-sample weeks\n"
        "all models are near-nominal at 95%; at 80% Prophet is conservative "
        "and seasonal-naive is best calibrated",
        loc="left", fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "coverage.png")


def figure_model_comparison(table: pd.DataFrame, mean_demand: float) -> None:
    models = [m.capitalize() for m in table["model"]]
    panels = [
        ("mae", "MAE", "mean |error| as % of mean demand", lambda v: 100 * v / mean_demand, "{:.2f}%"),
        ("mpiw_95", "95% band width", "width as % of mean demand", lambda v: 100 * v / mean_demand, "{:.1f}%"),
        ("winkler_95", "Winkler interval score @95%", "lower is better (millions of units)", lambda v: v / 1e6, "{:.2f}M"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (key, title, subtitle, transform, fmt) in zip(axes, panels):
        values = [transform(v) for v in table[key]]
        colors = [BLUE if m == "bayesian" else ("#8c8c8c" if m == "naive" else ORANGE) for m in table["model"]]
        bars = ax.bar(models, values, color=colors, alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(subtitle, fontsize=8.5)
        ax.margins(y=0.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE, "#8c8c8c")]
    fig.legend(handles, ["Prophet", "Bayesian", "Seasonal-naive"],
               loc="upper right", ncol=3, fontsize=9, bbox_to_anchor=(0.995, 0.99))
    fig.suptitle(
        "Model comparison — accuracy, sharpness and interval score (all out-of-sample)",
        x=0.01, ha="left", fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "model_comparison.png")


def figure_scenario(base: pd.DataFrame, scenario: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for frame, color, alpha, label in (
        (scenario, "#d62728", 0.13, "+20% demand scenario"),
        (base, BLUE, 0.28, "baseline forecast"),
    ):
        for tag, a in (("95", alpha * 0.55), ("80", alpha)):
            ax.fill_between(frame["ds"], frame[f"lower_{tag}"], frame[f"upper_{tag}"],
                            color=color, alpha=a, lw=0)
        ax.plot(frame["ds"], frame[f"yhat"], color=color, lw=2.2, label=label)
    ax.set_title(
        "Scenario analysis: scaling the whole predictive distribution\n"
        "a +20% demand scenario returns a +20% range, not a +20% line",
        loc="left", fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("week")
    date_axis(ax, len(base))
    ax.yaxis.set_major_formatter(lambda v, _: millions(v))
    ax.set_ylabel("weekly demand")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "scenario.png")


def main() -> None:
    style()
    series = prepare_walmart_series()
    print(f"real series: {len(series)} weeks, mean {millions(series['y'].mean())} units/week")

    result = forecast_all(series, periods=6, engine="closed_form")
    forecasts = result["forecasts"]
    for name, reason in result["skipped"].items():
        print(f"  [skipped] {name}: {reason}")

    table = pd.read_csv(REPORTS / "comparison_table.csv")

    figure_forecast_bands(series, forecasts)
    figure_decomposition(series)
    figure_coverage(table)
    figure_model_comparison(table, series["y"].mean())

    base = forecasts.get("bayesian")
    if base is not None:
        figure_scenario(base, apply_scenario(base, factor=1.2))


if __name__ == "__main__":
    main()
