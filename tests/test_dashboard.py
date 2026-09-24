"""Headless dashboard check: the app must render all five tabs without error.

This is the only smoke test in the suite, and it earns its place: a Streamlit
app can serve HTTP 200 while its script throws on first run. It runs against
both data sources and both Comparison-tab modes, so it needs no Kaggle
credentials and no optional heavy dependencies.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
SYNTHETIC = "Synthetic demo series (seed 42)"
LIVE = "Live re-run in this environment"


def test_dashboard_renders_all_tabs_on_real_data():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert len(app.tabs) == 5
    captions = " ".join(item.value for item in app.caption)
    assert "Real data: Walmart Recruiting" in captions
    # Regression guard: the app must use the configured log(demand) model.
    # It used to fall back to the level-space default, which made the live
    # re-run disagree with the committed measured table by 7-23%.
    assert "modelled on log(demand)" in captions


def _dataframes(app):
    tables = []
    for element in app.dataframe:
        value = getattr(element, "value", None)
        if value is not None and hasattr(value, "columns"):
            tables.append(value)
    return tables


def test_comparison_tab_defaults_to_measured_results_including_prophet():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert app.radio[0].value.startswith("Measured results")

    comparison_tables = [t for t in _dataframes(app) if "model" in t.columns]
    assert comparison_tables, "comparison table did not render"
    assert "prophet" in set(comparison_tables[0]["model"])

    # Regression guard: the sidebar defaults must match the committed measured
    # run, otherwise every visitor sees a "settings differ" warning on load.
    warnings = " ".join(item.value for item in app.warning)
    assert "differ from how this table was measured" not in warnings
    assert "not installed in this environment" not in warnings


def test_comparison_tab_live_mode_runs():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    app.radio[0].set_value(LIVE).run()
    assert not app.exception, [str(error.value) for error in app.exception]


def test_dashboard_switches_to_synthetic_and_labels_it():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    app.selectbox[0].select(SYNTHETIC).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    warnings = " ".join(item.value for item in app.warning)
    assert "Synthetic demo series" in warnings
