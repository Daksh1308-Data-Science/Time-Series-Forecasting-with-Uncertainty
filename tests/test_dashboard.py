"""Headless dashboard check: the app must render all five tabs without error.

This is the only smoke test in the suite, and it earns its place: a Streamlit
app can serve HTTP 200 while its script throws on first run. It runs against the
synthetic source (the sidebar default), so it needs no Kaggle credentials.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_renders_all_tabs_without_exception():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert len(app.tabs) == 5


def test_dashboard_warns_that_the_default_series_is_synthetic():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    warnings = " ".join(item.value for item in app.warning)
    assert "Synthetic demo series" in warnings
