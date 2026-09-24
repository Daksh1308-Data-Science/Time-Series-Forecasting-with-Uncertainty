"""Headless dashboard check: the app must render all five tabs without error.

This is the only smoke test in the suite, and it earns its place: a Streamlit
app can serve HTTP 200 while its script throws on first run. It runs against
both data sources, so it needs no Kaggle download (the processed Walmart series
is committed) and no credentials.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
SYNTHETIC = "Synthetic demo series (seed 42)"
WALMART = "Walmart — real Kaggle data"


def test_dashboard_renders_all_tabs_on_real_data():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert len(app.tabs) == 5
    captions = " ".join(item.value for item in app.caption)
    assert "Real data: Walmart Recruiting" in captions


def test_dashboard_switches_to_synthetic_and_labels_it():
    app = AppTest.from_file(str(DASHBOARD), default_timeout=900).run()
    app.selectbox[0].select(SYNTHETIC).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    warnings = " ".join(item.value for item in app.warning)
    assert "Synthetic demo series" in warnings
