import pandas as pd

from forecast.prophet_model import PROPHET_MISSING_MESSAGE, prophet_available


def test_prophet_available_returns_bool():
    assert isinstance(prophet_available(), bool)


def test_prophet_missing_message_is_platform_neutral():
    """A Cloud (Linux) visitor must not be told to run a Windows CmdStan command."""
    lowered = PROPHET_MISSING_MESSAGE.lower()
    for leaked in ("windows", "cmdstanpy", "install_cxx_toolchain", "mingw", "rt.exe"):
        assert leaked not in lowered, f"{leaked!r} leaked into the runtime message"
    # It must still say what to do and what the reader gets instead.
    assert "requirements-optional.txt" in PROPHET_MISSING_MESSAGE
    assert "Comparison tab" in PROPHET_MISSING_MESSAGE


def test_prophet_forecast_raises_the_shared_message_when_absent(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "prophet":
            raise ImportError("no prophet here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from forecast.prophet_model import prophet_forecast

    frame = pd.DataFrame({"ds": pd.date_range("2020-01-03", periods=60, freq="7D"), "y": 1.0})
    try:
        prophet_forecast(frame, periods=2)
    except ImportError as exc:
        assert str(exc) == PROPHET_MISSING_MESSAGE
    else:  # pragma: no cover - only when prophet is unexpectedly importable
        pass
