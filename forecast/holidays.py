"""Walmart holiday events (from the Kaggle competition documentation).

The competition data flags four special weeks per year. The date strings below
are the Friday week-dates used in ``features.csv``/``train.csv`` for those
weeks, so the event dates line up exactly with the ``IsHoliday`` flag.
"""

from __future__ import annotations

import pandas as pd

WALMART_HOLIDAYS: dict[str, list[str]] = {
    "Super Bowl": ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "Labor Day": ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "Christmas": ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}


def walmart_holidays() -> pd.DataFrame:
    """Return the event calendar as a long ``(holiday, ds)`` table.

    The shape is Prophet's ``holidays`` argument, so this can be passed straight
    to ``Prophet(holidays=...)``.
    """
    rows = [
        {"holiday": name, "ds": pd.Timestamp(date)}
        for name, dates in WALMART_HOLIDAYS.items()
        for date in dates
    ]
    return pd.DataFrame(rows, columns=["holiday", "ds"])


def holiday_indicators(
    ds: pd.Series | pd.DatetimeIndex,
    holiday_dates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """One-hot event columns aligned to ``ds`` (missing weeks are all zeros)."""
    dates = pd.Series(pd.to_datetime(ds)).reset_index(drop=True)
    table = walmart_holidays() if holiday_dates is None else pd.DataFrame(
        [
            {"holiday": name, "ds": pd.Timestamp(date)}
            for name, event_dates in holiday_dates.items()
            for date in event_dates
        ],
        columns=["holiday", "ds"],
    )
    out = {}
    for name in table["holiday"].unique():
        event_dates = set(table.loc[table["holiday"] == name, "ds"])
        out[name] = [1 if d in event_dates else 0 for d in dates]
    return pd.DataFrame(out, index=dates.index, columns=sorted(out))
