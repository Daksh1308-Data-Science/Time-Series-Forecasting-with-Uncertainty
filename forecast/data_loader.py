"""Real Walmart retail demand data: acquisition, loading, aggregation.

Source: the Kaggle competition *Walmart Recruiting - Store Sales Forecasting*
(45 stores, 99 departments, weekly sales 2010-02-05 .. 2012-10-26). Files
require a Kaggle account and accepting the competition rules.

Provenance rules for this project:
  * Real data is labelled "Walmart" everywhere it is shown.
  * The synthetic generator is labelled "synthetic" everywhere it is shown.
  * The processed weekly aggregate is committed; the raw competition CSVs are
    not (they are subject to the competition rules).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from .data_generator import normalize_weekly_ds

COMPETITION = "walmart-recruiting-store-sales-forecasting"
COMPETITION_URL = "https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting"
RAW_FILES = ("train.csv", "features.csv", "stores.csv")
CACHED_SERIES = Path("data") / "processed" / "weekly_total.csv"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def kaggle_credentials_present() -> bool:
    """True if a Kaggle API token is available to kagglehub."""
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json",
    ]
    return any(c.is_file() for c in candidates if str(c))


def download_walmart(raw_dir: Path | None = None, force: bool = False) -> Path:
    """Download the competition CSVs and return the directory holding them.

    Raises FileNotFoundError when no Kaggle token is configured and RuntimeError
    when Kaggle refuses the download (usually because the competition rules
    have not been accepted in the browser).
    """
    if not kaggle_credentials_present():
        raise FileNotFoundError(
            "No Kaggle credentials found. Create ~/.kaggle/kaggle.json "
            "(Kaggle > Settings > API > Create New Token) and accept the "
            f"competition rules at {COMPETITION_URL}."
        )
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("kagglehub is not installed; pip install kagglehub") from exc

    try:
        source = Path(kagglehub.competition_download(COMPETITION))
    except Exception as exc:  # noqa: BLE001 - kaggle raises many HTTP errors
        raise RuntimeError(
            f"Kaggle download failed ({exc}). Make sure you accepted the rules at {COMPETITION_URL}."
        ) from exc

    target = raw_dir or (project_root() / "data" / "raw")
    target.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        found = next((p for p in source.rglob(name) if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"{name} missing from the downloaded archive at {source}")
        destination = target / name
        if force or not destination.exists():
            shutil.copy2(found, destination)
    return target


def load_raw(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the raw competition CSVs (train, features, stores) from ``data_dir``."""
    data_dir = Path(data_dir) if data_dir else project_root() / "data" / "raw"
    frames = {}
    for name in RAW_FILES:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run forecast.data_loader.download_walmart() first."
            )
        frames[Path(name).stem] = pd.read_csv(path, parse_dates=["Date"])
    return frames


def aggregate_weekly_sales(
    train_df: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse store x department weekly sales into one weekly demand series.

    Returns columns ``ds`` (Friday of each week), ``y`` (total weekly sales) and
    ``IsHoliday`` (any Walmart special week).
    """
    sales = train_df.copy()
    sales["ds"] = normalize_weekly_ds(sales["Date"])
    weekly_y = sales.groupby("ds", as_index=True)["Weekly_Sales"].sum().sort_index()

    if features_df is not None and "IsHoliday" in features_df:
        feats = features_df.copy()
        feats["ds"] = normalize_weekly_ds(feats["Date"])
        holiday = feats.groupby("ds", as_index=True)["IsHoliday"].max()
    else:
        holiday = sales.groupby("ds", as_index=True)["IsHoliday"].max()

    holiday = holiday.reindex(weekly_y.index).fillna(False)
    return pd.DataFrame(
        {
            "ds": weekly_y.index,
            "y": weekly_y.to_numpy(dtype=float),
            "IsHoliday": holiday.to_numpy(dtype=bool),
        }
    ).reset_index(drop=True)


def load_cached_series(cache_path: Path | None = None) -> pd.DataFrame:
    """Load the committed processed weekly series (``data/processed``)."""
    cache_path = Path(cache_path) if cache_path else project_root() / CACHED_SERIES
    if not cache_path.is_file():
        raise FileNotFoundError(f"processed series not found at {cache_path}")
    df = pd.read_csv(cache_path, parse_dates=["ds"])
    return df


def prepare_walmart_series(
    data_dir: Path | None = None,
    download: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return the real weekly Walmart series (cached aggregate if available).

    Resolution order: processed cache -> raw CSVs -> (optionally) download.
    """
    if use_cache:
        try:
            return load_cached_series()
        except FileNotFoundError:
            pass

    try:
        frames = load_raw(data_dir)
    except FileNotFoundError:
        if not download:
            raise
        download_walmart(data_dir)
        frames = load_raw(data_dir)

    series = aggregate_weekly_sales(frames["train"], frames.get("features"))
    if use_cache:
        cache_path = project_root() / CACHED_SERIES
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(cache_path, index=False)
    return series
