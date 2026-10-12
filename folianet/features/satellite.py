"""The NDVI data source is still a placeholder.

NDVI describes how green or dense vegetation is. This file currently returns
repeatable made-up values with data_available=False. The predictor does not
use these values to adjust disease scores.

To use real data, add a source in _fetch_real_ndvi(). Possible sources include
Sentinel Hub (https://docs.sentinel-hub.com) and Google Earth Engine."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Union
import hashlib


@dataclass
class SatelliteFeatures:
    ndvi: float                 # -1..1 (vegetation density / vigor)
    data_available: bool = True
    source: str = "synthetic"


def _to_date(d: Union[str, date, datetime]) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def _synthetic_ndvi(lat: float, lon: float, target: date) -> float:
    seed = int(hashlib.sha256(f"ndvi{lat:.2f}{lon:.2f}{target}".encode()).hexdigest(), 16)
    return round(0.3 + 0.5 * ((seed % 1000) / 1000.0), 3)   # 0.30..0.80


def _fetch_real_ndvi(lat: float, lon: float, target: date) -> Optional[float]:
    """Add a satellite data source here. Return None if no data is available.

    For example, Sentinel Hub can return NDVI for a small area around the
    coordinates. Keep any API keys in environment variables, not in this file.
    """
    return None


def fetch_ndvi(
    lat: float,
    lon: float,
    target_date: Union[str, date, datetime],
) -> SatelliteFeatures:
    target = _to_date(target_date)
    real = _fetch_real_ndvi(lat, lon, target)
    if real is not None:
        return SatelliteFeatures(ndvi=float(real), data_available=True, source="satellite")
    return SatelliteFeatures(ndvi=_synthetic_ndvi(lat, lon, target),
                             data_available=False, source="synthetic")
