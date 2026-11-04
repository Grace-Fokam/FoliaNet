"""NDVI (how green and dense plants are). Still a placeholder.

For now this returns fake values with data_available=False, and they aren't
used in the score. To add real data, fill in _fetch_real_ndvi(). Sentinel Hub
(https://docs.sentinel-hub.com) or Google Earth Engine would work."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Union
import hashlib


@dataclass
class SatelliteFeatures:
    ndvi: float                 # -1 to 1
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
    """TODO: hook up a real satellite source. Return None if there's no data.

    Keep API keys in environment variables, not here.
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
