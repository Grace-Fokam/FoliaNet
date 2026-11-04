"""Scores how good the weather is for each disease, from 0 to 1.

Simple rules, not a model. Tweak them in folianet/diseases.py. No PyTorch needed."""

from dataclasses import dataclass
from typing import Dict, Optional

from folianet.diseases import DISEASES, Disease, Favorability


@dataclass
class WeatherFeatures:
    temp_mean_c: float          # average temp (C)
    humidity_mean_pct: float    # average humidity (%)
    precip_7d_mm: float         # rain over the last 7 days (mm)
    leaf_wetness_hours: float   # estimated hours the leaves were wet
    gdd: float                  # growing degree days
    data_available: bool = True # False if the values are fake fallback


def _temperature_suitability(temp: float, fav: Favorability) -> float:
    """0 if too hot or cold, 1 in the best range, in between otherwise."""
    if temp <= fav.t_min or temp >= fav.t_max:
        return 0.0
    if fav.t_opt_low <= temp <= fav.t_opt_high:
        return 1.0
    if temp < fav.t_opt_low:
        return (temp - fav.t_min) / max(fav.t_opt_low - fav.t_min, 1e-6)
    return (fav.t_max - temp) / max(fav.t_max - fav.t_opt_high, 1e-6)


def _humidity_factor(humidity: float, fav: Favorability) -> float:
    if fav.humidity_min <= 0:
        return 0.0
    if humidity <= fav.humidity_min:
        return 0.0
    # Goes up evenly from humidity_min to 100%.
    return min((humidity - fav.humidity_min) / max(100 - fav.humidity_min, 1e-6), 1.0)


def _wetness_factor(wetness_hours: float, fav: Favorability) -> float:
    if fav.wetness_hours <= 0:
        return 0.0
    return min(wetness_hours / fav.wetness_hours, 1.0)


def disease_favorability(weather: WeatherFeatures, disease: Disease) -> float:
    """Score from 0 to 1 using temperature, humidity, and wetness."""
    if disease.is_healthy:
        return 0.0
    fav = disease.favorability
    t = _temperature_suitability(weather.temp_mean_c, fav)
    h = _humidity_factor(weather.humidity_mean_pct, fav)
    w = _wetness_factor(weather.leaf_wetness_hours, fav)
    # Temperature caps it; humidity and wetness fill it in.
    score = t * (0.5 * h + 0.5 * w)
    return float(max(0.0, min(score, 1.0)))


def crop_favorability(
    weather: WeatherFeatures,
    crop: str,
    ndvi: Optional[float] = None,
) -> Dict[str, float]:
    """Score each disease for this crop. Lower NDVI bumps the score up a bit."""
    canopy_modifier = 1.0
    if ndvi is not None:
        # NDVI 0.2 (stressed) -> 1.15x, NDVI 0.8 (healthy) -> ~0.95x.
        canopy_modifier = max(0.9, min(1.2, 1.25 - 0.4 * ndvi))

    out: Dict[str, float] = {}
    for class_name, disease in DISEASES.items():
        if disease.crop != crop or disease.is_healthy:
            continue
        out[disease.key] = min(1.0, disease_favorability(weather, disease) * canopy_modifier)
    return out
