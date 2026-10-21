"""Track where the weather and NDVI came from and which values can be used."""


def environmental_context(weather, satellite, ndvi_override=None):
    """Use real weather for scoring and leave out made-up NDVI."""
    ndvi = ndvi_override if ndvi_override is not None else satellite.ndvi
    ndvi_source = ("user_override_unverified" if ndvi_override is not None
                   else satellite.source)
    usable_ndvi = ndvi if ndvi_override is not None or satellite.data_available else None
    return {
        "ndvi": ndvi,
        "usable_ndvi": usable_ndvi,
        "available": weather.data_available,
        "data_sources": {
            "weather": weather.data_available,
            "satellite": satellite.data_available if ndvi_override is None else False,
        },
        "provenance": {
            "image": "trained_checkpoint",
            "weather": "open_meteo" if weather.data_available else "synthetic_fallback",
            "ndvi": ndvi_source,
            "ndvi_used_in_favorability": bool(weather.data_available and usable_ndvi is not None),
            "environment_used_in_risk": weather.data_available,
            "leaf_wetness": "humidity_precipitation_proxy" if weather.data_available
                            else "synthetic_fallback",
            "confidence": "heuristic_not_calibrated_probability",
            "recommendation": "rule_based_advisory_text",
        },
    }
