"""Test the example and the rules for real, made-up, and user-supplied data."""

import json
import subprocess
import sys

import pytest

from folianet.context import environmental_context
from folianet.example import example_response
from folianet.features.environment import WeatherFeatures
from folianet.features.satellite import SatelliteFeatures


def test_example_returns_independent_objects():
    first = example_response()
    first["details"]["class_probabilities"].clear()
    assert example_response()["details"]["class_probabilities"]


def test_cli_matches_api_fixture():
    result = subprocess.run([sys.executable, "-m", "folianet.example"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == example_response()


@pytest.mark.parametrize("weather_real,satellite_real,override", [
    (False, False, None), (True, False, None), (False, True, None),
    (True, True, None), (False, False, 0.0), (True, True, 0.5),
])
def test_environment_policy_and_provenance(weather_real, satellite_real, override):
    weather = WeatherFeatures(26, 90, 20, 12, 112, weather_real)
    satellite = SatelliteFeatures(0.62, satellite_real,
                                  "satellite" if satellite_real else "synthetic")
    context = environmental_context(weather, satellite, override)
    assert context["available"] is weather_real
    assert context["data_sources"]["satellite"] is (satellite_real and override is None)
    expected_ndvi = override if override is not None else (0.62 if satellite_real else None)
    assert context["usable_ndvi"] == expected_ndvi
    assert context["provenance"]["ndvi_used_in_favorability"] is (
        weather_real and expected_ndvi is not None)
    if override is not None:
        assert context["provenance"]["ndvi"] == "user_override_unverified"


def test_browsable_json_matches_example():
    from pathlib import Path
    fixture = Path(__file__).resolve().parents[1] / "examples" / "prediction-response.json"
    assert json.loads(fixture.read_text()) == example_response()
