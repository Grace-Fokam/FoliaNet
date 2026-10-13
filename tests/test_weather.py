"""Weather aggregation checks using HTTP fixtures, never live downloads."""

from datetime import date
import sys
from types import SimpleNamespace

import pytest

from folianet.features.weather import fetch_weather


def install_response(monkeypatch, hourly, captured):
    def get(url, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"hourly": hourly})
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get))


def test_weather_window_and_aggregation(monkeypatch):
    captured = {}
    install_response(monkeypatch, {"temperature_2m": [26.0] * 168,
                                  "relative_humidity_2m": [95.0] * 168,
                                  "precipitation": [0.5] * 168}, captured)
    weather = fetch_weather(40, -88, date(2024, 7, 15))
    assert captured["params"]["start_date"] == "2024-07-09"
    assert captured["params"]["end_date"] == "2024-07-15"
    assert weather.data_available is True
    assert weather.temp_mean_c == 26
    assert weather.precip_7d_mm == 84
    assert weather.leaf_wetness_hours == 168
    assert weather.gdd == 112


@pytest.mark.parametrize("hourly", [
    {"temperature_2m": [26, 26], "relative_humidity_2m": [None, 95],
     "precipitation": [1, 0]},
    {"temperature_2m": [26], "relative_humidity_2m": [95]},
    {"temperature_2m": [26, 26], "relative_humidity_2m": [95],
     "precipitation": [1, 0]},
    {"temperature_2m": [float("nan")], "relative_humidity_2m": [95],
     "precipitation": [1]},
])
def test_incomplete_weather_is_labeled_fallback(monkeypatch, hourly):
    install_response(monkeypatch, hourly, {})
    assert fetch_weather(40, -88, "2024-07-15").data_available is False


def test_network_failure_is_labeled_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fail))
    assert fetch_weather(40, -88, "2024-07-15").data_available is False


def test_invalid_window():
    with pytest.raises(ValueError, match="positive"):
        fetch_weather(40, -88, "2024-07-15", window_days=0)
