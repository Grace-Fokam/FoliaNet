"""Exercise actual HTTP routing with an offline fixture and controlled predictor."""

import builtins
import io

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import api.main as api


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_predictor", None)
    monkeypatch.setenv("FOLIANET_CHECKPOINT", str(tmp_path / "missing.pt"))
    return TestClient(api.app)


def image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    return buf.getvalue()


def post(client, **overrides):
    data = {"lat": "40", "lon": "-88.2", "date": "2024-07-15"}
    data.update(overrides)
    return client.post("/predict", data=data,
                       files={"image": ("leaf.png", image_bytes(), "image/png")})


def test_example_and_health_do_not_import_ml_or_call_network(client, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("torch", "folianet.inference", "folianet.features.weather",
                            "folianet.features.satellite", "requests")):
            raise AssertionError(f"Example must not import {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert client.get("/health").json()["status"] == "ok"
    response = client.get("/example")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "illustrative_example"
    assert body["is_prediction"] is False
    assert body["details"]["model_version"] is None
    assert body["details"]["data_sources"] == {"weather": False, "satellite": False}
    assert "Hand-authored" in body["notice"]
    assert sum(body["details"]["class_probabilities"].values()) == pytest.approx(1)
    assert body == client.get("/example").json()
    assert post(client).status_code == 503


@pytest.mark.parametrize("overrides", [
    {"lat": "91"}, {"lon": "-181"}, {"lat": "nan"}, {"lon": "inf"},
    {"ndvi": "1.1"}, {"ndvi": "nan"}, {"date": "2024-02-30"}, {"date": "invalid"},
])
def test_invalid_context_is_422_before_model_loading(client, overrides):
    assert post(client, **overrides).status_code == 422


def test_invalid_image_is_400(client):
    response = client.post("/predict", data={"lat": 40, "lon": -88, "date": "2024-07-15"},
                           files={"image": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_missing_fields_are_422(client):
    assert client.post("/predict").status_code == 422


def test_checkpoint_load_failure_is_503_without_internal_details(client, monkeypatch, tmp_path):
    ckpt = tmp_path / "broken.pt"
    ckpt.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("FOLIANET_CHECKPOINT", str(ckpt))
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "folianet.inference":
            raise RuntimeError("private checkpoint details")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    response = post(client)
    assert response.status_code == 503
    assert "private checkpoint details" not in response.text
    assert api._predictor is None


def test_prediction_route_forwards_decoded_image_and_validated_context(client, monkeypatch):
    class Predictor:
        def predict(self, image, **context):
            assert image.mode == "RGB" and image.size == (4, 4)
            assert context["lat"] == 40 and context["lon"] == -88.2
            assert context["obs_date"].isoformat() == "2024-07-15"
            assert context["ndvi_override"] == 0.5
            return {"mode": "trained_model", "is_prediction": True}

    monkeypatch.setattr(api, "_predictor", Predictor())
    response = post(client, ndvi="0.5")
    assert response.status_code == 200
    assert response.json() == {"mode": "trained_model", "is_prediction": True}


def test_prediction_failure_is_500_without_internal_details(client, monkeypatch):
    class Predictor:
        def predict(self, *args, **kwargs):
            raise RuntimeError("private model details")

    monkeypatch.setattr(api, "_predictor", Predictor())
    response = post(client)
    assert response.status_code == 500
    assert "private model details" not in response.text
