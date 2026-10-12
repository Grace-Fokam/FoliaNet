"""The FoliaNet API. Start it with `python -m uvicorn api.main:app --reload`.

Use /example to see made-up output or /predict to run a trained model."""

from datetime import date as Date
import logging
import os

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from folianet.example import example_response

logger = logging.getLogger(__name__)
app = FastAPI(title="FoliaNet", version="0.1.0",
              description="Check leaf photos with /predict or see made-up output at /example.")
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get("FOLIANET_CHECKPOINT", "checkpoints/folianet_best.pt")
        # Check for the model file before importing PyTorch.
        # This also works when only the demo requirements are installed.
        if not os.path.isfile(ckpt):
            raise HTTPException(503, "No saved model file was found. Train with "
                                "python -m folianet.train --config configs/default.yaml "
                                "or set FOLIANET_CHECKPOINT to a trusted model file with matching settings. "
                                "GET /example returns made-up output, not a prediction.")
        try:
            from folianet.inference import FoliaNetPredictor
            predictor = FoliaNetPredictor(ckpt)
        except Exception:
            logger.exception("Unable to initialize FoliaNet predictor")
            raise HTTPException(503, "The model could not load. Check the server logs, installed "
                                "packages, and saved model settings.") from None
        _predictor = predictor
    return _predictor


@app.get("/health")
def health():
    """Check that the API is running. This does not check the model."""
    return {"status": "ok", "service": "folianet"}


@app.get("/example", summary="Made-up example, not a prediction")
def example():
    """Show a made-up response without checking a photo or downloading data."""
    return example_response()


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    lat: float = Form(..., ge=-90, le=90, allow_inf_nan=False),
    lon: float = Form(..., ge=-180, le=180, allow_inf_nan=False),
    date: Date = Form(...),
    ndvi: float | None = Form(None, ge=-1, le=1, allow_inf_nan=False),
):
    try:
        # Read the whole image here so broken files return a clear error.
        with Image.open(image.file) as uploaded:
            uploaded.load()
            img = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not read the uploaded image.") from None
    predictor = get_predictor()
    try:
        return predictor.predict(img, lat=lat, lon=lon, obs_date=date, ndvi_override=ndvi)
    except Exception:
        logger.exception("FoliaNet prediction failed")
        raise HTTPException(500, "Prediction failed. Check server logs for details.") from None
