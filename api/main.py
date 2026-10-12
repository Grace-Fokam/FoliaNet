"""FoliaNet API. Run `uvicorn api.main:app --reload`; see README for modes."""

from datetime import date as Date
import logging
import os

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from folianet.example import example_response

logger = logging.getLogger(__name__)
app = FastAPI(title="FoliaNet", version="0.1.0",
              description="Crop disease inference with a separate illustrative /example.")
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get("FOLIANET_CHECKPOINT", "checkpoints/folianet_best.pt")
        # Check before importing torch: a missing checkpoint has the same response
        # even in the lightweight demo installation.
        if not os.path.isfile(ckpt):
            raise HTTPException(503, "Model checkpoint unavailable. Train with "
                                "python -m folianet.train --config configs/default.yaml "
                                "or set FOLIANET_CHECKPOINT to a trusted compatible checkpoint. "
                                "GET /example provides illustrative output only.")
        try:
            from folianet.inference import FoliaNetPredictor
            predictor = FoliaNetPredictor(ckpt)
        except Exception:
            logger.exception("Unable to initialize FoliaNet predictor")
            raise HTTPException(503, "Model could not be loaded. Check server logs, "
                                "ML dependencies, and checkpoint compatibility.") from None
        _predictor = predictor
    return _predictor


@app.get("/health")
def health():
    """Liveness only; does not load or validate a checkpoint."""
    return {"status": "ok", "service": "folianet"}


@app.get("/example", summary="Illustrative response — no model inference")
def example():
    """Fixed offline fixture. Does not accept an image or call data providers."""
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
        # Decode inside the error boundary; Image.open alone only reads headers.
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
