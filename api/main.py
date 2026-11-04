"""FoliaNet API. Run: python -m uvicorn api.main:app --reload"""

from datetime import date as Date
import logging
import os

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from folianet.example import example_response

logger = logging.getLogger(__name__)
app = FastAPI(title="FoliaNet", version="0.1.0",
              description="Check a leaf photo with /predict, or see a fake example at /example.")
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get("FOLIANET_CHECKPOINT", "checkpoints/folianet_best.pt")
        # Check for the file first so this works without PyTorch installed.
        if not os.path.isfile(ckpt):
            raise HTTPException(503, "No model file found. Train one with "
                                "python -m folianet.train or set FOLIANET_CHECKPOINT. "
                                "(GET /example shows a fake response.)")
        try:
            from folianet.inference import FoliaNetPredictor
            predictor = FoliaNetPredictor(ckpt)
        except Exception:
            logger.exception("Couldn't load the FoliaNet model")
            raise HTTPException(503, "The model couldn't load. Check the server logs.") from None
        _predictor = predictor
    return _predictor


@app.get("/health")
def health():
    """Is the server up? (Doesn't check the model.)"""
    return {"status": "ok", "service": "folianet"}


@app.get("/example", summary="Fake example, not a prediction")
def example():
    """A fake response so you can see the format."""
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
        # Load the whole image now so a broken file gives a clean 400.
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
        raise HTTPException(500, "Prediction failed. Check the server logs.") from None
