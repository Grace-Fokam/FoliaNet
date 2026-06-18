# FoliaNet 🌽🌾

AI crop disease detection for **corn and wheat fungal diseases**. FoliaNet takes a
crop image plus location, date, and environmental/satellite context, and returns a
**risk score**, a **confidence level**, the **likely stressor**, and a
**recommended next action**.

> ⚠️ **Advisory tool.** FoliaNet supports scouting decisions; it does not replace a
> local agronomist or extension service. Disease thresholds, fungicide timing, and
> product/rate choices depend on local regulations, growth stage, and variety.

---

## How it works

FoliaNet fuses two signals:

1. **Image branch** — a transfer-learning CNN (EfficientNet-B0 by default) fine-tuned
   on labeled leaf images, producing disease-class probabilities.
2. **Environmental branch** — a transparent, rule-based agronomic module that turns
   weather (temperature, humidity, leaf-wetness, rainfall) and satellite NDVI into a
   per-disease *favorability* index.

```
   leaf image ─────────────▶ CNN classifier ──▶ disease probabilities ─┐
                                                                        ├─▶ late fusion ─▶ risk score
   lat / lon / date ─▶ weather + NDVI ─▶ agronomic favorability ────────┘                  confidence
                                                                                           likely stressor
                                                                                           recommended action
```

### A note on fusion (important)

The classic **PlantVillage** dataset has labeled images but **no paired weather or
location data**, so you cannot train a true early-fusion multimodal network on it
alone. FoliaNet therefore:

- **Trains the image classifier** on PlantVillage (real labels), and
- **Late-fuses** the environmental favorability at inference time
  (`risk = w_img · P(disease | image) + w_env · favorability`).

The model class (`FusionModel`) already supports **early fusion** — set
`model.tabular_dim > 0` and feed paired tabular features — for when you collect
image + environmental training data. Everything downstream stays the same.

### Corn vs. wheat

PlantVillage ships **corn** disease classes but **not wheat**. The data layer is
keyword-filtered (`data.crops`), so corn works immediately; add a wheat dataset whose
class folders match the keys in `folianet/diseases.py` to enable wheat. See
[`data/README.md`](data/README.md).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get the dataset:

```bash
pip install kagglehub
python -m folianet.data.download
# then set data.root in configs/default.yaml
```

## Train

```bash
python -m folianet.train --config configs/default.yaml
```

This saves `checkpoints/folianet_best.pt` (weights + class list + metadata).

## Predict (CLI)

```bash
python scripts/predict_example.py \
  --image path/to/leaf.jpg --lat 40.0 --lon -88.2 --date 2024-07-15
```

## Serve (API)

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F image=@path/to/leaf.jpg \
  -F lat=40.0 -F lon=-88.2 -F date=2024-07-15
```

### Output schema

```json
{
  "risk_score": 0.78,
  "risk_score_pct": 78.0,
  "risk_band": "high",
  "confidence": { "score": 0.81, "level": "high" },
  "likely_stressor": {
    "name": "Gray leaf spot",
    "pathogen": "Cercospora zeae-maydis",
    "crop": "Corn",
    "healthy": false
  },
  "recommended_action": "High pressure. Prioritize scouting now; ...",
  "details": {
    "image_disease_probability": 0.83,
    "environmental_favorability": 0.71,
    "class_probabilities": { "...": 0.0 },
    "weather": { "temp_mean_c": 26.1, "humidity_mean_pct": 92.0, "...": 0 },
    "ndvi": 0.62,
    "data_sources": { "weather": true, "satellite": false },
    "fusion_weights": { "image": 0.6, "environment": 0.4 },
    "model_version": "folianet-0.1.0"
  }
}
```

---

## Wiring real data sources

- **Weather** uses [Open-Meteo](https://open-meteo.com) (free, no key). Works as-is.
- **Satellite NDVI** ships as a documented stub in
  `folianet/features/satellite.py`. Implement `_fetch_real_ndvi()` with Sentinel Hub,
  Google Earth Engine, etc. Until then a synthetic NDVI is returned and flagged via
  `data_sources.satellite=false`. When environmental data is synthetic, FoliaNet
  automatically trusts the image model and lowers reported confidence.

## Project layout

```
folianet/
  diseases.py            disease metadata + agronomic favorability params
  config.py              YAML config loader
  data/                  dataset + download helper
  features/              weather, satellite, environmental favorability
  models/fusion_model.py image (+ optional tabular) network
  train.py               training loop
  inference.py           predictor: assembles the 4-field output
  recommend.py           stressor + risk band -> action text
api/main.py              FastAPI service
scripts/predict_example.py
tests/test_logic.py      torch-free unit tests (pytest)
```

## Tests

```bash
pytest -q
```

## Limitations & responsible use

- Trained on curated leaf images (PlantVillage); field photos vary in lighting,
  background, and stage — expect to fine-tune on your own images for production.
- The environmental module is a tunable heuristic, not a calibrated epidemiological
  model. Validate against local field outcomes before relying on the risk score.
- Any performance figures (e.g. crop-loss reduction) must be backed by **your own**
  pilot methodology; don't ship third-party numbers as FoliaNet's.
- Recommendations are advisory and general. Follow local regulations and extension
  guidance.

## License

MIT — see [LICENSE](LICENSE) (add your name).
