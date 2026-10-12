# FoliaNet 

AI crop disease detection for corn and wheat fungal diseases. FoliaNet takes a
crop image plus location, date, and environmental/satellite context and returns a
**risk score**, a **confidence level**, the **likely stressor**, and a
**recommended next action** in a single call.

*Original codebase was developed in September 2023. Reformatted version published in June 2026.*

---

## What it does

- Identifies the likely stressor from a leaf image using a transfer-learning CNN
  (EfficientNet-B0, PyTorch).
- Scores disease risk by fusing the image prediction with environmental context 
  (temperature, humidity, leaf-wetness, rainfall, and satellite NDVI).
- Reports a confidence level alongside every prediction.
- Recommends a next action tailored to the detected stressor and risk level.
- Serves over HTTP via a FastAPI endpoint, ready to drop into a product or app.

## How it works

```
   leaf image ─────────────▶ CNN classifier ──▶ disease probabilities ─┐
                                                                        ├─▶ fusion ─▶ risk score
   lat / lon / date ─▶ weather + NDVI ─▶ agronomic favorability ────────┘            confidence
                                                                                      likely stressor
                                                                                      recommended action
```

FoliaNet combines two complementary signals:

1. **Image branch**: a fine-tuned CNN produces disease-class probabilities from a leaf photo.
2. **Environmental branch**: weather and satellite NDVI for the field's location and date
   are converted into a per-disease favorability index using agronomic rules.

These are fused into the final risk score, confidence level, likely stressor, and
recommended action. The model architecture includes an optional **early-fusion** tabular branch,
but the current trainer and predictor supply image inputs only. Keep
`model.tabular_dim: 0`; paired-data training/inference requires further implementation.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See [`data/README.md`](data/README.md) for preparing a labeled image dataset, then
set `data.root` in `configs/default.yaml`.

## Train

```bash
python -m folianet.train --config configs/default.yaml
```

Saves `checkpoints/folianet_best.pt` (weights + class list + metadata).

## Predict (CLI)

```bash
python -m scripts.predict_example \
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

### Checkpoint-independent example (offline)

Browse the fully labeled [example JSON](examples/prediction-response.json) directly.

No trained checkpoint or dataset is distributed in this repository. To inspect the
intended response structure without PyTorch, training data, or provider calls:

```bash
pip install -r requirements-demo.txt
python -m folianet.example
uvicorn api.main:app --reload
# In another terminal:
curl http://127.0.0.1:8000/example
```

`GET /example` returns a fixed **hand-authored illustration**, with
`mode: "illustrative_example"`, `is_prediction: false`, a prominent notice,
`model_version: null`, and per-field provenance in `details.provenance`.
No image is analyzed and no model or weather/satellite provider is called.
Every numeric score, probability, weather value, NDVI value, and fusion weight
in this fixture is illustrative; the recommendation uses the existing rule-based
text. These values are not evidence of accuracy, calibrated confidence, or field
validation and must not guide field decisions. This endpoint never substitutes
for `/predict` when a model is unavailable.

### Genuine inference prerequisites and response

1. Install `requirements.txt` and assemble your own labeled data using
   [`data/README.md`](data/README.md). The repository does not include the dataset.
2. Configure `data.root`, keep `model.tabular_dim: 0`, and run the training command.
   The current trainer and predictor do not supply paired tabular inputs;
   early-fusion checkpoints are rejected by the predictor with a clear error.
3. Supply the resulting `checkpoints/folianet_best.pt` or set
   `FOLIANET_CHECKPOINT` for the API / `--checkpoint` for the CLI. Use only trusted
   checkpoints. A compatible checkpoint contains `state_dict`, `classes`, and
   matching `backbone`, `img_size`, and `tabular_dim` metadata.
4. Provide a readable leaf image, latitude in [-90, 90], longitude in [-180, 180],
   and a valid observation date. Optional `ndvi` must be finite and in [-1, 1].
5. Independently evaluate the trained classifier and tune/validate the heuristic
   risk and confidence outputs on your target crops and field conditions.

Use `python -m scripts.predict_example` from the repository root for the CLI
(the module invocation ensures local package imports resolve).

Successful `/predict` responses have `mode: "trained_model"` and
`is_prediction: true` plus `risk_score`, `risk_score_pct`, `risk_band`,
`confidence`, `likely_stressor`, `recommended_action`, and `details`.
`details` includes class probabilities, weather, NDVI, source-availability flags,
actual fusion weights, software model version, and explicit provenance.
`model_version` identifies the implementation version, not a unique checkpoint
or a validation result. The risk score is
an image/environment heuristic blend; confidence is a heuristic combining class
probability and entropy, not a calibrated disease probability.

Weather retrieval may fall back to synthetic values. Satellite retrieval is
currently unimplemented and returns synthetic NDVI. Synthetic weather never
enables environmental risk fusion, and synthetic NDVI never modifies favorability.
Real retrieved weather enables fusion; optional retrieved or user-supplied NDVI
can modify its favorability. A user override is labeled `user_override_unverified`
and does not claim satellite retrieval. Leaf wetness from retrieved weather is a
humidity/precipitation proxy, not a field sensor measurement. Inspect
`details.provenance`, `details.data_sources`, and `details.fusion_weights` on every
prediction; a trained image model can still run with fallback environmental data.

API errors: invalid/missing context → 422; unreadable image → 400; missing or
unloadable checkpoint/dependencies → 503; unexpected prediction failure → 500.
Internal failure details are logged on the server, not exposed to clients.
`/health` reports service liveness only, not model readiness or validity.

---

## Pluggable data sources

- Weather attempts [Open-Meteo](https://open-meteo.com) retrieval and uses a
  labeled synthetic fallback when retrieval fails.
- Satellite NDVI is pluggable; add your provider (Sentinel Hub, Google Earth Engine, etc.)
  in `folianet/features/satellite.py`.

## Project layout

```
folianet/
  diseases.py            disease metadata + agronomic favorability params
  config.py              YAML config loader
  data/                  dataset + download helper
  features/              weather, satellite, environmental favorability
  models/fusion_model.py image (+ optional tabular) network
  train.py               training loop
  inference.py           predictor: assembles the risk/confidence/stressor/action output
  recommend.py           stressor + risk band -> action text
api/main.py              FastAPI service
scripts/predict_example.py
tests/test_logic.py      unit tests (pytest)
```

## Tests

```bash
pip install -r requirements-demo.txt
python -m pytest -q
```

The lightweight suite covers the offline CLI/API example, input validation,
checkpoint errors, HTTP routing with a controlled predictor, and source/fusion
policy. These tests do not validate learned model accuracy, real provider
availability, or the full trained-checkpoint inference path.

## Disclaimer

- FoliaNet is an advisory tool that supports scouting decisions; follow local agronomy /
  extension guidance for treatment timing, rates, and product choice.
- Tune and validate risk thresholds against your own field data, and back any performance
  figures with your own pilot methodology.

## License

MIT [LICENSE](LICENSE). © Grace Fokam.
