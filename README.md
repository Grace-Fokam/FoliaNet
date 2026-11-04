# FoliaNet

FoliaNet looks at photos of corn and wheat leaves and flags possible fungal
diseases. It also checks the weather to see if conditions are good for that
disease to spread.

You get back a risk score, a confidence score, the likely disease, and what to
check next. Built with PyTorch (EfficientNet-B0) and FastAPI.

I wrote the original in September 2023 and cleaned it up for GitHub in June 2026.

## Original results (2023)

The original model was trained on about 26,000 labeled crop images and hit 94%
accuracy. I also made predictions 28% faster so it could run on a small field
device. The original weights, images, and logs aren't in this repo, so you
can't reproduce those numbers from this code.

## Try the example

There's no trained model or dataset in the repo, but you can see what a response
looks like in [the example JSON](examples/prediction-response.json), or run it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
python -m folianet.example
```

Or through the API:

```bash
python -m uvicorn api.main:app --reload
curl http://127.0.0.1:8000/example   # in another terminal
```

**The example is made up.** It doesn't look at a photo, load a model, or pull any
weather or satellite data. It's just there to show the format, so don't use it
for real crop decisions. You can tell because it says `mode: "illustrative_example"`,
`is_prediction: false`, and `model_version: null`.

## How it works

With a trained model, FoliaNet:

1. Runs the leaf photo through the image classifier.
2. Gets the weather for your location and date.
3. Checks the weather against simple rules for each disease.
4. Combines the two into one risk score.

The scores are rough estimates. A confidence of 0.8 doesn't mean an 80% chance
of being right. Test it on your own data before trusting it.

**Weather** comes from [Open-Meteo](https://open-meteo.com). If that fails, the
code uses fake fallback weather, labels it, and scores with the image alone.
Leaf wetness is guessed from humidity and rain, not measured.

**NDVI** (a measure of how green and dense plants are) is still a placeholder
that returns fake values, which aren't used in the score. You can plug in a real
source in `folianet/features/satellite.py` or pass your own with `ndvi`. Values
you pass are labeled `user_override_unverified`.

To see what went into a score, check `details.provenance`, `details.data_sources`,
and `details.fusion_weights`.

## Train a model

```bash
pip install -r requirements.txt
```

Set up your images using [data/README.md](data/README.md), point `data.root` in
`configs/default.yaml` at them, then run:

```bash
python -m folianet.train --config configs/default.yaml
```

The model saves to `checkpoints/folianet_best.pt`.

Leave `model.tabular_dim` at `0`. There's an unfinished branch in the model for
extra inputs, but nothing uses it yet, and the predictor won't load a
checkpoint that has it turned on.

Only load checkpoints (saved model files) you trust.

## Predict from the terminal

```bash
python -m scripts.predict_example \
  --image path/to/leaf.jpg --lat 40.0 --lon -88.2 --date 2024-07-15
```

Add `--checkpoint path/to/model.pt` if your model is somewhere else, or
`--ndvi 0.5` to pass your own NDVI.

## Use the API

```bash
python -m uvicorn api.main:app --reload
```

Then in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F image=@path/to/leaf.jpg \
  -F lat=40.0 -F lon=-88.2 -F date=2024-07-15
```

Set `FOLIANET_CHECKPOINT` to use a different model file. Latitude, longitude,
and date have to be valid, and `ndvi` (optional) has to be between -1 and 1.

A real prediction says `mode: "trained_model"` and `is_prediction: true`, and has:

- `risk_score`, `risk_score_pct`, `risk_band`: how risky it looks
- `confidence`: score and level
- `likely_stressor`: the disease, crop, and pathogen
- `recommended_action`: what to do next
- `details`: class probabilities, weather, NDVI, and where everything came from

`likely_stressor` always picks the top disease class, even if "healthy" scored
higher, so it's a thing to check, not a diagnosis. Look at the healthy
probability and risk score too.

`details.model_version` is the code version, not the specific checkpoint.

Errors:

| Code | Meaning |
| --- | --- |
| 400 | Couldn't read the image. |
| 422 | Missing or invalid input. |
| 503 | No model file, or it didn't load. |
| 500 | Prediction failed. Check the server logs. |

`/predict` never falls back to the fake example. `/health` only checks that the
server is up, not the model.

## Project layout

| Path | What's there |
| --- | --- |
| `folianet/diseases.py` | Disease list and weather rules |
| `folianet/data/` | Dataset loading |
| `folianet/features/` | Weather, NDVI, and weather scoring |
| `folianet/models/fusion_model.py` | The image model |
| `folianet/train.py` | Training |
| `folianet/inference.py` | Predictions and scoring |
| `folianet/recommend.py` | Advice text |
| `folianet/example.py` | The fake example |
| `api/main.py` | The API |
| `tests/` | Tests |

## Tests

```bash
pip install -r requirements-demo.txt
python -m pytest -q
```

These cover the example, API inputs and errors, disease rules, and fallback data.
They don't test model accuracy, a full trained-model prediction, or the live
weather and satellite services.

## Using it in a real field

This is a scouting helper, not a diagnosis. Check with a local agronomist or
extension service before treating anything, and test the model on your own
fields first. There are no verified accuracy or field results here.

## License

[MIT](LICENSE). © Grace Fokam.
