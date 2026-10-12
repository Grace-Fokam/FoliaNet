# FoliaNet

FoliaNet looks at photos of corn and wheat leaves to help check for possible
fungal diseases. It also uses weather information to estimate how favorable the
conditions are for a disease to develop.

The result includes a risk score, a confidence score, a possible disease, and a
suggestion for what to check next. The image model uses PyTorch and
EfficientNet-B0, and the API uses FastAPI.

The original code was developed in September 2023. This reformatted version was
published in June 2026.

## Try the example first

This repo doesn't include a trained model or the training images. You can still
see what an API response looks like in the [example JSON](examples/prediction-response.json).

To run the example yourself, open a terminal in the repo folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
python -m folianet.example
```

You can also get the same example through the API:

```bash
python -m uvicorn api.main:app --reload
```

Then open another terminal and run:

```bash
curl http://127.0.0.1:8000/example
```

**The example is made up to show the response format.** It doesn't look at a
photo, load a trained model, or download weather or satellite data. All of its
numbers are example values. The advice comes from the project's existing rules.
Don't use this example to make decisions about a crop.

The response says `mode: "illustrative_example"` and `is_prediction: false`.
It also explains where each part comes from in `details.provenance`.
`model_version` is `null` because no model was used.

## How predictions work

Once you have a trained model, FoliaNet:

1. Runs a leaf photo through the image classifier.
2. Looks up weather for the location and date you give it.
3. Uses simple disease rules to check whether those conditions favor disease.
4. Combines the image result with the weather score and returns a response.

The risk and confidence scores are estimates. For example, a confidence score
of 0.8 doesn't mean the prediction has an 80% chance of being right. You'll need
to test the model and these scoring rules on your own data.

### A few things to know about the data

Weather comes from [Open-Meteo](https://open-meteo.com). If the request fails,
the code returns made-up fallback weather and labels it that way. That fallback
weather isn't used to adjust the final risk score; the score uses the image
model alone.

The satellite code is still a placeholder. It returns made-up NDVI values,
which aren't used to adjust the disease score. NDVI is a number that describes
how green or dense vegetation is. You can add a real source in
`folianet/features/satellite.py`, or pass your own value with `ndvi`.
A value you supply is labeled `user_override_unverified`; it isn't treated as
data downloaded from a satellite.

Weather is needed for the environmental part of the risk score. If real weather
is available, real or user-supplied NDVI can also adjust that part of the score.
Leaf wetness is estimated from humidity and rain, not measured by a field sensor.

Check `details.provenance`, `details.data_sources`, and `details.fusion_weights`
to see where the values came from and what was used in the final score.

## Train a model

Install the full requirements in your virtual environment:

```bash
pip install -r requirements.txt
```

Follow [data/README.md](data/README.md) to prepare your labeled images. Set
`data.root` in `configs/default.yaml` to your image folder, then run:

```bash
python -m folianet.train --config configs/default.yaml
```

This saves the trained model to `checkpoints/folianet_best.pt`, along with its
class names and settings.

Keep `model.tabular_dim: 0`. The model code has an optional branch for extra
inputs, but the current training and prediction code only passes images to the
model. Using that branch would need more work. The predictor rejects checkpoints
that have it enabled.

If you already have a checkpoint, only load one you trust. It needs `state_dict`
and `classes`, with compatible `backbone`, `img_size`, and `tabular_dim` settings.
A checkpoint is the saved file containing the model's learned weights.

## Predict from the terminal

Run this from the repo folder after training:

```bash
python -m scripts.predict_example \
  --image path/to/leaf.jpg --lat 40.0 --lon -88.2 --date 2024-07-15
```

Replace the photo path, coordinates, and date with your own. The default model
path is `checkpoints/folianet_best.pt`. Use `--checkpoint path/to/model.pt` if
your file is somewhere else. You can also add `--ndvi 0.5` to supply an NDVI value.

## Use the prediction API

Start the server:

```bash
python -m uvicorn api.main:app --reload
```

Then send a photo from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F image=@path/to/leaf.jpg \
  -F lat=40.0 -F lon=-88.2 -F date=2024-07-15
```

The API uses the same default checkpoint path. To use a different file, set
`FOLIANET_CHECKPOINT` before starting the server.

Latitude must be between -90 and 90, longitude between -180 and 180, and the date
must be valid. If you include `ndvi`, it must be a finite number between -1 and 1.

A prediction response says `mode: "trained_model"` and `is_prediction: true`.
It includes:

- `risk_score`, `risk_score_pct`, and `risk_band`: the estimated risk.
- `confidence`: the model's confidence score and level.
- `likely_stressor`: the possible disease, crop, and pathogen.
- `recommended_action`: a suggestion based on the disease and risk band.
- `details`: class probabilities, weather, NDVI, data sources, and scoring weights.

`details.model_version` is the code version. It doesn't identify a specific
trained checkpoint or tell you how well that checkpoint was tested.

If something goes wrong, the API returns these status codes:

| Code | What it means |
| --- | --- |
| 400 | The uploaded image couldn't be read. |
| 422 | A required input is missing or invalid. |
| 503 | The checkpoint is missing or the model couldn't load. |
| 500 | Something failed while running the prediction. Check the server logs. |

`/predict` won't return the made-up example if the model is missing.
`/health` only checks that the service is running, not that the model is ready.

## Where things are

| File or folder | What's in it |
| --- | --- |
| `folianet/diseases.py` | Disease names and the rules for favorable conditions. |
| `folianet/data/` | Dataset loading and preparation helpers. |
| `folianet/features/` | Weather, NDVI, and environmental scoring. |
| `folianet/models/fusion_model.py` | The image model and optional extra-input branch. |
| `folianet/train.py` | Model training. |
| `folianet/inference.py` | Predictions and final scores. |
| `folianet/recommend.py` | Advice for each disease and risk band. |
| `folianet/example.py` | The made-up response example. |
| `api/main.py` | The HTTP API. |
| `tests/` | Automated tests. |

## Run the tests

```bash
pip install -r requirements-demo.txt
python -m pytest -q
```

These tests check the example, API inputs and errors, disease rules, and how
real and fallback data are handled. Some API tests use a stand-in predictor.
They don't test a trained model's accuracy, the full trained-model prediction
path, or whether the weather and satellite services work.

## Before using this in a field

FoliaNet is meant to help with scouting. Confirm what you find with a local
agronomist or extension service before deciding on treatment. Test your trained
model and scoring thresholds on your own field data. This repo doesn't provide
verified accuracy numbers or field-validation results.

## License

MIT [LICENSE](LICENSE). © Grace Fokam.
