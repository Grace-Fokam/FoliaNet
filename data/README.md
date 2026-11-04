# Training images

The dataset isn't in this repo. You'll need your own labeled corn or wheat leaf
photos, with one folder per class, like this inside `data/crop_images`:

- `Corn_(maize)___Common_rust_`
- `Corn_(maize)___Northern_Leaf_Blight`
- `Corn_(maize)___healthy`
- `Wheat___Leaf_rust`
- `Wheat___healthy`

Folder names should match the ones in `folianet/diseases.py`. Adding a new
disease? Add it there too.

Then in `configs/default.yaml`, set:

- `data.root` to the folder with your class folders
- `data.crops` to the crops you want, like `["Corn", "Wheat"]`

`data.crops` just filters by folder name, so you can train on one crop without
moving anything.

## Getting a dataset from Kaggle

```bash
pip install kagglehub
python -m folianet.data.download --kaggle <owner>/<dataset-slug>
```

Then point `data.root` at the downloaded class folders.

Check the dataset's license, and don't commit training images unless you're
allowed to.
