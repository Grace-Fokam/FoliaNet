# Training images

The dataset isn't included in this repo. To train FoliaNet, you'll need labeled
photos of corn or wheat leaves.

Put the images in separate folders for each disease or healthy class. For
example, you could have these folders inside `data/crop_images`:

- `Corn_(maize)___Common_rust_`
- `Corn_(maize)___Northern_Leaf_Blight`
- `Corn_(maize)___healthy`
- `Wheat___Leaf_rust`
- `Wheat___healthy`

Each folder should contain the images for that class. Use the class names in
`folianet/diseases.py`. If you're adding another disease, add its entry there too.

In `configs/default.yaml`, set:

- `data.root` to the folder containing all your class folders.
- `data.crops` to the crops you want to include, like `["Corn", "Wheat"]`.

The crop setting filters folder names, so you don't need to move images around
to train on just one crop.

## Downloading from Kaggle

If you have a Kaggle dataset you want to use:

```bash
pip install kagglehub
python -m folianet.data.download --kaggle <owner>/<dataset-slug>
```

Replace `<owner>/<dataset-slug>` with the dataset's name on Kaggle. Then set
`data.root` to the downloaded folder containing the class folders.

Check the dataset's license before using or sharing it. Keep your training
images out of commits unless you have a reason and permission to include them.
