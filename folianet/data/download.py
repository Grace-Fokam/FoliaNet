"""Show dataset setup steps or download a dataset from Kaggle.

Each disease or healthy class needs its own image folder. Set data.root in
configs/default.yaml to the folder containing those class folders.

For a Kaggle download:
    pip install kagglehub
    python -m folianet.data.download --kaggle <owner>/<dataset-slug>"""

import argparse

INSTRUCTIONS = """
Dataset setup:
  1. Put your images in separate folders for each disease or healthy class, e.g.
       <root>/Corn_(maize)___Common_rust_/*.jpg
       <root>/Corn_(maize)___healthy/*.jpg
     Use the folder names listed in folianet/diseases.py. Add an entry there
     if you want to include another disease.
  2. Set data.root in configs/default.yaml to that <root> folder.
  3. Set data.crops to the crops you want to include.
"""


def main():
    ap = argparse.ArgumentParser(description="Prepare a FoliaNet training dataset.")
    ap.add_argument("--kaggle", help="Optional Kaggle dataset slug, e.g. owner/dataset-name")
    args = ap.parse_args()

    if not args.kaggle:
        print(INSTRUCTIONS)
        return

    try:
        import kagglehub
        path = kagglehub.dataset_download(args.kaggle)
        print(f"Downloaded to: {path}")
        print("Set data.root in configs/default.yaml to the folder containing your class folders.")
    except Exception as e:
        print(f"Download unavailable ({e}).")
        print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
