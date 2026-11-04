"""Prints setup steps, or downloads a dataset from Kaggle.

    pip install kagglehub
    python -m folianet.data.download --kaggle <owner>/<dataset-slug>"""

import argparse

INSTRUCTIONS = """
Dataset setup:
  1. Make one folder per class, named like the ones in folianet/diseases.py:
       <root>/Corn_(maize)___Common_rust_/*.jpg
       <root>/Corn_(maize)___healthy/*.jpg
  2. Set data.root in configs/default.yaml to <root>.
  3. Set data.crops to the crops you want.
"""


def main():
    ap = argparse.ArgumentParser(description="Set up a FoliaNet dataset.")
    ap.add_argument("--kaggle", help="Kaggle dataset, like owner/dataset-name")
    args = ap.parse_args()

    if not args.kaggle:
        print(INSTRUCTIONS)
        return

    try:
        import kagglehub
        path = kagglehub.dataset_download(args.kaggle)
        print(f"Downloaded to: {path}")
        print("Now set data.root in configs/default.yaml to the class folders.")
    except Exception as e:
        print(f"Download failed ({e}).")
        print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
