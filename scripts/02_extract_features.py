import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_preprocessing.audio import load_audio
from feature_extraction.features import extract_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader((ROOT / "dataset" / "metadata_split.csv").open(encoding="utf-8")))
    features = []
    labels = []
    splits = []
    ids = []
    for number, row in enumerate(rows, 1):
        y, sr = load_audio(row["path"])
        features.append(extract_features(y, sr)[0])
        labels.append(row["class_label"])
        splits.append(row["dataset_split"])
        ids.append(row["audio_id"])
        if number % 100 == 0:
            print(f"Extracted {number}/{len(rows)}")
    (ROOT / "data").mkdir(exist_ok=True)
    np.savez_compressed(
        ROOT / "data" / "features.npz",
        x=np.asarray(features),
        y=np.asarray(labels),
        split=np.asarray(splits),
        audio_id=np.asarray(ids),
    )
    print(f"Saved {len(features)} feature rows to data/features.npz")


if __name__ == "__main__":
    main()
