import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_preprocessing.audio import load_audio
from feature_extraction.features import FEATURE_VERSION, extract_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "data" / "features.npz"
    destination.parent.mkdir(exist_ok=True)
    destination.unlink(missing_ok=True)
    rows = list(csv.DictReader((ROOT / "dataset" / "metadata_split.csv").open(encoding="utf-8")))
    features = []
    labels = []
    splits = []
    ids = []
    excluded = []
    for number, row in enumerate(rows, 1):
        try:
            y, sr = load_audio(row["path"])
        except (ValueError, OSError, EOFError) as exc:
            excluded.append({"path": row["path"], "reason": str(exc)})
            print(f"Skipping {row['path']}: {exc}")
            continue
        features.append(extract_features(y, sr)[0])
        labels.append(row["class_label"])
        splits.append(row["dataset_split"])
        ids.append(row["audio_id"])
        if number % 100 == 0:
            print(f"Extracted {number}/{len(rows)}")
    with (destination.parent / "excluded_features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    if not features:
        raise ValueError("No usable audio remains; inspect data/excluded_features.csv.")
    temporary = destination.with_name("features.pending.npz")
    np.savez_compressed(
        temporary,
        x=np.asarray(features),
        y=np.asarray(labels),
        split=np.asarray(splits),
        audio_id=np.asarray(ids),
        feature_version=np.asarray(FEATURE_VERSION),
    )
    temporary.replace(destination)
    print(f"Saved {len(features)} feature rows; skipped {len(excluded)}. See data/excluded_features.csv.")


if __name__ == "__main__":
    main()
