import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader((ROOT / "dataset" / "metadata_split.csv").open(encoding="utf-8")))
    target = ROOT / "gtm_model" / "training_audio"
    for row in rows:
        if row["dataset_split"] != "train":
            continue
        folder = target / row["class_label"]
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], folder / row["filename"])
    print(f"Prepared GTM training folders at {target}")


if __name__ == "__main__":
    main()
