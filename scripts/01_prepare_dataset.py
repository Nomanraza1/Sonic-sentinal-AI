import csv
import sys
import hashlib
from pathlib import Path
import soundfile as sf
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.class_map import SONIC_CLASSES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audio_dataset" / "originals"
DATA = ROOT / "dataset"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    rows = []
    excluded = []
    for label in SONIC_CLASSES:
        seen = set()
        for path in sorted((SOURCE / label).glob("*.wav")) if (SOURCE / label).exists() else []:
            digest = file_hash(path)
            if digest in seen or "(" in path.stem:
                excluded.append(
                    {"filename": str(path.relative_to(ROOT)), "reason": "duplicate source clip"}
                )
                continue
            seen.add(digest)
            info = sf.info(path)
            rows.append(
                {
                    "audio_id": path.stem,
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "class_label": label,
                    "duration": round(info.duration, 4),
                    "sampling_rate": info.samplerate,
                    "channels": info.channels,
                    "recording_environment": "",
                    "recording_device": "",
                    "source_distance": "",
                    "is_original": 1,
                }
            )
    labels = [row["class_label"] for row in rows]
    train, holdout = train_test_split(rows, test_size=0.30, random_state=7, stratify=labels)
    valid, test = train_test_split(
        holdout, test_size=0.50, random_state=7, stratify=[row["class_label"] for row in holdout]
    )
    for split, group in [("train", train), ("validation", valid), ("test", test)]:
        for row in group:
            row["dataset_split"] = split
    ordered = train + valid + test
    fields = list(ordered[0])
    for name in ["metadata_originals.csv", "metadata_split.csv"]:
        with (DATA / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ordered)
    with (ROOT / "config" / "excluded_clips.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    counts = {
        split: sum(row["dataset_split"] == split for row in ordered)
        for split in ["train", "validation", "test"]
    }
    present = sorted(set(labels))
    missing = [label for label in SONIC_CLASSES if label not in present]
    print(
        {
            "clips": len(ordered),
            "splits": counts,
            "classes": present,
            "missing": missing,
            "excluded": len(excluded),
        }
    )


if __name__ == "__main__":
    main()
