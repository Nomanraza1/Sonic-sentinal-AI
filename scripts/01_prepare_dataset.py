import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from augmentation.augment import make_variants
from audio_preprocessing.audio import load_audio, segments, validate_signal
from config.class_map import SONIC_CLASSES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "audio_dataset" / "raw"
PROCESSED = ROOT / "audio_dataset" / "processed"
DATA = ROOT / "dataset"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_clip(path, y, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")


def scan():
    rows = []
    ignored = []
    for label in SONIC_CLASSES:
        folder = RAW / label
        if not folder.exists():
            continue
        seen = set()
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            try:
                value = digest(path)
                if value in seen:
                    ignored.append({"filename": str(path.relative_to(ROOT)), "reason": "duplicate"})
                    continue
                info = sf.info(path)
                if info.duration < 0.2:
                    ignored.append({"filename": str(path.relative_to(ROOT)), "reason": "too short"})
                    continue
                seen.add(value)
                rows.append({"audio_id": f"{label}_{value[:12]}", "source_path": str(path.resolve()), "filename": path.name, "class_label": label, "source_hash": value, "duration": round(info.duration, 4), "sampling_rate": info.samplerate, "channels": info.channels})
            except Exception:
                ignored.append({"filename": str(path.relative_to(ROOT)), "reason": "unreadable"})
    return rows, ignored


def split_rows(rows):
    labels = [row["class_label"] for row in rows]
    train, holdout = train_test_split(rows, test_size=0.30, random_state=7, stratify=labels)
    valid, test = train_test_split(holdout, test_size=0.50, random_state=7, stratify=[row["class_label"] for row in holdout])
    for name, group in [("train", train), ("validation", valid), ("test", test)]:
        for row in group:
            row["dataset_split"] = name
    return train + valid + test


def build(rows, ignored=None):
    if ignored is None:
        ignored = []
    output = []
    for number, row in enumerate(rows, 1):
        try:
            y, sr = load_audio(row["source_path"])
        except (ValueError, OSError, EOFError) as exc:
            ignored.append({"filename": row["source_path"], "reason": str(exc)})
            continue
        for index, (clip, _, _) in enumerate(segments(y, sr)):
            try:
                validate_signal(clip)
            except ValueError as exc:
                ignored.append({"filename": f"{row['source_path']}#segment={index}", "reason": str(exc)})
                continue
            name = f"{row['audio_id']}_{index:02d}.wav"
            path = PROCESSED / row["dataset_split"] / row["class_label"] / name
            write_clip(path, clip, sr)
            item = dict(row, filename=name, path=str(path.resolve()), segment_index=index, is_original=1, augmentation="original")
            output.append(item)
            if row["dataset_split"] == "train":
                for kind, variant in make_variants(clip, sr, number + index).items():
                    augmented = path.with_name(f"{path.stem}_{kind}.wav")
                    write_clip(augmented, variant, sr)
                    output.append(dict(item, filename=augmented.name, path=str(augmented.resolve()), is_original=0, augmentation=kind))
    return output


def main():
    DATA.mkdir(exist_ok=True)
    rows, ignored = scan()
    if not rows:
        raise ValueError("No readable source clips were found.")
    originals = split_rows(rows)
    if PROCESSED.exists():
        for file in PROCESSED.rglob("*.wav"):
            file.unlink()
    prepared = build(originals, ignored)
    if not prepared:
        raise ValueError("No usable audio segments remain.")
    fields = ["audio_id", "filename", "path", "source_path", "source_hash", "class_label", "duration", "sampling_rate", "channels", "dataset_split", "segment_index", "is_original", "augmentation"]
    for name, group in [("metadata_originals.csv", [x for x in prepared if x["is_original"]]), ("metadata_split.csv", prepared)]:
        with (DATA / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(group)
    with (ROOT / "config" / "excluded_clips.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "reason"])
        writer.writeheader()
        writer.writerows(ignored)
    counts = {name: sum(row["dataset_split"] == name and row["is_original"] for row in prepared) for name in ["train", "validation", "test"]}
    print({"originals": len(originals), "prepared": len(prepared), "splits": counts})


if __name__ == "__main__":
    main()
