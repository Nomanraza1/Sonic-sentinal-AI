"""Save source splits and training augmentations, then train on frozen YAMNet features.

Run: python python_models/train_augmented.py
Use --prepare-only to save audio without fitting, or --features-only to also extract.
The existing deployed model is untouched until --activate is explicitly supplied.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_models import train_transfer as transfer
import numpy as np
import soundfile as sf
from audio_preprocessing.audio import load_audio, segments, validate_signal
from feature_extraction.yamnet import FEATURE_VERSION, extract_transfer_features, download_base

DATASET = ROOT / "audio_dataset" / "yamnet_augmented"
RUN = ROOT / "data" / "transfer_augmented"
VERSION = "shift-noise20-noise10-volume-v1"
AUGMENTATION = "training only: time shift, Gaussian noise at 20 dB and 10 dB SNR, volume scaling; validation/test original segments only"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def variants(clip, seed):
    rng = np.random.default_rng(seed)
    yield "original", clip
    shift = int(rng.integers(-3969, 3970))
    yield "shift", np.roll(clip, shift)
    power = float(np.sqrt(np.mean(clip ** 2)))
    for snr in (20, 10):
        noise = rng.normal(size=len(clip))
        noise *= power / (10 ** (snr / 20)) / max(float(np.sqrt(np.mean(noise ** 2))), 1e-12)
        yield f"noise_{snr}db", np.clip(clip + noise, -1, 1).astype(np.float32)
    yield "volume", clip * float(rng.uniform(0.72, 0.92))


def prepare():
    DATASET.mkdir(parents=True, exist_ok=True)
    rows, excluded = transfer.prepare_sources()
    originals, manifest = [], []
    for number, row in enumerate(rows, 1):
        source = Path(row["source_path"])
        # Preserve original bytes in inspectable train/validation/test folders.
        saved = DATASET / "splits" / row["dataset_split"] / row["class_label"] / source.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        if not saved.exists() or sha(saved) != row["source_hash"]:
            shutil.copy2(source, saved)
        originals.append(dict(row, saved_path=str(saved.relative_to(ROOT))))
        try:
            y, sr = load_audio(saved)
        except (ValueError, OSError, EOFError) as exc:
            excluded.append({"filename": str(saved), "reason": str(exc)})
            continue
        for index, (clip, start, end) in enumerate(segments(y, sr)):
            try:
                validate_signal(clip)
            except ValueError as exc:
                excluded.append({"filename": f"{saved}#{index}", "reason": str(exc)})
                continue
            seed = int(hashlib.sha256(f"{row['source_hash']}:{index}:{VERSION}".encode()).hexdigest()[:8], 16)
            copies = variants(clip, seed) if row["dataset_split"] == "train" else [("original", clip)]
            for kind, audio in copies:
                path = DATASET / "prepared" / row["dataset_split"] / row["class_label"] / f"{row['audio_id']}_{index:02d}_{kind}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                # FLOAT WAV retains the exact augmented samples used for extraction.
                sf.write(path, audio, sr, subtype="FLOAT")
                manifest.append(dict(row, path=str(path.relative_to(ROOT)), segment_index=index,
                                     segment_start=start, segment_end=end, augmentation=kind,
                                     augmentation_seed=seed, audio_sha256=sha(path)))
        if number % 100 == 0 or number == len(rows):
            print(f"Saved audio: {number}/{len(rows)} sources, {len(manifest)} clips", flush=True)
    validate_manifest(manifest)
    write_json(DATASET / "originals_manifest.json", originals)
    write_json(DATASET / "manifest.json", manifest)
    write_json(DATASET / "excluded.json", excluded)
    summary = {"augmentation_version": VERSION, "augmentation": AUGMENTATION, "seed": 7,
               "original_sources": {s: sum(r["dataset_split"] == s for r in originals) for s in ("train", "validation", "test")},
               "prepared_clips": {s: sum(r["dataset_split"] == s for r in manifest) for s in ("train", "validation", "test")},
               "variants": {k: sum(r["augmentation"] == k for r in manifest) for k in ("original", "shift", "noise_20db", "noise_10db", "volume")},
               "source_hash_leakage": False}
    write_json(DATASET / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


def validate_manifest(rows):
    owners = {}
    for row in rows:
        split = row["dataset_split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Invalid split")
        if owners.setdefault(row["source_hash"], split) != split:
            raise ValueError("Source leakage across splits")
        if split != "train" and row["augmentation"] != "original":
            raise ValueError("Augmentation outside training split")


def extract(manifest):
    validate_manifest(manifest)
    RUN.mkdir(parents=True, exist_ok=True)
    cache = RUN / "cache"
    cache.mkdir(exist_ok=True)
    download_base()
    vectors = []
    for number, row in enumerate(manifest, 1):
        audio_path = ROOT / row["path"]
        if sha(audio_path) != row["audio_sha256"]:
            raise ValueError(f"Prepared audio changed: {audio_path}")
        key = hashlib.sha256(f"{FEATURE_VERSION}:{row['audio_sha256']}".encode()).hexdigest()
        cached = cache / f"{key}.npy"
        if cached.exists():
            vector = np.load(cached, allow_pickle=False)
        else:
            y, sr = sf.read(audio_path, dtype="float32")
            vector = extract_transfer_features(y, sr)[0]
            pending = cached.with_suffix(".pending.npy")
            np.save(pending, vector)
            pending.replace(cached)
        if vector.shape != (2308,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid feature cache: {cached}")
        vectors.append(vector)
        if number % 100 == 0 or number == len(manifest):
            print(f"Augmented CNN features: {number}/{len(manifest)}", flush=True)
    pending = RUN / "features.pending.npz"
    np.savez_compressed(pending, x=np.asarray(vectors), y=np.asarray([r["class_label"] for r in manifest]),
                        split=np.asarray([r["dataset_split"] for r in manifest]),
                        audio_id=np.asarray([r["source_hash"] for r in manifest]),
                        feature_version=np.asarray(FEATURE_VERSION))
    pending.replace(RUN / "features.npz")
    write_json(RUN / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--reuse-audio", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((DATASET / "manifest.json").read_text()) if args.reuse_audio else prepare()
    if args.prepare_only:
        return
    extract(manifest)
    if args.features_only:
        return
    transfer.RUN = RUN
    if args.activate:
        backup = ROOT / "python_models" / "selection_before_augmentation.json"
        if not backup.exists():
            shutil.copy2(ROOT / "python_models" / "selection.json", backup)
    transfer.train(args.activate, artifact="yamnet_augmented", augmentation=AUGMENTATION)


if __name__ == "__main__":
    main()
