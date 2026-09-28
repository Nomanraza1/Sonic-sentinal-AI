"""Train a classifier on frozen pretrained audio-CNN features.

Run --extract once, then rerun without it to reuse the per-segment feature cache.
All candidates use training data only; validation chooses the winner.
Test data is evaluated only after selection. Existing artifacts are preserved.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_preprocessing.audio import load_audio, segments, validate_signal
from config.class_map import SONIC_CLASSES
from feature_extraction.yamnet import FEATURE_VERSION, download_base, extract_transfer_features
from python_models.train import scores

RUN = ROOT / "data" / "transfer_experiment"


def prepare_sources():
    spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts" / "01_prepare_dataset.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    rows, ignored = prepare.scan()
    if set(row["class_label"] for row in rows) != set(SONIC_CLASSES):
        raise ValueError("All ten class folders must contain readable source audio.")
    # Identical source files with conflicting labels must not cross splits.
    hashes = {}
    for row in rows:
        previous = hashes.setdefault(row["source_hash"], row["class_label"])
        if previous != row["class_label"]:
            raise ValueError(f"Duplicate audio has conflicting labels: {row['source_path']}")
    return prepare.split_rows(rows), ignored


def extract():
    RUN.mkdir(parents=True, exist_ok=True)
    cache = RUN / "cache"
    cache.mkdir(exist_ok=True)
    download_base()
    rows, excluded = prepare_sources()
    features, labels, splits, ids, manifest = [], [], [], [], []
    for number, row in enumerate(rows, 1):
        try:
            y, sr = load_audio(row["source_path"])
        except (ValueError, OSError, EOFError) as exc:
            excluded.append({"filename": row["source_path"], "reason": str(exc)})
            continue
        for index, (clip, _, _) in enumerate(segments(y, sr)):
            try:
                validate_signal(clip)
            except ValueError as exc:
                excluded.append({"filename": f"{row['source_path']}#{index}", "reason": str(exc)})
                continue
            key = hashlib.sha256(f"{FEATURE_VERSION}:{row['source_hash']}:{index}".encode()).hexdigest()
            path = cache / f"{key}.npy"
            if path.exists():
                vector = np.load(path, allow_pickle=False)
            else:
                vector = extract_transfer_features(clip, sr)[0]
                np.save(path, vector)
            features.append(vector)
            labels.append(row["class_label"])
            splits.append(row["dataset_split"])
            ids.append(row["source_hash"])
            manifest.append(dict(row, segment_index=index))
        if number % 100 == 0 or number == len(rows):
            print(f"CNN features: {number}/{len(rows)} sources, {len(features)} segments", flush=True)
    np.savez_compressed(RUN / "features.npz", x=np.asarray(features), y=np.asarray(labels),
                        split=np.asarray(splits), audio_id=np.asarray(ids),
                        feature_version=np.asarray(FEATURE_VERSION))
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RUN / "excluded.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")
    print("Prepared splits:", {name: splits.count(name) for name in set(splits)}, flush=True)


def candidates():
    for c in (0.1, 1, 10):
        yield f"logistic_c{c}", "mean", make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
    for c in (1, 10, 100):
        yield f"svm_mean_c{c}", "mean", make_pipeline(
            Normalizer(), SVC(C=c, gamma="scale", class_weight="balanced", probability=True, random_state=7))
    for c in (1, 10, 100):
        yield f"svm_combined_c{c}", "combined", make_pipeline(
            StandardScaler(), SVC(C=c, gamma="scale", class_weight="balanced", probability=True, random_state=7))


def train(activate=False, artifact="yamnet_transfer", augmentation="none; original segments only"):
    with np.load(RUN / "features.npz", allow_pickle=False) as source:
        x, y, split, ids = (source[name] for name in ("x", "y", "split", "audio_id"))
        if str(source["feature_version"]) != FEATURE_VERSION:
            raise ValueError("Re-extract outdated CNN features.")
    if x.ndim != 2 or x.shape[1] != 2308 or not np.isfinite(x).all():
        raise ValueError("Invalid CNN features.")
    masks = {name: split == name for name in ("train", "validation", "test")}
    for name, mask in masks.items():
        if set(y[mask]) != set(SONIC_CLASSES):
            raise ValueError(f"Missing classes in {name}.")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(ids[masks[a]]) & set(ids[masks[b]]):
            raise ValueError("Source leakage across splits.")
    results, best = {}, None
    for name, view, model in candidates():
        values = x[:, :1024] if view == "mean" else x
        print(f"Training {name}...", flush=True)
        model.fit(values[masks["train"]], y[masks["train"]])
        # Evaluate the same probability argmax used by the deployed application.
        predicted = model.classes_[model.predict_proba(values[masks["validation"]]).argmax(axis=1)]
        validation = scores(y[masks["validation"]], predicted, SONIC_CLASSES)
        results[name] = validation
        print(f"{name}: validation accuracy={validation['accuracy']:.4f}, macro-F1={validation['macro_f1']:.4f}", flush=True)
        if best is None or validation["macro_f1"] > best[0]:
            best = (validation["macro_f1"], name, view, model)
        (RUN / "validation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _, name, view, model = best
    values = x[:, :1024] if view == "mean" else x
    final = scores(y[masks["test"]], model.classes_[model.predict_proba(values[masks["test"]]).argmax(axis=1)], SONIC_CLASSES)
    model.feature_version_ = FEATURE_VERSION
    model.feature_view_ = view
    joblib.dump(model, ROOT / "python_models" / f"{artifact}.joblib")
    report = {"candidate": name, "feature_version": FEATURE_VERSION, "validation": results[name],
              "test": final, "split_counts": {k: int(v.sum()) for k, v in masks.items()},
              "source_counts": {k: len(set(ids[v])) for k, v in masks.items()},
              "selection_metric": "validation macro_f1", "prediction_rule": "probability argmax",
              "augmentation": augmentation, "seed": 7,
              "pretrained_model": "https://tfhub.dev/google/yamnet/1",
              "feature_archive_sha256": hashlib.sha256((RUN / "features.npz").read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
              "status": "final test evaluation complete"}
    (ROOT / "python_models" / f"{artifact}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if activate:
        selection = ROOT / "python_models" / "selection.json"
        backup = ROOT / "python_models" / "selection_before_transfer.json"
        if selection.exists() and not backup.exists():
            backup.write_bytes(selection.read_bytes())
        selection.write_text(json.dumps({"selected": artifact, "candidate": name,
            "feature_version": FEATURE_VERSION, "selection_metric": "validation macro_f1",
            "available_classes": SONIC_CLASSES, "missing_classes": []}, indent=2), encoding="utf-8")
    print(f"Winner: {name}; test accuracy={final['accuracy']:.4f}; macro-F1={final['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if args.extract or args.extract_only:
        extract()
    if not args.extract_only:
        train(args.activate)
