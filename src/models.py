import json
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
from config.class_map import SONIC_CLASSES
from config.settings import PYTHON_MODELS, GTM_MODEL, ACTIVE_PYTHON_MODEL
from feature_extraction.features import FEATURE_VERSION, extract_features


def unavailable(reason):
    return {"available": False, "reason": reason, "class": None, "scores": {}}


@lru_cache(maxsize=4)
def cached_python_model(path, modified):
    return joblib.load(path)


def predict_python(features=None, *, waveform=None, sr=None):
    selected = ACTIVE_PYTHON_MODEL
    selection = PYTHON_MODELS / "selection.json"
    if selection.exists():
        selected = json.loads(selection.read_text(encoding="utf-8"))["selected"]
    path = PYTHON_MODELS / f"{selected}.joblib"
    if not path.exists():
        return unavailable(f"Missing {path.name}")
    try:
        model = cached_python_model(str(path), path.stat().st_mtime_ns)
    except Exception as exc:
        return unavailable(f"Python model could not be loaded: {type(exc).__name__}")
    from feature_extraction.yamnet import FEATURE_VERSION as TRANSFER_VERSION

    version = getattr(model, "feature_version_", None)
    if version == TRANSFER_VERSION:
        if waveform is None or sr is None:
            return unavailable("The CNN model requires audio samples and a sample rate.")
        try:
            from feature_extraction.yamnet import extract_transfer_features

            features = extract_transfer_features(waveform, sr)
        except (ImportError, FileNotFoundError) as exc:
            return unavailable(f"CNN model unavailable: {exc}")
        if model.feature_view_ == "mean":
            features = features[:, :1024]
    elif version != FEATURE_VERSION:
        return unavailable("Python model uses outdated features. Rerun extraction and training.")
    elif features is None:
        features = extract_features(waveform, sr)
    raw = model.predict_proba(features)[0]
    labels = list(model.classes_)
    scores = {
        label: float(raw[labels.index(label)]) if label in labels else 0.0
        for label in SONIC_CLASSES
    }
    label = max(scores, key=scores.get)
    return {
        "available": True,
        "class": label,
        "scores": scores,
        "confidence": scores[label],
        "version": selected,
    }


def predict_gtm(y, sr):
    path = GTM_MODEL / "model.keras"
    labels = GTM_MODEL / "labels.json"
    if not path.exists() or not labels.exists():
        return unavailable("GTM model.keras or labels.json is missing")
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(path)
        names = json.loads(labels.read_text(encoding="utf-8"))
        if names != SONIC_CLASSES:
            return unavailable("GTM labels.json must contain the ten SonicSentinel classes in the configured order")
        clip = np.pad(y[: sr * 3], (0, max(0, sr * 3 - len(y))))[None, :, None]
        raw = model.predict(clip, verbose=0)[0]
        scores = {name: float(raw[i]) for i, name in enumerate(names)}
        label = max(scores, key=scores.get)
        return {
            "available": True,
            "class": label,
            "scores": scores,
            "confidence": scores[label],
            "version": "gtm-model.keras",
        }
    except Exception as e:
        return unavailable(f"GTM load failed: {e}")


# veriifed
