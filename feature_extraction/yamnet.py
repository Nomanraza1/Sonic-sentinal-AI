"""Frozen YAMNet CNN embeddings; the classifier is trained separately."""
from functools import lru_cache
from pathlib import Path
import tarfile
import urllib.request

import librosa
import numpy as np

from feature_extraction.features import extract_features

FEATURE_VERSION = "yamnet-mean-std-mfcc-v1"
MODEL_DIR = Path(__file__).resolve().parents[1] / "python_models" / "yamnet_base"
MODEL_URL = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"


def download_base():
    """Explicit setup step; inference never downloads a model implicitly."""
    if (MODEL_DIR / "saved_model.pb").exists():
        return
    archive = MODEL_DIR.with_suffix(".tar.gz")
    if not archive.exists():
        print("Downloading Google's pretrained YAMNet CNN...", flush=True)
        pending = archive.with_suffix(".pending")
        urllib.request.urlretrieve(MODEL_URL, pending)
        with tarfile.open(pending):
            pass
        pending.replace(archive)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(MODEL_DIR, filter="data")


@lru_cache(maxsize=1)
def base_model():
    if not (MODEL_DIR / "saved_model.pb").exists():
        raise FileNotFoundError("Missing YAMNet base. Run python_models/train_transfer.py --extract.")
    import tensorflow as tf

    model = tf.saved_model.load(str(MODEL_DIR))
    return model


def extract_transfer_features(y, sr):
    waveform = np.asarray(y, dtype=np.float32)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Expected finite mono audio.")
    waveform = librosa.util.normalize(waveform)
    resampled = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    _, embeddings, _ = base_model()(resampled.astype(np.float32))
    embeddings = embeddings.numpy()
    return np.concatenate([
        embeddings.mean(axis=0), embeddings.std(axis=0),
        extract_features(waveform, sr)[0],
    ]).astype(np.float32)[None, :]
