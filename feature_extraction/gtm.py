"""Preprocessing for the exported Teachable Machine audio classifier."""
import json
from functools import lru_cache

import librosa
import numpy as np
import tensorflow as tf

from config.class_map import SONIC_CLASSES
from config.settings import GTM_MODEL


@lru_cache(maxsize=1)
def gtm_metadata():
    return json.loads((GTM_MODEL / "metadata.json").read_text(encoding="utf-8"))


def gtm_labels():
    labels = gtm_metadata()["labels"]
    aliases = {"person_asking_help": "person_asking_for_help"}
    mapped = [aliases.get(label, label) for label in labels]
    if len(mapped) != len(SONIC_CLASSES) or set(mapped) != set(SONIC_CLASSES):
        raise ValueError("GTM metadata must identify each of the ten SonicSentinel classes")
    return mapped


def extract_gtm_features(waveform, sample_rate):
    """Return normalized one second log-mel windows in export input format."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio")
    meta = gtm_metadata()["features"]
    target_rate = meta["sample_rate"]
    if sample_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
    window = meta["window"]
    pad_to = meta["pad_to"]
    frames = []
    for start in range(0, max(len(audio), 1), window):
        clip = audio[start:start + window]
        if not clip.size:
            continue
        clip = np.pad(clip, (0, max(0, window - len(clip))))[:window]
        clip = np.pad(clip, (0, pad_to - window))
        spectrum = tf.signal.stft(
            clip, frame_length=meta["frame_length"],
            frame_step=meta["frame_step"], fft_length=meta["fft_length"],
        )
        power = tf.square(tf.abs(spectrum))
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            meta["mel_bins"], meta["fft_length"] // 2 + 1, target_rate,
            meta["fmin"], meta["fmax"], dtype=tf.float32,
        )
        mel = tf.matmul(power, mel_matrix)
        log_mel = tf.math.log(mel + meta["log_offset"])
        mean = tf.reduce_mean(log_mel)
        std = tf.math.reduce_std(log_mel)
        normalized = (log_mel - mean) / tf.maximum(std, 1e-6)
        frames.append(normalized.numpy()[..., np.newaxis])
    return np.stack(frames).astype(np.float32)
