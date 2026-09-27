import librosa
import numpy as np

FEATURE_VERSION = "mel-mfcc-v2"


def extract_features(y, sr):
    """Fixed-time spectral features shared by training and inference."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("Features require finite, nonempty mono audio.")
    y = librosa.util.normalize(y)
    y = np.pad(y, (0, max(0, 2048 - len(y))))
    magnitude = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(S=magnitude ** 2, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=20)
    groups = [
        log_mel, mfcc,
        librosa.feature.delta(mfcc, mode="nearest"),
        librosa.feature.delta(mfcc, order=2, mode="nearest"),
        librosa.feature.spectral_centroid(S=magnitude, sr=sr),
        librosa.feature.spectral_bandwidth(S=magnitude, sr=sr),
        librosa.feature.spectral_rolloff(S=magnitude, sr=sr),
        librosa.feature.spectral_flatness(S=magnitude),
        librosa.feature.zero_crossing_rate(y, hop_length=512),
        librosa.feature.rms(S=magnitude),
    ]
    values = np.concatenate([
        statistic(group, axis=1)
        for group in groups
        for statistic in (np.mean, np.std)
    ]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite audio features.")
    return values.reshape(1, -1)
