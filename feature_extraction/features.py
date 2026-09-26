import numpy as np
import librosa

def extract_features(y, sr):
    values = []
    groups = [librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32), librosa.feature.chroma_stft(y=y, sr=sr), librosa.feature.zero_crossing_rate(y), librosa.feature.rms(y=y), librosa.feature.spectral_centroid(y=y, sr=sr), librosa.feature.spectral_bandwidth(y=y, sr=sr), librosa.feature.spectral_rolloff(y=y, sr=sr)]
    for group in groups: values.extend([float(np.mean(group)), float(np.std(group))])
    return np.asarray(values, dtype=np.float32).reshape(1, -1)
