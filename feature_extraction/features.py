import numpy as np
from scipy.fft import dct


def stats(values):
    return [float(np.mean(values)), float(np.std(values))]


def band_stats(values):
    return list(np.mean(values, axis=0).astype(float)) + list(
        np.std(values, axis=0).astype(float)
    )


def extract_features(y, sr):
    size = int(np.ceil(len(y) / 32))
    frames = np.pad(y, (0, size * 32 - len(y))).reshape(32, size)
    window = np.hanning(len(frames[0]))
    spectra = np.asarray([np.abs(np.fft.rfft(frame * window)) for frame in frames])
    freqs = np.fft.rfftfreq(len(frames[0]), 1 / sr)
    power = spectra**2
    total = np.sum(power, axis=1) + 1e-9
    mel = np.stack(
        [band.mean(axis=1) for band in np.array_split(power, 32, axis=1)], axis=1
    )
    mfcc = dct(np.log(mel + 1e-9), type=2, axis=1, norm="ortho")[:, :13]
    chroma = np.zeros((len(frames), 12))
    bins = np.maximum(
        1, np.rint(12 * np.log2(np.maximum(freqs, 1) / 440) + 69).astype(int) % 12
    )
    for number in range(12):
        chroma[:, number] = np.sum(power[:, bins == number], axis=1)
    zcr = np.asarray([np.mean(np.abs(np.diff(np.signbit(frame)))) for frame in frames])
    rms = np.sqrt(np.mean(np.asarray(frames) ** 2, axis=1))
    centroid = np.sum(power * freqs, axis=1) / total
    bandwidth = np.sqrt(
        np.sum(power * (freqs - centroid[:, None]) ** 2, axis=1) / total
    )
    cumulative = np.cumsum(power, axis=1)
    rolloff = np.asarray(
        [
            freqs[min(len(freqs) - 1, np.searchsorted(cumulative[i], total[i] * 0.85))]
            for i in range(len(frames))
        ]
    )
    values = band_stats(mfcc) + band_stats(mel) + band_stats(chroma)
    for group in [zcr, rms, centroid, bandwidth, rolloff]:
        values.extend(stats(group))
    return np.asarray(values, dtype=np.float32).reshape(1, -1)
