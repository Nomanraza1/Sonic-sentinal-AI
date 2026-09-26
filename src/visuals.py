from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display


def make_images(y, sr, audio_id, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    waveform = folder / f"{audio_id}_waveform.png"
    spectrogram = folder / f"{audio_id}_spectrogram.png"
    fig, ax = plt.subplots(figsize=(10, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#16805f")
    ax.set(title="Waveform", xlabel="Time (seconds)")
    fig.tight_layout()
    fig.savefig(waveform, dpi=140)
    plt.close(fig)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=max)
    fig, ax = plt.subplots(figsize=(10, 3))
    image = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel spectrogram")
    fig.tight_layout()
    fig.savefig(spectrogram, dpi=140)
    plt.close(fig)
    return waveform, spectrogram


# verified
