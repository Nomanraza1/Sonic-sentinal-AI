import io
import numpy as np
import soundfile as sf
import librosa
from config.settings import SAMPLE_RATE, SEGMENT_SECONDS, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS

def validate_upload(name, raw):
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if ext not in ALLOWED_EXTENSIONS: raise ValueError('Unsupported audio format.')
    if not raw or len(raw) > MAX_UPLOAD_BYTES: raise ValueError('File is empty or exceeds 25 MB.')
    try: info = sf.info(io.BytesIO(raw))
    except Exception as e: raise ValueError('Audio decoding failed.') from e
    if info.duration < .2 or info.duration > 30: raise ValueError('Audio must be between 0.2 and 30 seconds.')
    if info.samplerate < 8000 or info.channels not in (1, 2): raise ValueError('Unsupported sampling rate or channel count.')
    return info

def load_audio(path_or_bytes):
    y, sr = librosa.load(path_or_bytes, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    if not len(y) or np.max(np.abs(y)) < 0.005: raise ValueError('Silent or unusable recording.')
    y = librosa.util.normalize(y)
    return y, sr

def segments(y, sr=SAMPLE_RATE):
    size = sr * SEGMENT_SECONDS
    for start in range(0, len(y), size):
        clip = y[start:start + size]
        clip = np.pad(clip, (0, max(0, size-len(clip))))
        yield clip, start/sr, min((start+size)/sr, len(y)/sr)

def quality(y):
    rms = float(np.sqrt(np.mean(y*y))); clipping = float(np.mean(np.abs(y) >= .99))
    noise = float(np.percentile(np.abs(y), 25))
    if rms < .01: return 'Unusable', {'rms': rms, 'clipping': clipping, 'noise': noise}
    if clipping > .03 or rms < .03: return 'Poor', {'rms': rms, 'clipping': clipping, 'noise': noise}
    if noise > .12: return 'Acceptable', {'rms': rms, 'clipping': clipping, 'noise': noise}
    return 'Good', {'rms': rms, 'clipping': clipping, 'noise': noise}

def fingerprint(y, sr):
    spec = np.abs(librosa.stft(y, n_fft=1024))
    bands = np.mean(spec[:128], axis=1)
    threshold = float(np.median(bands))
    return ''.join('1' if value >= threshold else '0' for value in bands)

def fingerprint_distance(one, two):
    if not one or not two or len(one) != len(two):
        return 999
    return sum(a != b for a, b in zip(one, two))
