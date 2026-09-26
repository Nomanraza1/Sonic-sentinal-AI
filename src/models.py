import json
from pathlib import Path
import joblib
import numpy as np
from config.class_map import SONIC_CLASSES
from config.settings import PYTHON_MODELS, GTM_MODEL, ACTIVE_PYTHON_MODEL

def unavailable(reason): return {'available': False, 'reason': reason, 'class': None, 'scores': {}}
def predict_python(features):
    path = PYTHON_MODELS / f'{ACTIVE_PYTHON_MODEL}.joblib'
    if not path.exists(): return unavailable(f'Missing {path.name}')
    model = joblib.load(path); raw = model.predict_proba(features)[0]
    labels = list(model.classes_); scores = {label: float(raw[labels.index(label)]) if label in labels else 0.0 for label in SONIC_CLASSES}
    label = max(scores, key=scores.get)
    return {'available': True, 'class': label, 'scores': scores, 'confidence': scores[label], 'version': ACTIVE_PYTHON_MODEL}
def predict_gtm(y, sr):
    # Expected TensorFlow/Keras export is documented in state.md.
    path = GTM_MODEL / 'model.keras'
    labels = GTM_MODEL / 'labels.json'
    if not path.exists() or not labels.exists(): return unavailable('GTM model.keras or labels.json is missing')
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path); names = json.loads(labels.read_text())
        clip = np.pad(y[:sr*3], (0, max(0, sr*3-len(y))))[None, :, None]
        raw = model.predict(clip, verbose=0)[0]; scores = {name: float(raw[i]) for i, name in enumerate(names)}
        label = max(scores, key=scores.get)
        return {'available': True, 'class': label, 'scores': scores, 'confidence': scores[label], 'version': 'gtm-model.keras'}
    except Exception as e: return unavailable(f'GTM load failed: {e}')
