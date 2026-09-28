import numpy as np

from config.class_map import SONIC_CLASSES
from feature_extraction.gtm import extract_gtm_features, gtm_labels
from src.models import predict_gtm


def test_gtm_preprocessing_matches_export_input_contract():
    labels = gtm_labels()
    assert len(labels) == len(SONIC_CLASSES)
    assert set(labels) == set(SONIC_CLASSES)
    features = extract_gtm_features(np.zeros(16000, dtype=np.float32), 16000)
    assert features.shape == (1, 31, 64, 1)
    assert np.isfinite(features).all()


def test_gtm_predicts_from_independent_audio_windows():
    time = np.arange(32000, dtype=np.float32) / 16000
    waveform = 0.2 * np.sin(2 * np.pi * 440 * time)
    result = predict_gtm(waveform, 16000)
    assert result["available"], result.get("reason")
    assert result["class"] in SONIC_CLASSES
    assert set(result["scores"]) == set(SONIC_CLASSES)
    assert np.isclose(sum(result["scores"].values()), 1.0, atol=1e-5)
    assert result["version"] == "gtm_model_final"
