import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from feature_extraction import yamnet
from src import models


def test_transfer_features_pool_cnn_frames(monkeypatch):
    class Embeddings:
        def numpy(self):
            return np.stack([np.ones(1024), np.ones(1024) * 3])
    seen = []
    def cnn(waveform):
        seen.append(waveform)
        return None, Embeddings(), None
    monkeypatch.setattr(yamnet, "base_model", lambda: cnn)
    monkeypatch.setattr(yamnet, "extract_features", lambda y, sr: np.zeros((1, 260)))
    result = yamnet.extract_transfer_features(np.ones(22050), 22050)
    assert result.shape == (1, 2308)
    assert np.allclose(result[0, :1024], 2)
    assert np.allclose(result[0, 1024:2048], 1)
    assert len(seen[0]) == 16000


def test_app_dispatches_audio_to_transfer_extractor(tmp_path, monkeypatch):
    model = LogisticRegression().fit(np.array([[0.] * 1024, [1.] * 1024]), ["gunshot", "background_noise"])
    model.feature_version_ = yamnet.FEATURE_VERSION
    model.feature_view_ = "mean"
    joblib.dump(model, tmp_path / "cnn.joblib")
    (tmp_path / "selection.json").write_text('{"selected":"cnn"}')
    monkeypatch.setattr(models, "PYTHON_MODELS", tmp_path)
    monkeypatch.setattr(yamnet, "extract_transfer_features", lambda y, sr: np.zeros((1, 2308)))
    result = models.predict_python(waveform=np.ones(16000), sr=16000)
    assert result["available"]
    assert result["class"] == "gunshot"
    assert not models.predict_python(np.zeros((1, 260)))["available"]
