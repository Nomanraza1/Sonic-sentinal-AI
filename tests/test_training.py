import csv
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction.features import FEATURE_VERSION, extract_features
from python_models import train
from src import models

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tone(seconds=1):
    t = np.arange(int(22050 * seconds)) / 22050
    return 0.3 * np.sin(2 * np.pi * 440 * t)


def test_features_are_fixed_size_and_finite():
    for audio in (tone(0.01), tone(1), tone(3)):
        result = extract_features(audio, 22050)
        assert result.shape == (1, 260)
        assert np.isfinite(result).all()
    with pytest.raises(ValueError):
        extract_features(np.array([np.nan]), 22050)


def test_preparation_excludes_internal_silent_segment(tmp_path, monkeypatch):
    prepare = script("01_prepare_dataset")
    monkeypatch.setattr(prepare, "PROCESSED", tmp_path / "processed")
    source = tmp_path / "source.wav"
    sf.write(source, np.concatenate([tone(3), np.zeros(22050 * 3), tone(3)]), 22050)
    rows = [{"source_path": str(source), "audio_id": "one", "dataset_split": "train", "class_label": "gunshot"}]
    ignored = []
    result = prepare.build(rows, ignored)
    assert len(result) == 8  # Two audible segments, each with three augmentations.
    assert len(ignored) == 1
    assert "#segment=1" in ignored[0]["filename"]


def extraction_fixture(tmp_path, monkeypatch):
    extraction = script("02_extract_features")
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    (tmp_path / "dataset").mkdir()
    (tmp_path / "data").mkdir()
    sf.write(tmp_path / "silent.wav", np.zeros(22050), 22050)
    sf.write(tmp_path / "tone.wav", tone(), 22050)
    rows = [dict(path=str(tmp_path / name), class_label="gunshot", dataset_split="train", audio_id=name)
            for name in ("silent.wav", "tone.wav")]
    with (tmp_path / "dataset" / "metadata_split.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return extraction


def test_extraction_skips_silence_and_replaces_stale_archive(tmp_path, monkeypatch):
    extraction = extraction_fixture(tmp_path, monkeypatch)
    (tmp_path / "data" / "features.npz").write_bytes(b"stale")
    extraction.main()
    with np.load(tmp_path / "data" / "features.npz") as result:
        assert result["x"].shape == (1, 260)
        assert result["audio_id"].tolist() == ["tone.wav"]
        assert str(result["feature_version"]) == FEATURE_VERSION
    assert "silent.wav" in (tmp_path / "data" / "excluded_features.csv").read_text()


def test_failed_extraction_removes_stale_archive(tmp_path, monkeypatch):
    extraction = extraction_fixture(tmp_path, monkeypatch)
    (tmp_path / "data" / "features.npz").write_bytes(b"stale")
    def fail(*args):
        raise RuntimeError("unexpected extraction failure")
    monkeypatch.setattr(extraction, "extract_features", fail)
    with pytest.raises(RuntimeError):
        extraction.main()
    assert not (tmp_path / "data" / "features.npz").exists()


def dataset():
    return dict(x=np.tile([[0., 1.], [1., 0.]], (6, 1)),
                y=np.tile(["gunshot", "background_noise"], 6),
                split=np.repeat(["train", "validation", "test"], 4),
                audio_id=np.array([str(i) for i in range(12)]),
                feature_version=np.asarray(FEATURE_VERSION))


def test_training_rejects_old_features_missing_classes_and_leakage():
    data = dataset()
    train.validate_dataset(data)
    data["feature_version"] = np.asarray("old")
    with pytest.raises(ValueError, match="Outdated"):
        train.validate_dataset(data)
    data = dataset()
    data["y"][4:8] = "gunshot"
    with pytest.raises(ValueError, match="missing classes"):
        train.validate_dataset(data)
    data = dataset()
    data["audio_id"][4] = data["audio_id"][0]
    with pytest.raises(ValueError, match="leakage"):
        train.validate_dataset(data)


def test_training_and_app_use_selected_scaled_model(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "python_models").mkdir()
    np.savez(tmp_path / "data" / "features.npz", **dataset())
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "candidate_models", lambda: {
        "svm_test": make_pipeline(StandardScaler(), SVC(random_state=7))})
    train.main()
    monkeypatch.setattr(models, "PYTHON_MODELS", tmp_path / "python_models")
    result = models.predict_python(np.array([[0., 1.]]))
    assert result["available"]
    assert result["version"] == "svm_test"
    assert sum(result["scores"].values()) == pytest.approx(1)
    metrics = json.loads((tmp_path / "python_models" / "svm_test_metrics.json").read_text())
    assert metrics["test"]["accuracy"] == 1
    model = joblib.load(tmp_path / "python_models" / "svm_test.joblib")
    del model.feature_version_
    joblib.dump(model, tmp_path / "python_models" / "svm_test.joblib")
    assert not models.predict_python(np.array([[0., 1.]]))["available"]
