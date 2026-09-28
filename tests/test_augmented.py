import numpy as np
import pytest
from python_models.train_augmented import variants, validate_manifest


def test_augmentations_are_reproducible_and_noise_is_at_requested_snr():
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(66150) / 22050)).astype(np.float32)
    first, second = dict(variants(y, 7)), dict(variants(y, 7))
    assert set(first) == {"original", "shift", "noise_20db", "noise_10db", "volume"}
    for key in first:
        assert np.array_equal(first[key], second[key])
        assert first[key].shape == y.shape
    for snr in (20, 10):
        observed = 10 * np.log10(np.mean(y ** 2) / np.mean((first[f"noise_{snr}db"] - y) ** 2))
        assert observed == pytest.approx(snr, abs=0.01)


def test_manifest_rejects_leakage_and_holdout_augmentation():
    row = dict(source_hash="one", dataset_split="train", augmentation="original")
    validate_manifest([row, dict(row, augmentation="noise_10db")])
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest([row, dict(row, dataset_split="test")])
    with pytest.raises(ValueError, match="outside"):
        validate_manifest([dict(row, dataset_split="validation", augmentation="shift")])
