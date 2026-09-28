# YAMNet training with saved splits and augmentations

The reproducible training entry point is `python python_models/train_augmented.py`.
It trains classifiers on frozen pretrained YAMNet features. It does not fine-tune the CNN weights.

## Saved audio

All source copies and prepared audio are inside `audio_dataset/yamnet_augmented/`:

```text
splits/
  train/<class>/<source recording>
  validation/<class>/<source recording>
  test/<class>/<source recording>
prepared/
  train/<class>/<source ID>_<segment>_<augmentation>.wav
  validation/<class>/<source ID>_<segment>_original.wav
  test/<class>/<source ID>_<segment>_original.wav
originals_manifest.json
manifest.json
excluded.json
summary.json
```

`splits` preserves the original recording bytes. `prepared` contains the exact float PCM WAV samples used for feature extraction. Original audio stays in `audio_dataset/raw/` as well.

The split is stratified at the original recording level, with random seed 7 and approximately 70% training, 15% validation and 15% testing. Each original and every derived segment or augmentation stays in the same partition. Source hashes must not cross partitions, and conflicting labels on identical files are rejected.

## Training augmentations

Each usable training segment has five versions:

1. Original segment.
2. A deterministic circular time shift of up to approximately 180 milliseconds.
3. Gaussian noise at a nominal 20 dB signal-to-noise ratio.
4. Gaussian noise at a nominal 10 dB signal-to-noise ratio.
5. A deterministic volume reduction between 72% and 92% of the original amplitude.

Noise is clipped to the valid amplitude range, so actual SNR can differ when clipping occurs. Inference normalization can remove much of the volume scaling effect; these copies do not necessarily add independent feature information. Augmented clips are not counted as new original recordings. Validation and test segments have no augmentation.

## Features and model selection

Feature caches, the final feature archive, its matching manifest and candidate results live under `data/transfer_augmented/`. Feature-cache keys include the feature version and saved audio checksum. Completed vectors are reused after interruption. Prepared audio is checksum-verified before it is used.

Nine candidate classifiers are compared by validation macro-F1. Only the selected classifier is evaluated on the test partition. Its artifact and report are `python_models/yamnet_augmented.joblib` and `python_models/yamnet_augmented_metrics.json`. The existing `yamnet_transfer.joblib` is preserved.

The test partition is the previously inspected project benchmark. Retraining does not make it a new independent evaluation. Use a fresh unseen collection for final generalization claims.

## Commands

```powershell
# Save audio and then extract features and train classifiers.
python python_models/train_augmented.py

# Save audio only.
python python_models/train_augmented.py --prepare-only

# Resume from the saved audio manifest; completed feature vectors are reused.
python python_models/train_augmented.py --reuse-audio
```

The command does not activate the new classifier by default. `--activate` changes the active selection after training and preserves `selection_before_augmentation.json`. Review the recorded validation results before deployment. The Colab equivalent is `notebooks/colab_augmented_training.ipynb`.

Audio and feature-cache directories are excluded from Git. Copy or archive them separately when handing over the dataset. A repository push alone does not distribute the splits or augmented recordings.

Actual completed-run measurements and activation status belong in `reports/augmented_training_results.md`. Until a successful run and report exist, the new training must not be presented as complete.
