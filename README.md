# SonicSentinel AI

For saved-model setup, administrator provisioning and evaluator steps, see [the installation guide](documentation/INSTALLATION.md). See [delivery readiness](reports/delivery_readiness.md) for the latest checks and remaining SRS work. The supplied GTM export is integrated as an independent second model. Its full 100-recording comparison and submission evidence still need to be completed.

SonicSentinel AI is a Flask application for uploaded-audio analysis and visible, permission-based live microphone monitoring. It records independent Python-model and GTM-model predictions, compares their confidence, evaluates audio quality, applies alert rules, and routes uncertain events to review.

## Run locally

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, then register an account.

The server loads both audio models before accepting requests, so the first analysis avoids model-loading delay. Startup takes longer while the models warm up. Audio charts are cached, and the review queue is paginated. If an older server is still running, stop it with Ctrl+C and restart to load code changes. Set `SONIC_PORT` to use an alternative port.

## Dataset and Colab preparation

The source recordings belong in `audio_dataset/raw/<class>/`. `scripts/01_prepare_dataset.py` creates a deterministic stratified 70/15/15 original split under `audio_dataset/processed`, writes metadata under `dataset`, and creates training-only shifted, noise-added, and volume-adjusted copies. No segment or augmentation crosses its original recording's split.

Run the Colab-ready notebook at `notebooks/colab_training.ipynb` after placing the updated repository in Google Drive and setting `REPO_DIR` in the notebook. It prepares the dataset, extracts features, compares random forest, Extra Trees, gradient boosting and scaled SVM candidates, and creates GTM training folders. Each step must succeed before the next runs.

Silent or unreadable sources and silent segments are excluded before augmentation and logged in `config/excluded_clips.csv`. Extraction also skips unusable files and logs them in `data/excluded_features.csv`. It invalidates the previous feature archive before starting, so a failed extraction cannot silently train on old data. Training checks class coverage and source-recording separation across all three splits.

The feature extractor uses fixed-window mel spectra, MFCCs and temporal derivatives with spectral statistics. SVM scaling is fitted only on training data; candidate selection uses validation macro-F1, then only the winner is evaluated on the test split. The app reads that winner from `python_models/selection.json`. Accuracy improvements must be measured on your recordings and are not guaranteed. **Existing models require fresh feature extraction and retraining** because the feature format has changed; the app reports old artifacts as unavailable until retrained.

Use exactly these class folder names:

`machinery_fault`, `glass_breaking`, `alarm_siren`, `vehicle_horn`, `animal_sound`, `gunshot`, `panic_scream`, `aggression`, `person_asking_for_help`, `background_noise`.

The supplied TensorFlow.js GTM export and its metadata are kept in `gtm_model/`. Run `python scripts/convert_gtm_tfjs.py` to rebuild the Python Keras artifact and ordered `labels.json` from the export. The app uses the metadata's 16 kHz, one-second log-mel windows, averages window probabilities for longer clips and sends the audio to GTM independently. It never sends the Python prediction to GTM. See [GTM integration notes](documentation/GTM_MODEL.md) for the input contract and validation limits.

## Audio CNN transfer learning

`notebooks/colab_transfer_training.ipynb` runs the stronger transfer-learning experiment. It uses Google's pretrained YAMNet CNN to produce audio embeddings and compares logistic-regression and SVM classifiers trained on your ten classes. The CNN weights remain frozen. This requires TensorFlow (already in `requirements.txt`) and Python 3.12 or newer for the model archive extraction.

To run locally with recordings in `audio_dataset/raw/<class>/`:

```powershell
python python_models/train_transfer.py --extract --activate
```

This preserves the classical model files, caches per-segment features under `data/transfer_experiment`, and saves `python_models/yamnet_transfer.joblib` and its metrics. `--activate` backs up the previous selection in `selection_before_transfer.json` before selecting the CNN-based classifier for the app. Omit `--activate` to evaluate without changing the active model. Run without `--extract` to reuse the finished feature archive; interrupting extraction is safe because completed segments are cached.

Source hashes are kept separate across the deterministic train/validation/test splits. Only original segments are used in this experiment. The classifier is chosen by validation macro-F1 and evaluated on the held-out test split using the same probability-argmax rule as the app. Reports include sample counts and dependency versions. A previously inspected test set is a benchmark, not a fresh independent assessment of real-world accuracy.

For deployment, keep `python_models/yamnet_base/` together with the classifier and selection file. This base model is downloaded once during training; the app never downloads it automatically. Install the scikit-learn version recorded in the metrics when moving a joblib model between environments, or retrain there. To restore the prior model, copy `selection_before_transfer.json` over `selection.json`.

## Saved splits and augmented YAMNet training

Run `python python_models/train_augmented.py` to save original train/validation/test copies and training-only augmented WAVs under `audio_dataset/yamnet_augmented/`, extract frozen YAMNet features, and compare classifiers. This creates a separate `yamnet_augmented.joblib` without overwriting the existing model. Use `--reuse-audio` to resume from saved audio and cached features.

See [the augmentation guide](documentation/AUGMENTED_TRAINING.md) for exact folder paths, the augmentation recipe, leakage checks and activation instructions. The Colab version is `notebooks/colab_augmented_training.ipynb`. Dataset files are local and Git-ignored, so they must be distributed separately.

## Tests

```powershell
python -m pytest
python scripts/verify_delivery.py
```

The delivery check uses ten existing test recordings to exercise both models through the Flask upload flow, then records warm upload/live timings and a dashboard query against 20,000 synthetic rows. It is an integration smoke check, not the SRS's required evaluation on 100 unseen recordings.

This is a controlled-test prototype and is not an emergency-response system.
