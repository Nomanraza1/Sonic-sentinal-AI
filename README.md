# SonicSentinel AI

SonicSentinel AI is a Flask application for uploaded-audio analysis and visible, permission-based live microphone monitoring. It records independent Python-model and GTM-model predictions, compares their confidence, evaluates audio quality, applies alert rules, and routes uncertain events to review.

## Run locally

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, then register an account.

## Dataset and Colab preparation

The source recordings belong in `audio_dataset/raw/<class>/`. `scripts/01_prepare_dataset.py` creates a deterministic stratified 70/15/15 original split under `audio_dataset/processed`, writes metadata under `dataset`, and creates training-only shifted, noise-added, and volume-adjusted copies. No segment or augmentation crosses its original recording's split.

Run the Colab-ready notebook at `notebooks/colab_training.ipynb` after placing the updated repository in Google Drive and setting `REPO_DIR` in the notebook. It prepares the dataset, extracts features, compares random forest, Extra Trees, gradient boosting and scaled SVM candidates, and creates GTM training folders. Each step must succeed before the next runs.

Silent or unreadable sources and silent segments are excluded before augmentation and logged in `config/excluded_clips.csv`. Extraction also skips unusable files and logs them in `data/excluded_features.csv`. It invalidates the previous feature archive before starting, so a failed extraction cannot silently train on old data. Training checks class coverage and source-recording separation across all three splits.

The feature extractor uses fixed-window mel spectra, MFCCs and temporal derivatives with spectral statistics. SVM scaling is fitted only on training data; candidate selection uses validation macro-F1, then only the winner is evaluated on the test split. The app reads that winner from `python_models/selection.json`. Accuracy improvements must be measured on your recordings and are not guaranteed. **Existing models require fresh feature extraction and retraining** because the feature format has changed; the app reports old artifacts as unavailable until retrained.

Use exactly these class folder names:

`machinery_fault`, `glass_breaking`, `alarm_siren`, `vehicle_horn`, `animal_sound`, `gunshot`, `panic_scream`, `aggression`, `person_asking_for_help`, `background_noise`.

After independently training the GTM model, place the converted Keras artifact at `gtm_model/model.keras` and an ordered JSON array of the same ten labels at `gtm_model/labels.json`. The app never sends the Python prediction to GTM.

## Audio CNN transfer learning

`notebooks/colab_transfer_training.ipynb` runs the stronger transfer-learning experiment. It uses Google's pretrained YAMNet CNN to produce audio embeddings and compares logistic-regression and SVM classifiers trained on your ten classes. The CNN weights remain frozen. This requires TensorFlow (already in `requirements.txt`) and Python 3.12 or newer for the model archive extraction.

To run locally with recordings in `audio_dataset/raw/<class>/`:

```powershell
python python_models/train_transfer.py --extract --activate
```

This preserves the classical model files, caches per-segment features under `data/transfer_experiment`, and saves `python_models/yamnet_transfer.joblib` and its metrics. `--activate` backs up the previous selection in `selection_before_transfer.json` before selecting the CNN-based classifier for the app. Omit `--activate` to evaluate without changing the active model. Run without `--extract` to reuse the finished feature archive; interrupting extraction is safe because completed segments are cached.

Source hashes are kept separate across the deterministic train/validation/test splits. Only original segments are used in this experiment. The classifier is chosen by validation macro-F1 and evaluated on the held-out test split using the same probability-argmax rule as the app. Reports include sample counts and dependency versions. A previously inspected test set is a benchmark, not a fresh independent assessment of real-world accuracy.

For deployment, keep `python_models/yamnet_base/` together with the classifier and selection file. This base model is downloaded once during training; the app never downloads it automatically. Install the scikit-learn version recorded in the metrics when moving a joblib model between environments, or retrain there. To restore the prior model, copy `selection_before_transfer.json` over `selection.json`.

## Tests

```powershell
python -m pytest
```

This is a controlled-test prototype and is not an emergency-response system.
