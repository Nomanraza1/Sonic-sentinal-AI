# SonicSentinel AI

Flask prototype for uploaded and live acoustic-event monitoring. It classifies the ten SRS categories with independently loaded Python and Google Teachable Machine models, compares their scores, applies configurable safety rules, and records events in SQLite.

## Setup

Use Python 3.11 or newer, then run:

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. Register an account first. Use Upload for WAV, MP3, FLAC, OGG, or M4A recordings (25 MB maximum, 0.2-30 seconds). Live microphone access always requires browser permission and is visibly indicated.

## Models

Set `ACTIVE_PYTHON_MODEL` in `config/settings.py` to choose the saved scikit-learn model. Put its `.joblib` artifact and metrics JSON in `python_models/`. Put the converted GTM Keras artifact at `gtm_model/model.keras` and its ordered labels in `gtm_model/labels.json`; see `state.md` for the exact contract.

To create the local Python artifacts from the supplied audio, run these commands in order:

```powershell
python scripts/01_prepare_dataset.py
python scripts/02_extract_features.py
python python_models/train.py
python scripts/03_prepare_gtm_data.py
```

The first command makes the stratified split. The second extracts shared features. The third trains and compares random forest, gradient boosting, and SVM, selects by validation macro F1, and evaluates the selected model once on test data. The final command creates the GTM training folders from training clips only. Export the GTM model through Teachable Machine as described in `state.md`; the application will load it when `model.keras` and `labels.json` are placed in `gtm_model/`.

## Testing

```powershell
python -m pytest
```

## Structure

`app.py` contains Flask routes. Processing lives in `audio_preprocessing/` and `feature_extraction/`; `src/` contains independent model and decision adapters; `database/` contains SQLite setup; `alert_rules/` contains configurable rules; `tests/` holds checks. `audio_dataset/` is preserved from the supplied workspace.

## Notes

This is a controlled-test prototype, not a certified emergency-response system. Do not rely on it as the sole basis for an emergency decision.

Current measured result: Gradient Boosting was selected using validation macro F1 (0.719); its untouched test accuracy is 0.672. This is below the SRS target. The available dataset has nine real classes; add `person_asking_for_help` clips and retrain before claiming ten-class coverage.
# verified
