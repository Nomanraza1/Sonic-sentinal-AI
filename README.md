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

Run the Colab-ready notebook at `notebooks/colab_training.ipynb` after placing the repository in Google Drive. It prepares the dataset, extracts features, trains and compares the three Python models, and creates GTM training folders. The notebook is not executed by this repository.

Use exactly these class folder names:

`machinery_fault`, `glass_breaking`, `alarm_siren`, `vehicle_horn`, `animal_sound`, `gunshot`, `panic_scream`, `aggression`, `person_asking_for_help`, `background_noise`.

After independently training the GTM model, place the converted Keras artifact at `gtm_model/model.keras` and an ordered JSON array of the same ten labels at `gtm_model/labels.json`. The app never sends the Python prediction to GTM.

## Tests

```powershell
python -m pytest
```

This is a controlled-test prototype and is not an emergency-response system.
