# Local installation and evaluator guide

The saved model was trained with Python 3.13.9, TensorFlow 2.20.0, scikit-learn 1.7.2, NumPy 2.3.5, and joblib 1.5.2. Windows is the locally verified environment. Other systems require their own installation and audio tests.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-trained-model.txt
.\.venv\Scripts\python.exe -m flask --app app create-user --role admin
.\.venv\Scripts\python.exe app.py
```

The create-user command prompts for username, email, and password. Create separate reviewer/operator/maintenance accounts by changing `--role`. Public registration creates ordinary users. Share evaluator credentials privately; do not commit real passwords.

Open http://127.0.0.1:5000 after model loading finishes. Startup performs a warm-up; a cold inference can exceed the eight-second upload target. Starting through a different WSGI server requires equivalent warm-up in each worker.

## Required files and storage

Keep `python_models/yamnet_transfer.joblib`, `python_models/selection.json`, and the entire `python_models/yamnet_base/` directory together. The base model is intentionally excluded from Git. The local `deliverables/model_snapshot_20260927/trained_model.zip` contains these artifacts; extract it at the repository root and verify against the companion `sha256.json`. A Git clone alone is insufficient to run the selected model. If that archive is unavailable, prepare the source dataset and use the documented transfer-training pipeline.

The database schema initializes automatically on app import. Default writable locations are `data/sonic_sentinel.db`, `uploads/`, and `instance/`. Optional environment variables are `SONIC_DATABASE_PATH`, `SONIC_UPLOAD_DIR`, `SONIC_RUNTIME_DIR`, and `SONIC_RULES_PATH`. The default rule file is `alert_rules/default.json`. Keep runtime storage private and persistent.

`imageio-ffmpeg` supplies the decoding executable used for formats needing conversion, including M4A. If decoding fails, verify that package is installed and the host can execute its bundled FFmpeg. Supported upload extensions are WAV, MP3, FLAC, OGG, and M4A; the upload size limit is 25 MiB.

## Evaluator walkthrough

1. Register or log in, upload permitted sample audio, and open its analysis.
2. Inspect metadata, playback, waveform, spectrogram, Python probabilities, quality, severity, and review status. Missing GTM results correctly require review.
3. Open microphone monitoring, allow browser permission, start monitoring, then stop it. Use localhost or HTTPS. Physical microphone/device behavior still needs manual verification.
4. Use history filters and dashboard summaries. Open a detection to download its report.
5. With a reviewer account, submit a review/override. Authorized roles can acknowledge alerts. An administrator can export CSV and update settings.
6. Retention cleanup deletes old data only after explicit confirmation; test it on disposable records.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_delivery.py
```

The second command requires the original local audio paths in `data/transfer_experiment/manifest.json`, the selected model and CNN base. It creates isolated runtime storage and writes `reports/delivery_e2e.json`. It currently verifies Python-only behavior while GTM is absent; it is not the dual-model acceptance suite. Its cold first prediction is reported separately from warm timing.

## GTM handoff

The previous session confirmed that GTM was still training. Preserve its original export, labels, metadata, project URL, sample counts, and training evidence. The current adapter expects `gtm_model/model.keras` and `labels.json`, but assumes a three-second raw waveform input. The actual export's preprocessing and input signature must be checked and the adapter matched before claiming support. Renaming a TensorFlow.js export is insufficient. After integration, validate both models on at least 100 unseen source recordings, with ten per class, and record every required comparison field.

See `reports/delivery_readiness.md` for the remaining acceptance gates.
