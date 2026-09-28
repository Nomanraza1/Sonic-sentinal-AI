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

Open http://127.0.0.1:5000 when the server starts. The app loads both audio models before accepting analysis requests, so the first upload does not carry the model-loading delay. Startup can take longer while the models warm up. The upload page displays progress while processing. Set `SONIC_PORT` to another local port if 5000 is already occupied. For example, `$env:SONIC_PORT="5001"` before running `python app.py`.

## Required files and storage

Keep `python_models/yamnet_transfer.joblib`, `python_models/selection.json`, and the entire `python_models/yamnet_base/` directory together. The base model is intentionally excluded from Git. The local `deliverables/model_snapshot_20260927/trained_model.zip` contains these artifacts; extract it at the repository root and verify against the companion `sha256.json`. The GTM source export, metadata and converted `model.keras` live under `gtm_model/`. A Git clone must include both model sets to run both classifiers. If the YAMNet archive is unavailable, prepare the source dataset and use the documented transfer-training pipeline.

The database schema initializes automatically on app import. Default writable locations are `data/sonic_sentinel.db`, `uploads/`, and `instance/`. Optional environment variables are `SONIC_DATABASE_PATH`, `SONIC_UPLOAD_DIR`, `SONIC_RUNTIME_DIR`, and `SONIC_RULES_PATH`. The default rule file is `alert_rules/default.json`. Keep runtime storage private and persistent.

`imageio-ffmpeg` supplies the decoding executable used for formats needing conversion, including M4A. If decoding fails, verify that package is installed and the host can execute its bundled FFmpeg. Supported upload extensions are WAV, MP3, FLAC, OGG, and M4A; the upload size limit is 25 MiB.

## Evaluator walkthrough

1. Register or log in, upload permitted sample audio, and open its analysis.
2. Inspect metadata, playback, waveform, spectrogram, Python and GTM probabilities, quality, severity, and review status. Model disagreement or weak audio quality can send an event for review.
3. Open microphone monitoring, allow browser permission, start monitoring, then stop it. Use localhost or HTTPS. Physical microphone/device behavior still needs manual verification.
4. Use history filters and dashboard summaries. Open a detection to download its report.
5. With a reviewer account, submit a review/override. Authorized roles can acknowledge alerts. An administrator can export CSV and update settings.
6. Retention cleanup deletes old data only after explicit confirmation; test it on disposable records.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_delivery.py
.\.venv\Scripts\python.exe scripts/convert_gtm_tfjs.py
```

The delivery check requires the original local audio paths in `data/transfer_experiment/manifest.json`, the selected Python model, its YAMNet base and the GTM export. It warms both models, creates isolated runtime storage, and writes `reports/delivery_e2e.json`. It checks one held-out example per class through real Flask uploads, model-score storage, analysis and playback routes, then records warm upload/live timing and a 20,000-row dashboard query. These examples are a smoke test, not the required evaluation on 100 unseen recordings.

## GTM handoff

The checked-in GTM export is a TensorFlow.js layers model. `scripts/convert_gtm_tfjs.py` maps the exported weights into `gtm_model/model.keras`; `gtm_model/metadata.json` supplies the class order and feature dimensions. The app resamples mono audio to 16 kHz, builds one-second log-mel inputs of shape 31 by 64 by 1, predicts each window independently and averages the scores. The `person_asking_help` export label is mapped to the app's `person_asking_for_help` label. See [GTM model notes](GTM_MODEL.md) for details and the known limits of the export evidence.

After conversion, run the full checks above. Then validate both models on at least 100 genuinely unseen source recordings, with ten per class, and record every required comparison field. The ten-recording smoke test does not satisfy this gate.

See `reports/delivery_readiness.md` for the remaining acceptance gates.
