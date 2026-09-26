# SonicSentinel AI state

## Done
- SQLite schema and initialization are in `database/db.py`: users, audio_files, detections, audit_log; foreign keys, WAL, ISO-8601 UTC timestamps and FK indexes are enabled.
- Shared preprocessing and feature extraction are in `audio_preprocessing/audio.py` and `feature_extraction/features.py`.
- Three-model training script is in `python_models/train.py`; application selection is the one-line `ACTIVE_PYTHON_MODEL` value in `config/settings.py`.
- Independent Python and GTM loaders, configurable JSON rules, upload classification, authentication, role checks, history, review queue and CSV export are implemented.
- Uploads now process every fixed-duration segment. Live microphone windows are captured as WAV in-browser and submitted to the same backend pipeline. Detection views generate waveform and Mel spectrogram images and provide downloadable HTML analysis reports.
- Dashboard and saved-metrics views, alert acknowledgement/dismissal/escalation, profile updates, reviewer overrides, administrator threshold/retention settings, retention cleanup and signal-fingerprint near-duplicate review routing are implemented.
- Created a real dataset manifest and stratified split: 2,700 original clips, 1,890 train, 405 validation, 405 test. One duplicate source clip is recorded in `config/excluded_clips.csv`. Extracted shared features, trained random forest, gradient boosting, and SVM, and selected gradient boosting by validation macro F1.

## In progress
- Improving data diversity and the missing help-request class before attempting another model cycle.

## Next
1. Expand test coverage for database, upload, audio-quality, duplicate, comparison, alert and role flows.
2. Validate against the missing dataset metadata and real model files when provided.

## Decisions
- The supplied repository contained `audio_dataset/` but no `dataset/`, `scripts/`, `gtm_upload/`, `MORNING_REPORT.md`, or `SRS_DATASET_AUDIT.md`. On 2026-09-26 the user explicitly authorized a stratified split, so `scripts/01_prepare_dataset.py` generated it. No augmentation process was run.
- `audio_dataset/` remains the existing source folder. The required SRS folders were added around it.
- `audio_files.uploaded_by` is nullable for dataset imports; user uploads supply the authenticated user.
- Admin-selected roles are accepted for local demonstration only. Production deployment should restrict role assignment to administrators.
- Validation-only tuning evaluated scaled SVM, ExtraTrees, k-NN and a full log-mel representation. None improved on Gradient Boosting validation macro F1 (0.719), so the selected model and compact shared feature set were retained. No score was fabricated and no test-set-based tuning was done.

## Model contracts
- Python: `python_models/<ACTIVE_PYTHON_MODEL>.joblib`, a scikit-learn classifier exposing `classes_` and `predict_proba`; feature vector is the MFCC, mel-band, chroma, ZCR, RMS, centroid, bandwidth and roll-off summary vector from `feature_extraction/features.py`, generated from 22,050 Hz mono 3-second clips. Metrics are `python_models/<name>_metrics.json`.
- GTM: `gtm_model/model.keras` and `gtm_model/labels.json`. The loader expects a TensorFlow/Keras audio model accepting `(batch, 66150, 1)` float mono samples at 22,050 Hz. `labels.json` is an ordered JSON list exactly matching model output positions and the ten `SONIC_CLASSES`. Missing files are a recorded unavailable GTM result, never a crash.

## Blocked / needs me
- `person_asking_for_help` has no audio clips. The trained Python models cover the nine available real classes; add voluntary, licensed help-request clips then rerun preparation, extraction and training for ten-class coverage.
- The GTM export cannot be generated locally without completing training/export in Google Teachable Machine. `gtm_model/training_audio/` now contains only the 1,890 training clips, arranged by class, ready for that export.

## SRS coverage
- [x] Core persistence, authentication, roles, upload validation, preprocessing, feature extraction, model adapters, comparison, live server-side windows, waveform/spectrogram, HTML analysis reports, history, manual review, audit and CSV export.
- [x] Alert acknowledgement/dismissal/escalation, retention/settings, near-duplicate comparison, dashboard, metrics view and profile management.
- [ ] Comprehensive test suite and validation using actual datasets/models.
- [x] Dataset split, common Python/GTM training dataset layout, feature extraction, three trained Python models, metrics and active-model selection.
- [ ] Tenth help-request class and exported GTM `model.keras`/`labels.json` are still required for full ten-class dual-model coverage.

## Work log
- 2026-09-25: User requested the complete SRS application. Read the 45-page SRS and build brief; created SRS project modules, schema, processing/model/rule baseline, Flask routes, templates, tests and requirements. Verified route smoke tests, compilation and one rule-engine test. Did not alter source audio or run any split/augmentation process.
- 2026-09-26: Continued toward SRS completion. Added server-side multi-segment processing, consent-based live microphone WAV windows, visual audio outputs and downloadable reports. Kept model fallback behavior intact; pytest passes.
- 2026-09-26: Added alert lifecycle actions, dashboard/metrics pages, profile management, explicit reviewer override, JSON-backed administrator settings, retention cleanup and perceptual-fingerprint duplicate review routing. Verified all new page routes and settings submission; pytest passes.
- 2026-09-26: User authorized training. Repaired a merge-corrupted `app.py` that prevented startup, prepared the dataset and GTM training folders, extracted features, trained three Python models and selected gradient boosting. Measured validation macro F1 is 0.719 and final test accuracy is 0.672; this is below the SRS target and is recorded honestly. Browser home-page validation passed.
- 2026-09-26: User requested 85-90% accuracy and human-readable code. Ran validation-only alternatives and retained the best honest result; formatted Python modules with Black and restored the compact shared feature cache after the log-mel experiment underperformed. Browser startup and pytest validation passed.
# verified
