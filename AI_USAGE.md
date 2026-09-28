# AI tool usage declaration

OpenAI Codex assisted with audio preprocessing, feature extraction, model training scripts, Flask application fixes, automated tests, and delivery documentation during September 2026.

Assistance requested included fixing silent-recording failures, comparing model accuracy, preserving trained models, reviewing the SRS, testing application flows, and completing interrupted delivery checks.

Affected areas include `audio_preprocessing/`, `feature_extraction/`, `python_models/`, `scripts/`, `notebooks/`, `app.py`, `src/`, `templates/`, `tests/`, `README.md`, and `reports/`. Git history records the actual changes.

Verification on 2026-09-28: 37 automated tests passed. Both saved models ran through Flask upload routes for one existing test recording in each class. Warm upload/live timing and a 20,000-record dashboard check completed. The model archive SHA-256 checksums matched the current Python model files. GTM's 10-recording smoke result is not evidence of unseen accuracy. Physical microphone capture and full SRS acceptance remain pending.

Final classification uses local trained model inference, not a generative-AI API. Automated rule tests use synthetic predictions to isolate application logic. The delivery smoke check uses real audio and model inference, but its ten selected examples are not an independent accuracy study.

Student modifications and verifying team members: the team must record its own review, modifications, module ownership, and verification here before submission. Co-author trailers do not establish that every member has reviewed or understood every change. No student verification is asserted by this declaration.
