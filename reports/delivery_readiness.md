# Delivery readiness review - 2026-09-28

Status: The GTM export is integrated and both models pass the application smoke checks. Full SRS delivery still needs a larger unseen comparison and submission evidence. This is not a blanket SRS-compliance certificate.

The previous chat, “Fix silent audio and improve model”, ended during delivery verification. Its code and models had subsequently reached remote `main` in commit `0d206e1`, including the team's co-author trailers. This review completes the interrupted test handoff and records outstanding work explicitly.

## Verified evidence

| Area | Evidence and result |
|---|---|
| Regression suite | `python -m pytest -q`: 37 passed on 2026-09-28, including real GTM preprocessing and inference |
| Saved model | Snapshot ZIP and current model files match every entry in `deliverables/model_snapshot_20260927/sha256.json` |
| Real web inference | `scripts/verify_delivery.py` passed upload, analysis and playback checks for one existing test recording per class. Both model scores were stored for all ten records. Python matched 9/10 and GTM matched 8/10 in these selected examples. |
| Model warm-up | Both models loaded successfully in 26.950 seconds before the test requests. The Flask entry point now warms both models before accepting requests. |
| Warm 30-second upload | 2.814 seconds, below the SRS eight-second target on this host |
| Warm three-second live request | 0.282 seconds, below the three-second target; this tests the endpoint, not microphone hardware |
| Database scale | Dashboard loaded with 20,000 synthetic events in 0.052 seconds; this does not establish concurrent-user or uptime compliance |
| Python benchmark | 86.62% test accuracy, 0.8596 macro-F1; five specified critical-class recalls exceed 85%; see `cnn_training_results.md` |
| Security/workflows | Automated coverage includes account roles, cross-user access, CSRF, upload validation, review, reporting, and alert logic; mocked dual-model responses are logic tests only |

Raw current results are in `delivery_e2e.json`. Benchmark figures are segment-level results on an already inspected project split, not unseen field accuracy.

## SRS acceptance gaps and next actions

| SRS area | Status / remaining action |
|---|---|
| Independent GTM inference | Integrated from the supplied TensorFlow.js export. Its metadata guided the input shape and mel features. The original preprocessing source and direct parity evidence were not supplied, so exact numerical parity with Teachable Machine remains unverified. |
| Dual-model accuracy and comparison | The smoke test saved separate labels, scores, confidence, agreement and review decisions for ten selected examples. Evaluate accuracy, macro-F1 and critical recall on at least 100 genuinely unseen recordings, with ten per class, before claiming SRS performance. |
| Dataset minimum | Existing experiment contains 2,999 unique source recordings. Obtain at least one additional legitimately sourced original and check per-class requirements; do not count augmentations as originals. Update provenance and splits before retraining. |
| Noise robustness | Weak: small diagnostic noise probe got 4/10 correct at 20 dB and 1/10 at 10 dB. Use training/validation noise augmentation and investigate domain mismatch, then assess a new untouched set. Do not tune on these probe answers. |
| Live microphone and browser behavior | Endpoint tested. Real permission denial, device loss, sustained capture, mobile layout, audio playback and physical critical-event demonstrations remain unverified. |
| Similar/overlapping events | Full unseen evaluation of confusable sounds, overlapping events, distance, echo and re-encoding remains outstanding. |
| Availability and concurrency | Single-process 20,000-row check passed; multi-user load and 99% uptime are not established. |
| Project report | Full required report and database dictionary still need delivery. System and GTM flow diagrams are now saved in `documentation/assets/`. This readiness report is not the project report. |
| Submission media | The technical blog has been updated with GTM integration and current limits. Demonstration MP4, screenshots, and completed GTM training evidence are still absent. |
| Team/integrity evidence | AI declaration added; team must fill in genuine review and contribution details. Existing co-authors must not be treated as proof of five-day work or technical understanding. |
| Deployment | No public application verified. Local execution guide provided as permitted fallback; supply evaluator access and permitted sample audio. |
| Dataset/model distribution | Raw audio and CNN base are Git-ignored. Supply permitted dataset and model archive separately, with provenance and checksums. |

## Handoff order

1. Provide the GTM project evidence, original preprocessing source and training sample counts.
2. Run the required dual-model comparison on 100 or more unseen recordings and review alert behavior.
3. Close dataset/provenance and noise-validation gaps without test leakage.
4. Perform physical browser/microphone and multi-user tests; record results and limitations.
5. Complete the full report, database dictionary, video, screenshots and student verification before submission.

Installation, admin provisioning, operating steps and troubleshooting are in `documentation/INSTALLATION.md`. GTM integration is present, but deliver the system as a tested prototype until the remaining evaluation and submission gates are closed.
