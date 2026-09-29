# SonicSentinel AI Demonstration Video Script

**Audience:** project judges and evaluators

**Target length:** 6–8 minutes

**Format:** screen recording with spoken narration

## Before recording

- Start the Flask application and wait for both model warm-ups to finish.
- Use a prepared evaluator account and permitted sample recording. Do not show passwords, personal recordings, or private account details.
- Keep one reviewer account available for the review-queue section and an administrator account available for the settings section.
- Confirm that the sample has a saved detection and that its audio, waveform, and spectrogram load.
- Test the microphone beforehand. If permission or hardware fails during recording, state that live capture needs a physical-device check and continue with the upload workflow; do not imply the microphone path was demonstrated.
- Use a disposable demonstration record for any review or alert action that changes stored data.

## Recording script

### 0:00–0:35 | Introduce the system

**On screen:** Open SonicSentinel AI in the browser and show the signed-in home page.

**Narration:** “This is SonicSentinel AI, a web prototype for reviewing potentially important sounds. It accepts uploaded audio and permission-based microphone input. It runs two independent classifiers, checks audio quality, applies configurable rules, and gives a person a way to review uncertain events. It is a decision-support prototype, not an emergency-response service.”

### 0:35–1:05 | Dashboard and navigation

**On screen:** Open the dashboard. Point out the summary cards, class counts, recent events, and any pending reviews or disagreements.

**Narration:** “The dashboard summarizes the detections visible to this account. Counts and categories come from stored application records. A disagreement or low-quality result can be useful to inspect, but these dashboard totals are not accuracy or false-alarm rates unless the events have independently verified labels.”

### 1:05–1:45 | Upload an audio sample

**On screen:** Open **Upload**, select a permitted WAV sample, and submit it. Show the processing status and the resulting detection link.

**Narration:** “For an upload, the app validates the file before inference. Supported formats include WAV, MP3, FLAC, OGG, and M4A. The current limit is 25 MiB per file, with a duration from 0.2 to 30 seconds. The audio is decoded, checked for usable signal, converted to mono, trimmed and segmented before it reaches the models.”

### 1:45–2:55 | Inspect one detection

**On screen:** Open the detection. Play the recording, then point out its metadata, waveform, spectrogram, Python result, GTM result, confidence values, agreement, quality, severity, and review status.

**Narration:** “This page keeps the two model outputs separate so we can see what each model predicted. The Python branch uses frozen YAMNet features and a trained SVM. The GTM branch uses its own log-mel input and convolutional model. The comparison and alert rules consider confidence, the top-two margin, audio quality, repeated detections, and model agreement for critical alerts. The waveform and spectrogram help a reviewer inspect the recording; they do not establish what caused a sound.”

“A model confidence score is an estimate from that model, not proof that an event occurred. Disagreement or poor audio can send the event for human review.”

### 2:55–3:35 | History and report

**On screen:** Open **History**, demonstrate a category or status filter, return to the detection, and download its analysis report.

**Narration:** “History can be filtered by fields such as filename, category, severity, quality, status, confidence, and date. The detection report is an HTML file with the recording details, model scores, and visualizations. The application does not currently produce a native PDF report.”

### 3:35–4:20 | Human review and alert action

**On screen:** Sign in with the prepared reviewer account, open **Review**, select a disposable pending event, choose the reviewer’s label, and add a short explanation. Show the saved result and, if appropriate for the demo data, acknowledge the alert.

**Narration:** “A reviewer can listen to an uncertain event, confirm or correct its label, and record a comment. The original model scores remain stored alongside the review decision, and the action is added to the audit trail. Authorized operational roles can also acknowledge, dismiss, escalate, or close an alert. These actions record a workflow decision; they do not send an emergency response outside this application.”

### 4:20–5:10 | Live microphone monitoring

**On screen:** Open **Microphone**, explain the permission indicator, start monitoring, show a processed window and its status, then stop monitoring and show that capture has ended.

**Narration:** “Live monitoring starts only after the user grants microphone permission and presses Start. The browser sends short WAV windows to the application about every three seconds. The interface displays the result and status, and the Stop control releases the microphone. This demonstration checks this computer and browser only; microphone behavior on other devices and long sessions needs separate testing.”

**If the physical microphone is unavailable:** Skip this segment. Say: “The live endpoint is implemented, but this recording could not verify physical microphone capture. I’ll show the upload workflow and stored analysis instead.” Do not substitute an upload and describe it as a live microphone test.

### 5:10–5:45 | Administrator settings

**On screen:** Sign in with the prepared administrator account and open **Settings**. Show confidence and margin thresholds, quality requirements, repeat-window count, and model-agreement option. Do not save a change during the video.

**Narration:** “Administrators can configure thresholds, required quality, repeat confirmation, retention, and category severity or critical status. The current defaults include a 65% minimum Python confidence, a 0.12 top-two margin, two confirming windows, and model agreement for critical alerts. Changes should be assessed against labeled validation examples before operational use.”

### 5:45–6:40 | Explain the evidence and limits

**On screen:** Show the saved model results or project report summary, then finish on the dashboard or project title.

**Narration:** “On the saved Python benchmark, the active model reached 86.62% accuracy and 0.8596 macro-F1 across 471 test segments from 450 held-out source recordings. Those segments are not 471 independent recordings, and this project split has already been inspected during development. The ten-recording application smoke check, with one selected sample per class, confirmed integration; Python matched nine labels and GTM matched eight. It is too small to establish unseen dual-model accuracy. The required larger evaluation, exact GTM preprocessing parity, broader noise testing, and physical-device checks remain open.”

“SonicSentinel AI demonstrates an end-to-end workflow for audio intake, independent model predictions, rule-based triage, and human review. Its results should support investigation, not replace a trained operator or local safety procedure.”

## Suggested closing card

**SonicSentinel AI**

Audio classification and human review prototype

Evaluation limits and remaining acceptance work are documented in the project report.
