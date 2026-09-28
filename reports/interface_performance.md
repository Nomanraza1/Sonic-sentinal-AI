# Interface and performance update

Verified locally on 28 September 2026.

The interface now uses a responsive light theme with sidebar navigation, dashboard summaries, audio upload feedback, confidence bars and accessible forms. No external font, chart library or React build is required.

## Findings and changes

- The existing process on port 5000 timed out after 15 seconds for the home page, login page and stylesheet. Its precise blocking cause could not be established. A fresh application process responds normally; the updated local instance is running on port 5001.
- Model warm-up now loads both classifiers before the server accepts requests. This avoids charging model loading to the first upload. Startup takes longer. Matplotlib imports only when charts are needed.
- Waveform and spectrogram requests previously decoded the recording and regenerated both charts on every request. Charts now share a cache with source-modification checks and a lock protecting rendering. Authorization is checked before serving them.
- Review pages now contain at most 25 recordings and audio uses `preload="none"`. This avoids loading an entire pending queue and its audio into one page.
- Removed conflicting legacy styling and the full-page reveal animation. Audio visuals load lazily and uploads display processing feedback.

## Verification

All 33 automated tests passed, including chart reuse/invalidation, bounded review pagination and existing authentication, upload, review and access-control tests. JavaScript syntax and Git whitespace checks passed. Browser checks covered desktop home/dashboard, sign-in, phone-sized uploads and mobile navigation.

Local HTTP timings against the isolated verification database with 20,000 synthetic records:

| Request | Time |
|---|---:|
| Overview | 18.1 ms |
| Dashboard | 55.7 ms |
| Event history | 18.4 ms |
| Review queue | 29.3 ms |
| Model metrics | 26.4 ms |
| Upload page | 27.9 ms |
| Detection details | 39.9 ms |
| Chart generation request | 507.9 ms |
| Cached waveform request | 10.9 ms |
| Cached spectrogram request | 8.9 ms |

These are local response measurements, not a concurrent-user benchmark or a guarantee for other machines. The real workspace server on port 5001 also returned HTTP 200 for home, sign-in, stylesheet and JavaScript, with measured responses under 100 ms.

Restart with `python app.py` after stopping the old server. If port 5000 remains occupied, set `$env:SONIC_PORT='5001'` first. The GTM export is now integrated. Its input parity and required evaluation on 100 unseen recordings remain open.
