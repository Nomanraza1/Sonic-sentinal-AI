# SonicSentinel AI — separate HTML pages

Open `index.html` for Home. The other eight pages are `upload.html`, `player.html`, `details.html`, `waveform.html`, `microphone.html`, `about.html`, `login.html` and `register.html`. All pages share `style.css` and `app.js`.

For best results, run a local server in this folder: `python -m http.server 8000`, then open `http://localhost:8000/`. Audio selection uses browser IndexedDB so a selected file stays available when navigating between pages. The browser may restrict large files or unsupported audio formats.

Login/Register are a frontend demonstration. Registration data is held in browser session storage; there is no backend, database or protected access. Never enter a real password. Add a server-side authentication system for real accounts.
