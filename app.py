import csv
import hashlib
import json
import io
import os
import secrets
import sqlite3
import uuid
import click
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from threading import Lock
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from audio_preprocessing.audio import (
    fingerprint,
    fingerprint_distance,
    load_audio,
    quality,
    segments,
    validate_upload,
    prepare_upload,
)
from config.class_map import DISPLAY_NAMES, SONIC_CLASSES
from config.settings import ROOT, UPLOAD_DIR, RUNTIME_DIR, RULES_PATH, MAX_UPLOAD_BYTES
from database.db import audit, connect, init_db, now
from src.models import predict_gtm, predict_python
from src.reports import write_report
from src.rules import decide

def make_images(*args, **kwargs):
    # Matplotlib is only needed for audio visuals, not navigation or sign-in.
    from src.visuals import make_images as render_images
    return render_images(*args, **kwargs)

app = Flask(__name__)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
secret_path = RUNTIME_DIR / ".secret_key"
if not os.environ.get("SONIC_SECRET_KEY") and not secret_path.exists():
    secret_path.write_text(secrets.token_hex(32), encoding="ascii")
app.config.update(SECRET_KEY=os.environ.get("SONIC_SECRET_KEY") or secret_path.read_text().strip(),
                  MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES * 10 + 1024 * 1024,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("SONIC_HTTPS") == "1")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
init_db()


@app.context_processor
def csrf_context():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return {"csrf_token": session["csrf_token"]}


@app.before_request
def protect_mutations():
    if request.method == "POST":
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(token, session.get("csrf_token", "missing")):
            abort(400, description="Form expired. Reload the page and try again.")


def authorize_audio(row):
    if not row:
        abort(404)
    if session.get("role") == "user" and row["uploaded_by"] != session["user_id"]:
        abort(403)


def visibility(alias="a"):
    return (f"{alias}.uploaded_by=?", [session["user_id"]]) if session.get("role") == "user" else ("1=1", [])


@app.errorhandler(413)
def too_large(error):
    return "Upload too large. Limit each file to 25 MB and each batch to ten files.", 413


def viewer():
    return {
        "user_id": session.get("user_id"),
        "username": session.get("username"),
        "role": session.get("role"),
    }


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapped


def roles(*allowed):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if session.get("role") not in allowed:
                abort(403)
            return fn(*args, **kwargs)

        return wrapped

    return decorate


def save_detection(audio_id, start, end, py, gtm, grade, details, overlap, result):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO detections(audio_id,segment_start,segment_end,python_class,python_scores,python_model_version,gtm_class,gtm_scores,gtm_model_version,agreement_status,confidence_difference,top_two_margin,quality,quality_details,overlap_detected,final_class,severity,alert_status,recommended_action,manual_review,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                audio_id,
                start,
                end,
                py.get("class"),
                json.dumps(py.get("scores", {})),
                py.get("version"),
                gtm.get("class"),
                json.dumps(gtm.get("scores", {})),
                gtm.get("version"),
                result["agreement_status"],
                result["confidence_difference"],
                result["top_two_margin"],
                grade,
                json.dumps(details),
                overlap,
                result["final_class"],
                result["severity"],
                result["alert_status"],
                result["recommended_action"],
                result["manual_review"],
                now(),
            ),
        )
        audit(con, session["user_id"], "prediction", "detection", cur.lastrowid)
        if result["alert_status"] == "Alert Generated":
            audit(con, session["user_id"], "alert", "detection", cur.lastrowid)
        return cur.lastrowid


def analyze(item, live=False):
    raw = item.read(MAX_UPLOAD_BYTES + 1)
    decoded, info = prepare_upload(item.filename, raw)
    digest = hashlib.sha256(raw).hexdigest()
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{secure_filename(item.filename) or 'audio.wav'}"
    original, original_sr = sf.read(io.BytesIO(decoded), dtype="float32", always_2d=True)
    grade, details = quality(original)
    y, sr = load_audio(io.BytesIO(decoded))
    if decoded is not raw:
        path = path.with_suffix(".wav")
    mark = fingerprint(y, sr)
    with connect() as con:
        exact = con.execute(
            "SELECT audio_id FROM audio_files WHERE file_hash=?", (digest,)
        ).fetchone()
        if exact and not live:
            raise ValueError("Duplicate audio: this recording has already been uploaded.")
        own_filter, own_params = visibility("audio_files")
        existing = con.execute(
            f"SELECT audio_id,perceptual_hash FROM audio_files WHERE perceptual_hash IS NOT NULL AND {own_filter}", own_params
        ).fetchall()
        nearest = min(
            (
                (fingerprint_distance(mark, row["perceptual_hash"]), row["audio_id"])
                for row in existing
            ),
            default=(999, None),
        )
        path.write_bytes(decoded)
        cur = con.execute(
            "INSERT INTO audio_files(uploaded_by,filename,stored_path,duration_s,sample_rate,channels,bit_depth,file_size,is_original,status,file_hash,perceptual_hash,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session["user_id"],
                item.filename,
                str(path),
                info.duration,
                info.samplerate,
                info.channels,
                str(getattr(info, "subtype", "")),
                len(raw),
                1,
                "Uploaded",
                None if live else digest,
                mark,
                now(),
            ),
        )
        audio_id = cur.lastrowid
        audit(
            con,
            session["user_id"],
            "microphone_session" if live else "upload",
            "audio_file",
            audio_id,
        )
    output = []
    for clip, start, end in segments(y, sr):
        py = predict_python(waveform=clip, sr=sr)
        gtm = predict_gtm(clip, sr)
        overlap = sum(score >= 0.25 for score in py.get("scores", {}).values()) >= 2
        stream_id = f"live:{session['user_id']}:{session.get('monitor_id', 'default')}" if live else f"upload:{audio_id}"
        result = decide(py, gtm, grade, overlap, stream_id, prefer_gtm=live)
        if not live and nearest[0] <= json.loads(RULES_PATH.read_text())["defaults"]["near_duplicate_distance"]:
            result["manual_review"] = True
            result["alert_status"] = "Manual Review"
            result["recommended_action"] += " Check possible near-duplicate audio."
        output.append(
            (
                save_detection(
                    audio_id, start, end, py, gtm, grade,
                    dict(details, python_status=py.get("reason", "Available"), gtm_status=gtm.get("reason", "Available")), overlap, result
                ),
                result,
            )
        )
    return output


@app.route("/")
def index():
    if not session.get("user_id"):
        return render_template("index.html", rows=[], display=DISPLAY_NAMES, user=viewer())
    clause, params = visibility()
    with connect() as con:
        rows = con.execute(
            f"SELECT d.* FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE {clause} ORDER BY d.created_at DESC LIMIT 10", params
        ).fetchall()
    return render_template(
        "index.html", rows=rows, display=DISPLAY_NAMES, user=viewer()
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username, email = request.form.get("username", "").strip(), request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not (3 <= len(username) <= 80 and "@" in email and len(email) <= 254 and 8 <= len(password) <= 256):
            flash("Use a 3-80 character username, a valid email and an 8-256 character password.", "error")
            return render_template("register.html", user=viewer()), 400
        try:
            role = "user"
            with connect() as con:
                cur = con.execute(
                    "INSERT INTO users(username,email,password_hash,role,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (
                        username,
                        email,
                        generate_password_hash(password),
                        role,
                        now(),
                        now(),
                    ),
                )
                audit(con, cur.lastrowid, "registration", "user", cur.lastrowid)
            flash("Account created. Please sign in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
    return render_template("register.html", user=viewer())


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        with connect() as con:
            row = con.execute(
                "SELECT * FROM users WHERE email=?",
                (request.form["email"].lower().strip(),),
            ).fetchone()
            if row and check_password_hash(
                row["password_hash"], request.form["password"]
            ):
                session.clear()
                session.update(
                    user_id=row["user_id"], username=row["username"], role=row["role"]
                )
                audit(con, row["user_id"], "login", "user", row["user_id"])
                return redirect(url_for("index"))
            audit(con, None, "failed_login", details=request.form.get("email"))
            flash("Invalid email or password.", "error")
    return render_template("login.html", user=viewer())


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        with connect() as con:
            duplicate = con.execute("SELECT 1 FROM users WHERE username=? AND user_id!=?", (username, session["user_id"])).fetchone()
            if not 3 <= len(username) <= 80 or duplicate:
                flash("Choose an available username with 3-80 characters.", "error")
                return redirect(url_for("profile"))
            con.execute(
                "UPDATE users SET username=?,display_name=?,updated_at=? WHERE user_id=?",
                (
                    request.form["username"].strip(),
                    request.form.get("display_name", "").strip(),
                    now(),
                    session["user_id"],
                ),
            )
            audit(con, session["user_id"], "profile_update", "user", session["user_id"])
        session["username"] = request.form["username"].strip()
        flash("Profile updated.", "success")
        return redirect(url_for("profile"))
    with connect() as con:
        account = con.execute(
            "SELECT * FROM users WHERE user_id=?", (session["user_id"],)
        ).fetchone()
    return render_template("profile.html", account=account, user=viewer())


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        outcomes = []
        items = request.files.getlist("audio")
        if not items or len(items) > 10:
            abort(400, description="Choose between one and ten audio files.")
        for item in items:
            try:
                for did, result in analyze(item):
                    outcomes.append((item.filename, did, result))
            except Exception as error:
                outcomes.append((item.filename, None, {"error": str(error)}))
        return render_template("upload.html", outcomes=outcomes, user=viewer())
    return render_template("upload.html", outcomes=None, user=viewer())


def live_comparison(did):
    with connect() as con:
        row = con.execute("SELECT * FROM detections WHERE detection_id=?", (did,)).fetchone()
    event_source = (
        "GTM" if row["gtm_class"] and row["final_class"] == row["gtm_class"]
        else "Python fallback" if row["python_class"]
        else "Unavailable"
    )
    return {"Python": row["python_class"] or "Unavailable", "GTM": row["gtm_class"] or "Unavailable",
            "Live event source": event_source,
            "Python confidence": max(json.loads(row["python_scores"]).values(), default=0),
            "GTM confidence": max(json.loads(row["gtm_scores"]).values(), default=0),
            "Agreement": row["agreement_status"], "Quality": row["quality"],
            "Confidence difference": row["confidence_difference"]}


@app.post("/microphone/start")
@login_required
def microphone_start():
    session["monitor_id"] = uuid.uuid4().hex
    return {"status": "ready"}


@app.post("/microphone/window")
@login_required
def microphone_window():
    item = request.files.get("audio")
    if not item:
        return {"error": "No audio window received."}, 400
    try:
        did, result = analyze(item, True)[0]
        return {
            "detection_id": did,
            "event": result["final_class"],
            "severity": result["severity"],
            "status": result["alert_status"],
            "url": url_for("detection", did=did),
            "comparison": live_comparison(did),
        }
    except ValueError as error:
        return {"error": str(error)}, 400


@app.route("/microphone")
@login_required
def microphone():
    session["monitor_id"] = uuid.uuid4().hex
    return render_template("microphone.html", user=viewer())


@app.route("/detection/<int:did>")
@login_required
def detection(did):
    with connect() as con:
        row = con.execute(
            "SELECT d.*,a.filename,a.stored_path,a.uploaded_by,a.duration_s,a.sample_rate,a.channels,a.bit_depth,a.file_size FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE d.detection_id=?",
            (did,),
        ).fetchone()
    authorize_audio(row)
    item = dict(row)
    item["python_scores"] = json.loads(item["python_scores"] or "{}")
    item["gtm_scores"] = json.loads(item["gtm_scores"] or "{}")
    return render_template(
        "details.html", item=item, display=DISPLAY_NAMES, user=viewer()
    )


@app.route("/audio/<int:did>")
@login_required
def audio_file(did):
    with connect() as con:
        row = con.execute(
            "SELECT stored_path,uploaded_by FROM audio_files a JOIN detections d ON d.audio_id=a.audio_id WHERE d.detection_id=?",
            (did,),
        ).fetchone()
    authorize_audio(row)
    path = Path(row["stored_path"])
    return send_from_directory(path.parent, path.name)


@app.route("/visual/<int:did>/<kind>.png")
@login_required
def visual(did, kind):
    if kind not in {"waveform", "spectrogram"}:
        abort(404)
    with connect() as con:
        row = con.execute(
            "SELECT a.audio_id,a.stored_path,a.uploaded_by FROM audio_files a JOIN detections d ON d.audio_id=a.audio_id WHERE d.detection_id=?",
            (did,),
        ).fetchone()
    authorize_audio(row)
    paths = recording_images(row)
    path = paths[0 if kind == "waveform" else 1]
    return send_from_directory(path.parent, path.name)


@app.route("/report/<int:did>")
@login_required
def report(did):
    with connect() as con:
        row = con.execute(
            "SELECT d.*,a.filename,a.uploaded_by,a.stored_path,a.duration_s,a.sample_rate,a.channels,a.bit_depth,a.file_size FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE d.detection_id=?",
            (did,),
        ).fetchone()
        authorize_audio(row)
        audit(con, session["user_id"], "report", "detection", did)
    images = recording_images(row)
    return send_file(write_report(dict(row), RUNTIME_DIR / "reports", images), as_attachment=True)


visual_lock = Lock()


def recording_images(row):
    """Reuse charts for immutable recordings; serialize Matplotlib rendering."""
    paths = tuple(RUNTIME_DIR / "visuals" / f"{row['audio_id']}_{kind}.png"
                  for kind in ("waveform", "spectrogram"))
    with visual_lock:
        source_modified = Path(row["stored_path"]).stat().st_mtime_ns
        if not all(path.exists() and path.stat().st_mtime_ns >= source_modified for path in paths):
            y, sr = load_audio(row["stored_path"])
            make_images(y, sr, row["audio_id"], RUNTIME_DIR / "visuals")
    return paths


@app.route("/history")
@login_required
def history():
    q = request.args.get("q", "")
    sev = request.args.get("severity", "")
    sql = "SELECT d.*,a.filename FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE (a.filename LIKE ? OR d.final_class LIKE ?)"
    params = [f"%{q}%", f"%{q}%"]
    clause, own_params = visibility()
    sql += f" AND {clause}"
    params.extend(own_params)
    if sev:
        sql += " AND d.severity=?"
        params.append(sev)
    for key, column in (("audio_id", "a.audio_id"), ("quality", "d.quality"),
                        ("category", "d.final_class"), ("status", "d.alert_status"), ("user_id", "a.uploaded_by")):
        if request.args.get(key):
            sql += f" AND {column}=?"
            params.append(request.args[key])
    for key, operator in (("from_date", ">="), ("to_date", "<=")):
        if request.args.get(key):
            sql += f" AND substr(d.created_at,1,10){operator}?"
            params.append(request.args[key])
    for key, operator in (("min_confidence", ">="), ("max_confidence", "<=")):
        if request.args.get(key):
            try:
                value = float(request.args[key])
                if not 0 <= value <= 1:
                    raise ValueError()
            except ValueError:
                abort(400, description="Confidence filters must be between zero and one.")
            sql += f" AND (SELECT MAX(value) FROM json_each(d.python_scores)){operator}?"
            params.append(value)
    page = max(1, request.args.get("page", 1, type=int))
    with connect() as con:
        rows = con.execute(sql + " ORDER BY d.created_at DESC LIMIT 100 OFFSET ?", params + [(page-1)*100]).fetchall()
    return render_template(
        "history.html",
        rows=rows,
        q=q,
        severity=sev,
        page=page,
        display=DISPLAY_NAMES,
        user=viewer(),
    )


@app.route("/dashboard")
@login_required
def dashboard():
    clause, params = visibility()
    with connect() as con:
        summary = con.execute(
            f"SELECT COUNT(*) total,SUM(severity IN ('High','Critical')) critical,SUM(manual_review=1 AND reviewer_decision IS NULL) pending,SUM(quality IN ('Poor','Unusable')) poor,SUM(agreement_status='Model Disagreement') disagreements,AVG((SELECT MAX(value) FROM json_each(d.python_scores))) confidence FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE {clause}", params
        ).fetchone()
        categories = con.execute(
            f"SELECT final_class,COUNT(*) count FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE {clause} GROUP BY final_class ORDER BY count DESC", params
        ).fetchall()
        recent = con.execute(
            f"SELECT d.* FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE {clause} ORDER BY d.created_at DESC LIMIT 15", params
        ).fetchall()
    return render_template(
        "dashboard.html",
        summary=summary,
        categories=categories,
        recent=recent,
        display=DISPLAY_NAMES,
        user=viewer(),
    )


@app.route("/metrics")
@login_required
def metrics():
    from config.settings import ACTIVE_PYTHON_MODEL

    selection = ROOT / "python_models" / "selection.json"
    selected = json.loads(selection.read_text())["selected"] if selection.exists() else ACTIVE_PYTHON_MODEL
    path = ROOT / "python_models" / f"{selected}_metrics.json"
    data = json.loads(path.read_text()) if path.exists() else None
    return render_template("metrics.html", data=data, selected=selected, user=viewer())


@app.route("/review", methods=["GET", "POST"])
@login_required
@roles("reviewer", "operator", "admin")
def review():
    if request.method == "POST":
        if request.form.get("decision") not in SONIC_CLASSES:
            abort(400, description="Choose one of the supported sound classes.")
        with connect() as con:
            if not con.execute("SELECT 1 FROM detections WHERE detection_id=?", (request.form.get("detection_id"),)).fetchone():
                abort(404)
            con.execute(
                "UPDATE detections SET reviewer_decision=?,reviewer_comment=?,reviewed_by=?,reviewed_at=?,final_class=?,alert_status='Reviewed' WHERE detection_id=?",
                (
                    request.form["decision"],
                    request.form.get("comment"),
                    session["user_id"],
                    now(),
                    request.form["decision"],
                    request.form["detection_id"],
                ),
            )
            audit(
                con,
                session["user_id"],
                "override",
                "detection",
                request.form["detection_id"],
            )
        return redirect(url_for("review"))
    page = max(1, request.args.get("page", 1, type=int))
    with connect() as con:
        rows = con.execute(
            "SELECT d.*,a.filename FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE d.manual_review=1 AND d.reviewer_decision IS NULL ORDER BY d.created_at DESC LIMIT 25 OFFSET ?",
            ((page - 1) * 25,),
        ).fetchall()
    return render_template(
        "review.html", rows=rows, page=page, display=DISPLAY_NAMES, user=viewer()
    )


@app.post("/alert/<int:did>/<action>")
@login_required
@roles("reviewer", "operator", "maintenance", "admin")
def alert_action(did, action):
    if action not in {"Acknowledged", "Dismissed", "Escalated", "Closed"}:
        abort(400)
    with connect() as con:
        if not con.execute("SELECT 1 FROM detections WHERE detection_id=?", (did,)).fetchone():
            abort(404)
        con.execute(
            "UPDATE detections SET alert_status=? WHERE detection_id=?", (action, did)
        )
        audit(con, session["user_id"], action.lower(), "detection", did)
    return redirect(url_for("dashboard"))


@app.route("/export.csv")
@login_required
@roles("admin")
def export_csv():
    path = RUNTIME_DIR / "reports" / "events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM detections ORDER BY created_at DESC"
        ).fetchall()
        audit(con, session["user_id"], "export", "report", details=str(path))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys() if rows else ["detection_id"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value)
                             for key, value in dict(row).items()})
    return send_file(path, as_attachment=True)


@app.route("/admin/settings", methods=["GET", "POST"])
@login_required
@roles("admin")
def admin_settings():
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    if request.method == "POST":
        try:
            for key in ("minimum_confidence", "top_two_margin"):
                value = float(request.form[key])
                if not 0 <= value <= 1:
                    raise ValueError()
                rules["defaults"][key] = value
            for key, lower, upper in (("repeat_windows", 1, 20), ("retention_days", 1, 3650), ("near_duplicate_distance", 0, 128)):
                value = int(request.form[key])
                if not lower <= value <= upper:
                    raise ValueError()
                rules["defaults"][key] = value
            rules["defaults"]["require_model_agreement_for_critical"] = request.form.get("require_agreement") == "on"
            qualities = request.form.getlist("required_quality")
            if not qualities or not set(qualities).issubset({"Good", "Acceptable"}):
                raise ValueError()
            rules["defaults"]["required_quality"] = qualities
            for label, category in rules["categories"].items():
                category["critical"] = request.form.get(f"critical_{label}") == "on"
                severity = request.form.get(f"severity_{label}", category["severity"])
                if severity not in {"Informational", "Low", "Medium", "High", "Critical"}:
                    raise ValueError()
                category["severity"] = severity
                category["action"] = request.form.get(f"action_{label}", category["action"]).strip()[:500]
            pending = RULES_PATH.with_suffix(".pending.json")
            pending.write_text(json.dumps(rules, indent=2), encoding="utf-8")
            pending.replace(RULES_PATH)
            with connect() as con:
                audit(con, session["user_id"], "settings_update", "rules")
            flash("Alert rules saved.", "success")
            return redirect(url_for("admin_settings"))
        except (ValueError, KeyError):
            flash("Invalid settings. Confidence must be 0-1; repeats 1-20; retention 1-3650 days; choose Good and/or Acceptable quality.", "error")
            return render_template("settings.html", rules=rules, user=viewer()), 400
    return render_template("settings.html", rules=rules, user=viewer())


@app.post("/admin/retention")
@login_required
@roles("admin")
def retention():
    if request.form.get("confirm") != "yes":
        abort(400, description="Confirm retention cleanup before proceeding.")
    rules = json.loads(RULES_PATH.read_text())
    cutoff = (datetime.now(timezone.utc) - timedelta(days=rules["defaults"]["retention_days"])).isoformat()
    with connect() as con:
        rows = con.execute("SELECT audio_id,stored_path FROM audio_files WHERE created_at < ?", (cutoff,)).fetchall()
        for row in rows:
            # Never remove an arbitrary stored path outside the configured upload directory.
            path = Path(row["stored_path"]).resolve()
            if not path.is_relative_to(UPLOAD_DIR.resolve()):
                abort(400, description="An expired audio path is outside upload storage; cleanup stopped.")
        for row in rows:
            con.execute("DELETE FROM detections WHERE audio_id=?", (row["audio_id"],))
            con.execute("DELETE FROM audio_files WHERE audio_id=?", (row["audio_id"],))
        audit(con, session["user_id"], "retention_cleanup", "audio_files", details=str(len(rows)))
    for row in rows:
        path = Path(row["stored_path"])
        with connect() as con:
            shared = con.execute("SELECT 1 FROM audio_files WHERE stored_path=?", (str(path),)).fetchone()
        if not shared:
            path.unlink(missing_ok=True)
        for image in (RUNTIME_DIR / "visuals").glob(f"{row['audio_id']}_*.png"):
            image.unlink(missing_ok=True)
    flash(f"Removed {len(rows)} expired recordings and associated detections.", "success")
    return redirect(url_for("admin_settings"))


@app.cli.command("create-user")
@click.option("--username", prompt=True)
@click.option("--email", prompt=True)
@click.option("--role", type=click.Choice(["user", "reviewer", "operator", "maintenance", "admin"]), default="admin")
@click.password_option()
def create_user(username, email, role, password):
    """Provision a trusted account locally; public registration always creates normal users."""
    if len(password) < 8:
        raise click.ClickException("Password must contain at least eight characters.")
    try:
        with connect() as con:
            cur = con.execute("INSERT INTO users(username,email,password_hash,role,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                              (username.strip(), email.strip().lower(), generate_password_hash(password), role, now(), now()))
            audit(con, cur.lastrowid, "account_provisioned", "user", cur.lastrowid)
    except sqlite3.IntegrityError:
        raise click.ClickException("Username or email already exists.")
    click.echo(f"Created {role} account: {username}")


def warmup_models():
    """Load both audio models before accepting interactive requests."""
    import numpy as np
    waveform = (0.1 * np.sin(2 * np.pi * 440 * np.arange(22050 * 3) / 22050)).astype(np.float32)
    python_result = predict_python(waveform=waveform, sr=22050)
    gtm_result = predict_gtm(waveform, 22050)
    if not python_result["available"]:
        app.logger.warning("Python warm-up: %s", python_result.get("reason"))
    if not gtm_result["available"]:
        app.logger.warning("GTM warm-up: %s", gtm_result.get("reason"))
    return {"python": python_result["available"], "gtm": gtm_result["available"]}


if __name__ == "__main__":
    # Load both classifiers once before accepting analysis requests.
    # Disabling the reloader prevents duplicate model copies in memory.
    warmup_models()
    app.run(debug=False, threaded=True, port=int(os.environ.get("SONIC_PORT", "5000")))
