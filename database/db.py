import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from config.settings import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','reviewer','operator','maintenance','admin')), display_name TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audio_files (audio_id INTEGER PRIMARY KEY, uploaded_by INTEGER REFERENCES users(user_id), filename TEXT NOT NULL, stored_path TEXT NOT NULL, sound_category TEXT, duration_s REAL, sample_rate INTEGER, channels INTEGER, bit_depth INTEGER, file_size INTEGER, recording_environment TEXT, recording_device TEXT, source_distance TEXT, is_original INTEGER, dataset_split TEXT, file_hash TEXT UNIQUE, perceptual_hash TEXT, status TEXT NOT NULL DEFAULT 'Uploaded', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS detections (detection_id INTEGER PRIMARY KEY, audio_id INTEGER NOT NULL REFERENCES audio_files(audio_id), segment_start REAL, segment_end REAL, python_class TEXT, python_scores TEXT, python_model_version TEXT, gtm_class TEXT, gtm_scores TEXT, gtm_model_version TEXT, agreement_status TEXT, confidence_difference REAL, top_two_margin REAL, quality TEXT, quality_details TEXT, overlap_detected INTEGER NOT NULL DEFAULT 0, final_class TEXT, severity TEXT, alert_status TEXT, recommended_action TEXT, manual_review INTEGER NOT NULL DEFAULT 0, reviewer_decision TEXT, reviewer_comment TEXT, reviewed_by INTEGER REFERENCES users(user_id), reviewed_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (audit_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id), action_type TEXT NOT NULL, target_type TEXT, target_id INTEGER, details TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_audio_user ON audio_files(uploaded_by); CREATE INDEX IF NOT EXISTS idx_detection_audio ON detections(audio_id); CREATE INDEX IF NOT EXISTS idx_detection_created ON detections(created_at); CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
"""

def now(): return datetime.now(timezone.utc).isoformat()
@contextmanager
def connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATABASE_PATH); con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON"); con.execute("PRAGMA journal_mode = WAL")
    try: yield con; con.commit()
    except: con.rollback(); raise
    finally: con.close()
def init_db():
    with connect() as con: con.executescript(SCHEMA)
def audit(con, user_id, action, target_type=None, target_id=None, details=None):
    con.execute("INSERT INTO audit_log(user_id,action_type,target_type,target_id,details,created_at) VALUES(?,?,?,?,?,?)", (user_id, action, target_type, target_id, details, now()))
#UPDATED VERIFIED