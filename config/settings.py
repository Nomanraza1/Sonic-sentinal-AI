from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "sonic_sentinel.db"
UPLOAD_DIR = ROOT / "uploads"
PYTHON_MODELS = ROOT / "python_models"
GTM_MODEL = ROOT / "gtm_model"
ACTIVE_PYTHON_MODEL = "gradient_boosting"
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
# UPSDATED
