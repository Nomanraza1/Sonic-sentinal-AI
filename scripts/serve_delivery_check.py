"""Local browser QA server using only the isolated verification database."""
import os,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
info=json.loads((ROOT/'reports/delivery_e2e.json').read_text())
base=Path(info['runtime_path'])
os.environ['SONIC_DATABASE_PATH']=str(base/'test.db')
os.environ['SONIC_UPLOAD_DIR']=str(base/'uploads')
os.environ['SONIC_RUNTIME_DIR']=str(base/'instance')
os.environ['SONIC_RULES_PATH']=str(base/'rules.json')
if not (base/'rules.json').exists():(base/'rules.json').write_bytes((ROOT/'alert_rules/default.json').read_bytes())
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='4'
os.environ['TF_NUM_INTEROP_THREADS']='1'
from app import app,warmup_models
warmup_models()
app.run(port=5057,host='127.0.0.1',debug=False)
