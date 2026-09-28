"""Exercise real model through Flask in isolated storage, with reproducible evidence."""
import os
import sys
import io
import json
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUNTIME=ROOT/"tmp"/f"delivery_{uuid.uuid4().hex[:8]}"
RUNTIME.mkdir(parents=True)
os.environ["SONIC_DATABASE_PATH"]=str(RUNTIME/"test.db")
os.environ["SONIC_UPLOAD_DIR"]=str(RUNTIME/"uploads")
os.environ["SONIC_RUNTIME_DIR"]=str(RUNTIME/"instance")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","1")

import numpy as np
import soundfile as sf
from werkzeug.security import generate_password_hash
import app as application
from audio_preprocessing.audio import load_audio,segments
from config.class_map import SONIC_CLASSES
from database.db import connect,now
from src.models import predict_python, predict_gtm

client=application.app.test_client()
warmup_started=time.perf_counter()
warmup_status=application.warmup_models()
warmup_seconds=time.perf_counter()-warmup_started
with connect() as con:
    con.execute("INSERT INTO users(username,email,password_hash,role,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                ("delivery_checker","delivery@example.test",generate_password_hash("Local-delivery-check-2026!"),"admin",now(),now()))

def post(url,data):
    client.get("/")
    with client.session_transaction() as session:token=session["csrf_token"]
    return client.post(url,data=dict(data,csrf_token=token))

assert post('/login',{'email':'delivery@example.test','password':'Local-delivery-check-2026!'}).status_code==302
manifest=json.loads((ROOT/'data/transfer_experiment/manifest.json').read_text())
selected=json.loads((ROOT/'python_models/selection.json').read_text())['selected']
result={'runtime_path':str(RUNTIME),'model':selected,'gtm':'gtm_model_final',
        'warmup':{'seconds':warmup_seconds,'models':warmup_status},
        'real_audio_flows':[],'performance':{},'noise_probe':[],'model_comparison':[]}
for label in SONIC_CLASSES:
    row=next(r for r in manifest if r['dataset_split']=='test' and r['class_label']==label and r['segment_index']==0)
    y,sr=load_audio(row['source_path']);clip=next(segments(y,sr))[0]
    stream=io.BytesIO();sf.write(stream,clip,sr,format='WAV')
    start=time.perf_counter()
    response=post('/upload',{'audio':(io.BytesIO(stream.getvalue()),f'{label}.wav')})
    elapsed=time.perf_counter()-start
    assert response.status_code==200 and b'Open analysis' in response.data,response.data.decode()[:1000]
    with connect() as con:d=dict(con.execute('SELECT * FROM detections ORDER BY detection_id DESC LIMIT 1').fetchone())
    assert len(json.loads(d['python_scores']))==10
    assert d['python_model_version']==selected
    assert len(json.loads(d['gtm_scores']))==10
    assert d['gtm_model_version']=='gtm_model_final'
    gtm=predict_gtm(clip,sr)
    assert gtm['available'],gtm.get('reason')
    assert d['gtm_class'] in SONIC_CLASSES
    result['model_comparison'].append({'truth':label,'python':d['python_class'],'gtm':d['gtm_class'],
        'python_confidence':json.loads(d['python_scores'])[d['python_class']],
        'gtm_confidence':json.loads(d['gtm_scores'])[d['gtm_class']],
        'agreement':d['python_class']==d['gtm_class'],'manual_review':bool(d['manual_review'])})
    for route in [f"/detection/{d['detection_id']}",f"/audio/{d['detection_id']}"]:
        assert client.get(route).status_code==200
    result['real_audio_flows'].append({'class':label,'python':d['python_class'],'gtm':d['gtm_class'],'seconds':elapsed,'detection_id':d['detection_id']})
    print(label,'python:',d['python_class'],'GTM:',d['gtm_class'],round(elapsed,3),'seconds',flush=True)
    for snr in (20,10):
        rng=np.random.default_rng(7)
        noise=rng.normal(0,np.sqrt(np.mean(clip**2))/(10**(snr/20)),len(clip)).astype(np.float32)
        noisy=np.clip(clip+noise,-1,1)
        prediction=predict_python(waveform=noisy,sr=sr)
        result['noise_probe'].append({'class':label,'snr_db':snr,'predicted':prediction['class'],'correct':prediction['class']==label})
result['smoke_summary']={
    'recordings':len(result['model_comparison']),
    'python_correct':sum(row['python']==row['truth'] for row in result['model_comparison']),
    'gtm_correct':sum(row['gtm']==row['truth'] for row in result['model_comparison']),
}
# Warm timing of an exact 30-second upload and a three-second live window.
y,sr=load_audio(next(r['source_path'] for r in manifest if r['dataset_split']=='test' and r['class_label']=='machinery_fault'))
clip=next(segments(y,sr))[0]
for seconds,route in ((30,'/upload'),(3,'/microphone/window')):
    waveform=np.tile(clip,int(seconds/3))
    stream=io.BytesIO();sf.write(stream,waveform,sr,format='WAV')
    start=time.perf_counter();response=post(route,{'audio':(io.BytesIO(stream.getvalue()),f'timing_{seconds}.wav')});elapsed=time.perf_counter()-start
    assert response.status_code==200
    if seconds==30:assert b'Open analysis' in response.data
    result['performance'][f'{seconds}_second_audio_seconds']=elapsed
    print('Timing',seconds,round(elapsed,3),flush=True)
# Test 20,000 persisted rows; these are synthetic load-test records, not evaluation audio.
with connect() as con:
    did=con.execute('SELECT audio_id FROM audio_files LIMIT 1').fetchone()[0]
    con.executemany('INSERT INTO detections(audio_id,final_class,severity,alert_status,quality,created_at) VALUES(?,?,?,?,?,?)',
                    [(did,'background_noise','Informational','Classified','Good',now()) for _ in range(20000)])
start=time.perf_counter();response=client.get('/dashboard');elapsed=time.perf_counter()-start
assert response.status_code==200
result['performance']['dashboard_20000_records_seconds']=elapsed
# Evidence only, not a model tuning loop.
report=ROOT/'reports/delivery_e2e.json';report.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Evidence saved',report,flush=True)
