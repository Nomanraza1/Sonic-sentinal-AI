"""Application flows in temporary storage; GTM stubs are explicitly contract tests."""
import io
import json
import importlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from werkzeug.security import generate_password_hash


@pytest.fixture
def web(tmp_path, monkeypatch):
    import config.settings as settings
    import database.db as db
    rules = tmp_path / "rules.json"
    rules.write_bytes((settings.ROOT / "alert_rules/default.json").read_bytes())
    monkeypatch.setattr(db, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(settings, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    module = importlib.import_module("app")
    monkeypatch.setattr(module, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "RULES_PATH", rules)
    from src import rules as decision
    monkeypatch.setattr(decision, "RULES_PATH", rules)
    decision.recent.clear()
    module.RUNTIME_DIR.mkdir(exist_ok=True)
    module.UPLOAD_DIR.mkdir(exist_ok=True)
    db.init_db()
    module.app.config.update(TESTING=True)
    client = module.app.test_client()
    return module, client, db


def post(client, url, data=None):
    client.get("/")
    with client.session_transaction() as session:
        token = session["csrf_token"]
    return client.post(url, data=dict(data or {}, csrf_token=token), follow_redirects=False)


def login(web, role="user", name="tester"):
    module, client, db = web
    with db.connect() as con:
        cur=con.execute("INSERT INTO users(username,email,password_hash,role,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                        (name, name+"@example.test", generate_password_hash("test-password"), role, db.now(), db.now()))
        uid=cur.lastrowid
    response=post(client,"/login",{"email":name+"@example.test","password":"test-password"})
    assert response.status_code==302
    return uid


def audio(seconds=1, amplitude=0.3, fmt="WAV", rate=16000, channels=1):
    t=np.arange(int(rate*seconds))/rate
    y=amplitude*np.sin(2*np.pi*440*t)
    if channels>1:y=np.tile(y[:,None],(1,channels))
    stream=io.BytesIO();sf.write(stream,y,rate,format=fmt);return stream.getvalue()


def prediction(label="gunshot"):
    from config.class_map import SONIC_CLASSES
    values={name:0.01 for name in SONIC_CLASSES};values[label]=0.91
    return {"available":True,"class":label,"scores":values,"confidence":0.91,"version":"test-contract-only"}


def stub_models(module, monkeypatch):
    monkeypatch.setattr(module,"predict_python",lambda **kwargs:prediction())
    monkeypatch.setattr(module,"predict_gtm",lambda *args:prediction())


def test_registration_authentication_and_role_escalation(web):
    module,client,db=web
    assert client.get('/upload').status_code==302
    assert client.post('/register',data={}).status_code==400
    bad=post(client,'/register',{'username':'ab','email':'no','password':'x'})
    assert bad.status_code==400
    good=post(client,'/register',{'username':'newuser','email':'new@example.test','password':'test-password','role':'admin'})
    assert good.status_code==302
    with db.connect() as con:assert con.execute('SELECT role FROM users').fetchone()['role']=='user'
    assert post(client,'/login',{'email':'new@example.test','password':'wrong'}).status_code==200
    assert post(client,'/login',{'email':'new@example.test','password':'test-password'}).status_code==302
    assert client.get('/admin/settings').status_code==403
    assert client.get('/export.csv').status_code==403
    assert post(client,'/logout').status_code==302


def test_upload_playback_visual_report_review_export(web,monkeypatch):
    module,client,db=web;login(web,'admin');stub_models(module,monkeypatch)
    response=post(client,'/upload',{'audio':(io.BytesIO(audio(4)),'event.wav')})
    assert response.status_code==200
    with db.connect() as con:
        rows=con.execute('SELECT * FROM detections ORDER BY detection_id').fetchall()
        assert len(rows)==2
        did=rows[0]['detection_id']
        assert rows[1]['alert_status']=='Alert Generated'
    for route in ['/','/dashboard','/history','/metrics','/profile','/review','/admin/settings',f'/detection/{did}',f'/audio/{did}']:
        assert client.get(route).status_code==200,route
    selected=json.loads((module.ROOT/'python_models/selection.json').read_text())['selected']
    assert selected.encode() in client.get('/metrics').data
    for kind in ['waveform','spectrogram']:
        result=client.get(f'/visual/{did}/{kind}.png');assert result.status_code==200
        assert result.data.startswith(b'\x89PNG')
    report=client.get(f'/report/{did}')
    assert report.status_code==200 and b'data:image/png;base64' in report.data
    assert b'Python confidence scores' in report.data
    assert post(client,'/review',{'detection_id':did,'decision':'made-up'}).status_code==400
    assert post(client,'/review',{'detection_id':did,'decision':'background_noise','comment':'corrected'}).status_code==302
    with db.connect() as con:
        row=con.execute('SELECT * FROM detections WHERE detection_id=?',(did,)).fetchone()
        assert row['python_class']=='gunshot' and row['final_class']=='background_noise'
    assert post(client,f'/alert/{did}/Acknowledged').status_code==302
    assert client.get('/export.csv').status_code==200
    duplicate=post(client,'/upload',{'audio':(io.BytesIO(audio(4)),'event.wav')})
    assert b'Duplicate audio' in duplicate.data
    assert len(list(module.UPLOAD_DIR.iterdir()))==1


def test_other_user_cannot_read_recordings(web,monkeypatch):
    module,client,db=web;login(web,name='owner');stub_models(module,monkeypatch)
    post(client,'/upload',{'audio':(io.BytesIO(audio()),'private.wav')})
    with db.connect() as con:did=con.execute('SELECT detection_id FROM detections').fetchone()[0]
    post(client,'/logout');login(web,name='other')
    for route in [f'/detection/{did}',f'/audio/{did}',f'/visual/{did}/waveform.png',f'/report/{did}']:
        assert client.get(route).status_code==403,route


def test_visuals_are_reused_and_refreshed_when_audio_changes(web, monkeypatch):
    import os
    module, client, db = web
    login(web)
    stub_models(module, monkeypatch)
    post(client, '/upload', {'audio': (io.BytesIO(audio()), 'cached.wav')})
    with db.connect() as con:
        row = con.execute('SELECT d.detection_id,a.stored_path FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id').fetchone()
    did = row['detection_id']
    render = module.make_images
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return render(*args, **kwargs)
    monkeypatch.setattr(module, 'make_images', counted)
    assert client.get(f'/visual/{did}/waveform.png').status_code == 200
    assert client.get(f'/visual/{did}/spectrogram.png').status_code == 200
    assert client.get(f'/report/{did}').status_code == 200
    assert len(calls) == 1
    path = Path(row['stored_path'])
    stamp = max(p.stat().st_mtime_ns for p in (module.RUNTIME_DIR / 'visuals').glob('*.png')) + 1_000_000
    os.utime(path, ns=(stamp, stamp))
    assert client.get(f'/visual/{did}/waveform.png').status_code == 200
    assert len(calls) == 2


def test_review_queue_is_bounded_and_paginated(web):
    module, client, db = web
    uid = login(web, 'reviewer')
    with db.connect() as con:
        aid = con.execute("INSERT INTO audio_files(uploaded_by,filename,stored_path,created_at) VALUES(?,?,?,?)",
                          (uid, 'queue.wav', 'unused.wav', db.now())).lastrowid
        con.executemany("INSERT INTO detections(audio_id,final_class,manual_review,created_at) VALUES(?,?,1,?)",
                        [(aid, 'gunshot', db.now()) for _ in range(30)])
    first = client.get('/review').data
    second = client.get('/review?page=2').data
    assert first.count(b'preload="none"') == 25
    assert second.count(b'preload="none"') == 5
    assert b'private.wav' not in client.get('/history').data
    assert b'gunshot' not in client.get('/dashboard').data


@pytest.mark.parametrize('name,raw',[
 ('empty.wav',b''),('bad.wav',b'not audio'),('bad.txt',b'invalid'),
 ('silent.wav',audio(amplitude=0)),('short.wav',audio(seconds=0.1)),
 ('long.wav',audio(seconds=31)),('rate.wav',audio(rate=4000)),('channels.wav',audio(channels=3)),
], ids=['empty','corrupt','extension','silence','too-short','too-long','sample-rate','channels'])
def test_invalid_uploads_do_not_persist(web,name,raw):
    module,client,db=web;login(web)
    result=post(client,'/upload',{'audio':(io.BytesIO(raw),name)})
    assert result.status_code==200 and b'class="error"' in result.data
    with db.connect() as con:assert con.execute('SELECT COUNT(*) FROM audio_files').fetchone()[0]==0
    assert not list(module.UPLOAD_DIR.iterdir())


@pytest.mark.parametrize('fmt,extension',[('WAV','wav'),('FLAC','flac'),('OGG','ogg'),('MP3','mp3')])
def test_supported_audio_formats(web,monkeypatch,fmt,extension):
    module,client,db=web;login(web);stub_models(module,monkeypatch)
    result=post(client,'/upload',{'audio':(io.BytesIO(audio(fmt=fmt)),f'sound.{extension}')})
    assert result.status_code==200 and b'Open analysis' in result.data


def test_live_windows_share_confirmation_and_stop_capture_contract(web,monkeypatch):
    module,client,db=web;login(web);stub_models(module,monkeypatch)
    assert client.get('/microphone').status_code==200
    assert post(client,'/microphone/start').status_code==200
    first=post(client,'/microphone/window',{'audio':(io.BytesIO(audio()),'live.wav')})
    second=post(client,'/microphone/window',{'audio':(io.BytesIO(audio()),'live.wav')})
    assert first.json['status']=='Classified'
    assert second.json['status']=='Alert Generated'
    assert second.json['comparison']['GTM']=='gunshot'
    assert post(client,'/microphone/start').status_code==200
    third=post(client,'/microphone/window',{'audio':(io.BytesIO(audio()),'live.wav')})
    assert third.json['status']=='Classified'


def test_live_event_uses_gtm_when_models_disagree(web,monkeypatch):
    module,client,db=web;login(web);post(client,'/microphone/start')
    monkeypatch.setattr(module,'predict_python',lambda **kwargs:prediction('person_asking_for_help'))
    monkeypatch.setattr(module,'predict_gtm',lambda *args:prediction('aggression'))
    response=post(client,'/microphone/window',{'audio':(io.BytesIO(audio(seconds=3)),'live.wav')})
    assert response.status_code==200
    assert response.json['event']=='aggression'
    assert response.json['status']=='Manual Review'
    assert response.json['comparison']['Python']=='person_asking_for_help'
    assert response.json['comparison']['GTM']=='aggression'
    assert response.json['comparison']['Live event source']=='GTM'
    with db.connect() as con:
        row=con.execute('SELECT final_class,python_class,gtm_class,severity FROM detections ORDER BY detection_id DESC LIMIT 1').fetchone()
    assert tuple(row)==('aggression','person_asking_for_help','aggression','High')


def test_missing_gtm_routes_for_review(web,monkeypatch):
    module,client,db=web;login(web)
    monkeypatch.setattr(module,'predict_python',lambda **kwargs:prediction())
    monkeypatch.setattr(module,'predict_gtm',lambda *args:{'available':False,'class':None,'scores':{},'reason':'GTM pending'})
    post(client,'/upload',{'audio':(io.BytesIO(audio()),'clip.wav')})
    with db.connect() as con:
        row=con.execute('SELECT * FROM detections').fetchone()
        assert row['manual_review']==1 and row['alert_status']=='Manual Review'
        assert json.loads(row['gtm_scores'])=={}


def test_settings_validation_and_retention_path_guard(web,monkeypatch,tmp_path):
    module,client,db=web;login(web,'admin');stub_models(module,monkeypatch)
    fields={'minimum_confidence':'0.8','top_two_margin':'0.2','repeat_windows':'3',
            'retention_days':'30','near_duplicate_distance':'8','require_agreement':'on','required_quality':['Good','Acceptable']}
    original=json.loads(module.RULES_PATH.read_text())
    for label,category in original['categories'].items():
        if category['critical']:fields[f'critical_{label}']='on'
    assert post(client,'/admin/settings',fields).status_code==302
    assert json.loads(module.RULES_PATH.read_text())['defaults']['minimum_confidence']==0.8
    fields['minimum_confidence']='2'
    assert post(client,'/admin/settings',fields).status_code==400
    assert json.loads(module.RULES_PATH.read_text())['defaults']['minimum_confidence']==0.8
    post(client,'/upload',{'audio':(io.BytesIO(audio()),'old.wav')})
    outside=tmp_path/'do-not-delete.txt';outside.write_text('keep')
    with db.connect() as con:con.execute("UPDATE audio_files SET created_at='2000-01-01',stored_path=?",(str(outside),))
    assert post(client,'/admin/retention',{'confirm':'yes'}).status_code==400
    assert outside.exists()


def test_m4a_upload(web,monkeypatch,tmp_path):
    import subprocess,imageio_ffmpeg
    module,client,db=web;login(web);stub_models(module,monkeypatch)
    source=tmp_path/'input.wav';source.write_bytes(audio())
    target=tmp_path/'input.m4a'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(source),'-c:a','aac',str(target)],check=True)
    result=post(client,'/upload',{'audio':(io.BytesIO(target.read_bytes()),'sound.m4a')})
    assert result.status_code==200 and b'Open analysis' in result.data


def test_clipping_and_low_signal_quality():
    from audio_preprocessing.audio import quality
    assert quality(np.ones(16000))[0]=='Poor'
    assert quality(np.ones(16000)*0.001)[0]=='Unusable'


def test_uncertain_windows_never_trigger_critical_alerts():
    from src.rules import decide,recent
    recent.clear()
    strong=prediction()
    weak=prediction();weak['scores']={'gunshot':0.51,'glass_breaking':0.49};weak['confidence']=0.51
    for _ in range(3):
        assert decide(weak,strong,'Good',False,'low')['alert_status']=='Manual Review'
        assert decide(strong,strong,'Good',True,'overlap')['alert_status']=='Manual Review'
        assert decide(strong,strong,'Poor',False,'poor')['alert_status']=='Manual Review'
    decide(strong,strong,'Good',False,'reset')
    decide(prediction('animal_sound'),prediction('animal_sound'),'Good',False,'reset')
    assert decide(strong,strong,'Good',False,'reset')['alert_status']=='Classified'


def test_retention_removes_only_expired_test_recordings(web,monkeypatch):
    module,client,db=web;login(web,'admin');stub_models(module,monkeypatch)
    post(client,'/upload',{'audio':(io.BytesIO(audio()),'expired.wav')})
    with db.connect() as con:con.execute("UPDATE audio_files SET created_at='2000-01-01'")
    assert post(client,'/admin/retention').status_code==400
    assert post(client,'/admin/retention',{'confirm':'yes'}).status_code==302
    with db.connect() as con:
        assert con.execute('SELECT COUNT(*) FROM audio_files').fetchone()[0]==0
        assert con.execute('SELECT COUNT(*) FROM detections').fetchone()[0]==0
        assert con.execute("SELECT COUNT(*) FROM audit_log WHERE action_type='retention_cleanup'").fetchone()[0]==1
    assert not list(module.UPLOAD_DIR.iterdir())
