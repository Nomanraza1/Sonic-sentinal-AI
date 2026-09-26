import csv, hashlib, json
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path
from flask import Flask, abort, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from audio_preprocessing.audio import fingerprint, fingerprint_distance, load_audio, quality, segments, validate_upload
from config.class_map import DISPLAY_NAMES
from config.settings import ROOT, UPLOAD_DIR
from database.db import audit, connect, init_db, now
from feature_extraction.features import extract_features
from src.models import predict_gtm, predict_python
from src.rules import decide
from src.visuals import make_images
from src.reports import write_report

app=Flask(__name__); app.config['SECRET_KEY']='change-this-secret-before-deployment'; UPLOAD_DIR.mkdir(exist_ok=True)
def login_required(f):
 @wraps(f)
 def w(*a,**k):
  if not session.get('user_id'): return redirect(url_for('login'))
  return f(*a,**k)
 return w
def roles(*ok):
 def d(f):
  @wraps(f)
  def w(*a,**k):
   if session.get('role') not in ok: abort(403)
   return f(*a,**k)
  return w
 return d
def viewer(): return {'user_id':session.get('user_id'),'username':session.get('username'),'role':session.get('role')}
def analyze_file(item, live=False):
 raw=item.read(); info=validate_upload(item.filename,raw); digest=hashlib.sha256(raw).hexdigest()
 path=UPLOAD_DIR/f'{digest[:16]}_{Path(item.filename).name}'; path.write_bytes(raw)
 y,sr=load_audio(str(path)); audio_fingerprint=fingerprint(y,sr)
 with connect() as c:
  old=c.execute('SELECT audio_id FROM audio_files WHERE file_hash=?',(digest,)).fetchone()
  if old and not live: raise ValueError(f'Duplicate of audio #{old["audio_id"]}.')
  matches=c.execute('SELECT audio_id,perceptual_hash FROM audio_files WHERE perceptual_hash IS NOT NULL').fetchall()
  nearest=min(((fingerprint_distance(audio_fingerprint,row['perceptual_hash']),row['audio_id']) for row in matches), default=(999,None))
  cur=c.execute('INSERT INTO audio_files(uploaded_by,filename,stored_path,duration_s,sample_rate,channels,bit_depth,file_size,is_original,status,file_hash,perceptual_hash,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(session['user_id'],item.filename,str(path),info.duration,info.samplerate,info.channels,str(getattr(info,'subtype','')),len(raw),1,'Uploaded',digest if not live else None,audio_fingerprint,now()))
  audio_id=cur.lastrowid; audit(c,session['user_id'],'microphone_session' if live else 'upload','audio_file',audio_id)
  if nearest[0] <= 8: audit(c,session['user_id'],'near_duplicate','audio_file',audio_id,f'Near audio #{nearest[1]}, distance {nearest[0]}')
 grade,details=quality(y); records=[]
 for clip,start,end in segments(y,sr):
  py=predict_python(extract_features(clip,sr)); gtm=predict_gtm(clip,sr); overlap=sum(x>=.25 for x in py.get('scores',{}).values())>=2; result=decide(py,gtm,grade,overlap)
  if nearest[0] <= 8: result['manual_review']=True; result['alert_status']='Manual Review'; result['recommended_action'] += ' Check possible near-duplicate audio.'
  records.append((store(audio_id,start,end,py,gtm,grade,details,overlap,result),result))
 return records
def store(audio_id, start, end, py, gtm, grade, details, overlap, result):
 with connect() as c:
  q='''INSERT INTO detections(audio_id,segment_start,segment_end,python_class,python_scores,python_model_version,gtm_class,gtm_scores,gtm_model_version,agreement_status,confidence_difference,top_two_margin,quality,quality_details,overlap_detected,final_class,severity,alert_status,recommended_action,manual_review,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
  cur=c.execute(q,(audio_id,start,end,py.get('class'),json.dumps(py.get('scores',{})),py.get('version'),gtm.get('class'),json.dumps(gtm.get('scores',{})),gtm.get('version'),result['agreement_status'],result['confidence_difference'],result['top_two_margin'],grade,json.dumps(details),overlap,result['final_class'],result['severity'],result['alert_status'],result['recommended_action'],result['manual_review'],now()))
  audit(c,session.get('user_id'),'prediction','detection',cur.lastrowid)
  if result['alert_status']=='Alert Generated': audit(c,session.get('user_id'),'alert','detection',cur.lastrowid)
  return cur.lastrowid
@app.route('/')
def index():
 with connect() as c: rows=c.execute('SELECT * FROM detections ORDER BY created_at DESC LIMIT 10').fetchall()
 return render_template('index.html',rows=rows,display=DISPLAY_NAMES,user=viewer())
@app.route('/register',methods=['GET','POST'])
def register():
 if request.method=='POST':
  try:
   role=request.form.get('role','user'); role=role if role in {'user','reviewer','operator','maintenance','admin'} else 'user'
   with connect() as c:
    cur=c.execute('INSERT INTO users(username,email,password_hash,role,created_at,updated_at) VALUES(?,?,?,?,?,?)',(request.form['username'].strip(),request.form['email'].lower().strip(),generate_password_hash(request.form['password']),role,now(),now())); audit(c,cur.lastrowid,'registration','user',cur.lastrowid)
   flash('Account created. Please sign in.','success'); return redirect(url_for('login'))
  except Exception: flash('Username or email already exists.','error')
 return render_template('register.html',user=viewer())
@app.route('/login',methods=['GET','POST'])
def login():
 if request.method=='POST':
  with connect() as c:
   row=c.execute('SELECT * FROM users WHERE email=?',(request.form['email'].lower().strip(),)).fetchone()
   if row and check_password_hash(row['password_hash'],request.form['password']): session.update(user_id=row['user_id'],username=row['username'],role=row['role']); audit(c,row['user_id'],'login','user',row['user_id']); return redirect(url_for('index'))
   audit(c,None,'failed_login',details=request.form.get('email')); flash('Invalid email or password.','error')
 return render_template('login.html',user=viewer())
@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('index'))
@app.route('/profile',methods=['GET','POST'])
@login_required
def profile():
 if request.method=='POST':
  with connect() as c:
   c.execute('UPDATE users SET username=?,display_name=?,updated_at=? WHERE user_id=?',(request.form['username'].strip(),request.form.get('display_name','').strip(),now(),session['user_id']))
   audit(c,session['user_id'],'profile_update','user',session['user_id'])
  session['username']=request.form['username'].strip(); flash('Profile updated.','success'); return redirect(url_for('profile'))
 with connect() as c: row=c.execute('SELECT * FROM users WHERE user_id=?',(session['user_id'],)).fetchone()
 return render_template('profile.html',account=row,user=viewer())
@app.route('/upload',methods=['GET','POST'])
@login_required
def upload():
 if request.method=='POST':
  outcomes=[]
  for item in request.files.getlist('audio'):
   try:
    records=analyze_file(item); did,result=records[0]; outcomes.append((item.filename,did,result))
   except Exception as e: outcomes.append((item.filename,None,{'error':str(e)}))
  return render_template('upload.html',outcomes=outcomes,user=viewer())
 return render_template('upload.html',outcomes=None,user=viewer())
@app.route('/detection/<int:did>')
@login_required
def detection(did):
 with connect() as c: row=c.execute('SELECT d.*,a.filename,a.stored_path FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE d.detection_id=?',(did,)).fetchone()
 if not row: abort(404)
 item=dict(row); item['python_scores']=json.loads(item['python_scores'] or '{}'); item['gtm_scores']=json.loads(item['gtm_scores'] or '{}'); return render_template('details.html',item=item,display=DISPLAY_NAMES,user=viewer())
@app.route('/visual/<int:did>/<kind>.png')
@login_required
def visual(did,kind):
 if kind not in {'waveform','spectrogram'}: abort(404)
 with connect() as c: row=c.execute('SELECT a.audio_id,a.stored_path FROM audio_files a JOIN detections d ON d.audio_id=a.audio_id WHERE d.detection_id=?',(did,)).fetchone()
 if not row: abort(404)
 y,sr=load_audio(row['stored_path']); files=make_images(y,sr,row['audio_id'],ROOT/'static'/'generated')
 return send_from_directory(files[0].parent,files[0 if kind=='waveform' else 1].name)
@app.route('/report/<int:did>')
@login_required
def report(did):
 with connect() as c: row=c.execute('SELECT d.*,a.filename FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE d.detection_id=?',(did,)).fetchone(); audit(c,session['user_id'],'report','detection',did)
 if not row: abort(404)
 item=dict(row); path=write_report(item,ROOT/'reports'); return send_file(path,as_attachment=True)
@app.route('/audio/<int:did>')
@login_required
def audio_file(did):
 with connect() as c: row=c.execute('SELECT stored_path FROM audio_files a JOIN detections d ON d.audio_id=a.audio_id WHERE d.detection_id=?',(did,)).fetchone()
 if not row: abort(404)
 path=Path(row['stored_path']); return send_from_directory(path.parent,path.name,as_attachment=False)
@app.route('/history')
@login_required
def history():
 q=request.args.get('q',''); sev=request.args.get('severity',''); sql='SELECT d.*,a.filename FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE (a.filename LIKE ? OR d.final_class LIKE ?)'; p=[f'%{q}%',f'%{q}%']
 if sev: sql+=' AND d.severity=?'; p.append(sev)
 with connect() as c: rows=c.execute(sql+' ORDER BY d.created_at DESC',p).fetchall()
 return render_template('history.html',rows=rows,q=q,severity=sev,display=DISPLAY_NAMES,user=viewer())
@app.route('/dashboard')
@login_required
def dashboard():
 with connect() as c:
  summary=c.execute("SELECT COUNT(*) total, SUM(severity IN ('High','Critical')) critical, ROUND(AVG(CASE WHEN python_scores!='{}' THEN 1 END)*100,1) model_ready FROM detections").fetchone()
  categories=c.execute('SELECT final_class,COUNT(*) count FROM detections GROUP BY final_class ORDER BY count DESC').fetchall()
  quality=c.execute('SELECT quality,COUNT(*) count FROM detections GROUP BY quality').fetchall()
  recent=c.execute('SELECT * FROM detections ORDER BY created_at DESC LIMIT 15').fetchall()
 return render_template('dashboard.html',summary=summary,categories=categories,quality=quality,recent=recent,display=DISPLAY_NAMES,user=viewer())
@app.route('/metrics')
@login_required
def metrics():
 path=ROOT/'python_models'/f'{__import__("config.settings",fromlist=["ACTIVE_PYTHON_MODEL"]).ACTIVE_PYTHON_MODEL}_metrics.json'
 data=json.loads(path.read_text()) if path.exists() else None
 return render_template('metrics.html',data=data,user=viewer())
@app.route('/review',methods=['GET','POST'])
@login_required
@roles('reviewer','operator','admin')
def review():
 if request.method=='POST':
  with connect() as c: c.execute("UPDATE detections SET reviewer_decision=?,reviewer_comment=?,reviewed_by=?,reviewed_at=?,final_class=?,alert_status='Reviewed' WHERE detection_id=?",(request.form['decision'],request.form.get('comment'),session['user_id'],now(),request.form['decision'],request.form['detection_id'])); audit(c,session['user_id'],'override','detection',request.form['detection_id'])
  return redirect(url_for('review'))
 with connect() as c: rows=c.execute('SELECT d.*,a.filename FROM detections d JOIN audio_files a ON a.audio_id=d.audio_id WHERE d.manual_review=1 AND d.reviewer_decision IS NULL ORDER BY d.created_at').fetchall()
 return render_template('review.html',rows=rows,display=DISPLAY_NAMES,user=viewer())
@app.post('/alert/<int:did>/<action>')
@login_required
@roles('reviewer','operator','maintenance','admin')
def alert_action(did,action):
 if action not in {'Acknowledged','Dismissed','Escalated'}: abort(400)
 with connect() as c:
  c.execute('UPDATE detections SET alert_status=? WHERE detection_id=?',(action,did)); audit(c,session['user_id'],action.lower(),'detection',did)
 flash(f'Alert {action.lower()}.','success'); return redirect(request.referrer or url_for('dashboard'))
@app.route('/admin/settings',methods=['GET','POST'])
@login_required
@roles('admin')
def admin_settings():
 path=ROOT/'alert_rules'/'default.json'; rules=json.loads(path.read_text())
 if request.method=='POST':
  defaults=rules['defaults']
  for key, cast in {'minimum_confidence':float,'top_two_margin':float,'repeat_windows':int,'retention_days':int,'near_duplicate_distance':int}.items(): defaults[key]=cast(request.form[key])
  path.write_text(json.dumps(rules,indent=2),encoding='utf-8')
  with connect() as c: audit(c,session['user_id'],'model_update','alert_rules',details='Updated alert thresholds and retention settings')
  flash('Configuration saved.','success'); return redirect(url_for('admin_settings'))
 return render_template('settings.html',rules=rules,user=viewer())
@app.post('/admin/retention')
@login_required
@roles('admin')
def retention_cleanup():
 rules=json.loads((ROOT/'alert_rules'/'default.json').read_text())
 cutoff=(datetime.now(timezone.utc)-timedelta(days=int(rules['defaults']['retention_days']))).isoformat()
 with connect() as c:
  rows=c.execute('SELECT audio_id,stored_path FROM audio_files WHERE created_at < ?',(cutoff,)).fetchall()
  for row in rows:
   path=Path(row['stored_path'])
   if path.parent.resolve()==UPLOAD_DIR.resolve() and path.exists(): path.unlink()
   c.execute('DELETE FROM audio_files WHERE audio_id=?',(row['audio_id'],))
  audit(c,session['user_id'],'retention_cleanup','audio_file',details=f'Removed {len(rows)} expired records')
 flash(f'Removed {len(rows)} expired audio records.','success'); return redirect(url_for('admin_settings'))
@app.route('/export.csv')
@login_required
@roles('admin')
def export_csv():
 path=Path('reports/events.csv')
 with connect() as c: rows=c.execute('SELECT * FROM detections').fetchall(); audit(c,session['user_id'],'export','report',details=str(path))
 with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['detection_id']); w.writeheader(); w.writerows(map(dict,rows))
 return send_file(path,as_attachment=True)
@app.route('/microphone')
@login_required
def microphone(): return render_template('microphone.html',user=viewer())
@app.post('/microphone/window')
@login_required
def microphone_window():
 item=request.files.get('audio')
 if not item: return {'error':'No audio window received.'},400
 try:
  did,result=analyze_file(item,live=True)[0]
  return {'detection_id':did,'event':result['final_class'],'severity':result['severity'],'status':result['alert_status'],'url':url_for('detection',did=did)}
 except ValueError as e: return {'error':str(e)},400
def import_dataset_metadata():
 for path in (Path('dataset/metadata_originals.csv'),Path('audio_dataset/metadata_originals.csv')):
  if path.exists(): return True
 return False
if __name__=='__main__': init_db(); import_dataset_metadata(); app.run(debug=True)
