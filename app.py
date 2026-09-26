#updated
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sonic_sentinel_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# 1. USERS TABLE (Only 'users' table will be created and used)
class User(UserMixin, db.Model):
    __tablename__ = 'users'  
    
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='user')
    
    # Check constraint for valid roles
    __table_args__ = (
        db.CheckConstraint("role IN ('user', 'reviewer', 'operator', 'maintenance', 'admin')", name="check_user_role"),
    )

    # Flask-Login ke liye user_id compatibility
    def get_id(self):
        return str(self.user_id)
    
    # Relationships
    audio_files = db.relationship('AudioFile', backref='uploader', lazy=True)
    audit_logs = db.relationship('AuditLog', backref='user', lazy=True)


# 2. AUDIO_FILES TABLE
class AudioFile(db.Model):
    __tablename__ = 'audio_files'
    
    audio_id = db.Column(db.Integer, primary_key=True)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    stored_path = db.Column(db.Text, nullable=False)
    sound_category = db.Column(db.String(100), nullable=True)
    duration_s = db.Column(db.Float, nullable=False)
    sample_rate = db.Column(db.Integer, nullable=False)
    channels = db.Column(db.Integer, nullable=False)
    recording_environment = db.Column(db.String(100), nullable=True)
    recording_device = db.Column(db.String(100), nullable=True)
    source_distance = db.Column(db.String(100), nullable=True)
    is_original = db.Column(db.Integer, nullable=False, default=1)
    dataset_split = db.Column(db.String(50), nullable=True)
    file_hash = db.Column(db.String(255), unique=True, nullable=True)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    detections = db.relationship('Detection', backref='audio_file', lazy=True)


# 3. DETECTIONS TABLE
class Detection(db.Model):
    __tablename__ = 'detections'
    
    detection_id = db.Column(db.Integer, primary_key=True)
    audio_id = db.Column(db.Integer, db.ForeignKey('audio_files.audio_id'), nullable=True)
    python_class = db.Column(db.Text, nullable=False)
    python_scores = db.Column(db.Text, nullable=False)
    python_model_version = db.Column(db.Text, nullable=False)
    gtm_class = db.Column(db.Text, nullable=True)
    gtm_scores = db.Column(db.Text, nullable=True)
    gtm_model_version = db.Column(db.Text, nullable=True)
    final_class = db.Column(db.Text, nullable=False)
    severity = db.Column(db.Text, nullable=False)
    alert_status = db.Column(db.Text, nullable=False)
    reviewer_decision = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


# 4. AUDIT_LOG TABLE
class AuditLog(db.Model):
    __tablename__ = 'audit_log'
    
    audit_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=True)
    action_type = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    # Updated to db.session.get to fix LegacyAPIWarning
    return db.session.get(User, int(user_id))


# ================= PAGES ROUTES =================

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
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
# VERIFIED