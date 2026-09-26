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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/player')
def player():
    return render_template('player.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/waveform')
def waveform():
    return render_template('waveform.html')

@app.route('/microphone')
def microphone():
    return render_template('microphone.html')

@app.route('/about')
def about():
    return render_template('about.html')


# ================= AUTHENTICATION ROUTES =================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Success: Welcome back!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Error: Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='scrypt')
        new_user = User(username=username, email=email, password_hash=hashed_password, role='user')
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
