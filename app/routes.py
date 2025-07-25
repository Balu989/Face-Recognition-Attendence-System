from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
import os
import sys
import subprocess



routes = Blueprint('routes', __name__)

# Define script directory (where face_recog.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = BASE_DIR  # face_train.py and face_recog.py are here


# ---------- Home ----------
@routes.route('/')
@login_required
def home():
    return render_template('index.html', user_name=current_user.name)


# ---------- Register Face ----------
@routes.route('/register', methods=['GET', 'POST'])
@login_required
def register_face():
    if request.method == 'POST':
        name = request.form['name']
        roll = request.form['roll']
        class_ = request.form['class_']
        section = request.form['section']

        subprocess.run([
            sys.executable,
            os.path.join(SCRIPT_PATH, 'face_train.py'),
            f"{name}_{roll}",
            class_,
            section
        ])

        return f"✅ Registered: {name} (Roll: {roll}, Class: {class_}, Section: {section})<br><a href='/'>Back to Home</a>"

    return render_template('register.html')


# ---------- Attendance Face Recognition ----------
@routes.route('/recognize', methods=['GET', 'POST'])
@login_required
def start_attendance():
    if request.method == 'POST':
        subject = request.form['subject']
        branch = request.form['branch']
        section = request.form['section']

        subprocess.run([
            sys.executable,
            os.path.join(SCRIPT_PATH, 'face_recog.py'),
            subject,
            branch,
            section
        ])

        return render_template(
            'success.html',
            message=f"✅ Attendance recorded for {branch.upper()}_{section.upper()} - {subject}",
            redirect_url='/'
        )

    return render_template('attendance.html')


# ---------- Download Excel ----------
@routes.route('/download_excel', methods=['POST'])
@login_required
def download_excel():
    branch = request.form['branch'].strip().lower()
    section = request.form['section'].strip().lower()
    excel_path = os.path.join('registered_data', f"{branch}_{section}", "students.xlsx")

    if os.path.exists(excel_path):
        return send_file(excel_path, as_attachment=True)
    else:
        return f"❌ File not found: {excel_path} <br><a href='/'>Back</a>"


# ---------- Evaluate Model ----------
@routes.route('/evaluate')
@login_required
def evaluate_accuracy():
    subprocess.run([sys.executable, os.path.join(SCRIPT_PATH, 'evaluate_model.py')])
    return "✅ Model evaluation complete. Check confusion_matrix.png and model_evaluation.txt. <a href='/'>Home</a>"


# ---------- Login ----------
@routes.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        email = request.form['email']
        password_input = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password_input):
            login_user(user)
            flash(f"👋 Welcome back, {user.name}!", "success")
            return redirect(url_for('routes.home'))
        else:
            flash("❌ Invalid email or password", "danger")

    return render_template('login.html')


# ---------- Signup ----------
@routes.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(email=email).first():
            flash("❌ Email already exists!", "danger")
            return redirect(url_for('routes.signup'))

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("✅ Account created! Please login.", "success")
        return redirect(url_for('routes.login'))

    return render_template('signup.html')


# ---------- Logout ----------
@routes.route('/logout')
@login_required
def logout():
    logout_user()
    flash("🚪 You have been logged out.", "info")
    return redirect(url_for('routes.login'))


# ---------- Dashboard, Students, Calendar ----------
@routes.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@routes.route('/students')
@login_required
def student_page():
    return render_template('students.html')


@routes.route('/calendar')
@login_required
def calendar_page():
    return render_template('calendar.html')
