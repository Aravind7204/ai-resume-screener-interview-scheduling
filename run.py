from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import os
import docx2txt
from sentence_transformers import SentenceTransformer, util
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import unicodedata
import PyPDF2
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from google import genai
from google.genai import types
import json
import time
from datetime import datetime, timedelta
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
import smtplib
from email.message import EmailMessage

# --- FLASK APP INITIALIZATION & CONFIG ---
app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_if_not_set')
app.config['SECURITY_PASSWORD_SALT'] = os.environ.get('SECURITY_PASSWORD_SALT', 'default_security_salt_if_not_set')

# Email configuration (replace with your actual email server details)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME") # Your new no-reply email
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD") # The App Password for the no-reply email

s = URLSafeTimedSerializer(app.config['SECRET_KEY'] + app.config['SECURITY_PASSWORD_SALT'])

UPLOAD_FOLDER = 'uploads'
DATABASE = 'database.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GEMINI API INTEGRATION & HELPERS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    print("GEMINI_API_KEY environment variable not set. Gemini client will not be initialized.")
    client = None
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}. Check your GEMINI_API_KEY.")
        client = None

# --- DATABASE INITIALIZATION (UPDATED FOR NEW FEATURES) ---
def init_db():
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL, user_type TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, description TEXT NOT NULL,
                hr_username TEXT NOT NULL, interview_date TEXT NOT NULL, interview_time TEXT,
                results_published INTEGER NOT NULL DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT, interview_id INTEGER NOT NULL, candidate_username TEXT NOT NULL,
                resume_filename TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Applied',
                raw_feedback TEXT, final_feedback TEXT,
                technical_skills TEXT, behavioral_skills TEXT, strengths TEXT, areas_for_development TEXT, general_notes TEXT, score REAL DEFAULT 0.0,
                FOREIGN KEY (interview_id) REFERENCES interviews (id)
            )
        ''')

        # Add new columns to applications table if they don't exist
        cursor.execute("PRAGMA table_info(applications)")
        columns = [info[1] for info in cursor.fetchall()]

        if 'technical_skills' not in columns:
            cursor.execute("ALTER TABLE applications ADD COLUMN technical_skills TEXT")
        if 'behavioral_skills' not in columns:
            cursor.execute("ALTER TABLE applications ADD COLUMN behavioral_skills TEXT")
        if 'strengths' not in columns:
            cursor.execute("ALTER TABLE applications ADD COLUMN strengths TEXT")
        if 'areas_for_development' not in columns:
            cursor.execute("ALTER TABLE applications ADD COLUMN areas_for_development TEXT")
        if 'general_notes' not in columns:
            cursor.execute("ALTER TABLE applications ADD COLUMN general_notes TEXT")
        if 'score' not in columns:
            cursor.execute("ALTER TABLE applications ADD COLUMN score REAL DEFAULT 0.0")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_invitations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, interview_id INTEGER NOT NULL, candidate_email TEXT NOT NULL,
                FOREIGN KEY (interview_id) REFERENCES interviews (id)
            )
        ''')
        db.commit()
        db.close()

def delete_old_interviews():
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        fifty_days_ago = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
        
        cursor.execute("SELECT id FROM interviews WHERE interview_date < ?", (fifty_days_ago,))
        old_interview_ids = [row[0] for row in cursor.fetchall()]

        for interview_id in old_interview_ids:
            try:
                # Delete associated applications
                cursor.execute("DELETE FROM applications WHERE interview_id = ?", (interview_id,))
                # Delete associated invitations
                cursor.execute("DELETE FROM interview_invitations WHERE interview_id = ?", (interview_id,))
                # Delete the interview itself
                cursor.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))

                # Delete associated upload folder
                interview_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(interview_id))
                if os.path.exists(interview_upload_folder):
                    import shutil
                    shutil.rmtree(interview_upload_folder)
                    print(f"Automatically deleted upload folder for interview {interview_id}")
                print(f"Automatically deleted old interview {interview_id} and its data.")
            except Exception as e:
                print(f"Error during automatic deletion of interview {interview_id}: {e}")
        db.commit()
        db.close()

def is_password_strong(password):
    if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"[a-z]", password) or not re.search(r"\d", password) or not re.search(r"[!@#$%^&*()_+-=\[\]{};':\"\\|,.<>\/?]", password):
        return False
    return True

def is_email_valid(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

model = SentenceTransformer("all-MiniLM-L6-v2")
spell = SpellChecker()
COMMON_TECH_TERMS = {'js', 'dev', 'github', 'linkedin', 'firebase', 'wireframe', 'wireframes', 'signups', 'efficiency'}
spell.word_frequency.load_words(COMMON_TECH_TERMS)
STOPWORDS = set(stopwords.words('english') + ["and", "the", "with", "for", "from", "that", "this", "your", "you", "a", "an", "in", "on", "to", "of", "or", "is", "are", "as", "be", "by", "at", "it", "using", "use", "also", "will", "can", "has", "have", "if", "must", "should", "etc", "e.g.", "i.e.", "role", "team", "years", "experience", "required", "preferred", "ability", "work"])
TECH_SKILLS = {"python", "java", "sql", "aws", "azure", "docker", "kubernetes", "react", "angular", "node js", "machine learning", "data science", "project management", "agile", "scrum", "excel", "git", "linux", "tensorflow", "pytorch", "javascript", "html", "css", "mongodb", "database", "backend", "frontend", "api", "rest", "ci/cd", "devops", "php", "html5", "jquery", "web developer", "analytical skills", "php mvc", "mvc"}
GENERIC_FILLERS = ["job description", "permanent category", "time permanent", "industry type", "traits looking", "code proud", "consulting department", "full time", "graduate post", "following traits", "technologies including", "craft writes", "understanding mvc", "basic knowledge", "in-depth knowledge", "including html5", "proud hit", "skillsjqueryfront endweb", "logical analytical", "developer php"]

def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"PyPDF2 read error: {e}")
    return text.strip()

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_text(file):
    if not file or file.filename == "":
        return ""
    filename = secure_filename(file.filename)
    temp_dir = os.path.join(app.root_path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, filename)
    file.save(filepath)
    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            text = read_pdf(filepath)
        elif filename.lower().endswith(".txt"):
            text = read_txt(filepath)
        elif filename.lower().endswith(".docx"):
            text = docx2txt.process(filepath)
    except Exception as e:
        print(f"File extraction error: {e}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    if text:
        text = unicodedata.normalize('NFKC', text).replace('\xa0', ' ').strip()
    return text.strip()

def extract_keywords(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\-\/]", " ", text)
    words = word_tokenize(text)
    filtered_words = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    keywords = set(filtered_words)
    bigrams = ngrams(filtered_words, 2)
    keywords.update([" ".join(g) for g in bigrams])
    return keywords

def compute_keyword_match(resume_text, jd_text):
    resume_words = extract_keywords(resume_text)
    jd_words = extract_keywords(jd_text)
    if not jd_words:
        return 0.0
    match_count = len(resume_words & jd_words)
    return round((match_count / len(jd_words)) * 100, 2)

def compute_ats_score(resume_text, jd_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(resume_emb, jd_emb).item() * 100
    keyword_score = compute_keyword_match(resume_text, jd_text)
    final_score = round(0.7 * semantic_sim + 0.3 * keyword_score, 2)
    return min(max(final_score, 0), 100), semantic_sim

def check_spelling(resume_text):
    words = re.findall(r'\b\w+\b', resume_text.lower())
    misspelled = spell.unknown(words)
    original_words = re.findall(r'\b\w+\b', resume_text)
    proper_nouns = {w.lower() for w in original_words if w and w[0].isupper()}
    return list(misspelled - proper_nouns)[:10]

def calculate_interview_score(technical_skills, behavioral_skills, strengths, areas_for_development, general_notes):
    if not client:
        print("Gemini client not initialized, cannot calculate AI score.")
        return 0.0 # Only return score

    # Construct a detailed prompt for Gemini to evaluate the candidate
    prompt = f"""
    As an expert HR professional, evaluate the following interview feedback for a candidate.
    Provide a bias-free assessment and assign an overall score from 0 to 100.

    Feedback details:
    - Technical Skills & Competencies: {technical_skills}
    - Behavioral & Soft Skills: {behavioral_skills}
    - Candidate Strengths: {strengths}
    - Areas for Development: {areas_for_development}
    - General Interview Notes (Overall Impression): {general_notes}

    Consider all aspects of the feedback to provide a fair and comprehensive score.
    The response should be in JSON format with a 'score' (float).
    Example: {{\"score\": 85.5}}
    """

    output_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "score": types.Schema(type=types.Type.NUMBER),
        },
        required=["score"]
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=output_schema, 
                temperature=0.2
            )
        )
        data = json.loads(response.text)
        score = min(max(float(data.get('score', 0.0)), 0.0), 100.0) # Ensure score is between 0 and 100
        return score
    except Exception as e:
        print(f"Error calculating AI interview score with Gemini: {e}")
        return 0.0

def generate_suggestions_gemini(resume_text, jd_text, ats_score, semantic_score):
    if not client:
        return ["⚠️ Gemini API is not initialized. Please check your GEMINI_API_KEY variable."]
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)
    raw_missing_keywords = [word for word in jd_keywords if word not in resume_keywords]
    missing_skills = [word for word in raw_missing_keywords if (word in TECH_SKILLS or len(word.split()) > 1) and word not in GENERIC_FILLERS]
    misspelled = check_spelling(resume_text)
    prompt = f"""
    Analyze the RESUME content against the JOB DESCRIPTION.
    - Final ATS Match Score: {ats_score:.1f}%
    - Semantic Match Score: {semantic_score:.1f}%
    - Missing Core Keywords: {', '.join(missing_skills)}
    - Potential Typos: {', '.join(misspelled)}
    RESUME (Snippet):\n---\n{resume_text[:1500]}\n---\nJOB DESCRIPTION (Snippet):\n---\n{jd_text[:1500]}\n---\n
    Provide actionable feedback in three categories as a list of clear, concise bullet points (3-4 points max per category).
    """
    output_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "ContextualFit": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "KeywordDensity": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "ClarityAndFormat": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
        },
        required=["ContextualFit", "KeywordDensity", "ClarityAndFormat"]
    )
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=output_schema, temperature=0.15
            )
        )
        data = json.loads(response.text)
        suggestions = [f"🎉 **Gemini AI Analysis (Score: {ats_score:.1f}%)**", "--- Contextual Fit ---"]
        suggestions.extend(data.get("ContextualFit", []))
        suggestions.append("--- Keyword Density ---")
        suggestions.extend(data.get("KeywordDensity", []))
        suggestions.append("--- Clarity & Typos ---")
        suggestions.extend(data.get("ClarityAndFormat", []))
        if ats_score < 60:
            suggestions.append("🚨 **Action Required:** Your score is low. Focus on Keyword Density fixes.")
        elif ats_score >= 80:
            suggestions.append("✅ **Excellent:** Your resume is ready to submit!")
        return suggestions
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return [f"❌ Gemini API failed. Error: {str(e)[:100]}...", f"Static Feedback (Score: {ats_score:.1f}%): Focus on terms like: {', '.join(missing_skills[:5])}."]

def generate_rejection_feedback(reasons):
    if not client:
        return "Thank you for your interest. After careful consideration, we have decided not to move forward with your application at this time."
    prompt = f"""
    Act as a compassionate but professional HR manager. A candidate has been rejected for the following reasons:
    - {', '.join(reasons)}
    Based ONLY on the reasons provided, write a single, professional, and encouraging feedback paragraph for the candidate.
    Do not be overly negative. Focus on areas for growth. Start with "Thank you for taking the time to interview with us."
    Do not invent new reasons for the rejection. Keep it concise.
    """
    try:
        response = client.generate_text(prompt=prompt, temperature=0.5)
        return response.result.strip()
    except Exception as e:
        print(f"Gemini feedback generation failed: {e}")
        feedback = "Thank you for your interest. After careful consideration, we have decided not to move forward. "
        if reasons:
            feedback += f"We recommend focusing on strengthening skills related to: {', '.join(reasons)}."
        return feedback

# --- DECORATORS ---
def login_required(role="any"):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_type' not in session:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for('dashboard'))
            if role != "any" and session['user_type'] != role:
                flash("You do not have permission to access this page.", "error")
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- MAIN & AUTH ROUTES ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        user_type = request.form.get("user_type")
        if not user_type:
            flash("You must select a role (HR or Candidate) to register.", "error")
            return redirect(url_for('register', role=request.form.get('default_role', 'candidate')))
        if not is_email_valid(email) or not is_password_strong(password):
            flash("Invalid email or weak password.", "error")
            return redirect(url_for('register', role=user_type))
        password_hash = generate_password_hash(password)
        try:
            db = sqlite3.connect(DATABASE)
            cursor = db.cursor()
            cursor.execute("INSERT INTO users (username, email, password_hash, user_type) VALUES (?, ?, ?, ?)", (username, email, password_hash, user_type))
            db.commit()
            flash("Registration successful! Please log in.", "success")
        except sqlite3.IntegrityError:
            flash("Email address already registered.", "error")
        finally:
            db.close()
        return redirect(url_for('dashboard'))
    default_role = request.args.get('role', 'candidate')
    return render_template("register.html", default_role=default_role)

@app.route("/dashboard")
def dashboard():
    if 'user_type' in session:
        return redirect(url_for(f"{session['user_type']}_dashboard"))
    return render_template("dashboard.html")

def handle_login(user_type):
    email = request.form.get("email")
    password = request.form.get("password")
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("SELECT password_hash, username FROM users WHERE email = ? AND user_type = ?", (email, user_type))
    user_record = cursor.fetchone()
    db.close()
    if user_record and check_password_hash(user_record[0], password):
        session['user_type'] = user_type
        session['username'] = user_record[1]
        session['email'] = email
        flash("Successfully logged in.", "success")
        return redirect(url_for(f'{user_type}_dashboard'))
    else:
        flash("Invalid credentials or user type.", "error")
        return redirect(url_for('dashboard'))

@app.route("/login/hr", methods=["POST"])
def login_hr():
    return handle_login('hr')

@app.route("/login/candidate", methods=["POST"])
def login_candidate():
    return handle_login('candidate')

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute("SELECT email, user_type FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        db.close()

        if user:
            token = s.dumps(email, salt='password-reset-salt')
            reset_link = url_for('reset_password', token=token, _external=True)

            msg = EmailMessage()
            msg['Subject'] = 'Password Reset Request'
            msg['From'] = app.config['MAIL_USERNAME']
            msg['To'] = email
            msg.set_content(f"""
                Hello,

                You have requested a password reset for your AI Resume Screener account.
                Please click on the following link to reset your password: {reset_link}

                If you did not request this, please ignore this email.

                Thanks,
                AI Resume Screener Team
                """)

            try:
                with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as smtp:
                    smtp.starttls()
                    smtp.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
                    smtp.send_message(msg)
                flash("A password reset link has been sent to your email address.", "info")
                print(f"Password reset link sent to {email}: {reset_link}") # For debugging if email fails
            except Exception as e:
                flash("Error sending password reset email. Please check server configuration.", "error")
                print(f"Email sending failed: {e}")
        else:
            flash("Email address not found.", "error")

        return redirect(url_for('forgot_password'))
    return render_template("forgot_password.html")

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)  # Token valid for 1 hour
    except SignatureExpired:
        flash("The password reset link has expired.", "error")
        return redirect(url_for('forgot_password', token=token))
    except BadTimeSignature:
        flash("Invalid password reset token.", "error")
        return redirect(url_for('forgot_password', token=token))

    if request.method == "POST":
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")

        if not is_password_strong(new_password):
            flash("New password is weak. Please use a strong password (8+ chars, uppercase, lowercase, digit, special char).", "error")
            return render_template("forgot_password.html", token=token)

        if new_password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("forgot_password.html", token=token)

        password_hash = generate_password_hash(new_password)
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", (password_hash, email))
        db.commit()
        db.close()

        flash("Your password has been reset successfully! Please log in.", "success")
        return redirect(url_for('dashboard'))

    return render_template("forgot_password.html", token=token)

# --- ANALYZER & ATS ROUTES ---
@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    if request.method == "POST":
        resume_file = request.files.get("resume")
        jd_text_input = request.form.get("jobdesc_text", "").strip()
        jd_file = request.files.get("jobdesc")
        resume_text = extract_text(resume_file)
        jd_text = extract_text(jd_file) if jd_file and jd_file.filename else jd_text_input
        if not resume_text or not jd_text:
            session['warning'] = "⚠️ Resume or Job Description could not be extracted."
            return redirect(url_for('analyzer'))
        ats_score, semantic_score = compute_ats_score(resume_text, jd_text)
        suggestions = generate_suggestions_gemini(resume_text, jd_text, ats_score, semantic_score)
        session['result'] = {'ats_score': ats_score, 'suggestions': suggestions}
        return redirect(url_for('analyzer'))
    results = session.pop('result', None)
    warning = session.pop('warning', None)
    return render_template("analyzer.html", result=results, warning=warning)

@app.route("/ats", methods=["GET", "POST"])
def ats():
    if request.method == "POST":
        jd_text_input = request.form.get("jobdesc_text", "").strip()
        jd_file = request.files.get("job_description")
        resume_files = request.files.getlist("resumes")
        jd_text = extract_text(jd_file) if jd_file and jd_file.filename else jd_text_input
        if not jd_text:
            session['warning'] = "⚠️ Job Description must be provided."
            return redirect(url_for('ats'))
        results = []
        if not resume_files or all(f.filename == '' for f in resume_files):
            session['warning'] = "⚠️ No resume files were uploaded."
            return redirect(url_for('ats'))
        for resume_file in resume_files:
            if resume_file and resume_file.filename != '':
                resume_text = extract_text(resume_file)
                if resume_text:
                    ats_score, semantic_score = compute_ats_score(resume_text, jd_text)
                    results.append({'filename': resume_file.filename, 'score': ats_score, 'semantic_score': semantic_score})
        results.sort(key=lambda x: x['score'], reverse=True)
        if not results:
            session['warning'] = "⚠️ Could not extract text from any uploaded resume."
            return redirect(url_for('ats'))
        session['shortlist_results'] = results
        return redirect(url_for('ats'))
    results = session.pop('shortlist_results', None)
    warning = session.pop('warning', None)
    return render_template("ats.html", results=results, warning=warning)

# --- DASHBOARD & INTERVIEW ROUTES ---
@app.route("/hr/dashboard")
@login_required(role='hr')
def hr_dashboard():
    return render_template("hr_dashboard.html")

@app.route("/candidate/dashboard")
@login_required(role='candidate')
def candidate_dashboard():
    return render_template("candidate_dashboard.html")

@app.route("/hr/schedule", methods=["GET", "POST"])
@login_required(role='hr')
def schedule_interview():
    if request.method == "POST":
        title = request.form.get("title")
        description = request.form.get("description")
        interview_date = request.form.get("interview_date")
        interview_time = request.form.get("interview_time")
        candidate_emails_raw = request.form.get("candidate_emails")
        if not all([title, description, interview_date, candidate_emails_raw]):
            flash("Please fill all required fields.", "error")
            return redirect(url_for('schedule_interview'))
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute("INSERT INTO interviews (title, description, hr_username, interview_date, interview_time) VALUES (?, ?, ?, ?, ?)",
                       (title, description, session['username'], interview_date, interview_time))
        db.commit()
        interview_id = cursor.lastrowid
        invited_emails = [email.strip() for email in candidate_emails_raw.splitlines() if email.strip()]
        for email in invited_emails:
            if is_email_valid(email):
                cursor.execute("INSERT INTO interview_invitations (interview_id, candidate_email) VALUES (?, ?)", (interview_id, email))
        db.commit()
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], str(interview_id)), exist_ok=True)
        db.close()
        flash(f"Interview scheduled successfully! Invited {len(invited_emails)} candidate(s).", "success")
        return redirect(url_for('view_hr_interviews'))
    return render_template("schedule_interview.html")

@app.route("/hr/interviews")
@login_required(role='hr')
def view_hr_interviews():
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("""
        SELECT i.id, i.title, i.hr_username, i.interview_date, i.interview_time, COUNT(a.id)
        FROM interviews i LEFT JOIN applications a ON i.id = a.interview_id
        WHERE i.hr_username = ? GROUP BY i.id, i.title, i.hr_username, i.interview_date, i.interview_time
        ORDER BY i.interview_date DESC
    """, (session['username'],))
    interviews_data = cursor.fetchall()
    db.close()
    hr_interviews = [{'id': r[0], 'title': r[1], 'created_by': r[2], 'interview_date': r[3], 'interview_time': r[4], 'applicant_count': r[5]} for r in interviews_data]
    return render_template("view_hr_interviews.html", interviews=hr_interviews)

@app.route("/hr/interview/delete/<int:interview_id>", methods=["POST"])
@login_required(role='hr')
def delete_interview(interview_id):
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    try:
        # Verify the HR user owns this interview
        cursor.execute("SELECT id FROM interviews WHERE id = ? AND hr_username = ?", (interview_id, session['username']))
        if not cursor.fetchone():
            flash("Interview not found or you do not have permission to delete it.", "error")
            return redirect(url_for('view_hr_interviews'))

        # Delete associated applications
        cursor.execute("DELETE FROM applications WHERE interview_id = ?", (interview_id,))
        # Delete associated invitations
        cursor.execute("DELETE FROM interview_invitations WHERE interview_id = ?", (interview_id,))
        # Delete the interview itself
        cursor.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
        db.commit()

        # Delete associated upload folder
        interview_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(interview_id))
        if os.path.exists(interview_upload_folder):
            import shutil
            shutil.rmtree(interview_upload_folder)
            print(f"Deleted upload folder: {interview_upload_folder}")

        flash("Interview and all associated data successfully deleted.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error deleting interview: {e}", "error")
        print(f"Error deleting interview {interview_id}: {e}")
    finally:
        db.close()
    return redirect(url_for('view_hr_interviews'))

@app.route("/candidate/interviews")
@login_required(role='candidate')
def view_candidate_interviews():
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("SELECT interview_id, status, final_feedback FROM applications WHERE candidate_username = ?", (session['username'],))
    applications = {row[0]: {'status': row[1], 'feedback': row[2]} for row in cursor.fetchall()}
    cursor.execute("""
        SELECT i.id, i.title, i.description, i.hr_username, i.interview_date, i.interview_time, i.results_published
        FROM interviews i JOIN interview_invitations inv ON i.id = inv.interview_id
        WHERE inv.candidate_email = ? ORDER BY i.interview_date DESC
    """, (session['email'],))
    interviews_data = cursor.fetchall()
    db.close()
    all_interviews = []
    for row in interviews_data:
        interview_id = row[0]
        app_details = applications.get(interview_id)
        all_interviews.append({
            'id': interview_id, 'title': row[1], 'description': row[2], 'created_by': row[3],
            'interview_date': row[4], 'interview_time': row[5], 'results_published': row[6],
            'application': app_details
        })
    return render_template("view_candidate_interviews.html", interviews=all_interviews)

@app.route("/application_details/<int:interview_id>")
@login_required(role='candidate')
def application_details(interview_id):
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("SELECT id, title, description, hr_username, interview_date, interview_time, results_published FROM interviews WHERE id = ?", (interview_id,))
    interview_data = cursor.fetchone()
    if not interview_data:
        db.close()
        flash("Interview not found.", "error")
        return redirect(url_for('candidate_dashboard'))
    interview = {
        'id': interview_data[0], 'title': interview_data[1], 'description': interview_data[2],
        'created_by': interview_data[3], 'interview_date': interview_data[4], 'interview_time': interview_data[5],
        'results_published': interview_data[6]
    }
    cursor.execute("SELECT id, status, final_feedback, score FROM applications WHERE interview_id = ? AND candidate_username = ?", (interview_id, session['username']))
    application_data = cursor.fetchone()
    application = None
    if application_data:
        application = {'id': application_data[0], 'status': application_data[1], 'feedback': application_data[2], 'score': application_data[3]}
    db.close()
    return render_template("application_details.html", interview=interview, application=application)

@app.route("/apply_for_interview/<int:interview_id>", methods=["POST"])
@login_required(role='candidate')
def apply_for_interview(interview_id):
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    # Check if the interview exists
    cursor.execute("SELECT id FROM interviews WHERE id = ?", (interview_id,))
    if not cursor.fetchone():
        db.close()
        flash("Interview not found.", "error")
        return redirect(url_for('candidate_dashboard'))

    # Check if candidate has already applied
    cursor.execute("SELECT id FROM applications WHERE interview_id = ? AND candidate_username = ?", (interview_id, session['username']))
    if cursor.fetchone() is not None:
        db.close()
        flash("You have already submitted an application for this role.", "warning")
        return redirect(url_for('application_details', interview_id=interview_id))

    file = request.files.get('resume')
    if file and file.filename:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], str(interview_id))
        os.makedirs(save_path, exist_ok=True)
        file.save(os.path.join(save_path, filename))
        cursor.execute("INSERT INTO applications (interview_id, candidate_username, resume_filename, status) VALUES (?, ?, ?, ?)",
                       (interview_id, session['username'], filename, 'Applied'))
        db.commit()
        db.close()
        flash("Successfully applied!", "success")
    else:
        flash('No file selected. Please upload your resume.', 'error')
        db.close()
        return redirect(url_for('application_details', interview_id=interview_id))

    return redirect(url_for('application_details', interview_id=interview_id))

@app.route("/interview/<int:interview_id>")
@login_required(role='hr')
def interview_details(interview_id):
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("SELECT title, description, results_published FROM interviews WHERE id = ? AND hr_username = ?", (interview_id, session['username']))
    interview_data = cursor.fetchone()
    if not interview_data:
        db.close(); flash("Interview not found or you do not have permission.", "error"); return redirect(url_for('hr_dashboard'))
    interview = {'id': interview_id, 'title': interview_data[0], 'description': interview_data[1], 'results_published': interview_data[2]}
    cursor.execute("SELECT candidate_username, resume_filename, status, score FROM applications WHERE interview_id = ? ORDER BY candidate_username", (interview_id,))
    applicants_data = cursor.fetchall()
    db.close()
    applicants = [{'candidate_username': r[0], 'resume_filename': r[1], 'status': r[2], 'score': r[3]} for r in applicants_data]

    interview_completed = any(app['status'] != 'Applied' for app in applicants)
    waiting_list_candidates = [app for app in applicants if app['status'] == 'Waiting List']
    return render_template("interview_details.html", interview=interview, applicants=applicants, interview_completed=interview_completed, waiting_list_candidates=waiting_list_candidates)

@app.route("/interview/<int:interview_id>/start/<candidate_username>")
@login_required(role='hr')
def live_interview(interview_id, candidate_username):
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("SELECT title FROM interviews WHERE id = ? AND hr_username = ?", (interview_id, session['username']))
    interview_data = cursor.fetchone()
    if not interview_data:
        db.close(); flash("Interview not found or you do not have permission.", "error"); return redirect(url_for('hr_dashboard'))
    cursor.execute("SELECT resume_filename FROM applications WHERE interview_id = ? AND candidate_username = ?", (interview_id, candidate_username))
    application_data = cursor.fetchone()
    if not application_data:
        db.close(); flash("Applicant not found for this interview.", "error"); return redirect(url_for('interview_details', interview_id=interview_id))
    interview_title = interview_data[0]
    resume_filename = application_data[0]

    # Fetch existing feedback and score
    cursor.execute("SELECT technical_skills, behavioral_skills, strengths, areas_for_development, general_notes, score FROM applications WHERE interview_id = ? AND candidate_username = ?", (interview_id, candidate_username))
    feedback_data = cursor.fetchone()
    
    existing_feedback = {
        'technical_skills_select': [],
        'technical_skills_extra': '',
        'behavioral_skills_select': [],
        'behavioral_skills_extra': '',
        'strengths_select': [],
        'strengths_extra': '',
        'areas_for_development_select': [],
        'areas_for_development_extra': '',
        'general_notes_select': [],
        'general_notes_extra': '',
    }

    # If the database columns contain combined JSON for select and extra notes
    if feedback_data:
        feedback_column_names = ['technical_skills', 'behavioral_skills', 'strengths', 'areas_for_development', 'general_notes']
        for i, key_prefix in enumerate(feedback_column_names):
            db_value = feedback_data[i]
            if db_value and isinstance(db_value, str) and db_value.strip().startswith('{'):
                try:
                    parsed_data = json.loads(db_value)
                    existing_feedback[f'{key_prefix}_select'] = parsed_data.get('select', [])
                    existing_feedback[f'{key_prefix}_extra'] = parsed_data.get('extra', '')
                except json.JSONDecodeError:
                    # Fallback for old data that might look like JSON but isn't complete, or corrupted JSON
                    existing_feedback[f'{key_prefix}_extra'] = db_value
            elif db_value:
                # Fallback for old data that is just a plain string
                existing_feedback[f'{key_prefix}_extra'] = db_value


    interview_score = feedback_data[5] if feedback_data else 0.0
    

    db.close()
    return render_template("live_interview.html", interview_id=interview_id, interview_title=interview_title, candidate_username=candidate_username, resume_filename=resume_filename, existing_feedback=existing_feedback, interview_score=interview_score)

@app.route("/interview/<int:interview_id>/decision/<candidate_username>", methods=["POST"])
@login_required(role='hr')
def submit_decision(interview_id, candidate_username):
    decision = request.form.get('decision')
    if not decision:
        flash("No decision was selected.", "error")
        return redirect(url_for('live_interview', interview_id=interview_id, candidate_username=candidate_username))

    # Retrieve structured feedback
    technical_skills_select = request.form.getlist('technical_skills_select')
    technical_skills_extra = request.form.get('technical_skills_extra', '')
    behavioral_skills_select = request.form.getlist('behavioral_skills_select')
    behavioral_skills_extra = request.form.get('behavioral_skills_extra', '')
    strengths_select = request.form.getlist('strengths_select')
    strengths_extra = request.form.get('strengths_extra', '')
    areas_for_development_select = request.form.getlist('areas_for_development_select')
    areas_for_development_extra = request.form.get('areas_for_development_extra', '')
    general_notes_select = request.form.getlist('general_notes_select')
    general_notes_extra = request.form.get('general_notes_extra', '')

    # Retrieve waiting list notes (if applicable)
    waiting_list_rejection_notes = request.form.get('waiting_list_rejection_notes', '')
    waiting_list_rejection_reasons = request.form.getlist('waiting_list_rejection_reasons')

    # Combine selected options and extra notes into JSON strings for database storage
    technical_skills_combined = json.dumps({'select': technical_skills_select, 'extra': technical_skills_extra})
    behavioral_skills_combined = json.dumps({'select': behavioral_skills_select, 'extra': behavioral_skills_extra})
    strengths_combined = json.dumps({'select': strengths_select, 'extra': strengths_extra})
    areas_for_development_combined = json.dumps({'select': areas_for_development_select, 'extra': areas_for_development_extra})
    general_notes_combined = json.dumps({'select': general_notes_select, 'extra': general_notes_extra})

    final_status = ""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()

    # For AI scoring, pass a consolidated string of all feedback for each category
    ai_technical_skills = ", ".join(technical_skills_select) + (f". {technical_skills_extra}" if technical_skills_extra else "")
    ai_behavioral_skills = ", ".join(behavioral_skills_select) + (f". {behavioral_skills_extra}" if behavioral_skills_extra else "")
    ai_strengths = ", ".join(strengths_select) + (f". {strengths_extra}" if strengths_extra else "")
    ai_areas_for_development = ", ".join(areas_for_development_select) + (f". {areas_for_development_extra}" if areas_for_development_extra else "")
    ai_general_notes = ", ".join(general_notes_select) + (f". {general_notes_extra}" if general_notes_extra else "")

    interview_score = calculate_interview_score(
        ai_technical_skills,
        ai_behavioral_skills,
        ai_strengths,
        ai_areas_for_development,
        ai_general_notes
    )

    if decision == 'accepted':
        final_status = 'Accepted'
        cursor.execute("UPDATE applications SET status = ?, technical_skills = ?, behavioral_skills = ?, strengths = ?, areas_for_development = ?, general_notes = ?, score = ? WHERE interview_id = ? AND candidate_username = ?",
                       (final_status, technical_skills_combined, behavioral_skills_combined, strengths_combined, areas_for_development_combined, general_notes_combined, interview_score, interview_id, candidate_username))
    elif decision == 'waiting_list':
        final_status = 'Waiting List'
        # Store waiting list notes and potential rejection notes in raw_feedback
        raw_feedback_json = json.dumps({'rejection_notes': waiting_list_rejection_notes, 'rejection_reasons': waiting_list_rejection_reasons})
        cursor.execute("UPDATE applications SET status = ?, raw_feedback = ?, technical_skills = ?, behavioral_skills = ?, strengths = ?, areas_for_development = ?, general_notes = ?, score = ? WHERE interview_id = ? AND candidate_username = ?",
                       (final_status, raw_feedback_json, technical_skills_combined, behavioral_skills_combined, strengths_combined, areas_for_development_combined, general_notes_combined, interview_score, interview_id, candidate_username))
    elif decision == 'rejected':
        final_status = 'Rejected'
        rejection_reasons = request.form.getlist('rejection_reasons')
        final_feedback_paragraph = generate_rejection_feedback(rejection_reasons)
        raw_feedback_json = json.dumps({'reasons': rejection_reasons})
        cursor.execute("UPDATE applications SET status = ?, raw_feedback = ?, final_feedback = ?, technical_skills = ?, behavioral_skills = ?, strengths = ?, areas_for_development = ?, general_notes = ?, score = ? WHERE interview_id = ? AND candidate_username = ?",
                       (final_status, raw_feedback_json, final_feedback_paragraph, technical_skills_combined, behavioral_skills_combined, strengths_combined, areas_for_development_combined, general_notes_combined, interview_score, interview_id, candidate_username))
    
    db.commit()
    db.close()
    flash(f"Decision for {candidate_username} recorded as '{final_status}'.", "success")
    return redirect(url_for('interview_details', interview_id=interview_id))

@app.route("/interview/<int:interview_id>/publish", methods=["POST"])
@login_required(role='hr')
def publish_results(interview_id):
    action = request.form.get('action', 'publish')
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    if action == 'publish':
        cursor.execute("UPDATE interviews SET results_published = 1 WHERE id = ?", (interview_id,))
        db.commit()
        db.close()
        flash("Results have been published to candidates.", "success")
        return redirect(url_for('interview_details', interview_id=interview_id))
    elif action == 'recollect':
        cursor.execute("UPDATE interviews SET results_published = 0 WHERE id = ?", (interview_id,))
        db.commit()
        db.close()
        flash("Results have been recalled.", "warning")
        return redirect(url_for('interview_details', interview_id=interview_id))
    elif action == 'reject_waiting_list':
        cursor.execute("SELECT candidate_username, raw_feedback FROM applications WHERE interview_id = ? AND status = 'Waiting List'", (interview_id,))
        waiting_list = cursor.fetchall()
        for candidate, raw_feedback_json in waiting_list:
            feedback_data = {}
            if raw_feedback_json:
                try:
                    parsed_data = json.loads(raw_feedback_json)
                    feedback_data['rejection_notes'] = parsed_data.get('rejection_notes', '')
                    
                    # If specific rejection reasons were saved, use them. Otherwise, provide a default.
                    if parsed_data.get('rejection_reasons') and isinstance(parsed_data['rejection_reasons'], list):
                        feedback_data['reasons'] = parsed_data['rejection_reasons']
                    elif parsed_data.get('reasons') and isinstance(parsed_data['reasons'], list):
                        feedback_data['reasons'] = parsed_data['reasons']
                    else:
                        feedback_data['reasons'] = ['position has been filled']

                except json.JSONDecodeError:
                    # Fallback for old or corrupted data
                    feedback_data['reasons'] = ['position has been filled']
                    feedback_data['rejection_notes'] = '' # Ensure it's defined
            else:
                feedback_data['reasons'] = ['position has been filled']
                feedback_data['rejection_notes'] = '' # Ensure it's defined
            
            # Use rejection_notes for additional_notes if available, otherwise use general notes
            notes_for_feedback = feedback_data['rejection_notes'] if feedback_data['rejection_notes'] else ''
            final_feedback = generate_rejection_feedback(feedback_data['reasons'])
            cursor.execute("UPDATE applications SET status = 'Rejected', final_feedback = ? WHERE interview_id = ? AND candidate_username = ?",
                           (final_feedback, interview_id, candidate))
        cursor.execute("UPDATE interviews SET results_published = 1 WHERE id = ?", (interview_id,))
        db.commit()
        db.close()
        flash("Waiting list candidates rejected and results published.", "success")
        return redirect(url_for('interview_details', interview_id=interview_id))
        
    db.commit()
    db.close()
    return redirect(url_for('interview_details', interview_id=interview_id))
    
@app.route('/uploads/<int:interview_id>/<filename>')
@login_required(role='hr')
def serve_resume(interview_id, filename):
    directory = os.path.join(app.config['UPLOAD_FOLDER'], str(interview_id))
    return send_from_directory(directory, filename)

if __name__ == "__main__":
    init_db()
    delete_old_interviews()
    app.run(debug=os.environ.get('FLASK_DEBUG') == 'True', host='0.0.0.0')
