# =========================
# IMPORTS
# =========================
from flask import Flask, request, jsonify, render_template, redirect, session, send_file
import numpy as np
import pickle
import sqlite3
import re
import math
import os   # ✅ ADDED

# NLP
import nltk
import spacy
from textblob import TextBlob
import language_tool_python
from nltk.corpus import stopwords

# ML Models
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec

# Security
from werkzeug.security import generate_password_hash, check_password_hash

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Metrics
from sklearn.metrics import mean_squared_error, cohen_kappa_score


# =========================
# APP INIT
# =========================
app = Flask(__name__)
app.secret_key = "secret123"


# =========================
# NLP SETUP (FIXED)
# =========================
nltk.download('stopwords')

# ✅ FIX spaCy for Render
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

tool = language_tool_python.LanguageTool('en-US')
stop_words = set(stopwords.words('english'))


# =========================
# DATABASE (FIXED PATH)
# =========================
DB_PATH = os.path.join(os.getcwd(), "models.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        max_marks INTEGER DEFAULT 5
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT,
        student_answer TEXT,
        correct_answer TEXT,
        score REAL,
        similarity REAL,
        feedback TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# =========================
# HELPER FUNCTIONS
# =========================
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words_local = {'the','is','and','of','to','in','a','an','on','for','with','that'}
    return list(set([w for w in words if w not in stop_words_local and len(w) > 3]))


def preprocess(text):
    text = text.lower().strip()

    # Spell correction
    text = str(TextBlob(text).correct())

    doc = nlp(text)
    words = []

    for token in doc:
        if token.text not in stop_words and token.is_alpha:
            words.append(token.lemma_)

    return " ".join(words)


def calculate_similarity(a, b):
    emb1 = sbert.encode(a)
    emb2 = sbert.encode(b)
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return round(float(sim * 100), 2)


def calculate_rmse(true_scores, predicted_scores):
    return round(math.sqrt(mean_squared_error(true_scores, predicted_scores)), 2)


def calculate_qwk(true, pred):
    true = [round(x) for x in true]
    pred = [round(x) for x in pred]
    return round(cohen_kappa_score(true, pred, weights='quadratic'), 2)


# =========================
# LOAD MODELS
# =========================
sbert = SentenceTransformer('all-MiniLM-L6-v2')
doc2vec_model = Doc2Vec.load("doc2vec.model")

with open("model.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)


def get_features(student_answer, correct_answer):
    combined = correct_answer + " " + student_answer
    clean_text = preprocess(combined)

    sbert_vec = sbert.encode(clean_text)
    doc_vec = doc2vec_model.infer_vector(clean_text.split())

    final_vec = np.concatenate([sbert_vec, doc_vec])
    final_vec = pca.transform([final_vec])

    return final_vec


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return redirect('/login')


# =========================
# AUTH
# =========================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        conn = get_db()
        cur = conn.cursor()

        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']

        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        if cur.fetchone():
            return "User already exists!"

        cur.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)",
                    (username, password, role))

        conn.commit()
        conn.close()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        conn = get_db()
        cur = conn.cursor()

        username = request.form['username']
        password = request.form['password']

        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            session['role'] = user['role']
            return redirect('/teacher' if user['role'] == "teacher" else '/student')

        return "Invalid credentials"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# =========================
# TEACHER
# =========================
@app.route('/teacher', methods=['GET', 'POST'])
def teacher():
    if 'role' not in session or session['role'] != 'teacher':
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    if request.method == 'POST':
        question = request.form['question']
        answer = request.form['answer']
        marks = request.form.get('marks', 5)

        if not question.strip():
            return "Question cannot be empty"

        cur.execute("INSERT INTO questions (question, answer, max_marks) VALUES (?,?,?)",
                    (question, answer, marks))
        conn.commit()

    cur.execute("""
    SELECT username, AVG(score) as avg_score
    FROM attempts
    GROUP BY username
    ORDER BY avg_score DESC
    """)
    leaderboard = cur.fetchall()

    cur.execute("SELECT * FROM questions")
    questions = cur.fetchall()

    conn.close()

    return render_template('teacher.html', questions=questions, leaderboard=leaderboard)


@app.route('/delete_question/<int:q_id>')
def delete_question(q_id):
    if 'role' not in session or session['role'] != 'teacher':
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM questions WHERE id=?", (q_id,))
    conn.commit()
    conn.close()

    return redirect('/teacher')


# =========================
# STUDENT
# =========================
@app.route('/student')
def student():
    if 'role' not in session or session['role'] != 'student':
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM questions")
    questions = cur.fetchall()

    conn.close()

    return render_template('student.html', questions=questions)


# =========================
# RUN
# =========================
if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port=10000)
