# =========================
# IMPORTS
# =========================
from flask import Flask, request, jsonify, render_template, redirect, session, send_file
import numpy as np
import pickle
import sqlite3
import re
import math

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

# Metrics (kept but optional)
from sklearn.metrics import mean_squared_error, cohen_kappa_score


# =========================
# APP INIT
# =========================
app = Flask(__name__)
app.secret_key = "secret123"


# =========================
# NLP SETUP
# =========================
nltk.download('stopwords')
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')
stop_words = set(stopwords.words('english'))


# =========================
# DATABASE
# =========================
def get_db():
    conn = sqlite3.connect("models.db")
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

    # Lemmatization + stopword removal
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


# Optional metrics (kept but not required)
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

@app.route('/student_history/<username>')
def student_history(username):
    if 'role' not in session or session['role'] != 'teacher':
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT question, student_answer, score, similarity, feedback, timestamp
    FROM attempts
    WHERE username=?
    ORDER BY timestamp DESC
    """, (username,))

    data = cur.fetchall()
    conn.close()

    return render_template("student_history.html", data=data, username=username)

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        student_answer = data['answer']
        correct_answer = data['correct']
        question_text = data.get('question', 'Unknown')

        if not student_answer.strip():
            return jsonify({"error": "Answer cannot be empty"})

        # =========================
        # FEATURE EXTRACTION
        # =========================
        features = get_features(student_answer, correct_answer)

        # =========================
        # GRAMMAR CHECK
        # =========================
        grammar_errors = len(tool.check(student_answer))

        # =========================
        # KEYWORD ANALYSIS
        # =========================
        correct_keywords = extract_keywords(correct_answer)
        student_keywords = extract_keywords(student_answer)

        matched = [w for w in student_keywords if w in correct_keywords]
        missing = [w for w in correct_keywords if w not in student_keywords]

        coverage = round((len(matched) / len(correct_keywords)) * 100, 2) if correct_keywords else 0

        # =========================
        # SIMILARITY
        # =========================
        similarity = calculate_similarity(student_answer, correct_answer)

        # =========================
        # ML + HYBRID SCORING
        # =========================
        ml_score = float(ml_model.predict(features)[0])

        sim_score = (similarity / 100) * 5
        coverage_score = (coverage / 100) * 5

        score = (0.5 * ml_score) + (0.3 * sim_score) + (0.2 * coverage_score)
        score = max(0, min(5, round(score, 2)))

        # =========================
        # RULE-BASED CORRECTIONS (CRITICAL FIX)
        # =========================
        if similarity < 20:
            score = 0
            feedback = "Incorrect answer"

        elif similarity < 40:
            score = min(score, 2)
            feedback = "Low relevance answer"

        elif similarity > 80:
            feedback = "Very close to correct answer"

        elif similarity > 60:
            feedback = "Partially correct"

        else:
            feedback = "Improve key concepts"

        # =========================
        # SAVE TO DATABASE
        # =========================
        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO attempts 
        (username, question, student_answer, correct_answer, score, similarity, feedback)
        VALUES (?,?,?,?,?,?,?)
        """, (
            session['user'],
            question_text,
            student_answer,
            correct_answer,
            score,
            similarity,
            feedback
        ))

        conn.commit()
        conn.close()

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            "score": score,
            "similarity": similarity,
            "feedback": feedback,
            "matched": matched[:5],
            "missing": missing[:5],
            "coverage": coverage,
            "grammar_errors": grammar_errors
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})
# =========================
# PROGRESS
# =========================
@app.route('/my_progress')
def my_progress():
    if 'user' not in session:
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT question, score, similarity, feedback, timestamp
    FROM attempts
    WHERE username=?
    ORDER BY timestamp ASC
    """, (session['user'],))

    data = cur.fetchall()
    conn.close()

    scores = [row[1] for row in data]
    labels = list(range(1, len(scores) + 1))

    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
    accuracy = round(sum([row[2] for row in data]) / len(data), 2) if data else 0

    badge = "🏆 Top Performer" if avg_score >= 4 else "🥈 Good Learner" if avg_score >= 3 else "📘 Beginner"

    return render_template("progress.html",
                           data=data,
                           scores=scores,
                           labels=labels,
                           avg_score=avg_score,
                           accuracy=accuracy,
                           badge=badge)

@app.route('/download_report_teacher/<username>')
def download_report_teacher(username):
    if 'role' not in session or session['role'] != 'teacher':
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT question, score, similarity, feedback
    FROM attempts
    WHERE username=?
    """, (username,))

    data = cur.fetchall()
    conn.close()

    file_path = f"{username}_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = [Paragraph(f"Report for {username}", styles['Title'])]

    for row in data:
        text = f"""
        Question: {row[0]} <br/>
        Score: {row[1]} <br/>
        Similarity: {row[2]}% <br/>
        Feedback: {row[3]} <br/><br/>
        """
        content.append(Paragraph(text, styles['Normal']))

    doc.build(content)

    return send_file(file_path, as_attachment=True)

# =========================
# LEADERBOARD
# =========================
@app.route('/leaderboard')
def leaderboard():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT username, AVG(score) as avg_score
    FROM attempts
    GROUP BY username
    ORDER BY avg_score DESC
    """)

    data = cur.fetchall()
    conn.close()

    return render_template('leaderboard.html', data=data)


# =========================
# PDF REPORT
# =========================
@app.route('/download_report')
def download_report():
    if 'user' not in session:
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT question, score, similarity, feedback
    FROM attempts
    WHERE username=?
    """, (session['user'],))

    data = cur.fetchall()
    conn.close()

    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = [Paragraph("Student Performance Report", styles['Title'])]

    for row in data:
        text = f"""
        Question: {row[0]} <br/>
        Score: {row[1]} <br/>
        Similarity: {row[2]}% <br/>
        Feedback: {row[3]} <br/><br/>
        """
        content.append(Paragraph(text, styles['Normal']))

    doc.build(content)

    return send_file(file_path, as_attachment=True)


# =========================
# RUN
# =========================
if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port=10000)
