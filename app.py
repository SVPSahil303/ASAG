from flask import Flask, request, jsonify, render_template, redirect, session
import numpy as np
import pickle
import sqlite3

from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "secret123"

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

    # USERS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # QUESTIONS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        max_marks INTEGER DEFAULT 5
    )
    """)

    # ATTEMPTS
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
# LOAD MODELS
# =========================
sbert = SentenceTransformer('all-MiniLM-L6-v2')
doc2vec_model = Doc2Vec.load("doc2vec.model")

with open("model.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# =========================
# ML FUNCTIONS
# =========================
def preprocess(text):
    return text.lower().strip()

def get_features(text):
    text = preprocess(text)

    sbert_vec = sbert.encode(text)
    doc_vec = doc2vec_model.infer_vector(text.split())

    final_vec = np.concatenate([sbert_vec, doc_vec])
    final_vec = pca.transform([final_vec])

    return final_vec

def calculate_similarity(a, b):
    emb1 = sbert.encode(a)
    emb2 = sbert.encode(b)
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return round(float(sim * 100), 2)

# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    return redirect('/login')

# =========================
# AUTH
# =========================
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        conn = get_db()
        cur = conn.cursor()

        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']

        # Prevent duplicate users
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        if cur.fetchone():
            return "User already exists!"

        cur.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)",
                    (username, password, role))

        conn.commit()
        conn.close()

        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET','POST'])
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

            if user['role'] == "teacher":
                return redirect('/teacher')
            else:
                return redirect('/student')

        return "Invalid credentials"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# =========================
# TEACHER
# =========================
@app.route('/teacher', methods=['GET','POST'])
def teacher():
    if 'role' not in session or session['role'] != 'teacher':
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    # ADD QUESTION (FIXED 🔥)
    if request.method == 'POST':
        question = request.form['question']
        answer = request.form['answer']
        marks = request.form.get('marks', 5)

        if not question.strip():
            return "Question cannot be empty"

        cur.execute("""
        INSERT INTO questions (question, answer, max_marks)
        VALUES (?,?,?)
        """, (question, answer, marks))

        conn.commit()

    # LEADERBOARD
    cur.execute("""
    SELECT username, AVG(score) as avg_score
    FROM attempts
    GROUP BY username
    ORDER BY avg_score DESC
    """)
    leaderboard = cur.fetchall()

    # QUESTIONS
    cur.execute("SELECT * FROM questions")
    questions = cur.fetchall()

    conn.close()

    return render_template('teacher.html', questions=questions, leaderboard=leaderboard)

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
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        student_answer = data['answer']
        correct_answer = data['correct']
        question_text = data.get('question', 'Unknown')

        # EMPTY CHECK
        if not student_answer or student_answer.strip() == "":
            return jsonify({
                "error": "Answer cannot be empty",
                "score": 0,
                "similarity": 0,
                "feedback": "No answer provided"
            })

        # ML
        features = get_features(student_answer)
        score = ml_model.predict(features)[0]
        score = round(float(score), 2)

        similarity = calculate_similarity(student_answer, correct_answer)

        # SMART FEEDBACK
        if similarity > 80:
            feedback = "Very close to correct answer"
        elif similarity > 60:
            feedback = "Partially correct, improve explanation"
        else:
            feedback = "Try to include key concepts"

        # SAVE
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

        return jsonify({
            "score": score,
            "similarity": similarity,
            "feedback": feedback
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
    ORDER BY timestamp DESC
    """, (session['user'],))

    data = cur.fetchall()
    conn.close()

    return render_template("progress.html", data=data)

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
if __name__ == '__main__':
    init_db()
    app.run(debug=True)