import os
import sys
import psycopg2
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# CONFIG
# =========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "dbname=placement_db user=postgres password=postgres host=localhost"
)

# =========================
# PYINSTALLER PATH FIX
# =========================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# =========================
# DATABASE
# =========================
def get_db():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE,
            password TEXT,
            role VARCHAR(20)
        )
    """)

    # Create admin if not exists
    cur.execute("SELECT * FROM users WHERE username='admin'")
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            ("admin", generate_password_hash("admin123"), "admin")
        )

    conn.commit()
    conn.close()

# =========================
# LOAD MODEL
# =========================
model = joblib.load(resource_path("models/placement_model.pkl"))
scaler = joblib.load(resource_path("models/scaler.pkl"))

FEATURES = [
    "cgpa",
    "internships",
    "projects",
    "certifications",
    "communication_skills",
    "backlogs"
]

# =========================
# FLASK APP
# =========================
app = Flask(__name__, template_folder=resource_path("templates"))
app.secret_key = "secure-placement-cloud"

probability_history = []

# =========================
# LOGIN
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT password, role FROM users WHERE username=%s",
            (username,)
        )
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session["logged_in"] = True
            session["username"] = username
            session["role"] = row[1]
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                (username, password, "student")
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except:
            conn.close()
            return render_template("register.html", error="User already exists")

    return render_template("register.html")

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        role=session["role"],
        features=FEATURES,
        probabilities=probability_history
    )

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    data = request.json

    values = [[
        min(max(data["cgpa"], 0), 10),
        min(data["internships"], 5),
        min(data["projects"], 6),
        min(data["certifications"], 10),
        min(max(data["communication_skills"], 1), 10),
        min(data["backlogs"], 5)
    ]]

    df = pd.DataFrame(values, columns=FEATURES)
    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]
    probability = round(model.predict_proba(scaled)[0][1] * 100, 2)

    probability_history.append(probability)

    return jsonify({
        "result": "Placed" if prediction == 1 else "Not Placed",
        "probability": probability
    })

@app.route("/logout")
def logout():
    session.clear()
    probability_history.clear()
    return redirect(url_for("login"))

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
