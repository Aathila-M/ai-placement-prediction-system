from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "placement_secret_key"

# ---------------- Load Model (Render-safe paths) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "placement_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

MODEL_ACCURACY = 92.5  # demo accuracy

# ---------------- Login ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["username"] = request.form.get("username")
        session["role"] = request.form.get("role", "student")
        return redirect(url_for("dashboard"))
    return render_template("login.html")

# ---------------- Dashboard ----------------
@app.route("/", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    prediction = None
    probability = None

    if request.method == "POST":
        cgpa = float(request.form["cgpa"])
        internships = int(request.form["internships"])
        projects = int(request.form["projects"])
        certifications = int(request.form["certifications"])
        communication = int(request.form["communication"])
        backlogs = int(request.form["backlogs"])

        X = np.array([[cgpa, internships, projects,
                       certifications, communication, backlogs]])

        X_scaled = scaler.transform(X)

        result = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        prediction = "PLACED ✅" if result == 1 else "NOT PLACED ❌"
        probability = round(prob * 100, 2)

    return render_template(
        "dashboard.html",
        prediction=prediction,
        probability=probability,
        role=session.get("role"),
        accuracy=MODEL_ACCURACY
    )

# ---------------- Logout ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- Render Required Entry ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
