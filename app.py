from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "placement_secret_key"

# ---------------- Load Model ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "placement_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run train_model.py first to create placement_model.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
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

        result = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

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

if __name__ == "__main__":
    app.run(debug=True)
