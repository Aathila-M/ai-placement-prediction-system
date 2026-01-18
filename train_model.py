import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# CREATE MODELS FOLDER
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# FEATURES & LABEL
# =========================
X = df.drop("placed", axis=1)
y = df["placed"]   # 1 = Placed, 0 = Not Placed

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN–TEST SPLIT (WITH STRATIFY)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,          # ✅ IMPORTANT (balanced classes)
    random_state=42
)

# =========================
# MODEL (HIGH ACCURACY)
# =========================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", round(accuracy, 4))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL & SCALER
# =========================
joblib.dump(model, "models/placement_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Placement model & scaler saved successfully")
