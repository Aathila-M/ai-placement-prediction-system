import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

data = {
    "cgpa": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
    "internships": [0, 1, 1, 2, 2, 3, 3],
    "projects": [1, 2, 2, 3, 3, 4, 4],
    "certifications": [0, 1, 1, 2, 2, 3, 3],
    "communication": [4, 5, 6, 7, 8, 9, 9],
    "backlogs": [3, 2, 1, 1, 0, 0, 0],
    "placed": [0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df.drop("placed", axis=1)
y = df["placed"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("placement_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ placement_model.pkl created successfully")
# To run this script, execute: python train_model.py