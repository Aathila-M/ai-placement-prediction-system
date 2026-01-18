import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("dataset.csv")

X = df.drop("placed", axis=1).values
y = df["placed"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM [samples, timesteps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

model = Sequential([
    LSTM(64, input_shape=(1, X_scaled.shape[2])),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_scaled, y, epochs=20, batch_size=16)

model.save("models/lstm_placement_model.h5")

print("✅ LSTM placement model saved")
