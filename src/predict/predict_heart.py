import joblib
import pandas as pd
import os
from src.config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")

# Load pipeline (IMPORTANT: full pipeline saved)
pipeline = joblib.load(MODEL_PATH)


def predict_heart(data: dict):
    df = pd.DataFrame([data])

    # Feature engineering (same as training)
    df["Age_HR"] = df["Age"] * df["MaxHR"]
    df["BP_Chol"] = df["RestingBP"] * df["Cholesterol"]
    df["Oldpeak_ST"] = df["Oldpeak"] * (df["ST_Slope"] == "Flat").astype(int)

    # Prediction
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    # Convert to readable output
    if prediction == 1:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return {
        "risk": risk,
        "probability": float(probability)
    }