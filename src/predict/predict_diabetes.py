import joblib
import pandas as pd
import os
from src.config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")

model = joblib.load(MODEL_PATH)


def predict_diabetes(data: dict):
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        risk = "Diabetic"
    else:
        risk = "Non-Diabetic"

    return {
        "result": risk,
        "probability": float(probability)
    }