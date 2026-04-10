import pickle
import json
import numpy as np
import os
from src.config import MODEL_PATH, ENCODER_PATH, COLUMNS_PATH

# Debug check (optional)
print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))
encoder = pickle.load(open(ENCODER_PATH, "rb"))
columns = json.load(open(COLUMNS_PATH))


def predict_disease(symptoms: list):
    # Validate symptoms
    valid_symptoms = [s for s in symptoms if s in columns]

    if not valid_symptoms:
        return {
            "error": "No valid symptoms provided"
        }

    # Create input vector
    input_data = [0] * len(columns)

    for symptom in valid_symptoms:
        index = columns.index(symptom)
        input_data[index] = 1

    input_array = np.array(input_data).reshape(1, -1)

    # Prediction
    prediction = model.predict(input_array)[0]
    disease = encoder.inverse_transform([prediction])[0]

    # Top 3 predictions
    probs = model.predict_proba(input_array)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3_diseases = encoder.inverse_transform(top3_idx)

    return {
        "prediction": disease,
        "top3": top3_diseases.tolist(),
        "used_symptoms": valid_symptoms
    }