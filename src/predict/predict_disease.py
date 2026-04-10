import pickle
import json
import numpy as np
from src.config import MODEL_PATH, ENCODER_PATH, COLUMNS_PATH

# Load everything once (efficient)
model = pickle.load(open(MODEL_PATH, "rb"))
encoder = pickle.load(open(ENCODER_PATH, "rb"))
columns = json.load(open(COLUMNS_PATH))


def predict_disease(symptoms: list):
    # Create input vector
    input_data = [0] * len(columns)

    # Fill symptoms
    for symptom in symptoms:
        if symptom in columns:
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
        "top3": top3_diseases.tolist()
    }