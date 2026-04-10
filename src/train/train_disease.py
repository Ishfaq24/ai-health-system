import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("../../data/skin-disease.csv")

print("Dataset Shape:", df.shape)

# ==============================
# 2. CLEAN DATA
# ==============================
# Drop useless column if exists
df = df.drop(columns=["Unnamed: 133"], errors='ignore')

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# ==============================
# 3. SPLIT FEATURES & TARGET
# ==============================
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Save feature names (VERY IMPORTANT)
columns = list(X.columns)
json.dump(columns, open("../../columns.json", "w"))

# ==============================
# 4. LABEL ENCODING
# ==============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
pickle.dump(le, open("../../models/label_encoder.pkl", "wb"))

# ==============================
# 5. ADD NOISE (REALISTIC DATA)
# ==============================
X_noisy = X.copy()

noise = np.random.binomial(1, 0.05, X_noisy.shape)  # 5% noise
X_noisy = np.logical_xor(X_noisy, noise).astype(int)

# ==============================
# 6. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y_encoded, test_size=0.2, random_state=42
)

# ==============================
# 7. MODEL TRAINING
# ==============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 8. EVALUATION
# ==============================
y_pred = model.predict(X_test)

# Decode predictions
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\nClassification Report (Real Labels):\n")
print(classification_report(y_test_labels, y_pred_labels))

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

# Train accuracy
train_acc = model.score(X_train, y_train)
print("Train Accuracy:", train_acc)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 9. SAVE MODEL
# ==============================
pickle.dump(model, open("../../models/disease_model.pkl", "wb"))

print("\n✅ Model, encoder, and columns saved successfully!")