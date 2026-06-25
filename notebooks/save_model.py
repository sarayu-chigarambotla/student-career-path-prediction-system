import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/student_career_dataset.csv")

X = df.drop("career_path", axis=1)
y = df["career_path"]

# Encode categorical features
encoders = {}
for col in ["education_level", "stream"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, _, y_train, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save everything
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/career_rf_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")
joblib.dump(encoders, "../model/encoders.pkl")

print("✅ Model, scaler, and encoders saved successfully")
