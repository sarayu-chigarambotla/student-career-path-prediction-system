import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("../data/student_career_dataset.csv")

# Separate features and target
X = df.drop("career_path", axis=1)
y = df["career_path"]

# Encode categorical features
categorical_cols = ["education_level", "stream"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Preprocessing Completed Successfully")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

