import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("../data/student_career_dataset.csv")

# =========================
# 2. Split Features & Target
# =========================
X = df.drop("career_path", axis=1)
y = df["career_path"]

# =========================
# 3. Encode Categorical Data
# =========================
categorical_cols = ["education_level", "stream"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# =========================
# 4. Feature Scaling
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. Train Random Forest Model
# =========================
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42
)

rf_model.fit(X_train, y_train)

# =========================
# 7. Career Prediction
# =========================
y_pred = rf_model.predict(X_test)

# Save predictions
results = df.iloc[y_test.index].copy()
results["predicted_career"] = y_pred

print("✅ Random Forest training & prediction completed")
print(results.head())
# plt.figure(figsize=(8,5))
# sns.countplot(x="predicted_career", data=results)
# plt.title("Predicted Career Distribution")
# plt.xlabel("Career Path")
# plt.ylabel("Number of Students")
# plt.tight_layout()
# plt.show()
# importance = rf_model.feature_importances_
# features = X.columns

# imp_df = pd.DataFrame({
#     "Feature": features,
#     "Importance": importance
# }).sort_values(by="Importance", ascending=False)

# plt.figure(figsize=(9,6))
# sns.barplot(x="Importance", y="Feature", data=imp_df)
# plt.title("Feature Importance in Career Prediction (Random Forest)")
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(8,5))
# sns.boxplot(x="predicted_career", y="hours_per_week", data=results)
# plt.title("Weekly Effort vs Predicted Career Path")
# plt.tight_layout()
# plt.show()
# =========================
# 8. Skill Requirement Mapping
# =========================

career_skill_requirements = {
    0: {  # Software Developer
        "python_skill": 7,
        "web_dev_skill": 6,
        "programming_score": 75
    },
    1: {  # Data Analyst
        "data_analysis_skill": 7,
        "math_score": 70,
        "python_skill": 6
    },
    2: {  # ML Engineer
        "ml_skill": 8,
        "python_skill": 8,
        "math_score": 75
    },
    3: {  # Cybersecurity
        "programming_score": 70,
        "internship_experience": 1
    },
    4: {  # Core Engineer
        "math_score": 65,
        "projects_done": 4
    },
    5: {  # Higher Studies
        "gpa": 8.5,
        "certifications": 3
    },
    6: {  # Management
        "communication_skill": 7,
        "projects_done": 3
    }
}
# =========================
# 9. Skill Gap Analysis Function
# =========================

def skill_gap_analysis(student_row, predicted_career):
    required_skills = career_skill_requirements[predicted_career]
    gaps = {}

    for skill, required_value in required_skills.items():
        current_value = student_row[skill]
        if current_value < required_value:
            gaps[skill] = required_value - current_value

    return gaps
# =========================
# 10. Apply Skill Gap Analysis to Predictions
# =========================

results["skill_gaps"] = results.apply(
    lambda row: skill_gap_analysis(row, row["predicted_career"]),
    axis=1
)

print("\n🔍 Sample Skill Gap Analysis:")
print(results[["predicted_career", "skill_gaps"]].head())

career_names = {
    0: "Software Developer",
    1: "Data Analyst",
    2: "Machine Learning Engineer",
    3: "Cybersecurity Specialist",
    4: "Core Engineer",
    5: "Higher Studies",
    6: "Management"
}

results["predicted_career_name"] = results["predicted_career"].map(career_names)

print("\n🎯 Readable Output:")
print(results[["predicted_career_name", "skill_gaps"]].head())

