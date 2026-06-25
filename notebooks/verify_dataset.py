import pandas as pd

# Load dataset
df = pd.read_csv("../data/student_career_dataset.csv")

# Basic checks
print("Dataset Shape:", df.shape)
print("\nCareer Path Distribution:\n", df["career_path"].value_counts())
print("\nStatistical Summary:\n", df.describe())
