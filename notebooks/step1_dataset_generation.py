import numpy as np
import pandas as pd
import random

np.random.seed(42)

N = 1400  # total students

education_levels = ["UG", "PG"]
streams = ["CS", "IT", "ECE", "EEE", "ME", "Civil"]

career_paths = {
    0: "Software Developer",
    1: "Data Analyst",
    2: "Machine Learning Engineer",
    3: "Cybersecurity Specialist",
    4: "Core Engineer",
    5: "Higher Studies",
    6: "Management"
}

data = []

for _ in range(N):
    education = random.choice(education_levels)
    stream = random.choice(streams)

    gpa = round(np.random.uniform(5.0, 10.0), 2)
    math = np.random.randint(40, 100)
    prog = np.random.randint(30, 100)

    python_skill = np.random.randint(0, 11)
    ml_skill = np.random.randint(0, 11)
    da_skill = np.random.randint(0, 11)
    web_skill = np.random.randint(0, 11)
    comm_skill = np.random.randint(0, 11)

    hours = np.random.randint(0, 41)
    projects = np.random.randint(0, 11)
    certs = np.random.randint(0, 6)

    internship = np.random.randint(0, 2)
    hackathon = np.random.randint(0, 2)

    # 🎯 CAREER LOGIC (CONTROLLED)
    if prog > 75 and python_skill > 7 and projects > 4:
        career = 0  # Software Developer
    elif da_skill > 7 and math > 70:
        career = 1  # Data Analyst
    elif ml_skill > 7 and python_skill > 7:
        career = 2  # ML Engineer
    elif prog > 65 and internship == 1:
        career = 3  # Cybersecurity
    elif stream in ["ME", "Civil", "EEE"]:
        career = 4  # Core Engineer
    elif gpa > 8.5 and certs > 3:
        career = 5  # Higher Studies
    else:
        career = 6  # Management

    data.append([
        education, stream, gpa, math, prog,
        python_skill, ml_skill, da_skill, web_skill, comm_skill,
        hours, projects, certs, internship, hackathon, career
    ])

columns = [
    "education_level", "stream", "gpa", "math_score", "programming_score",
    "python_skill", "ml_skill", "data_analysis_skill", "web_dev_skill", "communication_skill",
    "hours_per_week", "projects_done", "certifications",
    "internship_experience", "hackathon_participation", "career_path"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("../data/student_career_dataset.csv", index=False)

print("✅ Dataset generated successfully!")
print(df.head())