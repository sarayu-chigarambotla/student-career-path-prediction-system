from flask import Flask, render_template, request, session, redirect
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "career_prediction_secret"

# =========================
# LOAD ML COMPONENTS
# =========================
model = joblib.load("model/career_rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

# =========================
# CAREER LABELS
# =========================
career_names = {
    0: "Software Engineer",
    1: "Data Analyst",
    2: "Machine Learning Engineer",
    3: "Cybersecurity Specialist",
    4: "Core Engineer",
    5: "Higher Studies",
    6: "Management"
}

# =========================
# CAREER DETAILS (STATIC)
# =========================
career_details_master = {
    "Software Engineer": {
        "salary": "₹8–25 LPA",
        "growth": "+22%",
        "domain": "IT & Technology",
        "description": "Design, develop, and maintain scalable software systems.",
        "work_env": "Hybrid / Remote"
    },
    "Data Analyst": {
        "salary": "₹6–20 LPA",
        "growth": "+30%",
        "domain": "Data Analytics",
        "description": "Analyze data, build dashboards, and generate insights.",
        "work_env": "Hybrid"
    },
    "Machine Learning Engineer": {
        "salary": "₹12–35 LPA",
        "growth": "+40%",
        "domain": "Artificial Intelligence",
        "description": "Build and deploy machine learning models at scale.",
        "work_env": "Hybrid"
    },
    "Cybersecurity Specialist": {
        "salary": "₹8–22 LPA",
        "growth": "+30%",
        "domain": "Cyber Security",
        "description": "Protect systems from cyber threats and vulnerabilities.",
        "work_env": "Onsite / Hybrid"
    },
    "Core Engineer": {
        "salary": "₹6–18 LPA",
        "growth": "+18%",
        "domain": "Core Engineering",
        "description": "Work on core engineering systems.",
        "work_env": "Onsite"
    },
    "Management": {
        "salary": "₹7–20 LPA",
        "growth": "+25%",
        "domain": "Business & Leadership",
        "description": "Lead teams and manage business decisions.",
        "work_env": "Onsite / Hybrid"
    },
    "Higher Studies": {
        "salary": "—",
        "growth": "—",
        "domain": "Academics",
        "description": "Pursue higher education and research.",
        "work_env": "University"
    }
}

# =========================
# SKILL REQUIREMENTS
# =========================
career_skill_requirements = {
    0: {"python_skill": 7, "programming_score": 75},
    1: {"data_analysis_skill": 7, "math_score": 70},
    2: {"ml_skill": 8, "python_skill": 8},
    3: {"programming_score": 70},
    4: {"math_score": 65},
    5: {"gpa": 8.5},
    6: {"communication_skill": 7}
}

# =========================
# FEATURE ORDER
# =========================
FEATURE_ORDER = [
    "education_level",
    "stream",
    "gpa",
    "math_score",
    "programming_score",
    "python_skill",
    "ml_skill",
    "data_analysis_skill",
    "web_dev_skill",
    "communication_skill",
    "hours_per_week",
    "projects_done",
    "certifications",
    "internship_experience",
    "hackathon_participation"
]

# =========================
# HELPER
# =========================
def get_context():
    return session.get("result")

# =========================
# BASIC ROUTES
# =========================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("form.html", form_data=None)

# =========================
# NAVBAR ROUTES
# =========================
@app.route("/overview")
def overview():
    data = get_context()
    if not data:
        return redirect("/form")
    return render_template("overview.html", active="overview", **data)

@app.route("/career-paths")
def career_paths():
    data = get_context()
    if not data:
        return redirect("/form")

    return render_template(
        "career_paths.html",
        active="career",
        top_careers=data["top_careers"],
        career_details=data["career_details"]
    )

@app.route("/skills-analysis")
def skills_analysis_page():
    data = get_context()
    if not data:
        return redirect("/form")

    # 🔐 SAFE ACCESS (THIS IS THE FIX)
    skills = data.get("skills_analysis")


    return render_template(
        "skills_analysis.html",
        active="skills",
        skills=skills
    )


@app.route("/strengths")
def strengths():
    data = get_context()
    if not data:
        return redirect("/form")

    return render_template(
        "strengths.html",
        active="strengths",
        skills=data.get("skills_analysis")   # ✅ THIS LINE FIXES EVERYTHING
    )


@app.route("/roadmap")
def roadmap():
    data = get_context()
    if not data:
        return redirect("/form")
    
    career_name = data["career"]
    domain = "Data Analytics & AI"
    match_score = data["confidence"]

    # Override: selected career from modal
    selected_career = request.args.get("selected")
    if selected_career:
        career_name = selected_career
        domain = selected_career

    # build roadmap object ONLY for roadmap tab
    roadmap_data = {
        "target": {
            "name": data["career"],
            "domain": "Data Analytics & AI",
            "match": data["confidence"]
        },
        "sections": [
            {
                "title": "Immediate Actions",
                "duration": "0-3 months",
                "tasks": [
                    {"text": "Master core skills", "category": "Skills", "priority": "High"},
                    {"text": "Build 2-3 portfolio projects", "category": "Portfolio", "priority": "High"},
                    {"text": "Pursue relevant certifications", "category": "Certification", "priority": "Medium"},
                    {"text": "Create LinkedIn & GitHub profiles", "category": "Branding", "priority": "Medium"},
                ]
            },
            {
                "title": "Short-term Goals",
                "duration": "3-6 months",
                "tasks": [
                    {"text": "Secure an internship", "category": "Experience", "priority": "High"},
                    {"text": "Contribute to open-source projects", "category": "Open Source", "priority": "Medium"},
                    {"text": "Attend industry meetups", "category": "Networking", "priority": "Medium"},
                    {"text": "Complete specialized courses", "category": "Learning", "priority": "Medium"},
                ]
            },
            {
                "title": "Medium-term Goals",
                "duration": "6-12 months",
                "tasks": [
                    {"text": "Apply for entry-level roles", "category": "Job Search", "priority": "High"},
                    {"text": "Develop niche specialization", "category": "Specialization", "priority": "High"},
                    {"text": "Grow professional network", "category": "Networking", "priority": "Medium"},
                    {"text": "Start content creation", "category": "Branding", "priority": "Low"},
                ]
            },
            {
                "title": "Long-term Vision",
                "duration": "1-2 years",
                "tasks": [
                    {"text": "Secure role at reputed company", "category": "Career", "priority": "High"},
                    {"text": "Mentor juniors", "category": "Growth", "priority": "Medium"},
                    {"text": "Pursue advanced degree/certifications", "category": "Education", "priority": "Medium"},
                    {"text": "Build personal brand", "category": "Leadership", "priority": "Low"},
                ]
            }
        ]
    }

    return render_template(
        "roadmap.html",
        active="roadmap",
        roadmap=roadmap_data
    )
@app.route("/what-if")
def what_if():
    form_data = session.get("user_form_data", None)
    return render_template("form.html", form_data=form_data)


@app.route("/ai-insights")
def ai_insights():
    data = get_context()
    if not data:
        return redirect("/form")
    return render_template("ai_insights.html", active="ai", **data)


def build_roadmap_for(career):
    ROADMAPS = {
        "Data Scientist": [
            "Learn Python & Statistics",
            "Master ML Algorithms",
            "Build ML Projects",
            "Apply for DS Roles",
            "Grow into Senior DS"
        ],
        "Software Engineer": [
            "DSA & Programming",
            "System Design",
            "Backend / Frontend Skills",
            "Apply for SDE Roles",
            "Tech Lead"
        ],
        "Machine Learning Engineer": [
            "Math & ML Foundations",
            "Deep Learning",
            "Model Deployment",
            "MLOps",
            "Senior ML Engineer"
        ],
        "Cybersecurity Specialist": [
            "Networking Basics",
            "Security Fundamentals",
            "Ethical Hacking",
            "Certifications",
            "Security Architect"
        ],
        "Core Engineer": [
            "Engineering Fundamentals",
            "Domain Tools",
            "Industry Projects",
            "Core Company Roles",
            "Senior Engineer"
        ],
        "Management": [
            "Communication Skills",
            "MBA / Leadership",
            "Team Handling",
            "Business Strategy",
            "Senior Manager"
        ],
        "Higher Studies": [
            "Strong Academics",
            "Research Papers",
            "Entrance Exams",
            "Masters / PhD",
            "Research Career"
        ]
    }

    return {
        "target": career,
        "steps": ROADMAPS.get(career, [])
    }

# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    student_name = request.form.get("full_name", "Student")
    gpa = float(request.form["gpa"])
    coding_hours = int(request.form["hours_per_week"])
    internship_months = int(request.form.get("internship_months", 0))

    input_data = {
        "education_level": encoders["education_level"].transform(
            [request.form["education_level"]]
        )[0],
        "stream": encoders["stream"].transform(
            [request.form["stream"]]
        )[0],
        "gpa": gpa,
        "math_score": int(request.form["math_score"]),
        "programming_score": int(request.form["programming_score"]),
        "python_skill": int(request.form["python_skill"]),
        "ml_skill": int(request.form["ml_skill"]),
        "data_analysis_skill": int(request.form["data_analysis_skill"]),
        "web_dev_skill": int(request.form["web_dev_skill"]),
        "communication_skill": int(request.form["communication_skill"]),
        "hours_per_week": coding_hours,
        "projects_done": int(request.form["projects_done"]),
        "certifications": int(request.form["certifications"]),
        "internship_experience": int(request.form["internship_experience"]),
        "hackathon_participation": int(request.form["hackathon_participation"]),
        "technical_skill": int(request.form["technical_skill"]),
        "problem_solving": int(request.form["problem_solving"]),
        "leadership": int(request.form["leadership"]),
        "teamwork": int(request.form["teamwork"])
    }

    input_vector = np.array([input_data[f] for f in FEATURE_ORDER]).reshape(1, -1)
    input_scaled = scaler.transform(input_vector)

    probs = model.predict_proba(input_scaled)[0]
    top_indices = probs.argsort()[::-1][:3]

    top_careers = [
        {"name": career_names[i], "prob": round(probs[i] * 100, 1)}
        for i in top_indices
    ]

    top_prediction = top_indices[0]
    career_name = career_names[top_prediction]
    confidence = round(probs[top_prediction] * 100, 1)

    skill_gaps = {}
    for skill, req in career_skill_requirements[top_prediction].items():
        if input_data.get(skill, 0) < req:
            skill_gaps[skill] = req - input_data[skill]
    
    # Save all form inputs for What-If analysis
    session["user_form_data"] = dict(request.form)


    # Skills Analysis (COMPUTED SAFELY)
    # ---------- SKILLS ANALYSIS DATA ----------
    def get_overall_label(score):
        if score < 40:
            return "Poor"
        elif score < 60:
            return "Average"
        elif score < 75:
            return "Good"
        elif score < 90:
            return "Very Good"
        else:
            return "Excellent"

    overall_score = int(
    (
        input_data["technical_skill"]
        + input_data["problem_solving"]
        + input_data["communication_skill"]
        + input_data["leadership"]
        + input_data["teamwork"]
    ) / 5
)

    skills_analysis_data = {
    "overall_score": overall_score,
    "overall_label": get_overall_label(overall_score),

    "technical_skills": {
        "Technical Skills": input_data["technical_skill"],
        "Problem Solving": input_data["problem_solving"],
        "Aptitude": input_data["math_score"]
    },

    "soft_skills": {
        "Communication": input_data["communication_skill"],
        "Leadership": input_data["leadership"],
        "Teamwork": input_data["teamwork"]
    }
}




    career_details = {
        c["name"]: career_details_master.get(c["name"], {})
        for c in top_careers
    }

    session["result"] = {
        "student_name": student_name,
        "gpa": gpa,
        "internship_months": internship_months,
        "coding_hours": coding_hours,
        "career": career_name,
        "confidence": confidence,
        "top_careers": top_careers,
        "gaps": skill_gaps,
        "career_details": career_details,
        "skills_analysis": skills_analysis_data
    }

    return redirect("/overview")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)