# resume_analyzer.py
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import openai
import streamlit as st

# 1️⃣ First, try environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# 2️⃣ Load from .env if missing (local dev)
if not api_key:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

# 3️⃣ Raise error if still not found
if not api_key:
    raise ValueError("OpenAI API key not found.")

openai.api_key = api_key
client = OpenAI()

# ---------- FIX: safe model loading ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()
# ---------------------------------------------

# -------------------------------
# Step 1: Extract required skills
# -------------------------------
def extract_job_skills(job_description: str) -> list[str]:
    prompt = f"""
    Extract all skills, tools, programming languages, software, and technologies required in this job description.
    Return ONLY a JSON object like:
    {{ "skills": ["skill1", "skill2", ...] }}
    Do not include any other text.

    Job Description:
    {job_description}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return [s.lower().strip() for s in data.get("skills", []) if isinstance(s, str)]

    except Exception as e:
        print("Error extracting job skills:", e)
        return []

# -------------------------------
def extract_resume_skills(resume_text: str, job_skills: list[str]) -> list[str]:
    resume_text_lower = resume_text.lower()
    matched_skills = [skill for skill in job_skills if skill.lower() in resume_text_lower]
    return matched_skills

# -------------------------------
def compute_responsibility_score(resume_text: str, job_description: str) -> float:
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = embedding_model.encode(job_description, convert_to_tensor=True)

    similarity = util.cos_sim(resume_embedding, job_embedding).item()
    score = max(min(similarity, 1.0), 0.0) * 100
    return round(score, 2)

# -------------------------------
def evaluate_resume_against_job(resume_text: str, job_description: str) -> dict:
    job_skills = extract_job_skills(job_description)
    resume_skills_found = extract_resume_skills(resume_text, job_skills)

    if job_skills:
        skill_score = len(resume_skills_found) / len(job_skills) * 100
    else:
        skill_score = 0.0

    responsibility_score = compute_responsibility_score(resume_text, job_description)
    semantic_score = responsibility_score

    final_score = round(
        0.5 * skill_score +
        0.3 * responsibility_score +
        0.2 * semantic_score,
        2
    )

    return {
        "resume_skills": resume_skills_found,
        "job_skills": job_skills,
        "skill_score": round(skill_score, 2),
        "responsibility_score": round(responsibility_score, 2),
        "embedding_score": round(semantic_score, 2),
        "final_score": final_score
    }
