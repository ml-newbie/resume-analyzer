# analyzer.py
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import openai

# 1️⃣ First, try environment variable (Streamlit Cloud or manually set env)
api_key = os.environ.get("OPENAI_API_KEY")

# 2️⃣ If not set, try loading from .env file (for local development)
if not api_key:
    load_dotenv()  # loads variables from .env
    api_key = os.environ.get("OPENAI_API_KEY")

# 3️⃣ Raise error if still not found
if not api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in env or .env file.")

# 4️⃣ Set for OpenAI
openai.api_key = api_key

# Initialize OpenAI client
client = OpenAI()

# Initialize embeddings model (you can swap for larger model if needed)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# Step 1: Extract required skills from Job Description
# -------------------------------
def extract_job_skills(job_description: str) -> list[str]:
    """
    Dynamically extract required skills from a job description using LLM.
    Returns a list of lowercase skills.
    """
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
# Step 2: Extract matched skills from Resume
# -------------------------------
def extract_resume_skills(resume_text: str, job_skills: list[str]) -> list[str]:
    """
    Find which job skills appear anywhere in the resume text.
    Case-insensitive.
    """
    resume_text_lower = resume_text.lower()
    matched_skills = [skill for skill in job_skills if skill.lower() in resume_text_lower]
    return matched_skills


# -------------------------------
# Step 3: Embedding-based Responsibility Match
# -------------------------------
def compute_responsibility_score(resume_text: str, job_description: str) -> float:
    """
    Compute semantic similarity between resume and job description for responsibilities.
    Returns a percentage (0-100).
    """
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = embedding_model.encode(job_description, convert_to_tensor=True)

    similarity = util.cos_sim(resume_embedding, job_embedding).item()
    # Convert similarity [-1,1] to 0-100%
    score = max(min(similarity, 1.0), 0.0) * 100
    return round(score, 2)


# -------------------------------
# Step 4: Compute overall resume-job match
# -------------------------------
def evaluate_resume_against_job(resume_text: str, job_description: str) -> dict:
    """
    Full evaluation pipeline:
    - Dynamic skill extraction
    - Skill match scoring
    - Responsibility match scoring (embeddings)
    - Semantic similarity scoring
    - Weighted final score
    """
    # 1️⃣ Extract job skills
    job_skills = extract_job_skills(job_description)

    # 2️⃣ Extract resume skills
    resume_skills_found = extract_resume_skills(resume_text, job_skills)

    # 3️⃣ Compute skill score
    if job_skills:
        skill_score = len(resume_skills_found) / len(job_skills) * 100
    else:
        skill_score = 0.0

    # 4️⃣ Compute responsibility match score (embedding similarity)
    responsibility_score = compute_responsibility_score(resume_text, job_description)

    # 5️⃣ Compute overall semantic similarity (optional global text similarity)
    semantic_score = responsibility_score  # Can also compute separately if desired

    # 6️⃣ Weighted final score
    # You can adjust weights: skills 50%, responsibilities 30%, semantic 20%
    final_score = round(0.5 * skill_score + 0.3 * responsibility_score + 0.2 * semantic_score, 2)

    # 7️⃣ Return results
    return {
        "resume_skills": resume_skills_found,
        "job_skills": job_skills,
        "skill_score": round(skill_score, 2),
        "responsibility_score": round(responsibility_score, 2),
        "embedding_score": round(semantic_score, 2),
        "final_score": final_score
    }
