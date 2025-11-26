# app.py
#------------------------------------------------------------------------------ #
# GitHub commands to push changes:
# git init
# git add .
# git commit -m "Initial commit of resume analyzer app"
# git remote add origin https://github.com/ml-newbie/resume-analyzer.git
# git branch -M main
# git push -u origin main
# All future pushes can be done with: git push
#------------------------------------------------------------------------------ #

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from cv_analyzer import evaluate_resume_against_job

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# -------------------------------
# Header
# -------------------------------
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:20px;border-radius:5px">
        <h1 style="color:#0f1c4d;text-align:center;">📄 AI Resume Analyzer</h1>
        <p style="text-align:center;color:#555;">Upload your resume and paste a job description to see the match score</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------------------
# Resume Upload & Job Description
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
with col2:
    job_description = st.text_area("Paste the job description here", height=200)

# -------------------------------
# Resume Text Extraction
# -------------------------------
def read_resume(file):
    """Extract text from uploaded resume"""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text

    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    else:
        return file.getvalue().decode("utf-8")


resume_text = None
if uploaded_file:
    resume_text = read_resume(uploaded_file)
    st.success("✔ Resume uploaded successfully!")

st.write("")

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("🚀 Evaluate Resume"):
    if not uploaded_file or not job_description.strip():
        st.warning("⚠ Please upload a resume and enter a job description before evaluating.")
    else:
        with st.spinner("Analyzing resume..."):
            results = evaluate_resume_against_job(resume_text, job_description)

        # -------------------------------
        # Results Card
        # -------------------------------
        st.markdown("---")
        st.markdown(
            '<div style="background-color:#f9f9f9;padding:20px;border-radius:5px;">',
            unsafe_allow_html=True
        )

        # Skills Match
        st.subheader("✅ Skills Match")
        st.write(f"**Resume Skills Found:** {results['resume_skills']}")
        st.write(f"**Job Required Skills:** {results['job_skills']}")
        st.write(f"**Skill Match Score:** {results['skill_score']}%")

        missing_skills = set(results['job_skills']) - set(results['resume_skills'])
        if missing_skills:
            st.write(f"**Missing Skills:** {list(missing_skills)}")

        st.write("")

        # Responsibility Match
        st.subheader("📊 Responsibility Match")
        st.write(f"**Responsibility Score (Embedding similarity):** {results['responsibility_score']}%")

        st.write("")

        # Final Score
        st.subheader("🏆 Final Resume Match Score")
        st.markdown(
            f"""
            <div style="background-color:#eee;width:100%;border-radius:5px;">
                <div style="width:{results['final_score']}%;background-color:#4caf50;padding:5px;border-radius:5px;color:white;text-align:center;">
                    {results['final_score']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    '<p style="font-size:10px; color:gray; text-align:center;">© 2025 John Merwin. All rights reserved.</p>',
    unsafe_allow_html=True
)
