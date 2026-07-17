"""Streamlit web app for the resume screener.

Reuses the same extraction.py / file_readers.py / schemas.py logic as the
CLI (main.py) — this file is only a UI layer on top of the existing pipeline.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

# Streamlit Cloud secrets (set in "Advanced settings") are only exposed via
# st.secrets, not as real environment variables. config.py reads with
# os.getenv(), so bridge them here BEFORE importing anything that pulls in
# config.py. Locally (running via `.env` + python-dotenv) there's no
# secrets.toml at all, which raises rather than returning empty -- guard it.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass  # no secrets.toml (e.g. running locally with .env) -- fine

from extraction import ExtractionError, extract_job_description, parse_resume, score_candidate
from file_readers import UnreadableFileError, read_resume

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="centered")

st.title("📄 Resume Screener")
st.caption("Paste a job description, upload a resume, and get an instant fit score.")

with st.form("screen_form"):
    job_description_text = st.text_area(
        "Job description",
        height=250,
        placeholder="Paste the full job description here...",
    )
    uploaded_file = st.file_uploader("Resume", type=["pdf", "docx"])
    submitted = st.form_submit_button("Score resume")

if submitted:
    if not job_description_text.strip():
        st.error("Please paste a job description.")
    elif uploaded_file is None:
        st.error("Please upload a resume (PDF or DOCX).")
    else:
        try:
            with st.spinner("Reading resume..."):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)
                resume_text = read_resume(tmp_path)
                tmp_path.unlink(missing_ok=True)

            with st.spinner("Analyzing job description..."):
                job = extract_job_description(job_description_text)

            with st.spinner("Parsing resume..."):
                resume = parse_resume(resume_text)

            with st.spinner("Scoring candidate fit..."):
                result = score_candidate(job, resume)

            st.divider()
            st.subheader(result.candidate_name or resume.name or "Candidate")
            st.metric("Match score", f"{result.score:.0f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Matching skills**")
                if result.details.matching_skills:
                    for skill in result.details.matching_skills:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("_None listed_")
            with col2:
                st.markdown("**❌ Missing skills**")
                if result.details.missing_skills:
                    for skill in result.details.missing_skills:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("_None listed_")

            if result.details.experience_requirement_met is not None:
                st.markdown(
                    f"**Experience requirement met:** "
                    f"{'Yes' if result.details.experience_requirement_met else 'No'}"
                )

            st.markdown("**Verdict**")
            st.info(result.details.verdict or "No verdict provided.")

        except UnreadableFileError as exc:
            st.error(f"Couldn't read that resume: {exc}")
        except ExtractionError as exc:
            st.error(f"Something went wrong analyzing this with the LLM: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
            st.error(f"Unexpected error: {exc}")

st.divider()
st.caption("Built with Groq (Llama 3.3 70B) + Pydantic structured outputs.")