"""All LLM calls live here: job description parsing, resume parsing, match scoring.

Every call is wrapped with retry/backoff (tenacity) instead of a blind time.sleep(),
and every response is validated against a Pydantic schema so a malformed LLM
response fails loudly and specifically instead of crashing the whole pipeline.
"""
import json
import logging

from groq import Groq, APIStatusError, APIConnectionError
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import GROQ_API_KEY, MODEL_NAME, MAX_RETRIES
from schemas import JobDescription, Resume, MatchResult

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

RETRYABLE_EXCEPTIONS = (APIStatusError, APIConnectionError, json.JSONDecodeError, ValidationError)

_retry_policy = retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)


class ExtractionError(Exception):
    """Raised when the LLM response can't be parsed into the expected schema
    after all retries are exhausted."""


@_retry_policy
def _call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON, will retry: %s", raw[:200])
        raise


def extract_job_description(job_description_text: str) -> JobDescription:
    schema = JobDescription.model_json_schema()
    system_prompt = f"""You are an expert HR assistant.

Your job is to analyze job descriptions and extract structured information from them.

Return ONLY valid JSON matching this schema:
{schema}

IMPORTANT:
Do NOT return the schema itself.
Do NOT return fields like "properties", "title" or "type".
Fill the schema with actual information extracted from the job description.

If minimum experience is not mentioned, return null.
If information for a list is missing, return an empty list.
Do not invent information.
"""
    user_prompt = f"Analyze the following job description:\n{job_description_text}"

    try:
        data = _call_llm_json(system_prompt, user_prompt)
        return JobDescription(**data)
    except RETRYABLE_EXCEPTIONS as exc:
        raise ExtractionError(f"Failed to extract job description after retries: {exc}") from exc


def parse_resume(resume_text: str) -> Resume:
    schema = Resume.model_json_schema()
    system_prompt = f"""You are an expert resume parser.

Extract information from the resume based on its meaning, not only based on
exact section headings. Different resumes may use different headings, e.g.
Experience, Professional Experience, Work History, Employment, Internships —
these may all contain relevant experience.

Skills may appear in the skills section, work experience, internships, or projects.

Return ONLY valid JSON matching this schema:
{schema}

Important rules:
1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned across the entire resume.
"""
    user_prompt = f"Parse the following resume:\n{resume_text}"

    try:
        data = _call_llm_json(system_prompt, user_prompt)
        return Resume(**data)
    except RETRYABLE_EXCEPTIONS as exc:
        raise ExtractionError(f"Failed to parse resume after retries: {exc}") from exc


def score_candidate(job: JobDescription, resume: Resume) -> MatchResult:
    schema = MatchResult.model_json_schema()
    system_prompt = f"""You are an HR recruiter comparing a candidate's resume
against a job description.

Return ONLY valid JSON matching this schema:
{schema}

Provide:
1. candidate_name
2. details.matching_skills
3. details.missing_skills
4. details.experience_requirement_met (true/false)
5. score: overall match percentage from 0 to 100
6. details.verdict: a short, concise final verdict
"""
    user_prompt = f"""
JOB DESCRIPTION:
{job.model_dump_json(indent=2)}

CANDIDATE RESUME:
{resume.model_dump_json(indent=2)}
"""
    try:
        data = _call_llm_json(system_prompt, user_prompt)
        return MatchResult(**data)
    except RETRYABLE_EXCEPTIONS as exc:
        raise ExtractionError(f"Failed to score candidate after retries: {exc}") from exc
