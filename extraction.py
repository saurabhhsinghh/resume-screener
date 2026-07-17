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

Skills can be stated two ways, and you must catch BOTH:
1. Explicitly listed (e.g. a "Skills" section: "Python, SQL, React")
2. Demonstrated implicitly through what the candidate describes doing —
   read achievements, project bullets, and work experience descriptions
   carefully and infer the underlying skill/concept being demonstrated.
   Examples of this kind of inference:
   - "Solved 1000+ DSA problems, LeetCode rating 1829" -> the candidate has
     Data Structures and Algorithms skill, even though those exact words
     may not appear as a labeled skill.
   - "Built JWT-secured REST APIs" -> the candidate has REST API design
     and JWT/authentication skills.
   - "Optimized SQL queries for a payables report" -> the candidate has
     SQL and database skills.
   Do this for every achievement, project, and experience bullet, not only
   the ones that happen to name a skill directly.

Return ONLY valid JSON matching this schema:
{schema}

Important rules:
1. Do not invent information — every skill you extract must be traceable to
   something actually stated or clearly demonstrated in the resume text.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned OR demonstrated across the entire resume,
   including the Achievements/Certifications sections, not just a
   dedicated Skills section.
6. If total_experience_years is not explicitly stated, estimate it by
   summing the durations of the work experience entries (using their dates)
   rather than leaving it null when the dates are available.
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

Before listing a required or preferred skill as MISSING, double-check the
candidate's full profile (skills, experiences, projects, certifications) for
any mention or clear demonstration of that skill or a close equivalent —
including skills demonstrated through project/achievement descriptions, not
only skills that are explicitly named. Only mark a skill as missing if it is
genuinely absent after this check. A skill wrongly marked as missing is worse
than a skill wrongly marked as matching, so err on the side of re-checking.

For experience_requirement_met: if total_experience_years is available, compare
it directly against the job's minimum_experience. If total_experience_years is
null but the candidate's experience entries have durations/dates, estimate
total experience from those before concluding the requirement isn't met.

Provide:
1. candidate_name
2. details.matching_skills
3. details.missing_skills (only after the verification check above)
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