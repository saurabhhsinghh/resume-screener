"""Pydantic models shared across the screening pipeline."""
from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    role: str
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    minimum_experience: float | None = None
    education_requirements: list[str] = []
    responsibilities: list[str] = []


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


class MatchDetails(BaseModel):
    """Structured breakdown instead of a free-form dict."""
    matching_skills: list[str] = []
    missing_skills: list[str] = []
    experience_requirement_met: bool | None = None
    verdict: str = ""


class MatchResult(BaseModel):
    candidate_name: str | None = None
    score: float = Field(ge=0, le=100)
    details: MatchDetails
