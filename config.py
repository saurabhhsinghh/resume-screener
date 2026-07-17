"""Centralized configuration. Reads from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file or shell environment."
        )
    return value


GROQ_API_KEY = _require_env("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
RESUME_FOLDER = Path(os.getenv("RESUME_FOLDER", "resumes"))
JOB_DESCRIPTION_FILE = Path(os.getenv("JOB_DESCRIPTION_FILE", "job_description.txt"))

# Rate limiting / retry behaviour
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
