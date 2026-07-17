"""Per-candidate pipeline: read file -> parse resume -> score against job."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from config import REQUEST_DELAY_SECONDS
from extraction import ExtractionError, parse_resume, score_candidate
from file_readers import UnreadableFileError, read_resume
from schemas import JobDescription, MatchResult, Resume

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    file_name: str
    resume: Resume | None
    match: MatchResult | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def process_candidate(file_path: Path, job: JobDescription) -> CandidateResult:
    """Process a single resume file end to end. Never raises — failures are
    captured on the result so one bad resume doesn't kill the whole batch run.
    """
    try:
        resume_text = read_resume(file_path)
        parsed_resume = parse_resume(resume_text)
        time.sleep(REQUEST_DELAY_SECONDS)

        match = score_candidate(job, parsed_resume)
        time.sleep(REQUEST_DELAY_SECONDS)

        return CandidateResult(file_name=file_path.name, resume=parsed_resume, match=match)

    except (UnreadableFileError, ExtractionError) as exc:
        logger.error("Failed to process %s: %s", file_path.name, exc)
        return CandidateResult(file_name=file_path.name, resume=None, match=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort guard for a batch job
        logger.exception("Unexpected error processing %s", file_path.name)
        return CandidateResult(file_name=file_path.name, resume=None, match=None, error=f"Unexpected error: {exc}")
