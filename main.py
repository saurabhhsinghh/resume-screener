"""CLI entrypoint for the resume screening pipeline.

Usage:
    python main.py --resumes ./resumes --job-description ./job_description.txt
    python main.py --resumes ./resumes --job-description ./job_description.txt --top 3
"""
import argparse
import logging
from pathlib import Path

from extraction import ExtractionError, extract_job_description
from file_readers import SUPPORTED_EXTENSIONS
from pipeline import process_candidate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen resumes against a job description using an LLM.")
    parser.add_argument("--resumes", type=Path, default=Path("resumes"), help="Folder containing resume files (.pdf/.docx)")
    parser.add_argument("--job-description", type=Path, default=Path("job_description.txt"), help="Path to a text file with the job description")
    parser.add_argument("--top", type=int, default=2, help="Number of top/bottom candidates to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.job_description.exists():
        raise SystemExit(f"Job description file not found: {args.job_description}")
    if not args.resumes.exists() or not args.resumes.is_dir():
        raise SystemExit(f"Resume folder not found: {args.resumes}")

    job_description_text = args.job_description.read_text(encoding="utf-8")

    logger.info("Extracting structured job description...")
    try:
        job = extract_job_description(job_description_text)
    except ExtractionError as exc:
        raise SystemExit(f"Could not parse job description: {exc}")

    resume_files = sorted(
        p for p in args.resumes.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not resume_files:
        raise SystemExit(f"No .pdf/.docx resumes found in {args.resumes}")

    results = []
    for file_path in resume_files:
        logger.info("Processing: %s", file_path.name)
        result = process_candidate(file_path, job)
        if result.succeeded:
            logger.info("  -> score: %.1f", result.match.score)
        else:
            logger.warning("  -> skipped (%s)", result.error)
        results.append(result)

    scored = [r for r in results if r.succeeded]
    failed = [r for r in results if not r.succeeded]

    scored.sort(key=lambda r: r.match.score, reverse=True)
    top_n = scored[: args.top]
    bottom_n = scored[-args.top :] if len(scored) > args.top else []

    print(f"\n{'=' * 50}")
    print(f"TOP {args.top} CANDIDATES")
    print("=" * 50)
    for r in top_n:
        print(f"\n{r.match.candidate_name or r.resume.name or r.file_name} - {r.match.score:.1f}%")
        print(f"  Matching skills: {', '.join(r.match.details.matching_skills) or 'none listed'}")
        print(f"  Missing skills:  {', '.join(r.match.details.missing_skills) or 'none listed'}")
        print(f"  Verdict: {r.match.details.verdict}")

    if bottom_n:
        print(f"\n{'=' * 50}")
        print(f"LOWEST {args.top} CANDIDATES")
        print("=" * 50)
        for r in bottom_n:
            print(f"\n{r.match.candidate_name or r.resume.name or r.file_name} - {r.match.score:.1f}%")
            print(f"  Verdict: {r.match.details.verdict}")

    if failed:
        print(f"\n{'=' * 50}")
        print(f"SKIPPED ({len(failed)})")
        print("=" * 50)
        for r in failed:
            print(f"  {r.file_name}: {r.error}")


if __name__ == "__main__":
    main()
