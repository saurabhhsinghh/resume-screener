# LLM Resume Screener

Screens a folder of resumes (PDF/DOCX) against a job description using an LLM,
and ranks candidates by fit — matching skills, missing skills, and an overall
match score.

## Why

Manually screening dozens of resumes against a JD is slow and inconsistent.
This tool extracts structured data from both the job description and each
resume using an LLM constrained to a strict schema (Pydantic), then asks the
model to score the fit — turning free-text resumes into a ranked, comparable
shortlist in minutes.

## How it works

```
job_description.txt ──┐
                       ├─► extract_job_description() ──► JobDescription (structured)
resumes/*.pdf|.docx ───┘
        │
        ▼
  read_resume() ──► parse_resume() ──► Resume (structured)
                                              │
                                              ▼
                     score_candidate(job, resume) ──► MatchResult (score + verdict)
```

Each resume is processed independently, so one malformed PDF or a bad LLM
response doesn't fail the whole batch — it's logged and skipped, and the run
continues.

## Project structure

```
schemas.py        # Pydantic models: JobDescription, Resume, MatchResult
config.py          # env-driven configuration
file_readers.py    # PDF/DOCX -> plain text, with error handling
extraction.py       # all LLM calls: JD parsing, resume parsing, scoring
                     #   -> retries with exponential backoff (tenacity)
pipeline.py         # per-candidate orchestration, isolates failures
main.py             # CLI entrypoint
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
```

## Usage

### CLI (batch mode)

```bash
python main.py --resumes ./resumes --job-description ./job_description.txt --top 2
```

### Web app

A single-resume, interactive version is also available via Streamlit — paste a
job description, upload one resume, get an instant score.

```bash
streamlit run app.py
```

`app.py` is a thin UI layer only — it calls the exact same `extraction.py` /
`file_readers.py` functions as the CLI, so there's no duplicated logic between
the two interfaces.

## Design notes

- **Structured outputs**: every LLM call is constrained to a Pydantic JSON
  schema, so downstream code works with typed objects instead of raw text.
- **Resilience**: API calls retry with exponential backoff on transient
  failures (rate limits, connection errors, malformed JSON); a fixed delay
  between calls respects provider rate limits.
- **Isolation**: a single candidate failure (unreadable file, LLM error)
  doesn't take down the batch — it's reported at the end alongside the results.

## Possible next steps

- Streamlit/FastAPI front-end for uploading a JD + resumes through a browser
- Persist results to a database instead of stdout
- Parallelize candidate processing (async / thread pool) within rate limits
- Unit tests with mocked LLM responses
