## Phase 1: Fundamentals (1–2 Weeks)
[CI status badge placeholder]

This repository is set up to help you complete Phase 1 of the AI Resume Screening & Job Matching System.

### What to be confident with
- **Python OOP + File Handling**: Reading/writing files, organizing code into classes/functions.
- **Pandas, NumPy**: Data cleaning and manipulation.
- **Basic SQL (SQLite/MySQL)**: Store parsed resumes and job descriptions.
- **NLP Basics**: Tokenization, stopwords, TF‑IDF, word embeddings (basics).

### Output of this phase
- **Be able to extract text from a PDF and process it into clean text.**

### Environment Setup

1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

> Note: The script will download small NLTK resources (tokenizer + stopwords) the first time it runs.

### Usage: Extract + Clean PDF Text

```bash
python phase1_pdf_process.py --input path\to\resume.pdf --output outputs\resume_clean.txt --show-tfidf --top-k 20
```

Flags:
- **--input**: path to PDF file
- **--output** (optional): where to save cleaned text
- **--show-tfidf** (optional): print top TF‑IDF terms
- **--top-k** (optional): how many top TF‑IDF terms to show

### Next Steps (suggested drills)
- Implement a small SQLite DB with tables `resumes(id, path, text)` and `jobs(id, title, description)`.
- Write a loader that inserts cleaned text from `outputs/resume_clean.txt` into the DB.
- Try vectorizing multiple resumes with TF‑IDF and compute cosine similarity to a sample job description.
- Replace TF‑IDF with word embeddings (e.g., `sentence-transformers`) when ready.

## Resume Extraction and Parsing

### Multi-format extraction (PDF/DOCX/TXT)
```bash
python resume_cli.py --input path\to\resume.pdf --dump-text --output-json outputs\resume_parsed.json
python resume_cli.py --input path\to\resume.docx --output-json outputs\resume_parsed.json
```

The CLI will:
- Extract raw text from `.pdf` (pypdf → pdfminer fallback) or `.docx` (docx2txt) or `.txt`.
- Parse candidate fields: Name, Skills, Education, Experience.
- Print JSON to stdout and optionally write to `--output-json`.

Sample job descriptions are in `sample_job_descriptions/`.

## Phase 3: Matching Engine

## Web App (Streamlit)

## REST API (Flask)

## Deployment

### Streamlit Cloud
1) Push this repo to GitHub.
2) In Streamlit Cloud, create an app pointing to `app_streamlit.py`.
3) Set Python version to 3.11 and use this repo’s `requirements.txt`.
4) Optional secrets: none required.

### Heroku
```bash
heroku create your-app
heroku stack:set container
git push heroku main
```
This repo includes a `Dockerfile` and `Procfile` for web dyno.

### PythonAnywhere (free tier)
- Create a new web app (manual config), set WSGI to run `app_streamlit.py` via `web: streamlit run ...` (or deploy Flask `api_flask.py`).
- Use a virtualenv, install `requirements.txt`.

### AWS (free tier ideas)
- EC2: build Docker image and run container exposing 8501.
- App Runner: connect to GitHub and auto-deploy container.
- Lightsail: simple instance; run Docker or Python service.

## Demo Dataset and Seeding
Place files:
- Resumes: `demo_resumes/` (20–30 files; PDF/DOCX/TXT)
- JDs: `demo_jds/` (5–10 .txt files)

Seed into SQLite (`app.db`):
```bash
python seed_demo_data.py
```

## Demo Video Checklist (2–4 minutes)
- Show uploading JD and multiple resumes.
- Toggle SBERT, view rankings and missing skills.
- Open Dashboard: skill distribution pie and average scores.
- Use Recommendations to suggest jobs for a resume.
- Briefly show API endpoints with a curl example.

## CI
GitHub Actions workflow at `.github/workflows/ci.yml` installs dependencies and runs smoke tests on pushes and PRs. After pushing to GitHub, replace the badge placeholder above with:
`[![CI](https://github.com/<your-org-or-user>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org-or-user>/<your-repo>/actions/workflows/ci.yml)`

### Run API
```bash
pip install -r requirements.txt
python api_flask.py
```

### Endpoints
- POST `/upload/resume` form-data: `file=@resume.pdf`
- POST `/upload/jd` form-data: `file=@jd.txt` or fields `text`, `title`
- GET `/resumes` → list stored resumes
- GET `/jds` → list stored JDs
- POST `/rank` json: `{ "jd_id": 1, "resume_ids": [2,3], "use_sbert": true }`

### Examples
```bash
curl -F "file=@path/to/resume.pdf" http://localhost:8000/upload/resume | jq
curl -F "file=@sample_job_descriptions/jd_data_scientist.txt" http://localhost:8000/upload/jd | jq
curl http://localhost:8000/resumes | jq
curl http://localhost:8000/jds | jq
curl -H "Content-Type: application/json" -d '{"jd_id":1, "resume_ids":[1], "use_sbert":false}' http://localhost:8000/rank | jq
```

### Run the app
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Features:
- Upload resumes (PDF/DOCX/TXT) and a job description (TXT/DOCX/PDF)
- Choose TF‑IDF or toggle Sentence‑Transformers
- View match scores and missing skills per candidate
- SQLite persistence in `app.db`

### Rank resumes for a JD (TF‑IDF baseline)
```bash
python rank_resumes.py --jd sample_job_descriptions/jd_data_scientist.txt --resumes path\to\resume1.pdf path\to\resume2.docx --output outputs\ranking_tfidf.json
```

### Rank resumes with Sentence Transformers (BERT embeddings)
```bash
python rank_resumes.py --jd sample_job_descriptions/jd_data_scientist.txt --resumes path\to\resume1.pdf path\to\resume2.docx --use-sbert --model sentence-transformers/all-MiniLM-L6-v2 --output outputs\ranking_sbert.json
```

Scoring includes:
- **TF‑IDF similarity** and optional **SBERT similarity**.
- **Section scores**: skills_match, experience_match, education_match.
- **Overall match** = average of available section scores.


