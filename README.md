# MoodMentor

**AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System**
*Personalized Well-Being Through AI-Powered Emotional Analysis*

> **Status: Milestone 1.** Text ingestion and baseline sentiment analysis.
> Advanced emotion classification, personalized recommendations, authentication,
> and the MySQL database are planned for later milestones and are not included yet.

## What it does

MoodMentor reads written text (journal entries, messages, emails) and reports
whether its overall tone is **Positive**, **Negative**, or **Neutral**, with
positive, negative, neutral, and compound scores.

You can:
- Type or paste text and analyze it.
- Upload a `.txt` file (the whole file as one entry, or one entry per line).
- Upload a `.csv` file, choose the column that contains the text, and analyze every row.

> **Important:** MoodMentor gives a basic, automatic reading of the tone of
> written text. It is **not** a medical or psychological diagnosis and cannot
> reliably tell how a person truly feels.

## How it works

```
User enters text or uploads a file
            |
            v
   Streamlit frontend   (frontend/app.py)
            |  HTTP (REST, JSON)
            v
     FastAPI backend    (backend/app/main.py)
            |
            v
 Text preprocessing     cleaning -> tokenization -> stop words -> lemmatization
            |
            v
  VADER sentiment analysis (scores the original text)
            |
            v
 Label + compound score returned to Streamlit
            |
            v
 Sentiment report shown to the user
```

The frontend and backend are separate applications. The frontend contains no
analysis logic; it only sends text to the API and shows what comes back.

Design notes:
- **VADER scores the original text**, not the preprocessed text, because cleaning
  removes capital letters, punctuation, and words that VADER uses. The processed
  text is shown separately.
- **Negation words** such as "not", "no", and "never" are kept during stop-word
  filtering, and contractions such as "don't" become "do not".
- **Labels** follow VADER's standard thresholds: compound score >= 0.05 is
  Positive, <= -0.05 is Negative, otherwise Neutral.
- Uploaded files are only read as text. Their contents are never executed.

## Project structure

```
MoodMentor/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI routes
│   │   ├── sentiment.py     # preprocessing + VADER logic
│   │   ├── schemas.py       # request/response models
│   │   ├── agents.py        # placeholder (later milestone)
│   │   ├── database.py      # placeholder (later milestone)
│   │   ├── models.py        # placeholder (later milestone)
│   │   └── security.py      # placeholder (later milestone)
│   ├── tests/
│   │   ├── check_sentiment.py
│   │   └── check_error_handling.py
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit interface
│   ├── .env.example
│   └── requirements.txt
├── sample_data/             # small files for manual testing
├── .gitignore
└── README.md
```

## Requirements

- Windows with PowerShell
- Python 3.10 or newer (developed and tested with Python 3.14)
- VS Code (optional but recommended)
- Internet access for the first installation

## Setup and installation

Run all commands from the project root (the folder that contains `backend` and
`frontend`) in PowerShell.

**1. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. If PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**2. Install the dependencies**

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install -r frontend\requirements.txt
```

**3. Download the language resources**

```powershell
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt_tab stopwords
```

**4. Create the settings files**

```powershell
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
```

| File | Setting | Default | Meaning |
|---|---|---|---|
| `backend\.env` | `FRONTEND_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Frontend addresses allowed by CORS |
| `frontend\.env` | `API_BASE_URL` | `http://127.0.0.1:8000` | Address of the backend |

The defaults work as they are. Never commit real `.env` files; they are listed in `.gitignore`.

## Running the application

Use **two terminals**, both opened in the project root with `(.venv)` active.

**Terminal 1: backend**

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Wait for `Application startup complete.` The interactive API documentation is at
http://127.0.0.1:8000/docs

**Terminal 2: frontend**

```powershell
python -m streamlit run frontend\app.py
```

The app opens at http://localhost:8501. The sidebar shows **Backend: connected**
when the two applications can talk to each other.

To stop either one, click its terminal and press `Ctrl + C`.

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/` and `/health` | Check that the API is running |
| POST | `/api/analyze` | Analyze one text: `{"text": "..."}` |
| POST | `/api/analyze/batch` | Analyze several texts: `{"texts": ["...", "..."]}` |

Limits: 10,000 characters per entry, 500 entries per batch request (the frontend
sends larger files in several requests), 2 MB per uploaded file.

Example:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/analyze -Method Post -ContentType "application/json" -Body '{"text": "I am very happy today. Everything is going well!"}'
```

Each result contains `original_text`, `processed_text`, `positive_score`,
`negative_score`, `neutral_score`, `compound_score`, and `sentiment`.

| Status | Meaning |
|---|---|
| 200 | Success |
| 400 | Empty or whitespace-only text (or a blank entry in a batch) |
| 422 | Invalid request, such as a missing field or text that is too long |
| 503 | An NLTK or spaCy resource is missing (the message says how to install it) |
| 500 | Unexpected error (details are logged on the server, never shown to users) |

## Testing

### Automated checks

With `(.venv)` active, from the project root:

```powershell
python -m backend.tests.check_sentiment
python -m backend.tests.check_error_handling
```

Each ends with `RESULT: ... passed, 0 failed`. Some red log lines appear during
the second script; those are expected.

### Manual acceptance tests

Start both applications, then use the files in `sample_data`.

| # | Test | How | Expected |
|---|---|---|---|
| 1 | Positive | Type `I am very happy today. Everything is going well!` | Positive |
| 2 | Negative | Type `I feel very sad and disappointed today.` | Negative |
| 3 | Neutral | Type `The meeting starts at 10 AM tomorrow.` | Neutral |
| 4 | Negation | Type `I am not happy with the result.` | Negative |
| 5 | Empty input | Click **Analyze Sentiment** with an empty box | Warning message |
| 6 | Whitespace only | Type spaces, then click the button | Warning message |
| 7 | Valid TXT | **Upload TXT**, `journal.txt`, one entry per line | Table with 4 entries |
| 8 | Valid CSV | **Upload CSV**, `journal.csv`, column `entry` | Table with 4 entries |
| 9 | CSV without text | **Upload CSV**, `numbers_only.csv` | "No column with text was found" |
| 10 | Backend unavailable | Stop the backend, then analyze text | "Cannot reach the analysis backend" message |

Also try `empty.txt`, `spaces_only.txt`, `empty.csv`, `headers_only.csv`, and
`malformed.csv`; each should show a clear message and no crash.

For every result, check that the label matches the compound score:
>= 0.05 Positive, <= -0.05 Negative, otherwise Neutral.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` or `pip` not recognized | Install Python from python.org and tick "Add python.exe to PATH" |
| No `(.venv)` in the prompt | Run `.\.venv\Scripts\Activate.ps1` from the project root |
| `No module named backend` | You are not in the project root folder |
| Sidebar says **Backend: not reachable** | Start the backend (Terminal 1) and check `API_BASE_URL` in `frontend\.env` |
| 503 error mentioning spaCy or NLTK | Run the download commands in step 3 of the installation |
| Port already in use | Stop the other process or use another port, and update `API_BASE_URL` to match |

## Limitations

- VADER is a rule-based tool designed mainly for English text. It does not
  understand sarcasm, context, or mixed feelings well.
- Results are a rough indication of tone, not of a person's emotional state.
- The first analysis after starting the backend takes a few seconds while language models load.

## Roadmap (not part of Milestone 1)

Later milestones are planned to add deeper emotion classification, personalized
well-being recommendations, MySQL storage, and user accounts.