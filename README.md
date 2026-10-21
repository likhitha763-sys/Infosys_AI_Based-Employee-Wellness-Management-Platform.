# MoodMentor

**AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System**
*Personalized Well-Being Through AI-Powered Emotional Analysis*

> **Status: Milestone 1.** User accounts (landing page, registration, login) and
> baseline text-sentiment analysis, with results saved to a database.
> Advanced emotion classification and personalized recommendations are planned
> for later milestones and are not included.

## What it does

MoodMentor reads written text (journal entries, messages, emails) and reports
whether its overall tone is **Positive**, **Negative**, or **Neutral**, with
positive, negative, neutral, and compound scores. Every result is saved to the
logged-in user's account so it can be reviewed later.

> **Important:** MoodMentor gives a basic, automatic reading of the tone of
> written text. It is **not** a medical or psychological diagnosis and cannot
> reliably tell how a person truly feels.

## Features

- **Landing page** that explains the project, with links to register and log in.
- **Registration and login** with form validation and secure password storage.
- **Text analysis:** type or paste text and analyze it.
- **File uploads:** `.txt` (whole file or one entry per line) and `.csv`
  (choose the text column), with every entry analyzed.
- **Saved history:** every result is stored for the logged-in user, shown newest
  first with paging and a full report for each entry.
- **Clear error handling** for empty input, bad files, an unreachable backend,
  and expired logins.

## How it works

```
Visitor opens the app
        |
        v
  Streamlit frontend (frontend/)
  landing - register - login - analysis page
        |  HTTP (REST, JSON, login token)
        v
  FastAPI backend (backend/app/main.py)
        |
        +--> Accounts: users.py + security.py --> database (users table)
        |
        +--> Text preprocessing (sentiment.py)
                cleaning -> tokenization -> stop words -> lemmatization
                |
                v
             VADER sentiment analysis (scores the original text)
                |
                v
        Result saved (analyses.py --> database, analyses table)
                |
                v
        Result returned to Streamlit and shown as a report
```

The frontend and backend are separate applications. The frontend contains no
analysis or account logic; it only sends requests to the API and shows what comes back.

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
│   │   ├── database.py      # database connection
│   │   ├── models.py        # users and analyses tables
│   │   ├── users.py         # account logic (create user, check login)
│   │   ├── analyses.py      # saving and reading results
│   │   ├── security.py      # password hashing and login tokens
│   │   └── agents.py        # placeholder (later milestone)
│   ├── tests/               # automated checks
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app.py               # chooses which page to show
│   ├── views.py             # landing, register, login pages, sidebar
│   ├── analysis_page.py     # logged-in page: analyze, uploads, history
│   ├── api_client.py        # HTTP calls to the backend
│   ├── validation.py        # form rules
│   ├── tests/               # automated checks
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

Generate a secret key for login tokens:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Open `backend\.env` and replace `change-me` after `SECRET_KEY=` with the value
that was printed. Keep it private and never commit it.

| File | Setting | Default | Meaning |
|---|---|---|---|
| `backend\.env` | `FRONTEND_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Frontend addresses allowed by CORS |
| `backend\.env` | `DATABASE_URL` | `sqlite:///./moodmentor.db` | Where accounts and results are stored |
| `backend\.env` | `SECRET_KEY` | (you generate it) | Signs login tokens; must be at least 32 characters |
| `backend\.env` | `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | How long a login lasts |
| `frontend\.env` | `API_BASE_URL` | `http://127.0.0.1:8000` | Address of the backend |

The backend refuses to issue login tokens while `SECRET_KEY` is missing, still
`change-me`, or shorter than 32 characters. `.env` files and the database file
are listed in `.gitignore`.

## Running the application

Use **two terminals**, both opened in the project root with `(.venv)` active.

**Terminal 1: backend**

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Wait for `Application startup complete.` The database tables are created
automatically on startup. The interactive API documentation is at
http://127.0.0.1:8000/docs

**Terminal 2: frontend**

```powershell
python -m streamlit run frontend\app.py
```

The app opens at http://localhost:8501. The sidebar shows **Backend: connected**
when the two applications can talk to each other. To stop either one, click its
terminal and press `Ctrl + C`.

## Using the app

1. Open the landing page and choose **Create account**.
2. Register with your name, an email address, and a password (at least 8
   characters, with at least one letter and one number).
3. Log in. You arrive at the analysis page.
4. Use **Type text**, **Upload TXT**, or **Upload CSV** to analyze text. Every
   result is saved to your account.
5. Open **My history** to review earlier results.
6. Use **Log out** in the sidebar when you are done.

## API overview

| Method | Path | Login needed | Purpose |
|---|---|---|---|
| GET | `/` and `/health` | No | Check that the API is running |
| POST | `/api/auth/register` | No | Create an account |
| POST | `/api/auth/login` | No | Check email and password, return a login token |
| GET | `/api/auth/me` | Yes | Return the logged-in user |
| POST | `/api/analyze` | Yes | Analyze one text and save the result |
| POST | `/api/analyze/batch` | Yes | Analyze several texts and save the results |
| GET | `/api/history` | Yes | The user's saved results, newest first (`limit`, `offset`) |

Logged-in requests send the token in the header `Authorization: Bearer <token>`.
In `/docs`, use the **Authorize** button and paste the token.

Limits: 10,000 characters per entry, 500 entries per batch request (the frontend
sends larger files in several requests), 2 MB per uploaded file.

| Status | Meaning |
|---|---|
| 200 / 201 | Success / account created |
| 400 | Empty text, a weak password, or a blank entry in a batch |
| 401 | Not logged in, expired session, or wrong email or password |
| 409 | An account with this email already exists |
| 422 | Invalid request, such as a missing field or text that is too long |
| 503 | An NLTK or spaCy resource is missing (the message says how to install it) |
| 500 | Unexpected error (details are logged on the server, never shown to users) |

## Database

SQLite is used by default; the file `moodmentor.db` is created automatically in
the folder where the backend is started (the project root).

| Table | Contents |
|---|---|
| `users` | Name, email (unique), and a hashed password |
| `analyses` | Original and processed text, the four scores, the label, the time, and the owning user |

To start with an empty database, stop the backend, delete `moodmentor.db`, and
start the backend again. Other databases such as MySQL are a planned extension:
the code uses SQLAlchemy, so it would need a different `DATABASE_URL` and a
matching database driver package.

## Security notes

- Passwords are stored only as salted Argon2 hashes, never as plain text.
- Login tokens are signed JWTs that expire after `ACCESS_TOKEN_EXPIRE_MINUTES`.
- A wrong password and an unknown email give the same message, so the login
  form does not reveal which emails are registered.
- Every history query is limited to the logged-in user, so users cannot see
  each other's texts.
- Error messages shown to users never contain stack traces or secrets.
- Real `.env` files and the database file must not be committed.

## Testing

### Automated checks

With `(.venv)` active, from the project root:

```powershell
python -m backend.tests.check_sentiment
python -m backend.tests.check_error_handling
python -m backend.tests.check_database
python -m backend.tests.check_security
python -m backend.tests.check_users
python -m backend.tests.check_api
python -m frontend.tests.check_validation
python -m frontend.tests.check_analysis_page
```

Each ends with `RESULT: ... passed, 0 failed`. The database and API checks use a
temporary in-memory database and never touch your real data. Some red log lines
appear in `check_error_handling`; they are expected.

### Manual acceptance tests

Start both applications, then work through this table. Sample files are in `sample_data`.

| # | Test | Expected |
|---|---|---|
| 1 | Open the app while logged out | Landing page with Create account and Log in |
| 2 | Submit invalid registration forms | Clear error messages |
| 3 | Register, then log in | Analysis page with four tabs |
| 4 | Register the same email again | "An account with this email already exists." |
| 5 | Log in with a wrong password or unknown email | The same "Incorrect email or password." message |
| 6 | Analyze `I am very happy today. Everything is going well!` | Positive |
| 7 | Analyze `I feel very sad and disappointed today.` | Negative |
| 8 | Analyze `The meeting starts at 10 AM tomorrow.` | Neutral |
| 9 | Analyze `I am not happy with the result.` | Negative |
| 10 | Analyze empty text, then spaces only | Warning message |
| 11 | Open **My history** | The saved results, newest first |
| 12 | Upload `journal.txt` (one entry per line) | Table of 4 results |
| 13 | Upload `journal.csv`, column `entry` | Table of 4 results |
| 14 | Upload `numbers_only.csv` | "No column with text was found" |
| 15 | Upload `empty.txt`, `spaces_only.txt`, `empty.csv`, `headers_only.csv`, `malformed.csv` | A clear message each, no crash |
| 16 | Log in as a second user and open **My history** | Empty; no other user's texts |
| 17 | Stop the backend and click Analyze | "Cannot reach the backend" message |
| 18 | Call `POST /api/analyze` in `/docs` without authorizing | 401 |

For every result, check that the label matches the compound score:
>= 0.05 Positive, <= -0.05 Negative, otherwise Neutral.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` or `pip` not recognized | Install Python from python.org and tick "Add python.exe to PATH" |
| No `(.venv)` in the prompt | Run `.\.venv\Scripts\Activate.ps1` from the project root |
| `No module named backend` | You are not in the project root folder |
| Sidebar says **Backend: not reachable** | Start the backend and check `API_BASE_URL` in `frontend\.env` |
| Log in fails with "Login is temporarily unavailable" | `SECRET_KEY` in `backend\.env` is missing or shorter than 32 characters |
| 503 error mentioning spaCy or NLTK | Run the download commands in step 3 of the installation |
| You are logged out after refreshing the page | Expected; the login is kept only for the browser session |
| "Your session has expired" | Log in again; logins last `ACCESS_TOKEN_EXPIRE_MINUTES` |
| Port already in use | Stop the other process or use another port, and update `API_BASE_URL` to match |

## Limitations

- VADER is a rule-based tool designed mainly for English text. It does not
  understand sarcasm, context, or mixed feelings well.
- Results are a rough indication of tone, not of a person's emotional state.
- Refreshing the browser page logs you out, because the login token is kept
  only in the Streamlit session.
- There is no password reset or email verification yet.
- The first analysis after starting the backend takes a few seconds while the
  language models load.

## Roadmap (not part of Milestone 1)

Later milestones are planned to add deeper emotion classification, personalized
well-being recommendations, and further account features.