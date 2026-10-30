# MoodMentor

**AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System**
*Personalized Well-Being Through AI-Powered Emotional Analysis*

> **Status:** user accounts, sentiment analysis, emotion detection, and a mood
> dashboard, with results saved to a database. Personalized well-being
> recommendations are planned for a later milestone and are not included.

## What it does

MoodMentor reads written text (journal entries, messages, emails) and gives two
separate results:

| Result | Question it answers | Produced by |
|---|---|---|
| **Sentiment** | Is the overall tone positive, negative, or neutral, and how strongly? | VADER (rule-based) |
| **Emotion** | Which specific feeling does the text seem to express? | A pretrained emotion-classification model |

Every result is saved to the logged-in user's account, and a dashboard shows the
user's history as charts.

> **Important:** MoodMentor gives a basic, automatic reading of the tone of
> written text. It is **not** a medical or psychological diagnosis and cannot
> reliably tell how a person truly feels.

## Features

- **Landing page, registration, and login** with form validation and secure password storage.
- **Text analysis:** type or paste text, or upload `.txt` and `.csv` files.
- **Sentiment:** positive, negative, neutral, and compound scores with a label.
- **Emotion detection:** a detected emotion with an emoji and the model's confidence.
- **Mood dashboard:** summary cards, an emotional-rhythm line chart, an emotion
  doughnut, emotion and sentiment bar charts, emotion cards, and a recent-entries table.
- **Saved history:** every result is stored per user, shown newest first with paging.
- **Clear error handling:** empty input, bad files, an unreachable backend,
  expired logins, and a missing or switched-off emotion model.

## How it works

```
Visitor opens the app
        |
        v
  Streamlit frontend (frontend/)
  landing - register - login - dashboard - analysis - history
        |  HTTP (REST, JSON, login token)
        v
  FastAPI backend (backend/app/main.py)
        |
        +--> Accounts: users.py + security.py --> database (users table)
        |
        +--> Text preprocessing + VADER sentiment (sentiment.py)
        |
        +--> Emotion model (emotion.py)  [optional: sentiment still works without it]
        |
        +--> Results saved (analyses.py --> database, analyses table)
        |
        +--> Dashboard numbers calculated from saved results (dashboard.py)
        |
        v
  Results and chart data returned to Streamlit
```

The frontend and backend are separate applications. The frontend contains no
analysis or account logic. The charts are drawn from numbers calculated by the
backend from saved results, so nothing on the dashboard is made up.

Design notes:
- **VADER scores the original text**, not the preprocessed text, because cleaning
  removes capital letters, punctuation, and words that VADER uses.
- **Negation words** such as "not" are kept during stop-word filtering.
- **Sentiment labels** follow VADER's standard thresholds: compound >= 0.05 is
  Positive, <= -0.05 is Negative, otherwise Neutral.
- **Emotion is optional.** If it is switched off or the model cannot load, the
  analysis still returns the sentiment, with a short note.
- Uploaded files are only read as text. Their contents are never executed.

## Emotion detection

- **Model:** `SamLowe/roberta-base-go_emotions`, which recognizes 28 emotions. It
  runs on your computer (CPU) using the `transformers` and `torch` libraries.
  The first run downloads it, which needs an internet connection and several
  hundred megabytes of disk space.
- **Primary emotion:** the emotion with the highest score. The **confidence** is
  that score. It shows how strongly the model favors the emotion, and it is not
  a guarantee that the emotion is correct.
- **"No clear emotion":** if even the best score is below 0.30 (`MIN_CONFIDENCE`
  in `backend/app/emotion.py`), the result says so instead of guessing.
- **English only:** the model was trained on English text.

### Which of the requested emotions are supported

| Requested emotion | Model label | Support |
|---|---|---|
| Happiness 😊 | joy | Direct |
| Sadness 😢, Anger 😠, Fear 😨, Surprise 😮, Disgust 🤢, Love ❤️ | same names | Direct |
| Excitement 🤩, Confusion 🤔, Embarrassment 😳, Pride 🥹 | same names | Direct |
| Anxiety 😰 | nervousness | Close, but not identical. It is shown as "Nervousness". |
| Jealousy 😒, Shame 😔, Loneliness 🥺, Contentment 😌 | none | **Not supported by this model** |

The four unsupported emotions are never shown, because mapping nearby labels
onto them would be inaccurate. The model also detects emotions that were not in
the requested list (for example gratitude, optimism, curiosity, and relief), and
they are shown with their own emojis, which can be changed in `emotion.py`.

## Mood dashboard

The **Dashboard** tab shows, for the logged-in user:

| Part | What it shows |
|---|---|
| Summary cards | Number of entries, average compound score (with a +/- sign), most common sentiment, most common detected emotion |
| Emotional rhythm | Line chart of each entry's compound score in chronological order, with a zero line and hover details |
| Emotion mix | Doughnut chart of detected emotions with percentages |
| Emotion counts | Bar chart of entries per emotion |
| Sentiment mix | Bar chart of Positive, Neutral, and Negative counts |
| Emotion cards and recent entries | Top emotions as emoji cards, and a table of the newest entries |

Rules the dashboard follows:
- Everything comes from saved analyses. A user with no entries sees an
  explanatory message, not sample data.
- Entries saved before emotion detection existed count in the sentiment charts
  but not in the emotion charts, and the dashboard says how many have an emotion.
- "Most common emotion" ignores Neutral and "No clear emotion". Ties are shown as ties.
- The line chart shows the newest 20, 50, 100, or 200 entries, numbered by
  their position in the user's full history.

## Project structure

```
MoodMentor/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── sentiment.py     # preprocessing + VADER logic
│   │   ├── emotion.py       # emotion model, labels, and emojis
│   │   ├── dashboard.py     # dashboard numbers from saved results
│   │   ├── schemas.py       # request/response models
│   │   ├── database.py      # database connection and automatic column updates
│   │   ├── models.py        # users and analyses tables
│   │   ├── users.py         # account logic
│   │   ├── analyses.py      # saving and reading results
│   │   ├── security.py      # password hashing and login tokens
│   │   └── agents.py        # placeholder (later milestone)
│   ├── tests/               # automated checks
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app.py               # chooses which page to show
│   ├── views.py             # landing, register, login pages, sidebar
│   ├── analysis_page.py     # logged-in page: tabs, analysis, uploads, history
│   ├── dashboard_page.py    # the Dashboard tab
│   ├── charts.py            # Plotly chart builders
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
- Internet access for the first installation and the first model download
- About 2 GB of free disk space for the libraries and the emotion model

## Setup and installation

Run all commands from the project root (the folder that contains `backend` and
`frontend`) in PowerShell.

**1. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**2. Install the dependencies** (this includes PyTorch, which is a large download)

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

The emotion model downloads itself the first time it is used (or when the
backend starts), and is then kept in your user folder.

**4. Create the settings files**

```powershell
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
python -c "import secrets; print(secrets.token_hex(32))"
```

Open `backend\.env` and replace `change-me` after `SECRET_KEY=` with the value
that was printed. Keep it private and never commit it.

| File | Setting | Default | Meaning |
|---|---|---|---|
| `backend\.env` | `FRONTEND_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Frontend addresses allowed by CORS |
| `backend\.env` | `DATABASE_URL` | `sqlite:///./moodmentor.db` | Where accounts and results are stored |
| `backend\.env` | `SECRET_KEY` | (you generate it) | Signs login tokens; at least 32 characters |
| `backend\.env` | `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | How long a login lasts |
| `backend\.env` | `EMOTION_ENABLED` | `true` | Set to `false` to switch emotion detection off |
| `backend\.env` | `EMOTION_MODEL_NAME` | `SamLowe/roberta-base-go_emotions` | The emotion model to use |
| `frontend\.env` | `API_BASE_URL` | `http://127.0.0.1:8000` | Address of the backend |

## Running the application

Use **two terminals**, both opened in the project root with `(.venv)` active.

**Terminal 1: backend**

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Wait for `Application startup complete.` The database tables are created, and any
missing columns are added, automatically. A few seconds later the log shows
`Emotion model is ready.` (the first time it is downloaded first). The API
documentation is at http://127.0.0.1:8000/docs

**Terminal 2: frontend**

```powershell
python -m streamlit run frontend\app.py
```

The app opens at http://localhost:8501. To stop either one, click its terminal
and press `Ctrl + C`.

## Using the app

1. Open the landing page and choose **Create account**, then log in.
2. The **Dashboard** tab shows your history as cards and charts.
3. Use **Type text**, **Upload TXT**, or **Upload CSV** to analyze text. Each result
   shows the sentiment and the detected emotion, and is saved to your account.
4. **My history** lists every saved result, and a full report for any entry.
5. Use **Log out** in the sidebar when you are done.

## API overview

| Method | Path | Login needed | Purpose |
|---|---|---|---|
| GET | `/` and `/health` | No | Check that the API is running |
| POST | `/api/auth/register` | No | Create an account |
| POST | `/api/auth/login` | No | Check email and password, return a login token |
| GET | `/api/auth/me` | Yes | Return the logged-in user |
| POST | `/api/analyze` | Yes | Analyze one text and save the result |
| POST | `/api/analyze/batch` | Yes | Analyze several texts and save the results |
| GET | `/api/history` | Yes | The user's saved results, newest first |
| GET | `/api/dashboard` | Yes | Summary numbers and chart data (`rhythm_limit` 1 to 200) |

Analysis results contain the sentiment fields (`sentiment`, `compound_score`,
`positive_score`, `negative_score`, `neutral_score`, `original_text`,
`processed_text`) and the optional emotion fields (`predicted_emotion`,
`emotion_name`, `emotion_emoji`, `emotion_confidence`, `top_emotions`,
`emotion_note`). The emotion fields are empty when emotion detection is off or
unavailable, and for entries saved before it existed.

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
the folder where the backend is started.

| Table | Contents |
|---|---|
| `users` | Name, email (unique), and a hashed password |
| `analyses` | Original and processed text, the four scores, the sentiment label, the detected emotion and its confidence (both optional), the time, and the owning user |

When the backend starts, it adds any missing columns to existing tables without
deleting data. **Back up `moodmentor.db` before upgrading an existing project.**
To start with an empty database, stop the backend, delete `moodmentor.db`, and
start it again.

## Security notes

- Passwords are stored only as salted Argon2 hashes, never as plain text.
- Login tokens are signed JWTs that expire after `ACCESS_TOKEN_EXPIRE_MINUTES`.
- A wrong password and an unknown email give the same message.
- Every history and dashboard query is limited to the logged-in user.
- Error messages shown to users never contain stack traces or secrets.
- Text from entries is escaped before it is shown in chart hover boxes.
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
python -m backend.tests.check_emotion
python -m backend.tests.check_migration
python -m backend.tests.check_emotion_api
python -m backend.tests.check_dashboard
python -m frontend.tests.check_validation
python -m frontend.tests.check_analysis_page
python -m frontend.tests.check_charts
python -m frontend.tests.check_emotion_display
```

Each ends with `RESULT: ... passed, 0 failed`. The database and API checks use a
temporary in-memory database and never touch your real data. The emotion
logic checks use fake scores, so they are fast and repeatable. Some red log
lines are expected in the error-handling checks.

One more check loads the real emotion model (it may download it first) and
verifies that its output is well formed. It prints each sentence's emotion so
you can judge whether the results make sense:

```powershell
python -m backend.tests.check_emotion_model
```

### Manual acceptance tests

Start both applications, then:

| Area | Test | Expected |
|---|---|---|
| Accounts | Register, log in, try a wrong password | Clear messages; the same message for a wrong password and an unknown email |
| Sentiment | Analyze a positive, a negative, a neutral, and a negated sentence | Positive, Negative, Neutral, Negative |
| Emotion | Read the Detected emotion section | The emotion matches the first "Top model scores" entry |
| Input | Empty text, spaces only | A warning |
| Files | Upload `journal.txt`, `journal.csv`, `emotions.csv` | Results tables with emotion columns |
| Files | Upload `numbers_only.csv`, `empty.txt`, `malformed.csv` | A clear message, no crash |
| Dashboard | Compare the cards and charts with the database | The same counts |
| Privacy | Log in as a second user | An empty dashboard and history |
| Missing model | Set `EMOTION_MODEL_NAME=does-not-exist/nope` and restart | Sentiment still works; a note says emotion detection is unavailable |
| Backend down | Stop the backend and analyze text | "Cannot reach the backend" message |

For every result, check that the sentiment label matches the compound score:
>= 0.05 Positive, <= -0.05 Negative, otherwise Neutral.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` or `pip` not recognized | Install Python from python.org and tick "Add python.exe to PATH" |
| No `(.venv)` in the prompt | Run `.\.venv\Scripts\Activate.ps1` from the project root |
| `No module named backend` | You are not in the project root folder |
| Sidebar says **Backend: not reachable** | Start the backend and check `API_BASE_URL` in `frontend\.env` |
| Login fails with "temporarily unavailable" | `SECRET_KEY` in `backend\.env` is missing or shorter than 32 characters |
| Emotion section says unavailable or turned off | Check `EMOTION_ENABLED` and `EMOTION_MODEL_NAME` in `backend\.env`, and the backend log |
| The emotion model cannot download | Check your internet connection, free disk space, and run `python -m backend.tests.check_emotion_model` |
| Emotion charts are empty | Only entries analyzed with emotion detection on are included; analyze more text |
| `no such column: analyses.predicted_emotion` | Restart the backend; it adds missing columns on startup |
| 503 error mentioning spaCy or NLTK | Run the download commands in step 3 of the installation |
| You are logged out after refreshing the page | Expected; the login is kept only for the browser session |

## Limitations

- VADER is rule-based and designed mainly for English. It does not understand
  sarcasm, context, or mixed feelings well.
- The emotion model has moderate accuracy (its model card reports an overall F1
  score of about 0.61) and is weaker on rarer emotions. Its confidence is not a
  probability of being correct. It is English-only.
- Jealousy, Shame, Loneliness, and Contentment cannot be detected by this model.
  Anxiety is approximated by the model's "nervousness" label.
- The "No clear emotion" cutoff (0.30) is a starting value and may need tuning.
- Results are a rough indication of tone, not of a person's emotional state, and
  not a diagnosis.
- Refreshing the browser page logs you out, because the login is kept only in the Streamlit session.
- There is no password reset or email verification yet.
- The dashboard reloads its data whenever the page refreshes, which is fine for
  small histories but would need caching for very large ones.
- Entries analyzed before emotion detection was added have no emotion.

## Roadmap

Planned for later milestones: personalized well-being recommendations, further
account features, and optional use of a MySQL database.