"""Text preprocessing and VADER sentiment analysis for MoodMentor (Milestone 1).

This module contains no web code. FastAPI (main.py) calls analyze_text().

Preprocessing stages:
    1. Raw text input
    2. Text cleaning
    3. Tokenization
    4. Stop-word filtering
    5. Lemmatization
    6. Processed text output
"""

import re
from functools import lru_cache

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmptyTextError(ValueError):
    """Raised when the text is empty or contains only whitespace."""


class NLPResourceError(RuntimeError):
    """Raised when an NLTK or spaCy language resource is missing."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Words we keep even though NLTK lists them as stop words, because they
# change the meaning or strength of a sentiment ("not happy", "very sad").
WORDS_TO_KEEP = {"no", "nor", "not", "never", "very", "too", "but"}

# Contractions that do not follow the simple "n't -> not" rule.
SPECIAL_CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "shan't": "shall not",
    "ain't": "is not",
}

# General contraction endings, applied after the special cases.
CONTRACTION_ENDINGS = [
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'m\b", " am"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
    (r"'s\b", ""),
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")

# VADER's standard thresholds.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


# ---------------------------------------------------------------------------
# Loading resources (done once, then reused)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_stop_words():
    """Return NLTK's English stop words, minus the words we want to keep."""
    try:
        words = set(stopwords.words("english"))
    except LookupError as exc:
        raise NLPResourceError(
            "NLTK stop-word list is missing. "
            "Run: python -m nltk.downloader stopwords"
        ) from exc
    return frozenset(words - WORDS_TO_KEEP)


@lru_cache(maxsize=1)
def _get_spacy_model():
    """Load the spaCy English model (parser and NER are off to save time)."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise NLPResourceError(
            "spaCy model 'en_core_web_sm' is missing. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


@lru_cache(maxsize=1)
def _get_vader():
    """Create the VADER analyzer once."""
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_text(text):
    """Make sure the input is a non-empty string."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not text.strip():
        raise EmptyTextError("Text must not be empty or only whitespace.")


# ---------------------------------------------------------------------------
# Preprocessing stages
# ---------------------------------------------------------------------------


def clean_text(text):
    """Stage 2: lowercase, remove URLs/emails, expand contractions,
    remove punctuation, and tidy up spaces."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")  # curly quotes
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)

    for short_form, long_form in SPECIAL_CONTRACTIONS.items():
        text = text.replace(short_form, long_form)
    for pattern, replacement in CONTRACTION_ENDINGS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^\w\s]|_", " ", text)  # drop punctuation and emojis
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(cleaned_text):
    """Stage 3: split the cleaned text into word tokens using NLTK."""
    try:
        return word_tokenize(cleaned_text)
    except LookupError as exc:
        raise NLPResourceError(
            "NLTK tokenizer data is missing. "
            "Run: python -m nltk.downloader punkt_tab"
        ) from exc


def remove_stop_words(tokens):
    """Stage 4: remove common filler words but keep negations."""
    stop_words = _get_stop_words()
    return [token for token in tokens if token not in stop_words]


def lemmatize(tokens):
    """Stage 5: reduce each word to its base form using spaCy
    (for example "going" -> "go", "was" -> "be")."""
    if not tokens:
        return []
    nlp = _get_spacy_model()
    doc = nlp(" ".join(tokens))
    return [token.lemma_.lower() for token in doc if not token.is_space]


def preprocess_text(text):
    """Run all stages and return the result of each one."""
    _validate_text(text)

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered_tokens = remove_stop_words(tokens)
    lemmas = lemmatize(filtered_tokens)

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "processed_text": " ".join(lemmas),
    }


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------


def get_sentiment_label(compound_score):
    """Turn a VADER compound score into Positive, Negative or Neutral."""
    if compound_score >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound_score <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_text(text, use_processed_text=False):
    """Preprocess the text, score it with VADER, and return the result.

    By default VADER scores the ORIGINAL text, because cleaning removes
    punctuation, capital letters and words that VADER relies on.
    """
    steps = preprocess_text(text)  # also validates the input

    text_to_score = steps["processed_text"] if use_processed_text else text
    scores = _get_vader().polarity_scores(text_to_score)

    return {
        "original_text": text,
        "processed_text": steps["processed_text"],
        "positive_score": scores["pos"],
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "compound_score": scores["compound"],
        "sentiment": get_sentiment_label(scores["compound"]),
    }
    
    

def analyze_batch(texts, use_processed_text=False):
    """Analyze several texts and return one result per text.

    Raises EmptyTextError (naming the entry number) if any entry is blank.
    """
    results = []
    for position, text in enumerate(texts, start=1):
        try:
            results.append(analyze_text(text, use_processed_text))
        except EmptyTextError as exc:
            raise EmptyTextError(
                f"Entry {position} is empty or only whitespace."
            ) from exc
    return results