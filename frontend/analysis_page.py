"""The logged-in page: analyze text, upload files, and see saved history.

This file only handles the user interface. Sentiment analysis and saving are
done by the FastAPI backend. Uploaded files are only read as text; their
contents are never executed.
"""

import io
import json

import pandas as pd
import streamlit as st

import api_client
import model_performance_page
import dashboard_page
import views

MAX_TEXT_LENGTH = 10_000  # characters per entry (must match the backend)
MAX_FILE_SIZE_MB = 2
HISTORY_PAGE_SIZE = 10
FINETUNED_THRESHOLD = 0.70  # decision threshold chosen on the validation set (Milestone 2)

MENU_PAGES = [
    "Dashboard",
    "Wellness Check-in",
    "Emotion Analytics",
    "Model Performance",
    "My History",
    "Privacy & Data",
]

SAFETY_MESSAGE = (
    "**This text may describe thoughts of harming yourself.** MoodMentor is a "
    "text-analysis tool, not a counsellor, and it cannot help in a crisis. "
    "If you are in immediate danger, call your local emergency number now "
    "(112 in India). You can also talk to someone free of charge at any hour: "
    "in India, call Tele-MANAS on **14416** (or 1-800-891-4416). If you are "
    "somewhere else, please contact your local crisis line. Telling someone you "
    "trust how you feel can help too. You do not have to go through this alone."
)

# Column names that usually hold the text in a CSV file.
TEXT_COLUMN_HINTS = {
    "text", "entry", "message", "content", "journal", "comment",
    "review", "body", "sentence", "note", "description",
}


def get_token():
    """The login token stored for this browser session."""
    return st.session_state.get("token")

def has_consent():
    """True once the user has ticked the consent box this session."""
    return bool(st.session_state.get("consent"))


def saving_enabled():
    """True unless the user switched saving off."""
    return bool(st.session_state.get("save_results", True))


def render_consent():
    """The consent and saving choices, shown at the top of every page."""
    with st.expander("Consent and saving choices", expanded=not has_consent()):
        st.checkbox(
            "I understand that MoodMentor is not a diagnosis or clinical assessment, and "
            "I agree to my text being analysed by the sentiment and emotion models.",
            key="consent",
        )
        st.checkbox(
            "Save my results to my account history (untick to analyse without saving).",
            value=True,
            key="save_results",
        )
    if not has_consent():
        st.info("Please tick the first box under 'Consent and saving choices' to start analysing text.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_text(text):
    """Return an error message if the text is not acceptable, otherwise None."""
    if not text or not text.strip():
        return "Please enter some text before analyzing."
    if len(text) > MAX_TEXT_LENGTH:
        return f"Text is too long. Please keep it under {MAX_TEXT_LENGTH:,} characters."
    return None


# ---------------------------------------------------------------------------
# Reading uploaded files (read as text only, never executed)
# ---------------------------------------------------------------------------


def read_upload(uploaded_file, extension):
    """Basic checks shared by TXT and CSV uploads. Returns (bytes, error)."""
    if not uploaded_file.name.lower().endswith(extension):
        return None, f"Unsupported file type. Please upload a {extension} file."
    if uploaded_file.size == 0:
        return None, "The file is empty."
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return None, f"The file is too large. The maximum size is {MAX_FILE_SIZE_MB} MB."

    data = uploaded_file.getvalue()
    if b"\x00" in data:
        return None, "This does not look like a plain-text file."
    return data, None


def decode_text(data):
    """Decode bytes as UTF-8, falling back to Windows-1252. None if both fail."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_txt_entries(data, one_entry_per_line):
    """Return (list_of_entries, error) for a TXT file."""
    text = decode_text(data)
    if text is None:
        return None, "The file could not be read as text. Please save it as UTF-8."
    if not text.strip():
        return None, "The TXT file contains no text."
    return (text.splitlines() if one_entry_per_line else [text]), None


def read_csv_table(data):
    """Return (DataFrame, error) for a CSV file."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            table = pd.read_csv(
                io.BytesIO(data), dtype=str, keep_default_na=False, encoding=encoding
            )
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The CSV file is empty."
        except pd.errors.ParserError:
            return None, "The CSV file is malformed and could not be read. Please check its format."
        except Exception:
            return None, "The CSV file could not be read. Please check that it is a valid CSV file."
    else:
        return None, "The CSV file could not be decoded. Please save it as UTF-8."

    if table.empty:
        return None, "The CSV file has column headers but no data rows."
    table.columns = [str(column) for column in table.columns]
    return table, None


def column_has_text(series):
    """True if the column has non-blank values that are not all numbers."""
    values = series.str.strip()
    values = values[values != ""]
    if values.empty:
        return False
    return not pd.to_numeric(values, errors="coerce").notna().all()


def guess_text_column(table, columns):
    """Pick the most likely text column: by name, else the longest text."""
    for column in columns:
        if column.strip().lower() in TEXT_COLUMN_HINTS:
            return column
    return max(columns, key=lambda column: table[column].str.len().mean())


def prepare_entries(raw_entries):
    """Strip entries and drop blank or too-long ones.

    Returns (entries, blank_count, too_long_count).
    """
    entries, blank, too_long = [], 0, 0
    for raw in raw_entries:
        entry = raw.strip()
        if not entry:
            blank += 1
        elif len(entry) > MAX_TEXT_LENGTH:
            too_long += 1
        else:
            entries.append(entry)
    return entries, blank, too_long


# ---------------------------------------------------------------------------
# Report wording and display
# ---------------------------------------------------------------------------


def build_report(result):
    """Write a plain-language explanation of one result.

    Returns a list of sentences. The label itself always comes from the backend;
    this only describes it.
    """
    label = result["sentiment"]
    strength = abs(result["compound_score"])

    if strength >= 0.6:
        level = "strongly"
    elif strength >= 0.3:
        level = "moderately"
    else:
        level = "mildly"

    if label == "Neutral":
        lines = [
            "The overall tone of this text is neutral. It contains little "
            "emotional wording, or the positive and negative wording roughly "
            "balance out."
        ]
        if strength >= 0.03:
            lines.append(
                "The score is close to the edge of the neutral range, so a small "
                "change in wording could tip it either way."
            )
    else:
        lines = [f"The overall tone of this text leans {level} {label.lower()}."]
        if strength < 0.15:
            lines.append(
                "The score is close to the neutral range, so this is only a weak signal."
            )

    if result["positive_score"] >= 0.15 and result["negative_score"] >= 0.15:
        lines.append(
            "The text contains both positive and negative wording, so the "
            "feelings in it may be mixed."
        )
    if not result["processed_text"].strip():
        lines.append(
            "The text contains very little ordinary wording (for example only "
            "symbols or numbers), so treat this result with extra caution."
        )
    return lines


# ---------------------------------------------------------------------------
# Emotion display helpers (text and tables only; no screen code)
# ---------------------------------------------------------------------------


def emotion_text(result):
    """The emoji and name of the detected emotion, or 'Not analyzed'."""
    name = result.get("emotion_name")
    if not name:
        return "Not analyzed"
    return f"{result.get('emotion_emoji') or ''} {name}".strip()


def confidence_text(result):
    """The model's confidence as a whole-number percentage, or a dash."""
    confidence = result.get("emotion_confidence")
    return "—" if confidence is None else f"{confidence:.0%}"


def emotion_explanation(result):
    """A plain-language sentence about the emotion result."""
    emotion = result.get("predicted_emotion")
    if emotion is None:
        return result.get("emotion_note") or (
            "No emotion was saved for this entry. Entries analyzed before "
            "emotion detection was added do not have one."
        )
    if emotion == "no_clear_emotion":
        return "The emotion model was not confident about any single emotion for this text."
    if emotion == "neutral":
        return (
            "The emotion model found no strong emotion in this text. "
            "This is a separate result from the sentiment above."
        )
    return (
        f"The emotion model's best guess for this text is {result.get('emotion_name')}. "
        "This is separate from the sentiment above, and the model can be wrong, "
        "especially on short or unusual text."
    )


def top_emotions_text(result):
    """The model's top scores as one line, or None when they are not available."""
    top = result.get("top_emotions")
    if not top:
        return None
    scores = ", ".join(f"{item['emotion']} {item['score']:.2f}" for item in top)
    return f"Top model scores: {scores}"
def engine_text(result):
    """Which emotion engine produced this result, in words (None if unknown)."""
    engine = result.get("emotion_engine")
    if engine == "finetuned":
        return "Engine: fine-tuned BERT (MoodMentor, trained on GoEmotions)"
    if engine == "pretrained":
        return "Engine: pretrained RoBERTa (GoEmotions)"
    return None


def detected_emotions(result):
    """The top emotions that reach the fine-tuned model's threshold."""
    top = result.get("top_emotions") or []
    return [item for item in top if item["score"] >= FINETUNED_THRESHOLD]


def render_comparison(result):
    """Show the VADER baseline and the emotion model side by side for one text."""
    top = result.get("top_emotions")
    if not top:
        return

    st.markdown("#### Baseline vs transformer")
    left, right = st.columns(2)

    with left:
        st.markdown("**Baseline: VADER**")
        st.write(
            f"One label for the whole text: **{result['sentiment']}** "
            f"(compound {result['compound_score']:+.2f})."
        )
        st.caption("VADER counts positive and negative words. It cannot name individual emotions.")

    with right:
        st.markdown("**Transformer: emotion model**")
        engine = engine_text(result)
        if engine:
            st.caption(engine)
        if result.get("emotion_engine") == "finetuned":
            found = detected_emotions(result)
            if found:
                names = ", ".join(item["emotion"].replace("_", " ") for item in found)
                st.write(f"Emotions found: **{names}**")
            else:
                st.write(
                    f"No emotion reached the {FINETUNED_THRESHOLD:.0%} confidence threshold, "
                    "so the model is unsure. Its closest guesses are below."
                )
        else:
            st.write(f"Best guess: **{emotion_text(result)}**")
        for item in top:
            label = item["emotion"].replace("_", " ")
            score = min(max(float(item["score"]), 0.0), 1.0)
            st.progress(score, text=f"{label}: {score:.0%}")
        st.caption("Only the top 3 scores are shown.")

    note = result.get("emotion_note")
    if note and result.get("predicted_emotion"):
        st.warning(note)


def results_table(results):
    """The table for the results of an uploaded file (several entries)."""
    return pd.DataFrame(
        {
            "#": range(1, len(results) + 1),
            "Text": [result["original_text"] for result in results],
            "Emotion": [emotion_text(result) for result in results],
            "Confidence": [confidence_text(result) for result in results],
            "Sentiment": [result["sentiment"] for result in results],
            "Positive": [result["positive_score"] for result in results],
            "Negative": [result["negative_score"] for result in results],
            "Neutral": [result["neutral_score"] for result in results],
            "Compound": [result["compound_score"] for result in results],
        }
    )


def history_table(items):
    """The table for one page of saved history."""
    return pd.DataFrame(
        {
            "Date": [format_time(item["created_at"]) for item in items],
            "Text": [shorten(item["original_text"]) for item in items],
            "Emotion": [emotion_text(item) for item in items],
            "Sentiment": [item["sentiment"] for item in items],
            "Compound": [item["compound_score"] for item in items],
        }
    )


def render_result(result, key_prefix="single"):
    """Show the sentiment report for one analyzed text."""
    if result.get("safety_alert"):
        st.error(SAFETY_MESSAGE)
    st.subheader("Sentiment report")

    columns = st.columns(5)
    columns[0].metric("Sentiment", result["sentiment"])
    columns[1].metric("Compound", f"{result['compound_score']:.4f}")
    columns[2].metric("Positive", f"{result['positive_score']:.3f}")
    columns[3].metric("Neutral", f"{result['neutral_score']:.3f}")
    columns[4].metric("Negative", f"{result['negative_score']:.3f}")

    for line in build_report(result):
        st.write(line)
    st.caption(
        "The compound score summarizes the whole text from -1 (most negative) to "
        "+1 (most positive). Scores of +0.05 or higher count as positive and "
        "-0.05 or lower as negative. The positive, neutral, and negative scores "
        "show the share of the text in each category and add up to about 1."
    )

    st.markdown("#### Detected emotion")
    st.caption(
        "The emotion comes from a separate model and is different from the sentiment "
        "above. It works best on English text."
    )
    emotion_columns = st.columns(3)
    emotion_columns[0].metric("Emotion", emotion_text(result))
    emotion_columns[1].metric(
        "Model confidence",
        confidence_text(result),
        help="How strongly the model favors this emotion. It is not a guarantee that the emotion is correct.",
    )
    st.write(emotion_explanation(result))
    scores = top_emotions_text(result)
    if scores:
        st.caption(scores)
    render_comparison(result)

    left, right = st.columns(2)
    left.text_area(
        "Original text",
        value=result["original_text"],
        height=150,
        disabled=True,
        key=f"{key_prefix}_original",
    )
    right.text_area(
        "Processed text (used for display only)",
        value=result["processed_text"],
        height=150,
        disabled=True,
        key=f"{key_prefix}_processed",
    )
    st.caption(views.DISCLAIMER)


def render_results_table(results, key_prefix):
    """Show a table and summary for several entries, or one full report."""
    if len(results) == 1:
        render_result(results[0], key_prefix=f"{key_prefix}_detail_1")
        return

    st.subheader(f"Results for {len(results)} entries")

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for result in results:
        counts[result["sentiment"]] += 1
    average = sum(result["compound_score"] for result in results) / len(results)

    columns = st.columns(4)
    columns[0].metric("Positive entries", counts["Positive"])
    columns[1].metric("Neutral entries", counts["Neutral"])
    columns[2].metric("Negative entries", counts["Negative"])
    columns[3].metric("Average compound", f"{average:.3f}")

    if any(result.get("safety_alert") for result in results):
        st.error(SAFETY_MESSAGE)
    st.dataframe(results_table(results), hide_index=True)
    st.caption(
        "The average compound score is a rough overview only. "
        "Each entry is scored separately."
    )

    chosen = st.selectbox(
        "Show the full report for entry:",
        list(range(1, len(results) + 1)),
        key=f"{key_prefix}_entry",
    )
    render_result(results[chosen - 1], key_prefix=f"{key_prefix}_detail_{chosen}")


# ---------------------------------------------------------------------------
# Text and file tabs
# ---------------------------------------------------------------------------


def show_file_results(state_key, signature):
    """Show stored results, but only if they belong to the current file."""
    stored = st.session_state.get(state_key)
    if not stored or stored["signature"] != signature:
        return
    if stored["blank"]:
        st.info(f"{stored['blank']} blank entries were skipped.")
    if stored["too_long"]:
        st.warning(
            f"{stored['too_long']} entries were skipped because they are longer "
            f"than {MAX_TEXT_LENGTH:,} characters."
        )
    if stored.get("saved", True):
        st.success(f"{len(stored['results'])} results were saved to your history.")
    else:
        st.success(f"{len(stored['results'])} results. Nothing was saved, because saving is switched off.")
    render_results_table(stored["results"], f"{state_key}_{abs(hash(signature))}")


def run_file_analysis(state_key, signature, raw_entries):
    """Send the entries of an uploaded file to the backend and store the results."""
    if not has_consent():
        st.warning("Please tick the consent box at the top of the page before analysing a file.")
        return
    entries, blank, too_long = prepare_entries(raw_entries)
    if not entries:
        st.error("No usable text was found in the file.")
        return

    save = saving_enabled()
    with st.spinner(f"Analyzing {len(entries)} entries..."):
        results, error = api_client.analyze_batch(get_token(), entries, save=save)

    if error:
        st.error(error)
        return

    st.session_state[state_key] = {
        "signature": signature,
        "results": results,
        "blank": blank,
        "too_long": too_long,
        "saved": save,
    }


def render_text_section():
    """Tab 1: text box, Analyze button, and the resulting report."""
    st.subheader("Enter your text")
    text = st.text_area(
        "Type or paste a journal entry, message, or paragraph:",
        height=200,
        max_chars=MAX_TEXT_LENGTH,
        placeholder="For example: I am very happy today. Everything is going well!",
        key="text_input",
    )

    if st.button("Analyze Sentiment", type="primary", key="text_button"):
        if not has_consent():
            st.warning("Please tick the consent box at the top of the page before analysing text.")
            return
        error = validate_text(text)
        if error:
            st.warning(error)
            return

        save = saving_enabled()
        with st.spinner("Analyzing..."):
            result, api_error = api_client.analyze_text(get_token(), text, save=save)

        if api_error:
            st.error(api_error)
            return

        if save:
            st.success("Analysis complete. The result was saved to your history.")
        else:
            st.success("Analysis complete. Nothing was saved, because saving is switched off.")
        render_result(result, key_prefix=f"single_{abs(hash(result['original_text']))}")


def render_txt_section():
    """Tab 2: upload and analyze a TXT file."""
    st.subheader("Upload a TXT file")
    uploaded = st.file_uploader("Choose a .txt file", type=["txt"], key="txt_file")
    mode = st.radio(
        "How should the file be split?",
        ["Whole file as one entry", "One entry per line"],
        key="txt_mode",
    )
    if uploaded is None:
        st.caption("Upload a file to continue.")
        return

    data, error = read_upload(uploaded, ".txt")
    if error:
        st.error(error)
        return
    entries, error = read_txt_entries(data, mode == "One entry per line")
    if error:
        st.error(error)
        return

    signature = (uploaded.name, uploaded.size, mode)
    if st.button("Analyze TXT file", type="primary", key="txt_button"):
        run_file_analysis("txt_results", signature, entries)
    show_file_results("txt_results", signature)


def render_csv_section():
    """Tab 3: upload a CSV file, choose the text column, and analyze it."""
    st.subheader("Upload a CSV file")
    uploaded = st.file_uploader("Choose a .csv file", type=["csv"], key="csv_file")
    if uploaded is None:
        st.caption("Upload a file to continue. The first row should contain column names.")
        return

    data, error = read_upload(uploaded, ".csv")
    if error:
        st.error(error)
        return
    table, error = read_csv_table(data)
    if error:
        st.error(error)
        return

    st.caption(f"Found {len(table)} rows. Preview of the first rows:")
    st.dataframe(table.head(5), hide_index=True)

    text_columns = [column for column in table.columns if column_has_text(table[column])]
    if not text_columns:
        st.error(
            "No column with text was found in this CSV file. "
            f"Its columns are: {', '.join(table.columns)}. "
            "Each one is either empty or contains only numbers."
        )
        return

    default = guess_text_column(table, text_columns)
    column = st.selectbox(
        "Which column contains the text to analyze?",
        text_columns,
        index=text_columns.index(default),
        key="csv_column",
    )

    signature = (uploaded.name, uploaded.size, column)
    if st.button("Analyze CSV file", type="primary", key="csv_button"):
        run_file_analysis("csv_results", signature, table[column].tolist())
    show_file_results("csv_results", signature)


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------


def shorten(text, length=80):
    """Collapse whitespace and cut long text for table display."""
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 3] + "..."


def format_time(value):
    """Show a stored UTC time in a readable form."""
    try:
        return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(value)


def change_history_page(step):
    """Button callback: move to the previous or next history page."""
    st.session_state["history_page"] = max(0, st.session_state.get("history_page", 0) + step)


def render_history_section():
    """Tab 4: the user's saved analyses, newest first."""
    st.subheader("Your saved analyses")
    st.button("Refresh history", key="history_refresh")

    page = st.session_state.get("history_page", 0)
    data, error = api_client.get_history(
        get_token(), HISTORY_PAGE_SIZE, page * HISTORY_PAGE_SIZE
    )
    if error:
        st.error(error)
        return

    total, items = data["total"], data["items"]
    if total == 0:
        st.info("You have no saved analyses yet. Analyze some text and it will appear here.")
        return
    if not items:  # the page number is past the end
        st.session_state["history_page"] = 0
        st.rerun()

    st.dataframe(history_table(items), hide_index=True)

    last_page = (total - 1) // HISTORY_PAGE_SIZE
    previous_column, label_column, next_column = st.columns([1, 2, 1])
    previous_column.button(
        "Previous", key="history_previous", on_click=change_history_page,
        args=(-1,), disabled=page == 0,
    )
    label_column.caption(f"Page {page + 1} of {last_page + 1} ({total} saved analyses)")
    next_column.button(
        "Next", key="history_next", on_click=change_history_page,
        args=(1,), disabled=page >= last_page,
    )

    chosen = st.selectbox(
        "Show the full report for:",
        list(range(len(items))),
        format_func=lambda index: (
            f"{format_time(items[index]['created_at'])}  |  "
            f"{shorten(items[index]['original_text'], 50)}"
        ),
        key=f"history_select_{page}",
    )
    render_result(items[chosen], key_prefix=f"history_{items[chosen]['id']}")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def render_privacy_section():
    """Tab: download or delete the user's saved data."""
    st.subheader("Your data and privacy")
    st.write(
        "When saving is switched on, MoodMentor stores your text, its sentiment scores, "
        "and the detected emotion with its confidence in your account. The emotion result "
        "is an estimate, not a diagnosis. You can download or delete your saved analyses here."
    )

    st.markdown("#### Download my data")
    if st.button("Prepare my data export", key="export_prepare"):
        data, error = api_client.export_history(get_token())
        if error:
            st.error(error)
        else:
            st.session_state["export_json"] = json.dumps(
                data["items"], indent=2, ensure_ascii=False
            )
            st.session_state["export_count"] = data["total"]
    if st.session_state.get("export_json"):
        st.download_button(
            f"Download {st.session_state['export_count']} saved analyses (JSON)",
            data=st.session_state["export_json"],
            file_name="moodmentor_my_data.json",
            mime="application/json",
            key="export_download",
        )

    st.markdown("#### Delete my saved analyses")
    st.caption(
        "This permanently removes every saved analysis from your account and cannot be "
        "undone. Your account itself is not deleted."
    )
    confirmed = st.checkbox("I understand this cannot be undone.", key="delete_confirm")
    if st.button("Delete all my saved analyses", key="delete_button", disabled=not confirmed):
        data, error = api_client.delete_history(get_token())
        if error:
            st.error(error)
        else:
            for key in ("txt_results", "csv_results", "export_json", "export_count"):
                st.session_state.pop(key, None)
            st.session_state["history_page"] = 0
            st.success(f"Deleted {data['deleted']} saved analyses.")


def render_checkin():
    """Page: type text or upload a file to analyse."""
    st.title("Wellness Check-in")
    st.write("Describe how you feel, or upload entries from a TXT or CSV file.")
    text_tab, txt_tab, csv_tab = st.tabs(["Type text", "Upload TXT", "Upload CSV"])
    with text_tab:
        render_text_section()
    with txt_tab:
        render_txt_section()
    with csv_tab:
        render_csv_section()


def render_emotion_analytics():
    """Page: compare the VADER baseline with the emotion model on one text."""
    st.title("Emotion Analytics")
    st.write(
        "Compare the baseline (VADER) with the fine-tuned transformer on one text. "
        "VADER gives one overall label. The transformer can report several emotions, "
        "each with a confidence."
    )
    text = st.text_area(
        "Enter a sentence or a short paragraph:",
        height=150,
        max_chars=MAX_TEXT_LENGTH,
        placeholder="For example: I'm excited about the trip but also really nervous.",
        key="analytics_input",
    )
    if st.button("Compare", type="primary", key="analytics_button"):
        if not has_consent():
            st.warning("Please tick the consent box at the top of the page first.")
            return
        error = validate_text(text)
        if error:
            st.warning(error)
            return

        save = saving_enabled()
        with st.spinner("Analyzing..."):
            result, api_error = api_client.analyze_text(get_token(), text, save=save)
        if api_error:
            st.error(api_error)
            return

        if result.get("safety_alert"):
            st.error(SAFETY_MESSAGE)
        if result.get("top_emotions"):
            render_comparison(result)
        else:
            st.info(result.get("emotion_note") or "Emotion detection is not available right now.")
        st.caption(
            "Saved to your history." if save else "Nothing was saved, because saving is switched off."
        )
        st.caption(views.DISCLAIMER)


def render_dashboard():
    """The page shown to logged-in users: a sidebar menu picks the section."""
    user = st.session_state.get("user", {})
    choice = st.sidebar.radio("Menu", MENU_PAGES, key="menu_page")

    render_consent()

    if choice == "Dashboard":
        st.title(f"Welcome, {user.get('full_name', '')}")
        dashboard_page.render_mood_dashboard()
    elif choice == "Wellness Check-in":
        render_checkin()
    elif choice == "Emotion Analytics":
        render_emotion_analytics()
    elif choice == "My History":
        st.title("My History")
        render_history_section()
    elif choice == "Model Performance":
        model_performance_page.render_model_performance()    
    else:
        st.title("Privacy & Data")
        render_privacy_section()