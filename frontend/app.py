"""Streamlit frontend for MoodMentor: chooses which page to show.

Run from the project root:
    python -m streamlit run frontend\\app.py
"""

import logging

import streamlit as st

import views

logger = logging.getLogger("moodmentor.frontend")

PAGES = {
    "landing": views.render_landing,
    "register": views.render_register,
    "login": views.render_login,
    "home": views.render_home,
}


def choose_page(requested, logged_in):
    """Decide which page to show. Logged-out visitors cannot see the dashboard."""
    if requested not in PAGES:
        return "landing"
    if requested == "home" and not logged_in:
        return "login"
    if requested in ("login", "register") and logged_in:
        return "home"
    return requested


def main():
    st.set_page_config(page_title="MoodMentor", page_icon="💬", layout="wide")

    st.session_state.setdefault("page", "landing")
    logged_in = views.is_logged_in()
    page = choose_page(st.session_state["page"], logged_in)
    st.session_state["page"] = page

    views.render_sidebar(logged_in)

    try:
        PAGES[page]()
    except Exception:
        logger.exception("Unexpected error on page %s", page)
        st.error("Something unexpected went wrong. Please reload the page and try again.")


main()