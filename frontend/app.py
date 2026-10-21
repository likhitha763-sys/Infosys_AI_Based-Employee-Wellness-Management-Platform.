"""Streamlit frontend for MoodMentor: chooses which page to show.

Run from the project root:
    python -m streamlit run frontend\\app.py
"""

import logging

import streamlit as st

import analysis_page
import api_client
import views

logger = logging.getLogger("moodmentor.frontend")

PAGES = {
    "landing": views.render_landing,
    "register": views.render_register,
    "login": views.render_login,
    "home": analysis_page.render_dashboard,
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
    except api_client.SessionExpiredError:
        # The login token expired: forget it and ask the user to log in again.
        views.log_out()
        st.session_state["notice"] = "Your session has expired. Please log in again."
        st.session_state["page"] = "login"
        st.rerun()
    except Exception:
        logger.exception("Unexpected error on page %s", page)
        st.error("Something unexpected went wrong. Please reload the page and try again.")


main()