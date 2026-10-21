"""Pages of the MoodMentor frontend: landing, register, login, and dashboard.

This file only handles the user interface. Accounts and passwords are
handled by the FastAPI backend; requests go through api_client.py.
"""

import streamlit as st

import api_client
import validation

DISCLAIMER = (
    "MoodMentor gives a basic, automatic reading of the tone of written text. "
    "It is not a medical or psychological diagnosis, and it cannot reliably "
    "tell how a person truly feels."
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def is_logged_in():
    """True if a login token is stored in this browser session."""
    return bool(st.session_state.get("token"))


def go_to(page):
    """Button callback: switch to another page."""
    st.session_state["page"] = page


def log_in(token, user):
    """Remember the login and open the dashboard."""
    st.session_state["token"] = token
    st.session_state["user"] = user
    st.session_state["page"] = "home"


def log_out():
    """Button callback: forget the login and return to the landing page."""
    st.session_state.clear()
    st.session_state["page"] = "landing"


def centered_column():
    """A narrower column in the middle of the page, used for forms."""
    _, middle, _ = st.columns([1, 2, 1])
    return middle


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(logged_in):
    """Navigation buttons, the disclaimer, and the backend status."""
    with st.sidebar:
        st.header("MoodMentor")

        if logged_in:
            user = st.session_state.get("user", {})
            st.text(f"Signed in as {user.get('full_name', '')}")
            st.button("Dashboard", key="side_home", on_click=go_to, args=("home",))
            st.button("Log out", key="side_logout", on_click=log_out)
        else:
            st.button("Home", key="side_landing", on_click=go_to, args=("landing",))
            st.button("Log in", key="side_login", on_click=go_to, args=("login",))
            st.button("Register", key="side_register", on_click=go_to, args=("register",))

        st.divider()
        st.caption(DISCLAIMER)
        st.divider()
        st.caption(f"Backend address: {api_client.API_BASE_URL}")
        if api_client.backend_is_online():
            st.success("Backend: connected")
        else:
            st.error("Backend: not reachable")


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------


def render_landing():
    st.title("MoodMentor")
    st.subheader("Personalized Well-Being Through AI-Powered Emotional Analysis")
    st.write(
        "Write about your day, paste a message, or upload your notes. MoodMentor "
        "reads the text and tells you whether its overall tone is positive, "
        "negative, or neutral, and keeps a history so you can look back."
    )

    if is_logged_in():
        st.success("You are logged in.")
        st.button(
            "Open my dashboard", type="primary", key="landing_home",
            on_click=go_to, args=("home",),
        )
    else:
        left, right, _ = st.columns([1, 1, 3])
        left.button(
            "Create account", type="primary", key="landing_register",
            on_click=go_to, args=("register",),
        )
        right.button("Log in", key="landing_login", on_click=go_to, args=("login",))

    st.divider()
    st.subheader("How it works")
    steps = [
        ("1. Write or upload", "Type or paste a journal entry, message, or paragraph. You can also upload TXT and CSV files."),
        ("2. Analyze", "MoodMentor cleans the text and scores its tone as positive, negative, or neutral using VADER."),
        ("3. Look back", "Every result is saved to your account, so you can review your history."),
    ]
    for column, (title, text) in zip(st.columns(3), steps):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(text)

    st.subheader("Good to know")
    st.write(
        "- Your account and saved results are stored in the application's database. "
        "Passwords are stored in scrambled (hashed) form, never as plain text.\n"
        "- VADER is a rule-based tool that works best on English text. It can miss "
        "sarcasm, context, and mixed feelings."
    )
    st.caption(DISCLAIMER)
    st.caption(
        "If you are worried about your well-being, please talk to someone you "
        "trust or a qualified professional."
    )


# ---------------------------------------------------------------------------
# Registration page
# ---------------------------------------------------------------------------


def render_register():
    with centered_column():
        st.title("Create your account")

        with st.form("register_form"):
            full_name = st.text_input("Full name", max_chars=validation.MAX_NAME_LENGTH)
            email = st.text_input("Email address", max_chars=validation.MAX_EMAIL_LENGTH)
            password = st.text_input(
                "Password",
                type="password",
                max_chars=validation.MAX_PASSWORD_LENGTH,
                help=(
                    f"At least {validation.MIN_PASSWORD_LENGTH} characters, "
                    "with at least one letter and one number."
                ),
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                max_chars=validation.MAX_PASSWORD_LENGTH,
            )
            submitted = st.form_submit_button("Create account", type="primary")

        st.button(
            "I already have an account", key="register_to_login",
            on_click=go_to, args=("login",),
        )

        if not submitted:
            return

        error = validation.validate_registration(full_name, email, password, confirm_password)
        if error:
            st.error(error)
            return

        with st.spinner("Creating your account..."):
            _, api_error = api_client.register_user(full_name.strip(), email.strip(), password)
        if api_error:
            st.error(api_error)
            return

        st.session_state["flash"] = "Your account was created. Please log in."
        st.session_state["page"] = "login"
        st.rerun()


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------


def render_login():
    with centered_column():
        st.title("Log in")

        flash = st.session_state.pop("flash", None)
        if flash:
            st.success(flash)

        with st.form("login_form"):
            email = st.text_input("Email address", max_chars=validation.MAX_EMAIL_LENGTH)
            password = st.text_input(
                "Password", type="password", max_chars=validation.MAX_PASSWORD_LENGTH
            )
            submitted = st.form_submit_button("Log in", type="primary")

        st.button(
            "Create a new account", key="login_to_register",
            on_click=go_to, args=("register",),
        )

        if not submitted:
            return

        error = validation.validate_login(email, password)
        if error:
            st.error(error)
            return

        with st.spinner("Logging in..."):
            data, api_error = api_client.login_user(email.strip(), password)
        if api_error:
            st.error(api_error)
            return

        token = data.get("access_token")
        user = data.get("user")
        if not token or not user:
            st.error("The backend returned an unexpected response. Please try again.")
            return

        log_in(token, user)
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard (temporary; replaced by the analysis page in the next step)
# ---------------------------------------------------------------------------


def render_home():
    user = st.session_state.get("user", {})
    st.title(f"Welcome, {user.get('full_name', '')}")
    st.success("You are logged in.")
    st.info("The text-analysis page is added in the next step.")
    st.text(f"Signed in as {user.get('email', '')}")