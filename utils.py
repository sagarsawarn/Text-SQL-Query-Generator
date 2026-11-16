import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load API keys from a .env file next to this module (if present).
load_dotenv(Path(__file__).resolve().parent / ".env")

# Environment variables checked for each model provider, in order.
ENV_KEY_NAMES = {
    "GPT": ("OPENAI_API_KEY",),
    "Gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def get_env_api_key(model_type):
    """Return the API key configured in the environment/.env for a provider."""
    for name in ENV_KEY_NAMES.get(model_type, ()):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def get_api_key(model_type=None):
    """A key typed in the sidebar wins; otherwise fall back to the .env key."""
    model_type = model_type or st.session_state.get("model_type", "GPT")
    return st.session_state.get("api_key", "").strip() or get_env_api_key(model_type)


def init_session_state():
    defaults = {"model_type": "GPT", "model_name": "gpt-4o", "api_key": "", "text_tab_info": "", "text_tab_question": "", "excel_tab_question": "", "query_history": []}

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def widget_control(control_type="control_type"):
    if not (st.session_state["model_name"] and get_api_key()):
        return "Select a Model and add an API key (sidebar or .env file)"

    if control_type == "control_type":
        if not (st.session_state["text_tab_info"] and st.session_state["text_tab_question"]):
            return "Describe Your Data and/or Write Your Question"

    if control_type == "excel_tab_control":
        # "excel_tab_files" belongs to the file_uploader widget, so it must not be
        # pre-set in session state; read it defensively instead.
        if not (st.session_state.get("excel_tab_files") and st.session_state["excel_tab_question"]):
            return "Upload Your Excel(s) and/or Write Your Question"
