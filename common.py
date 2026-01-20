import streamlit as st
from i18n import TEXT

def init_lang(default="en"):
    """Initialize lang in session_state."""
    if "lang" not in st.session_state:
        st.session_state.lang = default

def t(key: str) -> str:
    """Translate helper."""
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))

def render_lang_toggle(location="sidebar"):
    """
    Render a language toggle button on every page.
    location: 'sidebar' or 'top'
    """
    init_lang()

    # Button label: show the target language name
    label = TEXT["en"]["lang_btn_to_zh"] if st.session_state.lang == "en" else TEXT["en"]["lang_btn_to_en"]
    btn_text = f"🌐 {label}"

    container = st.sidebar if location == "sidebar" else st

    if container.button(btn_text, use_container_width=True):
        st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"
        st.rerun()