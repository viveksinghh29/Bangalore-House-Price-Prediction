"""
Theme layer — loads the project's CSS and injects it into the Streamlit app.
Kept separate from prediction/validation logic so UI styling changes never
touch inference code.
"""

from pathlib import Path
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"


def load_theme() -> None:
    """Inject the project's custom CSS into the current Streamlit page."""
    if not STYLE_PATH.exists():
        return
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
