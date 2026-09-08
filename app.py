# Bangalore House Price Prediction — Streamlit App
import streamlit as st

from utils.theme import load_theme
from utils.hero import render_hero
from utils.inputs import render_prediction_card, WIDGET_KEYS
from utils.result import compute_result, render_result_card, render_empty_state
from utils.insights import render_model_insights

st.set_page_config(
    page_title="Bangalore House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_theme()
render_hero()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

inputs = render_prediction_card()

if inputs.reset_clicked:
    for key in WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.last_result = None
    st.rerun()

if inputs.submitted:
    with st.spinner("Estimating price..."):
        st.session_state.last_result = compute_result(inputs)

if st.session_state.last_result is not None:
    render_result_card(st.session_state.last_result)
else:
    render_empty_state()

render_model_insights()
