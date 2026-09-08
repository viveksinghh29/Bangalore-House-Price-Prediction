"""
Inference layer for the Bangalore House Price Prediction app.

Responsibilities:
    - Load the trained sklearn Pipeline (preprocessing + model) once.
    - Load supporting metadata (expected feature order, valid categories, metrics).
    - Build a single-row DataFrame from raw user input, in the exact column
      order the pipeline was trained on.
    - Run prediction and return a plain float (price in Lakhs INR).

This module contains no UI code (no widgets, no layout) -- it only uses
Streamlit's caching decorators, which is the idiomatic way to avoid
reloading the model/metadata on every rerun. It can still be run and
tested standalone (see the __main__ block); Streamlit's cache decorators
work safely outside of a live app session too.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Paths are resolved relative to this file, so the project works after
# cloning to any location on any OS (Windows-friendly, deployment-friendly).
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "Bangalore-project.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
CATEGORIES_PATH = MODEL_DIR / "categories.json"
METRICS_PATH = MODEL_DIR / "metrics.json"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained sklearn Pipeline (cached as a resource -- loaded once per process)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure model/Bangalore-project.joblib exists."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_feature_columns():
    """Load the exact feature order the model expects."""
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_categories():
    """Load the valid categorical values the encoder was fit on."""
    with open(CATEGORIES_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load saved evaluation metrics (used in the 'About the Model' section)."""
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def build_feature_row(location: str, availability: str, total_sqft: float,
                       bath: int, balcony: int, bhk: int) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the exact column order and dtypes
    the model was trained on: ['availability', 'location', 'total_sqft',
    'bath', 'balcony', 'bhk'].

    Raw values in, no defaults silently substituted — callers (validation.py /
    the UI) are responsible for ensuring values are sane before this is called.
    """
    feature_columns = load_feature_columns()

    row = {
        "availability": availability,
        "location": location,
        "total_sqft": float(total_sqft),
        "bath": int(bath),
        "balcony": int(balcony),
        "bhk": int(bhk),
    }

    missing = [col for col in feature_columns if col not in row]
    if missing:
        raise ValueError(f"build_feature_row is missing required columns: {missing}")

    # Reindex to the exact trained column order — this matters because the
    # ColumnTransformer's `remainder='passthrough'` columns are positional.
    return pd.DataFrame([row], columns=feature_columns)


def predict_price(location: str, availability: str, total_sqft: float,
                   bath: int, balcony: int, bhk: int) -> float:
    """
    Run the full inference pipeline and return the predicted price in Lakhs INR.
    Raises the underlying exception on failure — the UI layer is responsible
    for catching it and showing a friendly message (Phase 6/7).
    """
    model = load_model()
    X = build_feature_row(location, availability, total_sqft, bath, balcony, bhk)
    prediction = model.predict(X)[0]
    return float(prediction)


if __name__ == "__main__":
    # Standalone sanity check — run with: python -m utils.prediction
    # (from the project root) to verify the inference layer works before
    # connecting it to Streamlit.
    sample_categories = load_categories()
    sample_location = "Whitefield" if "Whitefield" in sample_categories["location"] else sample_categories["location"][0]
    sample_availability = "Ready To Move" if "Ready To Move" in sample_categories["availability"] else sample_categories["availability"][0]

    print("Feature columns:", load_feature_columns())
    print("Sample location used:", sample_location)
    print("Sample availability used:", sample_availability)

    price = predict_price(
        location=sample_location,
        availability=sample_availability,
        total_sqft=1200,
        bath=2,
        balcony=1,
        bhk=2,
    )
    print(f"Predicted price: {price:.2f} Lakhs")

    # Also test an out-of-training-set category to confirm handle_unknown='ignore'
    # degrades gracefully instead of throwing.
    price_unknown = predict_price(
        location="SomeTypoedLocationXYZ",
        availability="Immediate",
        total_sqft=1200,
        bath=2,
        balcony=1,
        bhk=2,
    )
    print(f"Predicted price with unknown categories: {price_unknown:.2f} Lakhs")
