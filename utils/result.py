"""
Result computation + display for the Bangalore House Price Prediction app.

Split into two parts on purpose:
    - compute_result(): pure logic (no Streamlit calls) -- runs inference,
      formats the price, collects validation warnings. Easy to unit test.
    - render_result_card() / render_empty_state(): pure rendering, given
      an already-computed result dict.
"""

from typing import Optional, TypedDict

import streamlit as st

from utils.prediction import predict_price
from utils.inputs import PredictionInputs
from utils.validation import validate_inputs


class PredictionResult(TypedDict):
    success: bool
    error_message: Optional[str]
    formatted_price: Optional[str]
    warnings: list[str]
    location: str
    total_sqft: float
    bhk: int
    bath: int
    balcony: int
    availability: str


def format_price_inr(price_in_lakhs: float) -> str:
    """
    Convert a raw price in Lakhs (the model's training unit) into a
    human-friendly Indian Crore/Lakh string. 100 Lakhs = 1 Crore.
    """
    if price_in_lakhs < 0:
        price_in_lakhs = 0.0
    if price_in_lakhs >= 100:
        return f"₹ {price_in_lakhs / 100:.2f} Crore"
    return f"₹ {price_in_lakhs:.2f} Lakh"


def compute_result(inputs: PredictionInputs) -> PredictionResult:
    """Run inference and assemble everything the result card needs to display."""
    warnings = validate_inputs(inputs.total_sqft, inputs.bhk)

    try:
        predicted_price = predict_price(
            location=inputs.location,
            availability=inputs.availability,
            total_sqft=inputs.total_sqft,
            bath=inputs.bath,
            balcony=inputs.balcony,
            bhk=inputs.bhk,
        )
    except Exception:
        return PredictionResult(
            success=False,
            error_message=(
                "Something went wrong while estimating the price. "
                "Please check your inputs and try again."
            ),
            formatted_price=None,
            warnings=warnings,
            location=inputs.location,
            total_sqft=inputs.total_sqft,
            bhk=inputs.bhk,
            bath=inputs.bath,
            balcony=inputs.balcony,
            availability=inputs.availability,
        )

    return PredictionResult(
        success=True,
        error_message=None,
        formatted_price=format_price_inr(predicted_price),
        warnings=warnings,
        location=inputs.location,
        total_sqft=inputs.total_sqft,
        bhk=inputs.bhk,
        bath=inputs.bath,
        balcony=inputs.balcony,
        availability=inputs.availability,
    )


def render_result_card(result: PredictionResult) -> None:
    """Render an already-computed result. Never touches the model directly."""
    if not result["success"]:
        st.error(result["error_message"])
        return

    for w in result["warnings"]:
        st.warning(w)

    location_display = "Other / Not listed" if result["location"] == "other" else result["location"]

    with st.container(key="result_card"):
        st.markdown('<div class="result-label">Estimated Property Price</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-price">{result["formatted_price"]}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-caption">Based on the property details you provided.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "\n".join([
                '<div class="result-summary">',
                '<div class="result-summary-item">',
                '<div class="k">Location</div>',
                f'<div class="v">{location_display}</div>',
                '</div>',
                '<div class="result-summary-item">',
                '<div class="k">Area</div>',
                f'<div class="v">{result["total_sqft"]:,.0f} sqft</div>',
                '</div>',
                '<div class="result-summary-item">',
                '<div class="k">BHK</div>',
                f'<div class="v">{result["bhk"]}</div>',
                '</div>',
                '<div class="result-summary-item">',
                '<div class="k">Bathrooms</div>',
                f'<div class="v">{result["bath"]}</div>',
                '</div>',
                '<div class="result-summary-item">',
                '<div class="k">Balconies</div>',
                f'<div class="v">{result["balcony"]}</div>',
                '</div>',
                '<div class="result-summary-item">',
                '<div class="k">Availability</div>',
                f'<div class="v">{result["availability"]}</div>',
                '</div>',
                '</div>',
            ]),
            unsafe_allow_html=True,
        )


def render_empty_state() -> None:
    """Shown before the first prediction, instead of blank space."""
    st.markdown(
        '<p class="text-muted" style="text-align:center; padding: 1rem 0;">'
        "Fill in the property details above and click <strong>Predict Price</strong> "
        "to see an estimate.</p>",
        unsafe_allow_html=True,
    )
