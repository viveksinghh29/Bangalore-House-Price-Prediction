"""
Prediction card UI — collects exactly the inputs the trained model needs:
location, availability, total_sqft, bath, balcony, bhk.

No feature is added here that wasn't part of the training data (see
utils/prediction.py's feature_columns.json for the source of truth).
Dropdown options come directly from categories.json, so a user can never
select a value the encoder wasn't fit on.

Widgets use explicit keys so app.py can reset them back to defaults via
the "Start Over" button.
"""

from dataclasses import dataclass

import streamlit as st

from utils.prediction import load_categories, load_metrics

# Widget keys, exposed so app.py can clear them on reset.
WIDGET_KEYS = [
    "pc_location", "pc_availability", "pc_sqft",
    "pc_bhk", "pc_bath", "pc_balcony",
]


@dataclass
class PredictionInputs:
    location: str
    availability: str
    total_sqft: float
    bath: int
    balcony: int
    bhk: int
    submitted: bool
    reset_clicked: bool


def _ordered_availability(options: list[str]) -> list[str]:
    """Put 'Ready To Move' first (the overwhelmingly common case), then the rest sorted."""
    rest = sorted(o for o in options if o != "Ready To Move")
    if "Ready To Move" in options:
        return ["Ready To Move"] + rest
    return rest


def _ordered_locations(options: list[str]) -> list[str]:
    """Alphabetical, with 'other' relabeled and pushed to the end as a catch-all."""
    known = sorted(o for o in options if o != "other")
    if "other" in options:
        return known + ["other"]
    return known


def render_prediction_card() -> PredictionInputs:
    categories = load_categories()
    metrics = load_metrics()

    location_options = _ordered_locations(categories["location"])
    availability_options = _ordered_availability(categories["availability"])

    bhk_lo, bhk_hi = metrics["bhk_range"]
    bath_lo, bath_hi = metrics["bath_range"]
    balcony_lo, balcony_hi = metrics["balcony_range"]
    sqft_lo, _sqft_hi = metrics["total_sqft_range"]

    with st.container(key="prediction_card"):
        st.markdown('<div class="card-heading">Property Details</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox(
                "Location",
                options=location_options,
                index=location_options.index("Whitefield") if "Whitefield" in location_options else 0,
                format_func=lambda x: "Other / Not listed" if x == "other" else x,
                help="Choose the closest matching neighborhood. Select 'Other / Not listed' if yours isn't shown.",
                key="pc_location",
            )
        with col2:
            total_sqft = st.number_input(
                "Total Area (sqft)",
                min_value=float(sqft_lo),
                max_value=10000.0,
                value=1200.0,
                step=50.0,
                help=f"Built-up area in square feet (model trained on {int(sqft_lo)}–52,272 sqft).",
                key="pc_sqft",
            )

        col3, col4 = st.columns(2)
        with col3:
            bhk = st.slider(
                "BHK (Bedrooms)",
                min_value=int(bhk_lo),
                max_value=int(bhk_hi),
                value=2,
                help=f"Number of bedrooms ({bhk_lo}–{bhk_hi}). Capped at {bhk_hi} to match training data.",
                key="pc_bhk",
            )
        with col4:
            bath = st.slider(
                "Bathrooms",
                min_value=int(bath_lo),
                max_value=min(int(bath_hi), 10),
                value=2,
                help="Number of bathrooms.",
                key="pc_bath",
            )

        col5, col6 = st.columns(2)
        with col5:
            balcony = st.slider(
                "Balconies",
                min_value=int(balcony_lo),
                max_value=int(balcony_hi),
                value=1,
                key="pc_balcony",
            )
        with col6:
            availability = st.selectbox(
                "Availability",
                options=availability_options,
                index=0,
                help="'Ready To Move' or a specific possession date used in the training data.",
                key="pc_availability",
            )

        st.write("")
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            submitted = st.button("Predict Price", type="primary", use_container_width=True)
        with btn_col2:
            reset_clicked = st.button("Start Over", use_container_width=True)

    return PredictionInputs(
        location=location,
        availability=availability,
        total_sqft=total_sqft,
        bath=bath,
        balcony=balcony,
        bhk=bhk,
        submitted=submitted,
        reset_clicked=reset_clicked,
    )
