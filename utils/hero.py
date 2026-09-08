"""
Hero section for the Bangalore House Price Prediction app.

Uses the Bangalore skyline image from:
app/assets/bangalore.jpg

The image itself contains the main title:
"BANGALORE HOUSE PRICE PREDICTION MODEL"

Therefore, no duplicate title is rendered over the image.
"""

import base64
from pathlib import Path

import streamlit as st


# ============================================================
# Asset Configuration
# ============================================================

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

PHOTO_PATH = ASSETS_DIR / "bangalore.png"


# ============================================================
# Load Hero Image
# ============================================================

def _photo_as_data_uri() -> str | None:
    """
    Convert the Bangalore hero image into a base64 data URI.

    Returns None if the image does not exist.
    """

    if not PHOTO_PATH.exists():
        return None

    encoded = base64.b64encode(
        PHOTO_PATH.read_bytes()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# Render Hero
# ============================================================

def render_hero() -> None:
    """
    Render the Bangalore skyline hero section.

    The uploaded Bangalore image is used as the background.
    """

    photo_uri = _photo_as_data_uri()

    # --------------------------------------------------------
    # Image available
    # --------------------------------------------------------

    if photo_uri:

        html = "\n".join([
            f'<div class="hero hero-with-image" '
            f'style="background-image: url(\'{photo_uri}\');">',

            '<div class="hero-image-overlay"></div>',

            '</div>',
        ])

    # --------------------------------------------------------
    # Image missing
    # --------------------------------------------------------

    else:

        html = "\n".join([
            '<div class="hero hero--fallback">',

            '<div class="hero-fallback-content">',

            '<span class="hero-badge">',
            'ML Powered • Bangalore Real Estate',
            '</span>',

            '<h1 class="hero-title">',
            'Bangalore House Price Prediction',
            '</h1>',

            '<p class="hero-subtitle">',
            'Estimate Bangalore property prices using machine learning.',
            '</p>',

            '</div>',

            '</div>',
        ])

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    st.markdown(
        html,
        unsafe_allow_html=True,
    )