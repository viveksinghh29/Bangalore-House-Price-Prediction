"""
Input validation for the Bangalore House Price Prediction app.

These are advisory, non-blocking checks -- they warn the user that an
estimate may be less reliable for unusual combinations, but never prevent
a prediction from running. The model can still produce an output for any
value within the widget bounds; this just adds honesty about where the
model is on shakier ground.
"""

# Matches the outlier rule applied to training data
# (rows with total_sqft / bhk < 300 were removed before training).
MIN_SQFT_PER_BHK = 300


def validate_inputs(total_sqft: float, bhk: int) -> list[str]:
    """Return a list of human-readable warning messages (empty if none)."""
    warnings: list[str] = []

    if bhk > 0 and (total_sqft / bhk) < MIN_SQFT_PER_BHK:
        warnings.append(
            f"{total_sqft:,.0f} sqft for {bhk} BHK is smaller than anything the model "
            "was trained on for that bedroom count — treat this estimate with extra caution."
        )

    if total_sqft > 10000:
        warnings.append(
            "This is a very large property. The model saw few examples this size, "
            "so the estimate may be less precise."
        )

    return warnings
