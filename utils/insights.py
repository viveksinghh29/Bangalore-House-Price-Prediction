"""
Optional model insights section — kept separate from the main prediction
experience (rendered inside a collapsed expander below it).

Only shows metrics that were actually computed during training and saved
to model/metrics.json. Nothing here is invented or rounded up.
"""

import streamlit as st

from utils.prediction import load_metrics


def render_model_insights() -> None:
    metrics = load_metrics()

    with st.expander("About the Model", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "Gradient Boosting")
            st.caption(metrics["model_type"])
        with col2:
            st.metric("R² Score (test)", f"{metrics['r2_test']:.3f}")
            st.caption("Higher is better, max 1.0")
        with col3:
            st.metric("Training Data", f"{metrics['n_rows_final']:,} rows")
            st.caption(f"{metrics['n_features']} input features")

        st.markdown(
            "\n".join([
                '<p class="text-muted" style="margin-top: 0.8rem;">',
                'This model is a <strong>Gradient Boosting Regressor</strong> trained on',
                f'{metrics["n_rows_final"]:,} cleaned property records from Bangalore, using',
                f'{metrics["n_features"]} features: location, availability, total area, bathrooms,',
                'balconies, and BHK. On held-out test data it achieves an',
                f'<strong>R\u00b2 of {metrics["r2_test"]:.3f}</strong>',
                f'(MAE \u2248 {metrics["mae"]:.1f} Lakhs, RMSE \u2248 {metrics["rmse"]:.1f} Lakhs).',
                '</p>',
                '<p class="text-muted">',
                f'For transparency: training-set R\u00b2 is {metrics["r2_train"]:.3f}, noticeably higher',
                'than the test-set score \u2014 a sign of some overfitting typical of tree-based models',
                'on a dataset this size. Treat predictions as estimates for exploration, not',
                'precise valuations.',
                '</p>',
            ]),
            unsafe_allow_html=True,
        )
