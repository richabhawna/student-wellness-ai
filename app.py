
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Student Wellness AI",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = "student_wellness_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file is missing. Upload student_wellness_model.pkl."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# ----------------------------------
# HEADER
# ----------------------------------

st.title("🧠 Student Wellness AI")

st.subheader(
    "Experimental wearable-based stress estimation"
)

st.write(
    """
    This application uses machine learning to explore the
    relationship between wearable-device measurements and
    self-reported student stress.
    """
)

st.info(
    "Research prototype: This application does not diagnose "
    "mental-health conditions or replace professional advice. "
    "Its predictive performance has not been established."
)

st.divider()

# ----------------------------------
# INPUT SECTION
# ----------------------------------

st.header("⌚ Enter wearable measurements")

st.caption(
    "Enter daily measurements. The example values are for "
    "demonstration only; they are not live smartwatch readings."
)

col1, col2 = st.columns(2)

with col1:

    daily_steps = st.number_input(
        "Daily steps",
        min_value=0,
        max_value=100000,
        value=6500,
        step=100
    )

    hrv_rmssd_mean = st.number_input(
        "Average HRV RMSSD (ms)",
        min_value=0.0,
        max_value=300.0,
        value=65.0,
        step=1.0
    )

    oxygen_mean = st.number_input(
        "Average blood oxygen (%)",
        min_value=0.0,
        max_value=100.0,
        value=97.0,
        step=0.5
    )

with col2:

    overall_score = st.number_input(
        "Sleep score",
        min_value=0.0,
        max_value=100.0,
        value=82.0,
        step=1.0
    )

    deep_sleep_in_minutes = st.number_input(
        "Deep sleep (minutes)",
        min_value=0.0,
        max_value=600.0,
        value=90.0,
        step=5.0
    )

    active_interval_fraction = st.slider(
        "Active interval fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
        help="Fraction of recorded activity intervals classified as non-sedentary."
    )

st.caption(
    "Active interval fraction ranges from 0 to 1. "
    "For example, 0.30 means 30% of recorded intervals were active."
)

st.divider()

# ----------------------------------
# PREDICTION
# ----------------------------------

if st.button(
    "Predict Stress Score",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame([{
        "daily_steps": daily_steps,
        "hrv_rmssd_mean": hrv_rmssd_mean,
        "oxygen_mean": oxygen_mean,
        "overall_score": overall_score,
        "deep_sleep_in_minutes": deep_sleep_in_minutes,
        "active_interval_fraction": active_interval_fraction
    }])

    try:

        prediction = float(model.predict(input_data)[0])

        st.session_state["prediction"] = prediction

        st.success("Prediction generated successfully!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------------
# DISPLAY RESULT
# ----------------------------------

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    st.header("📊 Prediction Result")

    st.metric(
        label="Estimated self-reported stress score",
        value=f"{prediction:.2f} / 100"
    )

    st.write(
        "This is an experimental model output, not a validated "
        "stress assessment or mental-health diagnosis."
    )

    st.caption(
        "The model's held-out participant evaluation produced "
        "MAE = 21.513, RMSE = 26.900 and R² = 0.004. "
        "Predictions should therefore not be relied upon for "
        "individual decisions."
    )

    st.divider()

    st.subheader("Your wearable inputs")

    st.dataframe(
        input_data if "input_data" in locals()
        else pd.DataFrame([{
            "Daily steps": daily_steps,
            "HRV RMSSD": hrv_rmssd_mean,
            "Oxygen": oxygen_mean,
            "Sleep score": overall_score,
            "Deep sleep minutes": deep_sleep_in_minutes,
            "Active interval fraction": active_interval_fraction
        }]),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("General wellness suggestions")

    st.markdown("""
    - Maintain a consistent sleep schedule.
    - Take regular breaks during study sessions.
    - Include suitable physical activity in your routine.
    - Monitor your stress and discuss persistent concerns
      with a qualified professional.
    """)

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "Student Wellness AI | Academic research prototype"
)

st.caption(
    "This system is for educational and research purposes only. "
    "It is not a medical device."
)
