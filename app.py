
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="German Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(
        BASE_DIR / "models/electricity_forecasting_model.joblib"
    )
    features = joblib.load(
        BASE_DIR / "models/feature_names.joblib"
    )
    return model, features

model, features = load_model()

# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("⚡ German Electricity Demand Forecasting")

st.markdown("""
**Machine Learning | Time-Series Forecasting | Germany**

Predict electricity demand one hour ahead using
a trained Gradient Boosting regression model.

**Data source:** Bundesnetzagentur — SMARD  
**Model:** Histogram-based Gradient Boosting  
**Forecast horizon:** One hour ahead
""")

st.divider()

# --------------------------------------------------
# 4. MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("Model Performance — Test Year 2025")

col1, col2, col3 = st.columns(3)

col1.metric("Mean Absolute Error", "500.46 MWh")
col2.metric("Root Mean Squared Error", "666.45 MWh")
col3.metric("MAE Reduction vs Baseline", "71.87%")

st.caption(
    "Results from the held-out 2025 test dataset. "
    "Performance on future data may differ."
)

st.divider()

# --------------------------------------------------
# 5. PREDICTION SECTION
# --------------------------------------------------

st.subheader("Electricity Demand Prediction")


# --------------------------------------------------
# 6. USER INPUTS
# --------------------------------------------------

st.markdown(
    "Enter the forecast date, time, and historical "
    "electricity demand values."
)

col1, col2 = st.columns(2)

with col1:
    forecast_date = st.date_input(
        "Forecast date"
    )

    forecast_hour = st.selectbox(
        "Forecast hour (Berlin time)",
        options=list(range(24)),
        index=12
    )

with col2:
    lag_1 = st.number_input(
        "Electricity demand 1 hour ago (MWh)",
        min_value=0.0,
        value=50000.0,
        step=100.0
    )

    lag_24 = st.number_input(
        "Electricity demand 24 hours ago (MWh)",
        min_value=0.0,
        value=50000.0,
        step=100.0
    )

    lag_168 = st.number_input(
        "Electricity demand 168 hours ago (MWh)",
        min_value=0.0,
        value=50000.0,
        step=100.0
    )

# Derive calendar features from the selected date
selected_date = pd.Timestamp(forecast_date)

day_of_week = selected_date.dayofweek
month = selected_date.month
is_weekend = int(day_of_week >= 5)

# Prepare the model input using the trained feature order
input_data = pd.DataFrame([{
    "hour": forecast_hour,
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": is_weekend,
    "lag_1": lag_1,
    "lag_24": lag_24,
    "lag_168": lag_168
}])[features]

# --------------------------------------------------
# 7. GENERATE PREDICTION
# --------------------------------------------------

if st.button("Predict Electricity Demand", type="primary"):

    prediction = model.predict(input_data)[0]

    st.success("Forecast generated successfully!")

    st.metric(
        "Predicted Electricity Demand",
        f"{prediction:,.2f} MWh"
    )

    st.caption(
        "Model prediction based on the historical demand "
        "values and calendar inputs provided. "
        "This is not a live electricity demand measurement."
    )




