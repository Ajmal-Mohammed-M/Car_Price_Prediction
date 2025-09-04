import streamlit as st
import pandas as pd
import pickle

# ===============================
# Load preprocessing pipeline & model
# ===============================
@st.cache_resource
def load_tools():
    with open("preprocessing_tools.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("catboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_tools()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Enter the car details below to predict the price.")

# ===============================
# Input fields (example: adjust to match your dataset columns)
# ===============================
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2015)
    mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=15.0)
    engine_size = st.number_input("Engine Size (CC)", min_value=600, max_value=5000, value=1500)

with col2:
    brand = st.selectbox("Car Brand", ["Toyota", "Hyundai", "Ford", "BMW", "Audi"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Create dataframe from user input
input_dict = {
    "year": year,
    "mileage": mileage,
    "engine_size": engine_size,
    "brand": brand,
    "fuel_type": fuel_type,
    "transmission": transmission
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        # Apply preprocessing (scaling + encoding)
        processed_input = preprocessor.transform(input_df)

        # Predict using CatBoost
        prediction = model.predict(processed_input)

        st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
