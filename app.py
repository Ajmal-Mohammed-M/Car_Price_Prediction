import streamlit as st
import pandas as pd
import pickle
import traceback

# ===============================
# Load preprocessing tools & model
# ===============================
@st.cache_resource
def load_tools():
    with open("preprocessing_tools.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("catboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_tools()

# ===============================
# Load dataset for filter options
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Preprocessed.csv")  # dataset before encoding/scaling
    return df

data = load_data()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the details below to get the predicted price of the car.")

# ===============================
# Show available columns (debugging)
# ===============================
st.sidebar.header("📑 Dataset Info")
st.sidebar.write("Columns detected in dataset:", list(data.columns))

# ===============================
# Column mapping
# 🔴 Change the values here to EXACTLY match your CSV headers
# Example: if your CSV header is 'Mileage(kmpl)' then put that instead of 'Mileage'
# ===============================
col_map = {
    "year": "Year",                 # Manufacturing Year column
    "mileage": "Mileage",           # Mileage column
    "engine_size": "Engine_Size",   # Engine Size column
    "brand": "Brand",               # Brand column
    "fuel_type": "Fuel_Type",       # Fuel Type column
    "transmission": "Transmission"  # Transmission column
}

# Validate columns
missing = [v for v in col_map.values() if v not in data.columns]
if missing:
    st.error(f"⚠️ These columns are missing in your Preprocessed.csv: {missing}")
    st.stop()

# ===============================
# Dynamic Input Fields
# ===============================
col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Manufacturing Year",
        min_value=int(data[col_map["year"]].min()),
        max_value=int(data[col_map["year"]].max()),
        value=int(data[col_map["year"]].median())
    )

    mileage = st.number_input(
        "Mileage",
        min_value=float(data[col_map["mileage"]].min()),
        max_value=float(data[col_map["mileage"]].max()),
        value=float(data[col_map["mileage"]].median())
    )

    engine_size = st.number_input(
        "Engine Size (CC)",
        min_value=int(data[col_map["engine_size"]].min()),
        max_value=int(data[col_map["engine_size"]].max()),
        value=int(data[col_map["engine_size"]].median())
    )

with col2:
    brand = st.selectbox("Car Brand", sorted(data[col_map["brand"]].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data[col_map["fuel_type"]].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data[col_map["transmission"]].dropna().unique()))

# Collect inputs into DataFrame
input_dict = {
    col_map["year"]: year,
    col_map["mileage"]: mileage,
    col_map["engine_size"]: engine_size,
    col_map["brand"]: brand,
    col_map["fuel_type"]: fuel_type,
    col_map["transmission"]: transmission
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        # Apply preprocessing
        processed_input = preprocessor.transform(input_df)

        # Predict with CatBoost
        prediction = model.predict(processed_input)

        st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")

    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
