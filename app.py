import streamlit as st
import pandas as pd
import pickle
import traceback
import re

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
# Show available columns
# ===============================
st.sidebar.header("📑 Dataset Info")
st.sidebar.write("Columns detected in dataset:", list(data.columns))

# ===============================
# Column mapping (using your real dataset headers)
# ===============================
col_map = {
    "year": "Year",
    "kilometer": "Kilometer",
    "engine": "Engine",
    "make": "Make",
    "model": "Model",
    "fuel_type": "Fuel Type",
    "transmission": "Transmission"
}

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

    kilometer = st.number_input(
        "Kilometers Driven",
        min_value=int(data[col_map["kilometer"]].min()),
        max_value=int(data[col_map["kilometer"]].max()),
        value=int(data[col_map["kilometer"]].median())
    )

    # Engine values like "1498 CC" → extract digits
    engine_options = data[col_map["engine"]].dropna().unique()
    engine = st.selectbox("Engine Size", engine_options)
    engine_numeric = int(re.findall(r"\d+", str(engine))[0]) if re.findall(r"\d+", str(engine)) else 0

with col2:
    make = st.selectbox("Car Make", sorted(data[col_map["make"]].dropna().unique()))
    model = st.selectbox("Car Model", sorted(data[col_map["model"]].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data[col_map["fuel_type"]].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data[col_map["transmission"]].dropna().unique()))

# ===============================
# Build input DataFrame
# ===============================
input_dict = {
    col_map["year"]: year,
    col_map["kilometer"]: kilometer,
    col_map["engine"]: engine,  # keep original string, preprocessing may handle it
    col_map["make"]: make,
    col_map["model"]: model,
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
