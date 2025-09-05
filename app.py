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
        tools = pickle.load(f)
    with open("catboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tools, model

tools, model = load_tools()

st.sidebar.subheader("🔧 Preprocessing Tools Info")
st.sidebar.write("Type:", type(tools))
if isinstance(tools, dict):
    st.sidebar.write("Keys:", list(tools.keys()))

# ===============================
# Load dataset for UI options
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Preprocessed.csv")

data = load_data()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the details below to get the predicted price of the car.")

# ===============================
# Input Fields (from Preprocessed.csv columns)
# ===============================
col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Manufacturing Year",
        min_value=int(data["Year"].min()),
        max_value=int(data["Year"].max()),
        value=int(data["Year"].median())
    )

    kilometer = st.number_input(
        "Kilometers Driven",
        min_value=int(data["Kilometer"].min()),
        max_value=int(data["Kilometer"].max()),
        value=int(data["Kilometer"].median())
    )

    engine_options = data["Engine"].dropna().unique()
    engine = st.selectbox("Engine Size", engine_options)

with col2:
    make = st.selectbox("Car Make", sorted(data["Make"].dropna().unique()))
    model_name = st.selectbox("Car Model", sorted(data["Model"].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data["Fuel Type"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data["Transmission"].dropna().unique()))

# ===============================
# Build raw input DataFrame (same as Preprocessed.csv format)
# ===============================
input_dict = {
    "Year": year,
    "Kilometer": kilometer,
    "Engine": engine,
    "Make": make,
    "Model": model_name,
    "Fuel Type": fuel_type,
    "Transmission": transmission
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Apply preprocessing correctly
# ===============================
def apply_preprocessing(df, tools):
    if isinstance(tools, dict):
        if "preprocessor" in tools:
            return tools["preprocessor"].transform(df)
        else:
            st.error("❌ Could not find 'preprocessor' in preprocessing tools.")
            return None
    else:
        return tools.transform(df)

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        processed_input = apply_preprocessing(input_df, tools)

        if processed_input is not None:
            prediction = model.predict(processed_input)
            st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")

    except Exception:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
