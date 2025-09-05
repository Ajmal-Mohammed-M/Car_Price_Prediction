import streamlit as st
import pandas as pd
import pickle
import traceback
import re
import numpy as np

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

# Sidebar debug info
st.sidebar.subheader("🔧 Preprocessing Tools Info")
st.sidebar.write("Type:", type(tools))
if isinstance(tools, dict):
    st.sidebar.write("Keys:", list(tools.keys()))

# ===============================
# Load datasets
# ===============================
@st.cache_data
def load_data():
    df_pre = pd.read_csv("Preprocessed.csv")  # before encoding/scaling
    df_proc = pd.read_csv("Processed.csv")    # after encoding/scaling
    return df_pre, df_proc

data_pre, data_proc = load_data()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the details below to get the predicted price of the car.")

# ===============================
# Column mapping (from Preprocessed.csv)
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
        min_value=int(data_pre[col_map["year"]].min()),
        max_value=int(data_pre[col_map["year"]].max()),
        value=int(data_pre[col_map["year"]].median())
    )

    kilometer = st.number_input(
        "Kilometers Driven",
        min_value=int(data_pre[col_map["kilometer"]].min()),
        max_value=int(data_pre[col_map["kilometer"]].max()),
        value=int(data_pre[col_map["kilometer"]].median())
    )

    # Engine values like "1198 cc" → extract digits
    engine_options = data_pre[col_map["engine"]].dropna().unique()
    engine_raw = st.selectbox("Engine Size", engine_options)
    engine_numeric = int(re.findall(r"\d+", str(engine_raw))[0]) if re.findall(r"\d+", str(engine_raw)) else 0

with col2:
    make = st.selectbox("Car Make", sorted(data_pre[col_map["make"]].dropna().unique()))
    model_name = st.selectbox("Car Model", sorted(data_pre[col_map["model"]].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data_pre[col_map["fuel_type"]].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data_pre[col_map["transmission"]].dropna().unique()))

# ===============================
# Build input with ALL training columns
# ===============================
# Start with empty row using Processed.csv columns
input_full = pd.DataFrame(columns=data_proc.drop("Price", axis=1, errors="ignore").columns)

# Fill values
row = {}
row[col_map["year"]] = year
row[col_map["kilometer"]] = kilometer
row[col_map["engine"]] = engine_numeric   # numeric version
row[col_map["make"]] = make
row[col_map["model"]] = model_name
row[col_map["fuel_type"]] = fuel_type
row[col_map["transmission"]] = transmission

# Fill missing features with default values
for col in input_full.columns:
    if col not in row:
        row[col] = 0 if data_proc[col].dtype != "object" else "Unknown"

input_full = pd.DataFrame([row], columns=input_full.columns)

st.subheader("🔍 Entered Details (Processed Input)")
st.write(input_full)

# ===============================
# Apply preprocessing safely
# ===============================
def apply_preprocessing(df, tools):
    if isinstance(tools, dict):
        if "preprocessor" in tools:
            return tools["preprocessor"].transform(df)
        elif "scaler" in tools or "encoder" in tools:
            # If manually split objects, just pass through
            return df  # assume already numeric
    else:
        return tools.transform(df)
    return df

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        processed_input = apply_preprocessing(input_full, tools)

        prediction = model.predict(processed_input)

        st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")

    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
