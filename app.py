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
# Column mapping (based on your Preprocessed.csv)
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
    model_name = st.selectbox("Car Model", sorted(data[col_map["model"]].dropna().unique()))
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
    col_map["model"]: model_name,
    col_map["fuel_type"]: fuel_type,
    col_map["transmission"]: transmission
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Apply preprocessing safely
# ===============================
def apply_preprocessing(df, tools):
    """
    Safely apply preprocessing depending on whether tools is a dict or a pipeline.
    """
    X = df.copy()

    # Case 1: tools is a dict
    if isinstance(tools, dict):
        if "preprocessor" in tools:
            X = tools["preprocessor"].transform(X)
        else:
            # Assume dict might contain encoder/scaler separately
            cat_cols = ["Make", "Model", "Fuel Type", "Transmission"]
            num_cols = ["Year", "Kilometer", "Engine"]

            X_num, X_cat = None, None

            if "scaler" in tools and all(col in df.columns for col in num_cols):
                try:
                    X_num = tools["scaler"].transform(df[num_cols])
                except Exception as e:
                    st.error("⚠️ Error applying scaler")
                    st.code(traceback.format_exc())

            if "encoder" in tools and all(col in df.columns for col in cat_cols):
                try:
                    X_cat = tools["encoder"].transform(df[cat_cols])
                except Exception as e:
                    st.error("⚠️ Error applying encoder")
                    st.code(traceback.format_exc())

            # Merge numeric + categorical
            parts = []
            if X_num is not None:
                parts.append(pd.DataFrame(X_num))
            if X_cat is not None:
                if not isinstance(X_cat, (pd.DataFrame, np.ndarray)):
                    X_cat = pd.DataFrame(X_cat)
                parts.append(pd.DataFrame(X_cat))
            if parts:
                X = pd.concat(parts, axis=1)

    # Case 2: tools is a pipeline (ColumnTransformer or Pipeline)
    else:
        X = tools.transform(X)

    return X

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        processed_input = apply_preprocessing(input_df, tools)

        prediction = model.predict(processed_input)

        st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")

    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
