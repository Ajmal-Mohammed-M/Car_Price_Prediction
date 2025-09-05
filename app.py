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
# Input Fields
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

    # Engine values like "1198 cc" → extract digits
    engine_options = data["Engine"].dropna().unique()
    engine_raw = st.selectbox("Engine Size", engine_options)
    engine_numeric = int(re.findall(r"\d+", str(engine_raw))[0]) if re.findall(r"\d+", str(engine_raw)) else 0

with col2:
    make = st.selectbox("Car Make", sorted(data["Make"].dropna().unique()))
    model_name = st.selectbox("Car Model", sorted(data["Model"].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data["Fuel Type"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data["Transmission"].dropna().unique()))

# ===============================
# Build raw input DataFrame
# ===============================
input_dict = {
    "Year": year,
    "Kilometer": kilometer,
    "Engine": engine_numeric,  # cleaned numeric version
    "Make": make,
    "Model": model_name,
    "Fuel Type": fuel_type,
    "Transmission": transmission
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Apply preprocessing manually
# ===============================
def apply_preprocessing(df, tools):
    try:
        cat_cols = ["Make", "Model", "Fuel Type", "Transmission"]
        num_cols = ["Year", "Kilometer", "Engine"]

        X_num, X_cat = None, None

        # Apply scaler
        if "scaler" in tools and all(col in df.columns for col in num_cols):
            X_num = tools["scaler"].transform(df[num_cols])
            X_num = pd.DataFrame(X_num, index=df.index)

        # Apply encoder
        if "encoder" in tools and all(col in df.columns for col in cat_cols):
            X_cat = tools["encoder"].transform(df[cat_cols])
            if not isinstance(X_cat, (pd.DataFrame, np.ndarray)):
                X_cat = pd.DataFrame(X_cat.toarray() if hasattr(X_cat, "toarray") else X_cat)

        # Merge
        parts = []
        if X_num is not None:
            parts.append(X_num)
        if X_cat is not None:
            parts.append(pd.DataFrame(X_cat, index=df.index))
        if parts:
            return pd.concat(parts, axis=1)

        return df
    except Exception:
        st.error("⚠️ Error during preprocessing")
        st.code(traceback.format_exc())
        return None

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
