import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import traceback

# ===============================
# Load preprocessing tools & model
# ===============================
@st.cache_resource
def load_tools_and_model():
    with open("preprocessing_tools.pkl", "rb") as f:
        tools = pickle.load(f)
    with open("catboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)   # ✅ already pickled
    return tools, model

tools, model = load_tools_and_model()

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
# Input Fields
# ===============================
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing Year",
        min_value=int(data_pre["Year"].min()),
        max_value=int(data_pre["Year"].max()),
        value=int(data_pre["Year"].median())
    )

    kilometer = st.number_input("Kilometers Driven",
        min_value=int(data_pre["Kilometer"].min()),
        max_value=int(data_pre["Kilometer"].max()),
        value=int(data_pre["Kilometer"].median())
    )

    engine_raw = st.selectbox("Engine Size", data_pre["Engine"].dropna().unique())
    engine_numeric = int(re.findall(r"\d+", str(engine_raw))[0]) if re.findall(r"\d+", str(engine_raw)) else 0

with col2:
    make = st.selectbox("Car Make", sorted(data_pre["Make"].dropna().unique()))
    model_name = st.selectbox("Car Model", sorted(data_pre["Model"].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data_pre["Fuel Type"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data_pre["Transmission"].dropna().unique()))

# ===============================
# Build input row
# ===============================
input_dict = {
    "Year": year,
    "Kilometer": kilometer,
    "Engine": engine_numeric,
    "Make": make,
    "Model": model_name,
    "Fuel Type": fuel_type,
    "Transmission": transmission
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Apply preprocessing
# ===============================
def apply_preprocessing(df, tools, reference_df):
    try:
        # Training schema (drop Price if exists)
        schema_cols = reference_df.drop("Price", axis=1, errors="ignore").columns

        # Force alignment
        aligned = pd.DataFrame(columns=schema_cols)

        # Fill with input values or defaults
        row = {}
        for col in schema_cols:
            if col in df.columns:
                row[col] = df[col].iloc[0]
            else:
                # Fill missing with median/mode from training data
                if reference_df[col].dtype != "object":
                    row[col] = reference_df[col].median()
                else:
                    row[col] = reference_df[col].mode()[0] if not reference_df[col].mode().empty else "Unknown"

        aligned.loc[0] = row

        # Numeric + categorical separation
        num_cols = [c for c in aligned.columns if aligned[c].dtype != "object"]
        cat_cols = [c for c in aligned.columns if aligned[c].dtype == "object"]

        # Scale numeric
        if num_cols:
            X_num = tools["scaler"].transform(aligned[num_cols])
            X_num = pd.DataFrame(X_num, index=aligned.index)
        else:
            X_num = pd.DataFrame()

        # Encode categoricals
        if cat_cols:
            X_cat = tools["encoder"].transform(aligned[cat_cols])
            if hasattr(X_cat, "toarray"):
                X_cat = X_cat.toarray()
            X_cat = pd.DataFrame(X_cat, index=aligned.index)
        else:
            X_cat = pd.DataFrame()

        return pd.concat([X_num, X_cat], axis=1)

    except Exception:
        st.error("⚠️ Error during preprocessing")
        st.code(traceback.format_exc())
        return None

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    try:
        processed_input = apply_preprocessing(input_df, tools, data_proc)

        if processed_input is not None:
            prediction = model.predict(processed_input)
            st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")

    except Exception:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
