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
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("catboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, label_encoders, model

scaler, label_encoders, model = load_tools_and_model()

# ===============================
# Load dataset (only Preprocessed.csv)
# ===============================
@st.cache_data
def load_data():
    df_pre = pd.read_csv("Preprocessed.csv")  # before encoding/scaling
    return df_pre

data_pre = load_data()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the details below to get the predicted price of the car.")

# ===============================
# Input Fields
# ===============================
st.subheader("📋 Enter Car Details")

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

    length = st.number_input("Car Length (mm)",
        min_value=float(data_pre["Length"].min()),
        max_value=float(data_pre["Length"].max()),
        value=float(data_pre["Length"].median())
    )

    width = st.number_input("Car Width (mm)",
        min_value=float(data_pre["Width"].min()),
        max_value=float(data_pre["Width"].max()),
        value=float(data_pre["Width"].median())
    )

    height = st.number_input("Car Height (mm)",
        min_value=float(data_pre["Height"].min()),
        max_value=float(data_pre["Height"].max()),
        value=float(data_pre["Height"].median())
    )

with col2:
    make = st.selectbox("Car Make", sorted(data_pre["Make"].dropna().unique()))
    model_name = st.selectbox("Car Model", sorted(data_pre["Model"].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data_pre["Fuel Type"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data_pre["Transmission"].dropna().unique()))
    location = st.selectbox("Location", sorted(data_pre["Location"].dropna().unique()))
    color = st.selectbox("Color", sorted(data_pre["Color"].dropna().unique()))
    owner = st.selectbox("Owner Type", sorted(data_pre["Owner"].dropna().unique()))
    seller_type = st.selectbox("Seller Type", sorted(data_pre["Seller Type"].dropna().unique()))

col3, col4 = st.columns(2)

with col3:
    engine_raw = st.selectbox("Engine Size", data_pre["Engine"].dropna().unique())
    max_power = st.selectbox("Max Power", data_pre["Max Power"].dropna().unique())
    max_torque = st.selectbox("Max Torque", data_pre["Max Torque"].dropna().unique())
    drivetrain = st.selectbox("Drivetrain", data_pre["Drivetrain"].dropna().unique())

with col4:
    seating_capacity = st.number_input("Seating Capacity",
        min_value=float(data_pre["Seating Capacity"].min()),
        max_value=float(data_pre["Seating Capacity"].max()),
        value=float(data_pre["Seating Capacity"].median())
    )

    fuel_tank_capacity = st.number_input("Fuel Tank Capacity (Litres)",
        min_value=float(data_pre["Fuel Tank Capacity"].min()),
        max_value=float(data_pre["Fuel Tank Capacity"].max()),
        value=float(data_pre["Fuel Tank Capacity"].median())
    )

# ===============================
# Build input row
# ===============================
input_dict = {
    "Make": make,
    "Model": model_name,
    "Year": year,
    "Kilometer": kilometer,
    "Fuel Type": fuel_type,
    "Transmission": transmission,
    "Location": location,
    "Color": color,
    "Owner": owner,
    "Seller Type": seller_type,
    "Engine": engine_raw,
    "Max Power": max_power,
    "Max Torque": max_torque,
    "Drivetrain": drivetrain,
    "Length": length,
    "Width": width,
    "Height": height,
    "Seating Capacity": seating_capacity,
    "Fuel Tank Capacity": fuel_tank_capacity,
}
input_df = pd.DataFrame([input_dict])

st.subheader("🔍 Entered Details")
st.write(input_df)

# ===============================
# Apply preprocessing
# ===============================
def apply_preprocessing(df, scaler, label_encoders, reference_df):
    try:
        # Training schema (drop Price if exists)
        schema_cols = reference_df.drop("Price", axis=1, errors="ignore").columns

        aligned = pd.DataFrame(columns=schema_cols)
        row = {}

        for col in schema_cols:
            if col in df.columns:
                row[col] = df[col].iloc[0]
            else:
                if reference_df[col].dtype != "object":
                    row[col] = reference_df[col].median()
                else:
                    row[col] = reference_df[col].mode()[0] if not reference_df[col].mode().empty else "Unknown"

        aligned.loc[0] = row

        # Numeric + categorical separation
        num_cols = [c for c in aligned.columns if aligned[c].dtype != "object" and c != "Price"]
        cat_cols = [c for c in aligned.columns if aligned[c].dtype == "object"]

        # Scale numeric
        if num_cols:
            X_num = scaler.transform(aligned[num_cols])
            X_num = pd.DataFrame(X_num, index=aligned.index)
        else:
            X_num = pd.DataFrame()

        # Encode categoricals
        X_cat = pd.DataFrame(index=aligned.index)
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    X_cat[col] = le.transform(aligned[col])
                except ValueError:
                    X_cat[col] = aligned[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                X_cat[col] = 0  # fallback

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
        processed_input = apply_preprocessing(input_df, scaler, label_encoders, data_pre)

        if processed_input is not None:
            prediction = model.predict(processed_input)
            st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")

    except Exception:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
