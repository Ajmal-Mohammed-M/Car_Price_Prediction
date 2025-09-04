import streamlit as st
import pandas as pd
import pickle

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
    df = pd.read_csv("Preprocessed.csv")  # before encoding/scaling
    return df

data = load_data()

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the details below to get the predicted price of the car.")

# ===============================
# Dynamic Input Fields from Dataset
# ===============================
col1, col2 = st.columns(2)

# Numeric features (example: adjust to match your dataset)
with col1:
    year = st.number_input("Manufacturing Year", 
                           min_value=int(data["year"].min()), 
                           max_value=int(data["year"].max()), 
                           value=int(data["year"].median()))
    
    mileage = st.number_input("Mileage (km/l)", 
                              min_value=float(data["mileage"].min()), 
                              max_value=float(data["mileage"].max()), 
                              value=float(data["mileage"].median()))
    
    engine_size = st.number_input("Engine Size (CC)", 
                                  min_value=int(data["engine_size"].min()), 
                                  max_value=int(data["engine_size"].max()), 
                                  value=int(data["engine_size"].median()))

# Categorical features (options taken from dataset unique values)
with col2:
    brand = st.selectbox("Car Brand", sorted(data["brand"].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data["fuel_type"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data["transmission"].dropna().unique()))

# Collect inputs into DataFrame
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
        # Apply preprocessing
        processed_input = preprocessor.transform(input_df)

        # Predict with CatBoost
        prediction = model.predict(processed_input)

        st.success(f"💰 Estimated Car Price: **₹ {prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
