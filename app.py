import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import os

# ------------------------------
# Load model and preprocessing tools
# ------------------------------
MODEL_PATH = "car_price_model.cbm"
ENCODER_PATH = "encoders.pkl"

# Load trained CatBoost model
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Load encoders if available, else create new dict
if os.path.exists(ENCODER_PATH):
    encoders = joblib.load(ENCODER_PATH)
else:
    encoders = {}

# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess_input(df):
    df = df.copy()

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Apply Label Encoding for categoricals
    for col in cat_cols:
        if col not in encoders:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        else:
            # Handle unseen categories safely
            df[col] = df[col].astype(str).map(
                lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
            )

    # Save encoders for later use
    joblib.dump(encoders, ENCODER_PATH)

    return df[num_cols + cat_cols]

# ------------------------------
# Prediction function
# ------------------------------
def predict_price(input_csv):
    # Load new input data
    df = pd.read_csv(input_csv)

    # Preprocess
    processed_df = preprocess_input(df)

    # Predict
    preds = model.predict(processed_df)

    # Attach predictions to dataframe
    df["PredictedPrice"] = preds
    return df

# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    input_file = "2025-09-05T05-44_export.csv"   # Change this to your uploaded CSV
    results = predict_price(input_file)

    # Save output
    results.to_csv("predictions_output.csv", index=False)
    print("✅ Predictions saved to predictions_output.csv")
