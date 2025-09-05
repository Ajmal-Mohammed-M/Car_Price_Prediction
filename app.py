import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
df = pd.read_csv("Preprocessed.csv")

# ---- Step 1: Standard Scaling for numeric columns (exclude Price) ----
scaler = StandardScaler()

# Exclude Price from scaling
numeric_cols = df.drop(columns=["Price"]).select_dtypes(include=["int64", "float64"]).columns

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ---- Step 2: Label Encoding for object columns ----
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le   # save encoder for each column

# ---- Save processed dataset ----
df.to_csv("Processed.csv", index=False)

# ---- Save scaler separately ----
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---- Save label encoders separately ----
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Preprocessing complete")
print("📂 Files saved: Processed.csv, scaler.pkl, label_encoders.pkl")
