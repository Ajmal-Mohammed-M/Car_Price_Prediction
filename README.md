# 🚗 Car Price Prediction (Streamlit App)

This is a Streamlit web app for predicting car prices using a **CatBoost model**.  
The app loads a preprocessing pipeline (`preprocessing_tools.pkl`) and a trained model (`catboost_best_model.pkl`) to generate predictions.

## 📂 Files
- `app.py` → Streamlit app code
- `requirements.txt` → Python dependencies
- `Preprocessed.csv` → dataset before encoding/scaling (used for dropdowns)
- `Processed.csv` → dataset after encoding/scaling (used for schema)
- `preprocessing_tools.pkl` → saved scaler + encoder
- `catboost_best_model.pkl` → trained CatBoost model
- `.gitignore` → ignore unnecessary files
- `README.md` → project documentation

## 🚀 Deployment
This project is designed to be deployed directly on **Streamlit Cloud**.

Steps:
1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Create a new app, select this repo, and deploy.

## ✅ Usage
- Select car details (`Year`, `Kilometer`, `Make`, `Model`, `Fuel Type`, `Transmission`, `Engine`).
- Click **Predict Price**.
- The app will preprocess your input and output the estimated price.
