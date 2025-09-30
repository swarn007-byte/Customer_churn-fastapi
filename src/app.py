from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, conint, confloat
from typing import Literal
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# -------------------------------
# App initialization
# -------------------------------
app = FastAPI(title="Churn Prediction API", version="1.0")

# -------------------------------
# Paths to model & scaler
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../notebook/churn_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../notebook/notebook/scaler.pkl")  # adjust if needed

# -------------------------------
# Globals
# -------------------------------
model = None
scaler = None

# -------------------------------
# Load model & scaler at startup
# -------------------------------
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        print("Loading model from:", MODEL_PATH)
        model = load_model(MODEL_PATH)
        print("Loading scaler from:", SCALER_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print("Error loading artifacts:", e)

# -------------------------------
# Pydantic input schema
# -------------------------------
class CustomerData(BaseModel):
    customerID: constr(strip_whitespace=True, min_length=1)
    gender: Literal["Male", "Female"]
    SeniorCitizen: conint(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: conint(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: confloat(ge=0)
    TotalCharges: confloat(ge=0)

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict_churn(customer: CustomerData):
    global model, scaler
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([customer.dict()])

        # Map Yes/No and Male/Female to numeric
        df.replace({"Yes": 1, "No": 0, "Male": 0, "Female": 1}, inplace=True)

        # Scale only numeric features the scaler was trained on
        numeric_cols = ["tenure", "MonthlyCharges"]
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Make prediction
        prediction_prob = model.predict(df)
        prediction_class = int((prediction_prob > 0.5)[0][0])

        return {
            "ProcessedData": df.to_dict(orient="records")[0],
            "ChurnProbability": float(prediction_prob[0][0]),
            "ChurnPrediction": "Yes" if prediction_class == 1 else "No"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
