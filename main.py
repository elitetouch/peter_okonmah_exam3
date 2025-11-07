from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Wine Quality Prediction API", description="Developed by Peter Okonmah ", version="1.0" )

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected input schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Map numeric predictions to labels
label_map = {0: "Low Quality", 1: "High Quality"}

@app.get("/")
def home():
    return {"message": "Welcome to the Wine Quality Prediction API built by Peter Okonmah"}


@app.post("/predict")
def predict_quality(data: WineFeatures):
    # Convert input to numpy array
    features = np.array([[ 
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]])

    # Scale input
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)[0]
    prediction_label = label_map[prediction]

    return {
        "prediction_numeric": int(prediction),
        "prediction_label": prediction_label
    }
