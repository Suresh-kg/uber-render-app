from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="Uber Fare Prediction API")

# Load model
with open("uber_fare_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Uber Fare Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["trip_distance_km"],
        data["trip_duration_min"],
        data["surge_multiplier"]
    ]])
    fare = model.predict(features)[0]
    return {"predicted_fare": round(float(fare), 2)}
