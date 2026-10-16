
# FareCast AI - FastAPI application
# Loads the trained model and provides a fare prediction endpoint

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pathlib import Path

# Create the FastAPI application
app = FastAPI(
    title="FareCast AI",
    description="A machine learning API for estimating ride booking value.",
    version="1.0.0"
)

# Load the trained ML pipeline from the project folder
MODEL_PATH = Path(__file__).parent / "farecast_model.joblib"
model = joblib.load(MODEL_PATH)


# Define the input format expected by the prediction API
class FareRequest(BaseModel):
    vehicle_type: str
    ride_distance: float | None = Field(default=None, gt=0, le=50)
    pickup_location: str
    drop_location: str
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)


# Homepage message
@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# Health-check endpoint to confirm that the API is running
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


# Predict the booking value from the supplied ride details
@app.post("/predict")
def predict_fare(request: FareRequest):
    try:
        # Convert request data into the column names used during training
        input_data = pd.DataFrame([{
            "Vehicle Type": request.vehicle_type,
            "Ride Distance": request.ride_distance,
            "Pickup Location": request.pickup_location,
            "Drop Location": request.drop_location,
            "Hour": request.hour,
            "Day of Week": request.day_of_week,
            "Month": request.month
        }])

        # Generate the fare estimate using the saved pipeline
        prediction = float(model.predict(input_data)[0])

        # Prevent negative estimates from being shown to users
        prediction = max(0.0, prediction)

        return {
            "estimated_fare": round(prediction, 2),
            "currency": "INR",
            "message": "Estimated value only; actual ride prices may differ."
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the input and saved model."
        ) from error

