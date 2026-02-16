from fastapi import FastAPI
from pydantic import BaseModel, create_model
from typing import Type
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import numpy as np
import pickle
import json
import os

app = FastAPI(title="Real Estate Prediction API")

# =========================
# Directory Setup
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
RF_DIR = os.path.join(MODEL_DIR, "rf_pkl")
XGB_DIR = os.path.join(MODEL_DIR, "xgb_pkl")

# =========================
# Load Models
# =========================

rf_single = pickle.load(open(os.path.join(RF_DIR, "rf_Single_Family.pkl"), "rb"))
xgb_condo = pickle.load(open(os.path.join(XGB_DIR, "xgb_Condo.pkl"), "rb"))
xgb_multi = pickle.load(open(os.path.join(XGB_DIR, "xgb_Multi-Family.pkl"), "rb"))
xgb_fallback = pickle.load(open(os.path.join(XGB_DIR, "xgb_fallback.pkl"), "rb"))

#sarimax_model = pickle.load(open(os.path.join(MODEL_DIR, "sarimax_model.pkl"), "rb"))


# Load compact SARIMAX model
sarimax_model = SARIMAXResults.load(
    os.path.join(MODEL_DIR, "sarimax_model_compact.pkl")
)

# =========================
# Load Feature Schema
# =========================

with open(os.path.join(MODEL_DIR, "feature_schema.json"), "r") as f:
    feature_columns = json.load(f)

# =========================
# Dynamic Feature Model
# =========================

feature_fields = {col: (float, ...) for col in feature_columns}

DynamicFeatureModel: Type[BaseModel] = create_model(
    "DynamicFeatureModel",
    **feature_fields
)

class PropertyInput(BaseModel):
    PROPERTY_TYPE: str
    features: DynamicFeatureModel

    model_config = {
        "json_schema_extra": {
            "example": {
                "PROPERTY_TYPE": "Condo",
                "features": {col: 1.0 for col in feature_columns}
            }
        }
    }

# =========================
# Valuation Endpoint
# =========================

@app.post("/predict_valuation/{property_id}")
def predict_valuation(data: PropertyInput):

    property_type = data.PROPERTY_TYPE

    # Convert Pydantic model to dict
    feature_dict = data.features.model_dump()

    # Ensure correct training order
    feature_vector = np.array(
        [feature_dict[col] for col in feature_columns]
    ).reshape(1, -1)

    if property_type == "Single Family":
        prediction = rf_single.predict(feature_vector)
    elif property_type == "Condo":
        prediction = xgb_condo.predict(feature_vector)
    elif property_type == "Multi-Family":
        prediction = xgb_multi.predict(feature_vector)
    else:
        prediction = xgb_fallback.predict(feature_vector)

    return {"predicted_property_value": float(prediction[0])}

# =========================
# Forecast Endpoint
# =========================

@app.get("/forecast_price")
def forecast_price(steps: int = 12):

    # Get last observed exogenous values
    last_exog = sarimax_model.model.exog[-1:]

    # Repeat it for future horizon
    future_exog = np.repeat(last_exog, steps, axis=0)

    forecast = sarimax_model.forecast(
        steps=steps,
        exog=future_exog
    )

    return {
        "forecast_horizon_months": steps,
        "forecast_values": forecast.tolist()
    }