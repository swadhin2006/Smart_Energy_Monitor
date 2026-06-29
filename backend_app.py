"""
Smart IoT Energy Meter — FastAPI Backend
Serves two endpoints:
  POST /predict/energy   → XGBoost energy forecast
  POST /predict/anomaly  → Isolation Forest anomaly detection
  GET  /health           → health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import os

# ── Load models ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))


def _load(filename):
    path = os.path.join(BASE, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{filename}' not found. "
            "Run the notebook first to generate model files."
        )
    return joblib.load(path)

try:
    forecast_model    = _load("energy_forecast_model.pkl")
    forecast_scaler   = _load("energy_forecast_scaler.pkl")
    forecast_features = _load("forecast_features.pkl")
    anomaly_model     = _load("anomaly_model.pkl")
    anomaly_scaler    = _load("anomaly_scaler.pkl")
    anomaly_features  = _load("anomaly_features.pkl")
    MODELS_LOADED = True
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    MODELS_LOADED = False

app = FastAPI(
    title="Smart IoT Energy Meter — ML API",
    description="AI/ML predictions for ESP32 IoT energy meter readings",
    version="1.0.0",
)

# ── Request / Response schemas ────────────────────────────────────────────────

class EnergyForecastRequest(BaseModel):
    voltage_v:          float = Field(..., example=220.5,  description="RMS Voltage (V)")
    current_a:          float = Field(..., example=2.3,    description="RMS Current (A)")
    power_factor:       float = Field(..., example=0.92,   description="Power factor (0-1)")
    temperature_c:      float = Field(..., example=28.0,   description="Ambient temperature (°C)")
    hour:               int   = Field(..., example=18,     description="Hour of day (0-23)")
    month:              int   = Field(..., example=6,      description="Month (1-12)")
    is_weekend:         int   = Field(..., example=0,      description="1 if weekend, else 0")
    is_peak_hour:       int   = Field(..., example=1,      description="1 if peak hour, else 0")
    power_roll_mean_4:  float = Field(..., example=480.0,  description="Rolling avg power (last 1h)")
    power_roll_mean_24: float = Field(..., example=460.0,  description="Rolling avg power (last 6h)")
    power_roll_std_4:   float = Field(..., example=15.0,   description="Rolling std power (last 1h)")
    energy_lag_1:       float = Field(..., example=0.12,   description="Energy 15 min ago (kWh)")
    energy_lag_4:       float = Field(..., example=0.11,   description="Energy 1 hour ago (kWh)")
    energy_lag_96:      float = Field(..., example=0.10,   description="Energy same time yesterday (kWh)")


class EnergyForecastResponse(BaseModel):
    predicted_energy_kwh: float
    predicted_cost_inr:   float
    tariff_per_kwh:       float = 7.0


class AnomalyRequest(BaseModel):
    voltage_v:       float = Field(..., example=220.5)
    current_a:       float = Field(..., example=2.3)
    power_w:         float = Field(..., example=460.0)
    power_factor:    float = Field(..., example=0.92)
    apparent_power:  float = Field(..., example=507.15)
    reactive_power:  float = Field(..., example=199.5)
    power_roll_std_4:float = Field(..., example=12.0)


class AnomalyResponse(BaseModel):
    is_anomaly:     bool
    anomaly_score:  float
    label:          str
    recommendation: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_models():
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run energy_meter_ml.ipynb first to train and save models."
        )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}


@app.post("/predict/energy", response_model=EnergyForecastResponse)
def predict_energy(req: EnergyForecastRequest):
    """Predict energy consumption (kWh) for the next 15-minute interval."""
    _check_models()

    apparent_power = req.voltage_v * req.current_a
    reactive_power = apparent_power * np.sqrt(max(1 - req.power_factor ** 2, 0))
    hour_sin  = np.sin(2 * np.pi * req.hour / 24)
    hour_cos  = np.cos(2 * np.pi * req.hour / 24)
    month_sin = np.sin(2 * np.pi * req.month / 12)
    month_cos = np.cos(2 * np.pi * req.month / 12)

    row = np.array([[
        req.voltage_v, req.current_a, req.power_factor, req.temperature_c,
        hour_sin, hour_cos, month_sin, month_cos,
        req.is_weekend, req.is_peak_hour,
        apparent_power, reactive_power,
        req.power_roll_mean_4, req.power_roll_mean_24, req.power_roll_std_4,
        req.energy_lag_1, req.energy_lag_4, req.energy_lag_96,
    ]])

    scaled     = forecast_scaler.transform(row)
    energy_kwh = float(forecast_model.predict(scaled)[0])
    energy_kwh = max(energy_kwh, 0.0)
    cost_inr   = round(energy_kwh * 7.0, 4)

    return EnergyForecastResponse(
        predicted_energy_kwh=round(energy_kwh, 5),
        predicted_cost_inr=cost_inr,
    )


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(req: AnomalyRequest):
    """Detect anomalies in live sensor readings."""
    _check_models()

    row = np.array([[
        req.voltage_v, req.current_a, req.power_w,
        req.power_factor, req.apparent_power,
        req.reactive_power, req.power_roll_std_4,
    ]])

    scaled     = anomaly_scaler.transform(row)
    raw        = anomaly_model.predict(scaled)[0]       # -1 or 1
    score      = float(-anomaly_model.score_samples(scaled)[0])
    is_anomaly = raw == -1

    if is_anomaly:
        if req.voltage_v > 250:
            rec = "Voltage spike detected. Check surge protector or ZMPT101B calibration."
        elif req.current_a > 10:
            rec = "Current surge detected. Check for short circuit or overloaded circuit."
        elif req.voltage_v < 10:
            rec = "Power dropout detected. Possible power outage or wiring fault."
        else:
            rec = "Unusual reading detected. Inspect device and wiring connections."
        label = "ANOMALY"
    else:
        rec   = "All readings normal."
        label = "NORMAL"

    return AnomalyResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        label=label,
        recommendation=rec,
    )
