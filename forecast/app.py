from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

from inference import load_artifacts, predict_next_24h_from_latest_window
from preprocess import ensure_feature_order, add_time_features, HISTORY_WINDOW

app = FastAPI(title="Weather 24h Forecast API")
model, scalers, meta = load_artifacts()

class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    timezone_str: str = "UTC"  # e.g., "America/New_York"

# NOTE: Replace this with your actual feature construction to match training
# For demo, we assume we trained only on 'temperature'

def fetch_openmeteo_hourly(lat: float, lon: float, hours: int = 200, timezone_str: str = "UTC") -> pd.DataFrame:
    """Fetch recent hourly data from Open-Meteo.
    This is a simple example; you should adjust parameters and variables to match training features.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    # Open-Meteo free API: https://open-meteo.com/en/docs
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m&start={start.strftime('%Y-%m-%dT%H:%M')}"
        f"&end={end.strftime('%Y-%m-%dT%H:%M')}&timezone={timezone_str}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Open-Meteo error: {r.text}")
    js = r.json()
    # Parse response
    times = js['hourly']['time']
    temps = js['hourly']['temperature_2m']
    df = pd.DataFrame({'time': pd.to_datetime(times), 'temperature': temps}).set_index('time')
    df = add_time_features(df)
    df = ensure_feature_order(df)
    return df

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(req: PredictRequest):
    df = fetch_openmeteo_hourly(req.latitude, req.longitude, hours=HISTORY_WINDOW + 48, timezone_str=req.timezone_str)
    if len(df) < HISTORY_WINDOW:
        raise HTTPException(status_code=400, detail=f"Not enough data: need {HISTORY_WINDOW} hours, got {len(df)}")
    # Take the latest history window
    latest_window = df.tail(HISTORY_WINDOW).values  # shape (HISTORY_WINDOW, num_features)
    pred = predict_next_24h_from_latest_window(model, scalers, meta, latest_window)
    return {"latitude": req.latitude, "longitude": req.longitude, "forecast_horizon_hours": 24, "prediction": pred}
