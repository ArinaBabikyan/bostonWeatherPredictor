
import json
from pathlib import Path
import numpy as np
import torch
import joblib

from .model_def import WeatherGRU

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model.pt'
SCALER_X_PATH = ARTIFACTS_DIR / 'scaler_X.joblib'
SCALER_Y_PATH = ARTIFACTS_DIR / 'scaler_y.joblib'
META_PATH = ARTIFACTS_DIR / 'metadata.json'

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None
_scalers = None
_meta = None

def load_artifacts():
    global _model, _scalers, _meta
    if _model is not None:
        return _model, _scalers, _meta
    _scalers = {
        'X': joblib.load(SCALER_X_PATH),
        'y': joblib.load(SCALER_Y_PATH),
    }
    with open(META_PATH, 'r') as f:
        _meta = json.load(f)
    _model = WeatherGRU(input_size=_meta['input_size'])
    sd = torch.load(MODEL_PATH, map_location='cpu')
    _model.load_state_dict(sd)
    _model.eval()
    _model.to(_device)
    return _model, _scalers, _meta

def predict_next_24h_from_latest_window(X_latest_window: np.ndarray) -> float:
    """
    X_latest_window: shape (history_window, num_features) in ORIGINAL units.
    Returns prediction in ORIGINAL units.
    """
    model, scalers, meta = load_artifacts()
    scaler_X = scalers['X']
    X_scaled = scaler_X.transform(X_latest_window)
    X_tensor = torch.tensor(X_scaled[None, ...], dtype=torch.float32).to(_device)
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()  # (1, 1)
    scaler_y = scalers['y']
    pred = scaler_y.inverse_transform(pred_scaled)  # (1, 1)
    return float(pred[0, 0])
