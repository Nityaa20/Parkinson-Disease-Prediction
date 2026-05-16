# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from schemas import ParkinsonsInput
import joblib
import numpy as np
import os

app = FastAPI(title="Parkinson's AI Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names_model = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

frontend_to_model = {
    "MDVP_Fo_Hz": "MDVP:Fo(Hz)",
    "MDVP_Fhi_Hz": "MDVP:Fhi(Hz)",
    "MDVP_Flo_Hz": "MDVP:Flo(Hz)",
    "MDVP_Jitter_percent": "MDVP:Jitter(%)",
    "MDVP_Jitter_Abs": "MDVP:Jitter(Abs)",
    "MDVP_RAP": "MDVP:RAP",
    "MDVP_PPQ": "MDVP:PPQ",
    "Jitter_DDP": "Jitter:DDP",
    "MDVP_Shimmer": "MDVP:Shimmer",
    "MDVP_Shimmer_dB": "MDVP:Shimmer(dB)",
    "Shimmer_APQ3": "Shimmer:APQ3",
    "Shimmer_APQ5": "Shimmer:APQ5",
    "MDVP_APQ": "MDVP:APQ",
    "Shimmer_DDA": "Shimmer:DDA",
    "NHR": "NHR",
    "HNR": "HNR",
    "RPDE": "RPDE",
    "DFA": "DFA",
    "spread1": "spread1",
    "spread2": "spread2",
    "D2": "D2",
    "PPE": "PPE"
}

@app.post("/predict")
def predict(data: ParkinsonsInput):
    input_dict = data.dict()
    # Create input array in model order
    input_ordered = [input_dict[f] for f in frontend_to_model.keys()]
    input_scaled = scaler.transform([input_ordered])

    probs = model.predict_proba(input_scaled)[0] * 100
    prediction = "Parkinson's Detected" if model.predict(input_scaled)[0] == 1 else "Healthy"

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]

    return {
        "prediction": prediction,
        "confidence": float(np.max(probs)),
        "probability_healthy": float(probs[0]),
        "probability_parkinsons": float(probs[1]),
        "top_features": [feature_names_model[i] for i in top_idx],
        "top_importances": [float(importances[i]) for i in top_idx]
    }

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
