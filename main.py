from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from typing import List, Optional
import os

app = FastAPI(title="PV Efficiency Model API")

MODEL_PATH = "cigs_model.pkl"
FEATURES = ["cu_ratio", "ga_ratio", "metal_se", "thickness_um"]

class TrainRow(BaseModel):
    sample_id: Optional[str] = None
    cu_ratio: float
    ga_ratio: float
    metal_se: float
    thickness_um: float
    eff: float

class PredictRow(BaseModel):
    sample_id: Optional[str] = None
    cu_ratio: float
    ga_ratio: float
    metal_se: float
    thickness_um: float

class TrainPayload(BaseModel):
    rows: List[TrainRow]

class PredictPayload(BaseModel):
    rows: List[PredictRow]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(payload: TrainPayload):
    df = pd.DataFrame([r.dict() for r in payload.rows])
    df = df.dropna(subset=FEATURES + ["eff"])

    X = df[FEATURES]
    y = df["eff"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return {"status": "trained", "rows_used": len(df)}

@app.post("/predict")
def predict(payload: PredictPayload):
    if not os.path.exists(MODEL_PATH):
        return {"error": "model not trained yet"}

    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([r.dict() for r in payload.rows])
    X = df[FEATURES]
    preds = model.predict(X)

    return {
        "predictions": [
            {"sample_id": payload.rows[i].sample_id, "pred_eff": float(preds[i])}
            for i in range(len(preds))
        ]
    }
