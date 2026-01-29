from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import joblib
import os
import json
import time
import requests

from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


app = FastAPI(title="PV Voc+Eff Model + BO + Upstage LLM")

# =======================
# Files / Columns
# =======================
FEATURES = ["cu_ratio", "ga_ratio", "metal_se", "thickness_um"]  # A
TARGETS = ["Voc", "Eff"]  # B

VOC_MODEL = "voc_model.pkl"
EFF_MODEL = "eff_model.pkl"
STATE_FILE = "state.json"

# =======================
# BO default bounds
# =======================
DEFAULT_BOUNDS = {
    "cu_ratio": (0.8, 1.1),
    "ga_ratio": (0.1, 0.4),
    "metal_se": (0.9, 1.2),
    "thickness_um": (1.5, 3.0),
}

UPSTAGE_URL = "https://api.upstage.ai/v1/generation/pro2"


# =======================
# Utils
# =======================
def load_state() -> Dict[str, float]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"Voc_max": 0.75, "Eff_max": 18.0}


def save_state(state: Dict[str, float]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def ensure_models_exist() -> None:
    if not (os.path.exists(VOC_MODEL) and os.path.exists(EFF_MODEL)):
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Train first with A+B (Voc, Eff).",
        )


def safe_json(x: Any, max_len: int = 3500) -> str:
    try:
        s = json.dumps(x if x is not None else {}, ensure_ascii=False)
    except Exception:
        s = "{}"
    if len(s) > max_len:
        s = s[:max_len] + "…(truncated)"
    return s


# =======================
# Schemas
# =======================
class TrainRow(BaseModel):
    cu_ratio: float
    ga_ratio: float
    metal_se: float
    thickness_um: float
    Voc: float
    Eff: float


class PredictRow(BaseModel):
    cu_ratio: float
    ga_ratio: float
    metal_se: float
    thickness_um: float


class TrainPayload(BaseModel):
    rows: List[TrainRow]


class PredictPayload(BaseModel):
    rows: List[PredictRow]


class OptimizePayload(BaseModel):
    n_calls: int = 25
    w_voc: float = 0.5
    w_eff: float = 0.5
    bounds: Optional[Dict[str, List[float]]] = None
    seed_A: Optional[List[Dict[str, float]]] = None


class ExplainPayload(BaseModel):
    suggested_A: Optional[Dict[str, float]] = None
    pred_at_suggested: Optional[Dict[str, float]] = None
    best_set: Optional[Dict[str, Any]] = None
    objective: Optional[Dict[str, Any]] = None


# =======================
# Health
# =======================
@app.get("/health")
def health():
    return {"status": "ok"}


# =======================
# Train
# =======================
@app.post("/train")
def train(payload: TrainPayload):
    df = pd.DataFrame([r.model_dump() for r in payload.rows])

    X = df[FEATURES]
    y_voc = df["Voc"]
    y_eff = df["Eff"]

    voc_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    eff_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)

    voc_model.fit(X, y_voc)
    eff_model.fit(X, y_eff)

    joblib.dump(voc_model, VOC_MODEL)
    joblib.dump(eff_model, EFF_MODEL)

    state = load_state()
    state["Voc_max"] = max(state["Voc_max"], float(df["Voc"].max()))
    state["Eff_max"] = max(state["Eff_max"], float(df["Eff"].max()))
    save_state(state)

    return {"status": "trained", "rows_used": len(df), "state": state}


# =======================
# Predict
# =======================
@app.post("/predict")
def predict(payload: PredictPayload):
    ensure_models_exist()

    X = pd.DataFrame([r.model_dump() for r in payload.rows])[FEATURES]
    voc_model = joblib.load(VOC_MODEL)
    eff_model = joblib.load(EFF_MODEL)

    voc_pred = voc_model.predict(X)
    eff_pred = eff_model.predict(X)

    return {
        "predictions": [
            {
                "A": X.iloc[i].to_dict(),
                "Voc": float(voc_pred[i]),
                "Eff": float(eff_pred[i]),
            }
            for i in range(len(X))
        ]
    }


# =======================
# Optimize (BO)
# =======================
@app.post("/optimize")
def optimize(payload: OptimizePayload):
    ensure_models_exist()

    voc_model = joblib.load(VOC_MODEL)
    eff_model = joblib.load(EFF_MODEL)

    state = load_state()
    Voc_max = max(1e-6, state["Voc_max"])
    Eff_max = max(1e-6, state["Eff_max"])

    bounds = DEFAULT_BOUNDS.copy()
    if payload.bounds:
        for k, v in payload.bounds.items():
            if k in bounds and len(v) == 2:
                bounds[k] = (float(v[0]), float(v[1]))

    space = [Real(*bounds[k], name=k) for k in FEATURES]

    w_voc, w_eff = payload.w_voc, payload.w_eff
    w_sum = max(1e-12, w_voc + w_eff)
    w_voc /= w_sum
    w_eff /= w_sum

    def score(voc, eff):
        return (w_voc * voc / Voc_max) + (w_eff * eff / Eff_max)

    cache: Dict[Tuple[float, float, float, float], float] = {}

    @use_named_args(space)
    def objective(**p):
        key = tuple(round(p[k], 6) for k in FEATURES)
        if key in cache:
            return cache[key]

        Xp = pd.DataFrame([dict(zip(FEATURES, key))])
        voc = float(voc_model.predict(Xp)[0])
        eff = float(eff_model.predict(Xp)[0])
        val = -score(voc, eff)
        cache[key] = val
        return val

    res = gp_minimize(objective, space, n_calls=payload.n_calls, random_state=42)
    A = dict(zip(FEATURES, map(float, res.x)))

    Xs = pd.DataFrame([A])
    voc_s = float(voc_model.predict(Xs)[0])
    eff_s = float(eff_model.predict(Xs)[0])

    return {
        "suggested_A": A,
        "pred_at_suggested": {"Voc": voc_s, "Eff": eff_s, "score": score(voc_s, eff_s)},
        "best_set": {"A": A, "Voc": voc_s, "Eff": eff_s},
        "objective": {
            "w_voc": w_voc,
            "w_eff": w_eff,
            "Voc_max": Voc_max,
            "Eff_max": Eff_max,
        },
    }


# =======================
# Explain (Upstage Pro2)
# =======================
@app.post("/explain")
def explain(payload: ExplainPayload):
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        return {
            "llm_explanation": "(UPSTAGE_API_KEY 미설정: 설명 생략)",
            "fallback": True,
        }

    prompt = f"""
너는 태양전지(CIGS) 연구실 선임 연구자다.

[BO 제안 A]
{safe_json(payload.suggested_A)}

[예측 결과]
{safe_json(payload.pred_at_suggested)}

[best_set]
{safe_json(payload.best_set)}

[objective]
{safe_json(payload.objective)}

다음을 전문 태양전지 연구자의 연구노트 톤으로 작성:
1) 추출된 효율, Voc 데이터가 왜 최적의 조합인지 설명
2) 다음 실험에 쓸 최적의 조합 추천
3) 다음 실험 체크리스트
4) 위 내용 한 문장 요약
""".strip()

    t0 = time.time()
    try:
        r = requests.post(
            UPSTAGE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": prompt,
                "max_output_tokens": 800,
            },
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("generated_text") or data.get("output") or ""
    except Exception as e:
        return {
            "llm_explanation": "(Upstage 호출 실패로 설명 생략)",
            "error": str(e),
            "fallback": True,
        }

    return {
        "llm_explanation": text.strip() or "(Upstage 출력 없음)",
        "meta": {"provider": "upstage-pro2", "latency_sec": round(time.time() - t0, 3)},
    }
