from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# OpenAI (new SDK)
from openai import OpenAI

app = FastAPI(title="PV Voc+Eff Model + BO + LLM")

# ====== 모델/상태 파일 ======
FEATURES = ["cu_ratio", "ga_ratio", "metal_se", "thickness_um"]  # A
TARGETS = ["Voc", "Eff"]  # B

VOC_MODEL = "voc_model.pkl"
EFF_MODEL = "eff_model.pkl"
STATE_FILE = "state.json"

# ====== 기본 탐색 범위(BO) ======
DEFAULT_BOUNDS = {
    "cu_ratio": (0.8, 1.1),
    "ga_ratio": (0.1, 0.4),
    "metal_se": (0.9, 1.2),
    "thickness_um": (1.5, 3.0),
}

# -----------------------
# Utils: state (정규화 기준)
# -----------------------
def load_state() -> Dict[str, float]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # 초기값은 너무 작으면 score 폭주하니까 "현실적인 디폴트"로 둠
    return {"Voc_max": 0.75, "Eff_max": 18.0}

def save_state(state: Dict[str, float]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

def ensure_models_exist():
    if not (os.path.exists(VOC_MODEL) and os.path.exists(EFF_MODEL)):
        raise HTTPException(status_code=400, detail="Model not trained yet. Train first with A+B (Voc, Eff).")

# -----------------------
# Schemas
# -----------------------
class TrainRow(BaseModel):
    # A
    cu_ratio: float
    ga_ratio: float
    metal_se: float
    thickness_um: float
    # B
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
    # BO 설정
    n_calls: int = 25
    top_k: int = 10  # 결과로 top 후보 몇 개 뽑을지(간단히 best_set만 써도 OK)
    w_voc: float = 0.5
    w_eff: float = 0.5
    # bounds override (선택)
    bounds: Optional[Dict[str, List[float]]] = None
    # BO seed (선택): 특정 시작점 제공 가능
    seed_A: Optional[List[Dict[str, float]]] = None

class ExplainPayload(BaseModel):
    # /optimize 응답 그대로 넣어도 되게 유연하게
    suggested_A: Optional[Dict[str, float]] = None
    pred_at_suggested: Optional[Dict[str, float]] = None
    best_set: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, float]] = None
    objective: Optional[Dict[str, Any]] = None

# -----------------------
# Health
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# Train: Voc & Eff (A+B 입력)
# -----------------------
@app.post("/train")
def train(payload: TrainPayload):
    df = pd.DataFrame([r.model_dump() for r in payload.rows])

    # 유효성
    for col in FEATURES + TARGETS:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")

    X = df[FEATURES]
    y_voc = df["Voc"]
    y_eff = df["Eff"]

    voc_model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42)
    eff_model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42)

    voc_model.fit(X, y_voc)
    eff_model.fit(X, y_eff)

    joblib.dump(voc_model, VOC_MODEL)
    joblib.dump(eff_model, EFF_MODEL)

    # state 갱신: 실험 데이터 기반으로만 갱신 (안전)
    state = load_state()
    state["Voc_max"] = max(state["Voc_max"], float(df["Voc"].max()))
    state["Eff_max"] = max(state["Eff_max"], float(df["Eff"].max()))
    save_state(state)

    return {"status": "trained", "rows_used": len(df), "state": state}

# -----------------------
# Predict: Voc & Eff (A 입력)
# -----------------------
@app.post("/predict")
def predict(payload: PredictPayload):
    ensure_models_exist()

    X = pd.DataFrame([r.model_dump() for r in payload.rows])[FEATURES]
    voc_model = joblib.load(VOC_MODEL)
    eff_model = joblib.load(EFF_MODEL)

    voc_pred = voc_model.predict(X)
    eff_pred = eff_model.predict(X)

    # 입력 A도 같이 반환하면 n8n에서 후처리 편해짐
    preds = []
    for i in range(len(X)):
        preds.append({
            "A": X.iloc[i].to_dict(),
            "Voc": float(voc_pred[i]),
            "Eff": float(eff_pred[i]),
        })

    return {"predictions": preds}

# -----------------------
# Optimize (BO): "최적 Voc+Eff 조합" + "다음 실험 A 추천"
# - A만 들어와도(=모델만 있으면) 항상 best_set 산출 가능
# -----------------------
@app.post("/optimize")
def optimize(payload: OptimizePayload):
    ensure_models_exist()

    voc_model = joblib.load(VOC_MODEL)
    eff_model = joblib.load(EFF_MODEL)

    state = load_state()
    Voc_max = max(1e-6, float(state["Voc_max"]))
    Eff_max = max(1e-6, float(state["Eff_max"]))

    # bounds 세팅
    bounds = DEFAULT_BOUNDS.copy()
    if payload.bounds:
        for k, v in payload.bounds.items():
            if k in bounds and isinstance(v, list) and len(v) == 2:
                bounds[k] = (float(v[0]), float(v[1]))

    space = [
        Real(bounds["cu_ratio"][0], bounds["cu_ratio"][1], name="cu_ratio"),
        Real(bounds["ga_ratio"][0], bounds["ga_ratio"][1], name="ga_ratio"),
        Real(bounds["metal_se"][0], bounds["metal_se"][1], name="metal_se"),
        Real(bounds["thickness_um"][0], bounds["thickness_um"][1], name="thickness_um"),
    ]

    w_voc = float(payload.w_voc)
    w_eff = float(payload.w_eff)

    def score_from_pred(voc: float, eff: float) -> float:
        # 정규화 가중합 (Voc + Eff)
        return (w_voc * (voc / Voc_max)) + (w_eff * (eff / Eff_max))

    @use_named_args(space)
    def objective(**params):
        X = pd.DataFrame([params])[FEATURES]
        voc = float(voc_model.predict(X)[0])
        eff = float(eff_model.predict(X)[0])
        return -score_from_pred(voc, eff)  # gp_minimize = minimize

    # seed 제공 가능
    x0 = None
    y0 = None
    if payload.seed_A:
        x0 = []
        y0 = []
        for s in payload.seed_A:
            row = [float(s["cu_ratio"]), float(s["ga_ratio"]), float(s["metal_se"]), float(s["thickness_um"])]
            x0.append(row)
            Xs = pd.DataFrame([{
                "cu_ratio": row[0],
                "ga_ratio": row[1],
                "metal_se": row[2],
                "thickness_um": row[3],
            }])[FEATURES]
            voc = float(voc_model.predict(Xs)[0])
            eff = float(eff_model.predict(Xs)[0])
            y0.append(-score_from_pred(voc, eff))

    res = gp_minimize(
        objective,
        space,
        n_calls=int(payload.n_calls),
        x0=x0,
        y0=y0,
        random_state=42
    )

    suggested_A = dict(zip(FEATURES, [float(x) for x in res.x]))

    # suggested_A에서의 예측 Voc/Eff
    Xs = pd.DataFrame([suggested_A])[FEATURES]
    voc_s = float(voc_model.predict(Xs)[0])
    eff_s = float(eff_model.predict(Xs)[0])
    score_s = float(score_from_pred(voc_s, eff_s))

    # "최적 Voc+Eff 세트" = best_set (여기서는 BO 최적점 자체를 best로 정의)
    best_set = {
        "A": suggested_A,
        "Voc": voc_s,
        "Eff": eff_s,
        "score": score_s
    }

    return {
        "suggested_A": suggested_A,                  # 다음 실험 조합
        "pred_at_suggested": {"Voc": voc_s, "Eff": eff_s, "score": score_s},
        "best_set": best_set,                        # 네가 말한 "최적 Voc+Eff 세트"
        "state": state,
        "objective": {
            "type": "weighted_normalized_sum",
            "w_voc": w_voc,
            "w_eff": w_eff,
            "Voc_max": Voc_max,
            "Eff_max": Eff_max
        }
    }

# -----------------------
# Explain (LLM): BO 결과를 연구자 친화 텍스트로 변환
# - n8n에서 /optimize 응답 그대로 POST 하면 됨
# -----------------------
@app.post("/explain")
def explain(payload: ExplainPayload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set on server.")

    client = OpenAI(api_key=api_key)

    # payload에서 핵심 뽑기
    suggested_A = payload.suggested_A or {}
    pred = payload.pred_at_suggested or {}
    best = payload.best_set or {}
    obj = payload.objective or {}

    prompt = f"""
너는 태양전지(특히 CIGS) 연구실의 선임 연구자다.
아래는 Bayesian Optimization이 제안한 다음 실험 조합(A)과,
해당 조합에서 예측된 Voc/Eff(효율)이다.

[BO 제안 A]
{json.dumps(suggested_A, ensure_ascii=False)}

[예측 결과 (해당 A에서)]
{json.dumps(pred, ensure_ascii=False)}

[최적 Voc+Eff 세트(best_set)]
{json.dumps(best, ensure_ascii=False)}

[목적 함수 정보]
{json.dumps(obj, ensure_ascii=False)}

요구사항:
1) 왜 이 A가 Voc와 Eff 관점에서 유망한지 (가정/직관 포함)
2) 실험에서 주의할 점 (현실적인 조성/공정 리스크)
3) 다음 실험 실행 체크리스트(간단히)
4) 결과를 한 문장 요약(연구노트 스타일)

출력은 한국어로, 연구노트 톤으로 작성해.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert photovoltaic (CIGS) researcher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return {"llm_explanation": resp.choices[0].message.content}
