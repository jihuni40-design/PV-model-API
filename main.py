from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
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
# Utils
# -----------------------
def load_state() -> Dict[str, float]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"Voc_max": 0.75, "Eff_max": 18.0}

def save_state(state: Dict[str, float]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

def ensure_models_exist():
    if not (os.path.exists(VOC_MODEL) and os.path.exists(EFF_MODEL)):
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Train first with A+B (Voc, Eff)."
        )

# -----------------------
# Schemas
# -----------------------
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
    top_k: int = 10
    w_voc: float = 0.5
    w_eff: float = 0.5
    bounds: Optional[Dict[str, List[float]]] = None
    seed_A: Optional[List[Dict[str, float]]] = None

class ExplainPayload(BaseModel):
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
# Train
# -----------------------
@app.post("/train")
def train(payload: TrainPayload):
    df = pd.DataFrame([r.model_dump() for r in payload.rows])

    for col in FEATURES + TARGETS:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")

    X = df[FEATURES]
    y_voc = df["Voc"]
    y_eff = df["Eff"]

    voc_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    eff_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    voc_model.fit(X, y_voc)
    eff_model.fit(X, y_eff)

    joblib.dump(voc_model, VOC_MODEL)
    joblib.dump(eff_model, EFF_MODEL)

    state = load_state()
    state["Voc_max"] = max(state["Voc_max"], float(df["Voc"].max()))
    state["Eff_max"] = max(state["Eff_max"], float(df["Eff"].max()))
    save_state(state)

    return {"status": "trained", "rows_used": len(df), "state": state}

# -----------------------
# Predict
# -----------------------
@app.post("/predict")
def predict(payload: PredictPayload):
    ensure_models_exist()

    X = pd.DataFrame([r.model_dump() for r in payload.rows])[FEATURES]
    voc_model = joblib.load(VOC_MODEL)
    eff_model = joblib.load(EFF_MODEL)

    voc_pred = voc_model.predict(X)
    eff_pred = eff_model.predict(X)

    preds = []
    for i in range(len(X)):
        preds.append({
            "A": X.iloc[i].to_dict(),
            "Voc": float(voc_pred[i]),
            "Eff": float(eff_pred[i]),
        })

    return {"predictions": preds}

# -----------------------
# Optimize (BO)
# -----------------------
@app.post("/optimize")
def optimize(payload: OptimizePayload):
    ensure_models_exist()

    voc_model = joblib.load(VOC_MODEL)
    eff_model = joblib.load(EFF_MODEL)

    state = load_state()
    Voc_max = max(1e-6, float(state["Voc_max"]))
    Eff_max = max(1e-6, float(state["Eff_max"]))

    bounds = DEFAULT_BOUNDS.copy()
    if payload.bounds:
        for k, v in payload.bounds.items():
            if k in bounds and isinstance(v, list) and len(v) == 2:
                bounds[k] = (float(v[0]), float(v[1]))

    space = [
        Real(*bounds["cu_ratio"], name="cu_ratio"),
        Real(*bounds["ga_ratio"], name="ga_ratio"),
        Real(*bounds["metal_se"], name="metal_se"),
        Real(*bounds["thickness_um"], name="thickness_um"),
    ]

    w_voc = float(payload.w_voc)
    w_eff = float(payload.w_eff)
    w_sum = max(1e-12, w_voc + w_eff)
    w_voc /= w_sum
    w_eff /= w_sum

    def score_from_pred(voc: float, eff: float) -> float:
        return (w_voc * (voc / Voc_max)) + (w_eff * (eff / Eff_max))

    cache: Dict[Tuple[float, float, float, float], float] = {}
    ROUND_N = 6

    def key_from_params(p: Dict[str, float]) -> Tuple[float, float, float, float]:
        return (
            round(p["cu_ratio"], ROUND_N),
            round(p["ga_ratio"], ROUND_N),
            round(p["metal_se"], ROUND_N),
            round(p["thickness_um"], ROUND_N),
        )

    @use_named_args(space)
    def objective(**params):
        k = key_from_params(params)
        if k in cache:
            return cache[k]

        Xp = pd.DataFrame([{
            "cu_ratio": k[0],
            "ga_ratio": k[1],
            "metal_se": k[2],
            "thickness_um": k[3],
        }])[FEATURES]

        voc = float(voc_model.predict(Xp)[0])
        eff = float(eff_model.predict(Xp)[0])
        neg = -score_from_pred(voc, eff)

        cache[k] = neg
        return neg

    x0, y0 = None, None
    if payload.seed_A:
        x0, y0 = [], []
        for s in payload.seed_A:
            k = key_from_params(s)
            x0.append(list(k))
            Xs = pd.DataFrame([dict(zip(FEATURES, k))])[FEATURES]
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

    suggested_A = dict(zip(FEATURES, map(float, res.x)))

    Xs = pd.DataFrame([suggested_A])[FEATURES]
    voc_s = float(voc_model.predict(Xs)[0])
    eff_s = float(eff_model.predict(Xs)[0])
    score_s = score_from_pred(voc_s, eff_s)

    return {
        "suggested_A": suggested_A,
        "pred_at_suggested": {"Voc": voc_s, "Eff": eff_s, "score": score_s},
        "best_set": {
            "A": suggested_A,
            "Voc": voc_s,
            "Eff": eff_s,
            "score": score_s
        },
        "state": state,
        "objective": {
            "type": "weighted_normalized_sum",
            "w_voc": w_voc,
            "w_eff": w_eff,
            "Voc_max": Voc_max,
            "Eff_max": Eff_max,
            "cache_size": len(cache),
            "cache_round": ROUND_N
        }
    }

# -----------------------
# Explain (LLM, GPT-4.1)
# -----------------------
@app.post("/explain")
def explain(payload: ExplainPayload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY not set on server."
        )

    client = OpenAI(api_key=api_key)

    prompt = f"""
너는 태양전지(특히 CIGS) 연구실의 선임 연구자다.

[BO 제안 A]
{json.dumps(payload.suggested_A or {}, ensure_ascii=False)}

[예측 결과]
{json.dumps(payload.pred_at_suggested or {}, ensure_ascii=False)}

[best_set]
{json.dumps(payload.best_set or {}, ensure_ascii=False)}

[objective]
{json.dumps(payload.objective or {}, ensure_ascii=False)}

연구노트 톤으로:
1) 왜 이 조합이 Voc/Eff 관점에서 유망한지
2) 실험 시 주의할 리스크
3) 다음 실험 체크리스트
4) 한 문장 요약
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        temperature=0.4
    )

    text = ""

    # ✅ SDK 객체 / dict 모두 대응
    try:
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                content = getattr(item, "content", None) or item.get("content", [])
                for c in content:
                    c_type = getattr(c, "type", None) or c.get("type")
                    if c_type == "output_text":
                        text += getattr(c, "text", None) or c.get("text", "")
    except Exception as e:
        # 절대 500으로 안 터지게 방어
        text = f"(LLM 응답 파싱 중 경고: {str(e)})"

    if not text.strip():
        text = "(LLM 출력이 비어 있습니다. 프롬프트 또는 모델 상태를 확인하세요.)"

    return {
        "llm_explanation": text.strip()
    }
