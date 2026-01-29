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


def ensure_models_exist() -> None:
    if not (os.path.exists(VOC_MODEL) and os.path.exists(EFF_MODEL)):
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Train first with A+B (Voc, Eff).",
        )


def _safe_json(x: Any, max_len: int = 4000) -> str:
    try:
        s = json.dumps(x if x is not None else {}, ensure_ascii=False)
    except Exception:
        s = "{}"
    if len(s) > max_len:
        s = s[:max_len] + "…(truncated)"
    return s


def _extract_output_text(resp: Any) -> str:
    """
    Robust extractor for OpenAI Responses API:
    resp.output[*].content[*] where content.type == "output_text"
    Works for SDK object and dict-like shapes.
    """
    # 1) output_text shortcut (if present)
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    if isinstance(resp, dict):
        t = resp.get("output_text")
        if isinstance(t, str) and t.strip():
            return t.strip()

    # 2) iterate output blocks
    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")

    if not output:
        return ""

    out_text = ""
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content", [])
        if not content:
            continue

        for c in content:
            c_type = getattr(c, "type", None)
            c_text = getattr(c, "text", None)
            if isinstance(c, dict):
                c_type = c.get("type", c_type)
                c_text = c.get("text", c_text)

            if c_type == "output_text" and isinstance(c_text, str):
                out_text += c_text

    return out_text.strip()


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

    voc_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    eff_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)

    voc_model.fit(X, y_voc)
    eff_model.fit(X, y_eff)

    joblib.dump(voc_model, VOC_MODEL)
    joblib.dump(eff_model, EFF_MODEL)

    state = load_state()
    state["Voc_max"] = max(float(state.get("Voc_max", 0.0)), float(df["Voc"].max()))
    state["Eff_max"] = max(float(state.get("Eff_max", 0.0)), float(df["Eff"].max()))
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
        preds.append(
            {
                "A": X.iloc[i].to_dict(),
                "Voc": float(voc_pred[i]),
                "Eff": float(eff_pred[i]),
            }
        )

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
    Voc_max = max(1e-6, float(state.get("Voc_max", 0.75)))
    Eff_max = max(1e-6, float(state.get("Eff_max", 18.0)))

    # bounds
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

    # weight normalize
    w_voc = float(payload.w_voc)
    w_eff = float(payload.w_eff)
    w_sum = max(1e-12, w_voc + w_eff)
    w_voc /= w_sum
    w_eff /= w_sum

    def score_from_pred(voc: float, eff: float) -> float:
        return (w_voc * (voc / Voc_max)) + (w_eff * (eff / Eff_max))

    # cache neg score by rounded params
    cache: Dict[Tuple[float, float, float, float], float] = {}
    ROUND_N = 6

    def key_from_params(p: Dict[str, float]) -> Tuple[float, float, float, float]:
        return (
            round(float(p["cu_ratio"]), ROUND_N),
            round(float(p["ga_ratio"]), ROUND_N),
            round(float(p["metal_se"]), ROUND_N),
            round(float(p["thickness_um"]), ROUND_N),
        )

    @use_named_args(space)
    def objective(**params):
        k = key_from_params(params)
        if k in cache:
            return cache[k]

        Xp = pd.DataFrame(
            [
                {
                    "cu_ratio": k[0],
                    "ga_ratio": k[1],
                    "metal_se": k[2],
                    "thickness_um": k[3],
                }
            ]
        )[FEATURES]

        voc = float(voc_model.predict(Xp)[0])
        eff = float(eff_model.predict(Xp)[0])
        neg = -float(score_from_pred(voc, eff))

        cache[k] = neg
        return neg

    # optional seeds
    x0, y0 = None, None
    if payload.seed_A:
        x0, y0 = [], []
        for s in payload.seed_A:
            k = key_from_params(s)
            x0.append(list(k))

            Xs = pd.DataFrame(
                [
                    {
                        "cu_ratio": k[0],
                        "ga_ratio": k[1],
                        "metal_se": k[2],
                        "thickness_um": k[3],
                    }
                ]
            )[FEATURES]

            voc = float(voc_model.predict(Xs)[0])
            eff = float(eff_model.predict(Xs)[0])
            y0.append(-float(score_from_pred(voc, eff)))

    res = gp_minimize(
        objective,
        space,
        n_calls=int(payload.n_calls),
        x0=x0,
        y0=y0,
        random_state=42,
    )

    suggested_A = dict(zip(FEATURES, map(float, res.x)))

    Xs = pd.DataFrame([suggested_A])[FEATURES]
    voc_s = float(voc_model.predict(Xs)[0])
    eff_s = float(eff_model.predict(Xs)[0])
    score_s = float(score_from_pred(voc_s, eff_s))

    best_set = {"A": suggested_A, "Voc": voc_s, "Eff": eff_s, "score": score_s}

    return {
        "suggested_A": suggested_A,
        "pred_at_suggested": {"Voc": voc_s, "Eff": eff_s, "score": score_s},
        "best_set": best_set,
        "state": state,
        "objective": {
            "type": "weighted_normalized_sum",
            "w_voc": w_voc,
            "w_eff": w_eff,
            "Voc_max": Voc_max,
            "Eff_max": Eff_max,
            "cache_size": len(cache),
            "cache_round": ROUND_N,
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
{_safe_json(payload.suggested_A)}

[예측 결과]
{_safe_json(payload.pred_at_suggested)}

[best_set]
{_safe_json(payload.best_set)}

[objective]
{_safe_json(payload.objective)}

다음을 한국어로, 태양전지 전문 연구자의 연구노트 톤으로 작성:
1) 추출된 효율, Voc 데이터가 왜 최적의 조합인지 설명
2) 다음 실험에 쓸 최적의 조합 추천
3) 위 내용 한 문장 요약
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

        text = ""
        if "generated_text" in data:
            text = data["generated_text"]
        elif "output" in data:
            text = data["output"]
        elif "outputs" in data and isinstance(data["outputs"], list):
            text = data["outputs"][0].get("text", "")
        elif "result" in data:
            text = data["result"].get("text", "")

    except Exception as e:
        return {
            "llm_explanation": "(Upstage 호출 실패로 설명 생략)",
            "error": str(e),
            "fallback": True,
        }

    return {
        "llm_explanation": text.strip() or "(Upstage 출력 없음)",
        "meta": {
            "provider": "upstage-pro2",
            "latency_sec": round(time.time() - t0, 3),
        },
    }
