from fastapi import FastAPI
from src.api.schemas import PredictRequest, PredictResponse
from src.api.model_loader import predict_proba_negative, load_assets

app = FastAPI(title="Tweet Sentiment API", version="1.0.0")

@app.on_event("startup")
def _startup():
    # Charge une fois au démarrage (important pour perf)
    load_assets()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = predict_proba_negative(req.text, max_len=96)
    label = "negative" if p >= 0.5 else "not_negative"
    return PredictResponse(label=label, proba_negative=p)
