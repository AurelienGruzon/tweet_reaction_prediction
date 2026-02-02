import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.model_loader import predict_proba_negative, load_assets
from src.api.schemas import PredictRequest, PredictResponse, FeedbackRequest
from src.api.feedback_store import init_db, add_feedback, update_bad_streak
from src.api.alerting import alert_if_needed

# Logging simple (stdout). Azure App Service récupère stdout/stderr -> Application Insights traces.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tweet-sentiment")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB feedback + chargement modèle (en thread pour éviter de bloquer l'event loop)
    init_db()
    await asyncio.to_thread(load_assets)
    yield


app = FastAPI(
    title="Tweet Sentiment API",
    version="1.0.4",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = predict_proba_negative(req.text, max_len=96)
    label = "negative" if p >= 0.5 else "not_negative"
    return PredictResponse(label=label, proba_negative=p)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # 1) Persist feedback
    add_feedback(
        text=req.text,
        predicted_label=req.predicted_label,
        proba_negative=req.proba_negative,
        is_correct=req.is_correct,
        true_label=req.true_label,
    )

    # 2) Update streak
    bad_streak = update_bad_streak(req.is_correct)

    # 3) Emit log signal for Azure Monitor (rule: traces contains "BAD_PREDICTION")
    if not req.is_correct:
        logger.warning(
            "BAD_PREDICTION bad_streak=%d predicted_label=%s proba_negative=%s true_label=%s",
            bad_streak,
            req.predicted_label,
            req.proba_negative,
            req.true_label,
        )

        # 4) Keep your optional hook (if it logs or does something else)
        alert_if_needed(
            bad_streak=bad_streak,
            predicted_label=req.predicted_label,
            proba_negative=req.proba_negative,
        )

    return {"status": "ok", "bad_streak": bad_streak}
