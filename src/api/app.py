import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


from src.api.model_loader import predict_proba_negative, load_assets
from src.api.schemas import PredictRequest, PredictResponse, FeedbackRequest
from src.api.feedback_store import init_db, add_feedback, update_bad_streak

# --- Azure Monitor OpenTelemetry -> Application Insights (table traces, etc.)
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

# --- Logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tweet-sentiment")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await asyncio.to_thread(load_assets)
    yield


app = FastAPI(title="Tweet Sentiment API", version="1.0.5", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

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
    add_feedback(
        text=req.text,
        predicted_label=req.predicted_label,
        proba_negative=req.proba_negative,
        is_correct=req.is_correct,
        true_label=req.true_label,
    )

    bad_streak = update_bad_streak(req.is_correct)

    if not req.is_correct:
        logger.warning(
            "BAD_PREDICTION bad_streak=%d predicted_label=%s proba_negative=%s true_label=%s",
            bad_streak,
            req.predicted_label,
            req.proba_negative,
            req.true_label,
        )

    return {"status": "ok", "bad_streak": bad_streak}
