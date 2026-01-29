import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.model_loader import predict_proba_negative, load_assets
from src.api.schemas import PredictRequest, PredictResponse, FeedbackRequest
from src.api.feedback_store import init_db, add_feedback, update_bad_streak
from src.api.alerting import alert_if_needed


import os
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


from pydantic import BaseModel

if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

tracer = trace.get_tracer("tweet-sentiment")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await asyncio.to_thread(load_assets)
    yield



app = FastAPI(
    title="Tweet Sentiment API",
    version="1.0.3",
    lifespan=lifespan,
)

FastAPIInstrumentor.instrument_app(app)


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
    # 1) Persist
    add_feedback(
        text=req.text,
        predicted_label=req.predicted_label,
        proba_negative=req.proba_negative,
        is_correct=req.is_correct,
        true_label=req.true_label,
    )

    # 2) Compteur erreurs consécutives
    bad_streak = update_bad_streak(req.is_correct)

    # 3) Traces (surtout quand c'est faux)
    if not req.is_correct:
        with tracer.start_as_current_span("prediction_not_validated") as span:
            span.set_attribute("predicted_label", req.predicted_label)
            if req.proba_negative is not None:
                span.set_attribute("proba_negative", float(req.proba_negative))
            span.set_attribute("is_correct", False)

        # 4) Alerte si >= 5 (configurable)
        alert_if_needed(
            bad_streak=bad_streak,
            predicted_label=req.predicted_label,
            proba_negative=req.proba_negative,
        )

    return {"status": "ok", "bad_streak": bad_streak}
