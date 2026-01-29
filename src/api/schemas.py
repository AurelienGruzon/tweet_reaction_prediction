from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    label: str
    proba_negative: float

class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    proba_negative: float | None = None
    is_correct: bool
    true_label: Optional[str] = None  # optionnel