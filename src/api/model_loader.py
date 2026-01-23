from pathlib import Path
import os
import mlflow
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/best"))
TOKENIZER_DIR = Path(os.getenv("TOKENIZER_DIR", "models/best_tokenizer"))

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_tokenizer = None

def load_assets():
    global _model, _tokenizer
    if _model is None:
        _model = mlflow.pytorch.load_model(str(MODEL_DIR)).to(_device)
        _model.eval()

    if _tokenizer is None:
        if not TOKENIZER_DIR.exists():
            raise RuntimeError(f"Tokenizer not found at {TOKENIZER_DIR}. Export it first.")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)

    return _model, _tokenizer

@torch.no_grad()
def predict_proba_negative(text: str, max_len: int = 96) -> float:
    model, tokenizer = load_assets()
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}
    out = model(**enc)
    p = softmax(out.logits, dim=1)[0, 1].item()  # proba classe 1 = "negative"
    return float(p)
