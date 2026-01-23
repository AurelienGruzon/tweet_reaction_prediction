import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


def build_bert_model(model_name: str, freeze_base: bool = False):
    """
    Binary classification via HuggingFace model with num_labels=2.
    We'll use softmax -> proba(class=1) for AUC/thresholding.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    if freeze_base:
        base = getattr(model, model.base_model_prefix, None)
        if base is None:
            # fallback: freeze everything except classifier head
            for name, p in model.named_parameters():
                if "classifier" not in name and "score" not in name:
                    p.requires_grad = False
        else:
            for p in base.parameters():
                p.requires_grad = False

    return model
