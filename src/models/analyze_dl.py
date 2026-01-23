#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from src.models.impl.text_common_torch import (
    load_split_csv, TextDataset, Vocab,
    compute_metrics
)
from src.models.impl.hf_dataset import HFDataset
from transformers import AutoTokenizer


EXPERIMENT_NAME = "sentiment_tweets_clean"
VAL_PATH = Path("data/processed/val.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_vocab(vocab_path: Path) -> Vocab:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    itos = payload["itos"]
    pad_id = payload["pad_id"]
    unk_id = payload["unk_id"]
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_id=pad_id, unk_id=unk_id)


def get_best_run_id(experiment_name: str, metric: str = "val_auc") -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=10,
    )
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")
    return runs[0].info.run_id


def threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        m, _ = compute_metrics(y_true, y_proba, threshold=float(t))
        rows.append({"threshold": float(t), **m})
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


@torch.no_grad()
def predict_proba_vocab_model(model, loader, device):
    model.eval()
    probs = []
    for xb, _yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def make_hf_loader(tokenizer, dataset, max_len: int, batch_size: int):
    def collate_fn(samples):
        texts, labels = zip(*samples)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        labels_t = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels_t,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)


@torch.no_grad()
def predict_proba_bert(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out = model(**batch)
        p = softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def main():
    client = MlflowClient()
    run_id = get_best_run_id(EXPERIMENT_NAME, metric="val_auc")
    run = client.get_run(run_id)

    run_name = run.data.tags.get("mlflow.runName", "unknown_run")
    params = run.data.params

    model_uri = f"runs:/{run_id}/model"
    print("Best run:", run_id)
    print("Run name :", run_name)
    print("Model URI:", model_uri)

    # Load VAL split
    X_val_text, y_val = load_split_csv(VAL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect if BERT run
    is_bert = ("bert_model" in params) or (params.get("model_name") == "bert") or run_name.startswith("bert_")

    if is_bert:
        # ---- BERT path ----
        model = mlflow.pytorch.load_model(model_uri).to(device)

        # Download tokenizer artifact directory
        # In our training code we logged a directory named something like "{run_name}_tokenizer"
        artifacts = client.list_artifacts(run_id)
        tok_dir_rel = None
        for a in artifacts:
            if a.is_dir and "tokenizer" in a.path:
                tok_dir_rel = a.path
                break
        if tok_dir_rel is None:
            raise RuntimeError("Tokenizer directory not found in MLflow artifacts for this BERT run.")

        tok_dir_local = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=tok_dir_rel))
        tokenizer = AutoTokenizer.from_pretrained(tok_dir_local, use_fast=True)

        max_len = int(params.get("bert_max_len", 96))
        ds_val = HFDataset(X_val_text, y_val)
        dl_val = make_hf_loader(tokenizer, ds_val, max_len=max_len, batch_size=512)

        proba = predict_proba_bert(model, dl_val, device)

    else:
        # ---- Vocab model path (simple/lstm) ----
        model = mlflow.pytorch.load_model(model_uri).to(device)

        # Find vocab.json in artifacts
        artifacts = client.list_artifacts(run_id)
        vocab_rel = None
        for a in artifacts:
            if a.path.endswith("vocab.json"):
                vocab_rel = a.path
                break
        if vocab_rel is None:
            raise RuntimeError("vocab.json not found in MLflow artifacts for this run.")

        vocab_local = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=vocab_rel))
        vocab = load_vocab(vocab_local)

        max_len = int(params["max_len"])
        ds_val = TextDataset(X_val_text, y_val, vocab, max_len)
        dl_val = DataLoader(ds_val, batch_size=2048, shuffle=False, num_workers=0)

        proba = predict_proba_vocab_model(model, dl_val, device)

    # ---- Common analysis (VAL) ----
    base_metrics, y_pred_05 = compute_metrics(y_val, proba, threshold=0.5)
    print("VAL @0.5:", base_metrics)

    tmp = pd.DataFrame({"text": X_val_text, "target": y_val})
    tmp["proba_negative"] = proba
    tmp["pred_05"] = y_pred_05

    fp = tmp[(tmp["target"] == 0) & (tmp["pred_05"] == 1)].sort_values("proba_negative", ascending=False)
    fn = tmp[(tmp["target"] == 1) & (tmp["pred_05"] == 0)].sort_values("proba_negative", ascending=True)

    # PREFIX to avoid overwriting baseline/DL
    prefix = run_name

    fp_path = ARTIFACTS_DIR / f"{prefix}_val_errors_false_positives.csv"
    fn_path = ARTIFACTS_DIR / f"{prefix}_val_errors_false_negatives.csv"
    fp.head(200).to_csv(fp_path, index=False)
    fn.head(200).to_csv(fn_path, index=False)

    sweep = threshold_sweep(y_val, proba)
    sweep_path = ARTIFACTS_DIR / f"{prefix}_val_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best = sweep.iloc[0].to_dict()
    print("Best threshold by F1 (VAL):", best)

    # Log artifacts back into SAME run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            f"{prefix}_val_f1_at_0.5": float(base_metrics["f1"]),
            f"{prefix}_val_auc_at_0.5": float(base_metrics["auc"]),
            f"{prefix}_best_threshold_f1": float(best["threshold"]),
            f"{prefix}_best_f1": float(best["f1"]),
        })
        mlflow.log_artifact(str(fp_path))
        mlflow.log_artifact(str(fn_path))
        mlflow.log_artifact(str(sweep_path))


if __name__ == "__main__":
    main()
