#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from src.models.impl.text_common_torch import (
    load_split_csv, build_vocab, infer_max_len_from_train, TextDataset,
    compute_metrics, save_confusion_png, save_roc_png, save_vocab
)
from src.models.impl.simple_torch import SimpleMeanPool
from src.models.impl.lstm_torch import BiLSTMClassifier
from transformers import AutoTokenizer
from torch.nn.functional import softmax

from src.models.impl.bert_torch import build_bert_model
from src.models.impl.hf_dataset import HFDataset
from src.models.impl.text_preprocess_nltk import preprocess_texts
from src.models.impl.embeddings_gensim import build_embedding_matrix


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "sentiment_tweets_clean"

SEED = 42


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, vocab_size: int, embedding_dim: int, pad_id: int, embedding_matrix=None, freeze_embeddings=False):
    if model_name == "simple":
        return SimpleMeanPool(vocab_size=vocab_size, embedding_dim=embedding_dim, pad_id=pad_id)
    if model_name == "lstm":
        return BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_id=pad_id,
            hidden_size=64,
            embedding_matrix=embedding_matrix,
            freeze_embeddings=freeze_embeddings,
        )
    raise ValueError(f"Unknown model: {model_name}")



@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def predict_proba_bert(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        # logits shape (B, 2)
        p = softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def make_hf_loader(tokenizer, dataset, max_len: int, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


def train_bert(args, X_train_text, y_train, X_val_text, y_val, X_test_text, y_test):
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    model = build_bert_model(args.bert_model, freeze_base=args.freeze_base).to(device)

    ds_train = HFDataset(X_train_text, y_train)
    ds_val = HFDataset(X_val_text, y_val)
    ds_test = HFDataset(X_test_text, y_test)

    # IMPORTANT: batch-size BERT doit être petit
    train_bs = min(args.batch_size, 32)
    eval_bs = max(train_bs, 32)

    dl_train = make_hf_loader(tokenizer, ds_train, max_len=args.bert_max_len, batch_size=train_bs, shuffle=True)
    dl_val = make_hf_loader(tokenizer, ds_val, max_len=args.bert_max_len, batch_size=eval_bs, shuffle=False)
    dl_test = make_hf_loader(tokenizer, ds_test, max_len=args.bert_max_len, batch_size=eval_bs, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = f"bert_{args.bert_model.replace('/','_')}_len{args.bert_max_len}_freeze{int(args.freeze_base)}"
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "framework": "pytorch_hf",
            "model_name": "bert",
            "bert_model": args.bert_model,
            "bert_max_len": args.bert_max_len,
            "freeze_base": bool(args.freeze_base),
            "batch_size_train": train_bs,
            "batch_size_eval": eval_bs,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "threshold": 0.5,
            "device": str(device),
            "split": "train/val/test (csv)",
        })

        best_val_auc = -1.0
        best_state = None

        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []

            for batch in dl_train:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                out = model(**batch)  # includes loss because labels provided
                loss = out.loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses)) if losses else 0.0
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # VAL
            t_pred = time.perf_counter()
            val_proba = predict_proba_bert(model, dl_val, device)
            val_pred_time = time.perf_counter() - t_pred
            val_metrics, _ = compute_metrics(y_val, val_proba, threshold=0.5)

            mlflow.log_metric("val_auc", float(val_metrics["auc"]), step=epoch)
            mlflow.log_metric("val_f1", float(val_metrics["f1"]), step=epoch)
            mlflow.log_metric("val_predict_time_sec", float(val_pred_time), step=epoch)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        fit_time = time.perf_counter() - t0
        mlflow.log_metric("fit_time_sec", float(fit_time))

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final VAL
        t1 = time.perf_counter()
        val_proba = predict_proba_bert(model, dl_val, device)
        val_pred_time = time.perf_counter() - t1
        val_metrics, val_pred = compute_metrics(y_val, val_proba, threshold=0.5)
        mlflow.log_metric("val_predict_time_sec_final", float(val_pred_time))
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Final TEST
        t2 = time.perf_counter()
        test_proba = predict_proba_bert(model, dl_test, device)
        test_pred_time = time.perf_counter() - t2
        test_metrics, test_pred = compute_metrics(y_test, test_proba, threshold=0.5)
        mlflow.log_metric("test_predict_time_sec", float(test_pred_time))
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Artifacts ROC + CM
        val_cm = ARTIFACTS_DIR / f"{run_name}_val_confusion_matrix.png"
        test_cm = ARTIFACTS_DIR / f"{run_name}_test_confusion_matrix.png"
        val_roc = ARTIFACTS_DIR / f"{run_name}_val_roc_curve.png"
        test_roc = ARTIFACTS_DIR / f"{run_name}_test_roc_curve.png"

        save_confusion_png(y_val, val_pred, val_cm, f"Confusion Matrix (VAL) - BERT")
        save_confusion_png(y_test, test_pred, test_cm, f"Confusion Matrix (TEST) - BERT")
        save_roc_png(y_val, val_proba, val_roc, f"ROC (VAL) - BERT")
        save_roc_png(y_test, test_proba, test_roc, f"ROC (TEST) - BERT")

        mlflow.log_artifact(str(val_cm))
        mlflow.log_artifact(str(test_cm))
        mlflow.log_artifact(str(val_roc))
        mlflow.log_artifact(str(test_roc))

        # Log tokenizer
        tok_dir = ARTIFACTS_DIR / f"{run_name}_tokenizer"
        tok_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tok_dir)
        mlflow.log_artifact(str(tok_dir))

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model")

        print("Run:", run_name)
        print("VAL :", val_metrics)
        print("TEST:", test_metrics)


def main(args):
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_text, y_train = load_split_csv(TRAIN_PATH)
    X_val_text, y_val = load_split_csv(VAL_PATH)
    X_test_text, y_test = load_split_csv(TEST_PATH)
    
    if args.model == "bert":
        return train_bert(args, X_train_text, y_train, X_val_text, y_val, X_test_text, y_test)
    

    X_train_text = preprocess_texts(X_train_text, mode=args.preprocess)
    X_val_text   = preprocess_texts(X_val_text, mode=args.preprocess)
    X_test_text  = preprocess_texts(X_test_text, mode=args.preprocess)



    vocab = build_vocab(X_train_text, max_words=args.max_words, min_freq=2)
    max_len = infer_max_len_from_train(X_train_text, vocab, percentile=args.max_len_percentile)

    embedding_matrix = None
    if args.embed != "random":
        embedding_matrix = build_embedding_matrix(
            texts=X_train_text,     # IMPORTANT: train only
            vocab=vocab,
            method=args.embed,
            dim=args.embedding_dim,
            seed=SEED,
            window=args.w2v_window,
            min_count=args.w2v_min_count,
            epochs=args.w2v_epochs,
        )



    ds_train = TextDataset(X_train_text, y_train, vocab, max_len)
    ds_val = TextDataset(X_val_text, y_val, vocab, max_len)
    ds_test = TextDataset(X_test_text, y_test, vocab, max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    model = build_model(
        args.model,
        vocab_size=len(vocab.itos),
        embedding_dim=args.embedding_dim,
        pad_id=vocab.pad_id,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=bool(args.freeze_embeddings),
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    run_name = f"dl_{args.model}_pre{args.preprocess}_emb{args.embed}_dim{args.embedding_dim}_mw{args.max_words}_len{max_len}"
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "framework": "pytorch",
            "model_name": args.model,
            "preprocess": args.preprocess,
            "embed": args.embed,
            "freeze_embeddings": bool(args.freeze_embeddings),
            "w2v_epochs": args.w2v_epochs,
            "w2v_window": args.w2v_window,
            "w2v_min_count": args.w2v_min_count,
            "max_words": args.max_words,
            "embedding_dim": args.embedding_dim,
            "max_len_percentile": args.max_len_percentile,
            "max_len": max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "threshold": 0.5,
            "device": str(device),
            "split": "train/val/test (csv)",
        })

        # Training loop (simple & stable)
        best_val_auc = -1.0
        best_state = None

        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, dl_train, optimizer, loss_fn, device)

            # quick val AUC each epoch to keep best
            t_pred = time.perf_counter()
            val_proba = predict_proba(model, dl_val, device)
            val_pred_time = time.perf_counter() - t_pred

            val_metrics, _ = compute_metrics(y_val, val_proba, threshold=0.5)

            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_auc", float(val_metrics["auc"]), step=epoch)
            mlflow.log_metric("val_f1", float(val_metrics["f1"]), step=epoch)
            mlflow.log_metric("val_predict_time_sec", float(val_pred_time), step=epoch)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        fit_time = time.perf_counter() - t0
        mlflow.log_metric("fit_time_sec", float(fit_time))

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final eval VAL
        t1 = time.perf_counter()
        val_proba = predict_proba(model, dl_val, device)
        val_pred_time = time.perf_counter() - t1
        val_metrics, val_pred = compute_metrics(y_val, val_proba, threshold=0.5)

        mlflow.log_metric("val_predict_time_sec_final", float(val_pred_time))
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Final eval TEST
        t2 = time.perf_counter()
        test_proba = predict_proba(model, dl_test, device)
        test_pred_time = time.perf_counter() - t2
        test_metrics, test_pred = compute_metrics(y_test, test_proba, threshold=0.5)

        mlflow.log_metric("test_predict_time_sec", float(test_pred_time))
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        val_cm = ARTIFACTS_DIR / f"{run_name}_val_confusion_matrix.png"
        test_cm = ARTIFACTS_DIR / f"{run_name}_test_confusion_matrix.png"
        val_roc = ARTIFACTS_DIR / f"{run_name}_val_roc_curve.png"
        test_roc = ARTIFACTS_DIR / f"{run_name}_test_roc_curve.png"
        vocab_path = ARTIFACTS_DIR / f"{run_name}_vocab.json"


        save_confusion_png(y_val, val_pred, val_cm, f"Confusion Matrix (VAL) - {args.model}")
        save_confusion_png(y_test, test_pred, test_cm, f"Confusion Matrix (TEST) - {args.model}")
        save_roc_png(y_val, val_proba, val_roc, f"ROC (VAL) - {args.model}")
        save_roc_png(y_test, test_proba, test_roc, f"ROC (TEST) - {args.model}")

        mlflow.log_artifact(str(val_cm))
        mlflow.log_artifact(str(test_cm))
        mlflow.log_artifact(str(val_roc))
        mlflow.log_artifact(str(test_roc))

        save_vocab(vocab_path, vocab)
        mlflow.log_artifact(str(vocab_path))

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model")

        print("Run:", run_name)
        print("VAL :", val_metrics)
        print("TEST:", test_metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simple", "lstm", "bert"], required=True)
    p.add_argument("--max-words", type=int, default=100_000)
    p.add_argument("--embedding-dim", type=int, default=100)
    p.add_argument("--max-len-percentile", type=float, default=95.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)  # monte à 5-8 si tu as le temps
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bert-model", type=str, default="distilbert-base-uncased")
    p.add_argument("--bert-max-len", type=int, default=96)
    p.add_argument("--freeze-base", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--preprocess", choices=["clean", "stem", "lemma"], default="clean")
    p.add_argument("--embed", choices=["random", "word2vec", "fasttext"], default="random")
    p.add_argument("--freeze-embeddings", action="store_true")
    p.add_argument("--w2v-epochs", type=int, default=10)
    p.add_argument("--w2v-window", type=int, default=5)
    p.add_argument("--w2v-min-count", type=int, default=2)


    args = p.parse_args()
    main(args)
