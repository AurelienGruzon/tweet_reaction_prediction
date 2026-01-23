#!/usr/bin/env python3

from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

import mlflow
import mlflow.sklearn


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def _eval_binary(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }, y_pred


def _save_confusion_png(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non_negative(0)", "negative(1)"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_roc_png(y_true, y_proba, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["text"].astype(str)
    y_train = train_df["target"].astype(int).to_numpy()

    X_val = val_df["text"].astype(str)
    y_val = val_df["target"].astype(int).to_numpy()

    X_test = test_df["text"].astype(str)
    y_test = test_df["target"].astype(int).to_numpy()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ))
    ])

    mlflow.set_experiment("sentiment_tweets_clean")

    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        mlflow.log_params({
            "dataset": "sentiment140_processed_v1",
            "split": "train/val/test",
            "vectorizer": "tfidf",
            "max_features": 100_000,
            "ngram_range": "1-2",
            "min_df": 2,
            "classifier": "logreg",
            "max_iter": 1000,
            "threshold": 0.5,
        })

        # Fit timing
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # Predict timing (VAL)
        t1 = time.perf_counter()
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_pred_time = time.perf_counter() - t1

        val_metrics, val_pred = _eval_binary(y_val, val_proba, threshold=0.5)

        # Predict timing (TEST)
        t2 = time.perf_counter()
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred_time = time.perf_counter() - t2

        test_metrics, test_pred = _eval_binary(y_test, test_proba, threshold=0.5)

        # Log times
        mlflow.log_metrics({
            "fit_time_sec": float(fit_time),
            "val_predict_time_sec": float(val_pred_time),
            "test_predict_time_sec": float(test_pred_time),
        })

        # Log metrics (prefix val_/test_)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Artifacts
        val_cm_png = ARTIFACT_DIR / "val_confusion_matrix.png"
        test_cm_png = ARTIFACT_DIR / "test_confusion_matrix.png"
        val_roc_png = ARTIFACT_DIR / "val_roc_curve.png"
        test_roc_png = ARTIFACT_DIR / "test_roc_curve.png"

        _save_confusion_png(y_val, val_pred, val_cm_png, "Confusion Matrix (VAL) - Baseline TFIDF+LogReg")
        _save_confusion_png(y_test, test_pred, test_cm_png, "Confusion Matrix (TEST) - Baseline TFIDF+LogReg")
        _save_roc_png(y_val, val_proba, val_roc_png, "ROC Curve (VAL) - Baseline TFIDF+LogReg")
        _save_roc_png(y_test, test_proba, test_roc_png, "ROC Curve (TEST) - Baseline TFIDF+LogReg")

        mlflow.log_artifact(str(val_cm_png))
        mlflow.log_artifact(str(test_cm_png))
        mlflow.log_artifact(str(val_roc_png))
        mlflow.log_artifact(str(test_roc_png))

        # Model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print("Baseline trained and logged to MLflow.")
        print("VAL :", val_metrics)
        print("TEST:", test_metrics)


if __name__ == "__main__":
    main()
