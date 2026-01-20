#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import mlflow
import mlflow.sklearn


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["text"].astype(str)
    y_train = train_df["target"].astype(int)

    X_test = test_df["text"].astype(str)
    y_test = test_df["target"].astype(int)

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

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_path = ARTIFACT_DIR / "confusion_matrix.csv"
    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"]
    ).to_csv(cm_path, index=True)

    mlflow.set_experiment("sentiment_tweets")

    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        mlflow.log_params({
            "dataset": "sentiment140_processed_v1",
            "vectorizer": "tfidf",
            "max_features": 100_000,
            "ngram_range": "1-2",
            "min_df": 2,
            "classifier": "logreg",
            "max_iter": 1000,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    print("Baseline trained.")
    print(metrics)
    print("Confusion matrix saved to:", cm_path)


if __name__ == "__main__":
    main()