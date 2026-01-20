from pathlib import Path
import pandas as pd
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

EXPERIMENT_NAME = "sentiment_tweets"
TEST_PATH = Path("data/processed/test.csv")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)


def get_best_run_id(experiment_name: str, metric: str = "f1") -> str:
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


def main():
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test["text"].astype(str)
    y_test = df_test["target"].astype(int).to_numpy()

    run_id = get_best_run_id(EXPERIMENT_NAME, metric="f1")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Predict proba if available, else fallback to decision function/predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: use predict as 0/1 and cast to float
        pred = model.predict(X_test)
        proba = pred.astype(float)

    # Default threshold 0.5
    y_pred_05 = (proba >= 0.5).astype(int)

    base_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_05)),
        "precision": float(precision_score(y_test, y_pred_05)),
        "recall": float(recall_score(y_test, y_pred_05)),
        "f1": float(f1_score(y_test, y_pred_05)),
    }
    cm = confusion_matrix(y_test, y_pred_05)

    print("Loaded model from:", model_uri)
    print("Metrics @ threshold=0.5:", base_metrics)
    print("Confusion matrix @0.5:\n", cm)

    # Error samples
    tmp = df_test.copy()
    tmp["proba_neg"] = proba
    tmp["pred_05"] = y_pred_05

    fp = tmp[(tmp["target"] == 0) & (tmp["pred_05"] == 1)].sort_values("proba_neg", ascending=False)
    fn = tmp[(tmp["target"] == 1) & (tmp["pred_05"] == 0)].sort_values("proba_neg", ascending=True)

    fp_path = OUT_DIR / "errors_false_positives.csv"
    fn_path = OUT_DIR / "errors_false_negatives.csv"
    fp.head(200).to_csv(fp_path, index=False)
    fn.head(200).to_csv(fn_path, index=False)

    print("Saved:", fp_path, fn_path)

    # Threshold sweep (optimize F1 + show recall-precision trade-off)
    thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
        })
    sweep = pd.DataFrame(rows).sort_values("f1", ascending=False)
    sweep_path = OUT_DIR / "threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best = sweep.iloc[0].to_dict()
    print("Best threshold by F1:", best)
    print("Saved:", sweep_path)


if __name__ == "__main__":
    main()
