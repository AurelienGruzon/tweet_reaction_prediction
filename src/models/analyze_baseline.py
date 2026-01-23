from pathlib import Path
import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

EXPERIMENT_NAME = "sentiment_tweets_clean"
VAL_PATH = Path("data/processed/val.csv")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)


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


def main():
    df_val = pd.read_csv(VAL_PATH)
    X_val = df_val["text"].astype(str)
    y_val = df_val["target"].astype(int).to_numpy()

    run_id = get_best_run_id(EXPERIMENT_NAME, metric="val_auc")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    proba = model.predict_proba(X_val)[:, 1]

    # metrics @0.5
    y_pred_05 = (proba >= 0.5).astype(int)
    base_metrics = {
        "val_accuracy_at_0.5": float(accuracy_score(y_val, y_pred_05)),
        "val_precision_at_0.5": float(precision_score(y_val, y_pred_05)),
        "val_recall_at_0.5": float(recall_score(y_val, y_pred_05)),
        "val_f1_at_0.5": float(f1_score(y_val, y_pred_05)),
        "val_auc": float(roc_auc_score(y_val, proba)),
    }

    print("Loaded model from:", model_uri)
    print("VAL metrics @ threshold=0.5:", base_metrics)

    # Error samples (VAL)
    tmp = df_val.copy()
    tmp["proba_negative"] = proba
    tmp["pred_05"] = y_pred_05

    fp = tmp[(tmp["target"] == 0) & (tmp["pred_05"] == 1)].sort_values("proba_negative", ascending=False)
    fn = tmp[(tmp["target"] == 1) & (tmp["pred_05"] == 0)].sort_values("proba_negative", ascending=True)

    fp_path = OUT_DIR / "val_errors_false_positives.csv"
    fn_path = OUT_DIR / "val_errors_false_negatives.csv"
    fp.head(200).to_csv(fp_path, index=False)
    fn.head(200).to_csv(fn_path, index=False)

    # Threshold sweep (VAL)
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred)),
            "recall": float(recall_score(y_val, y_pred)),
            "f1": float(f1_score(y_val, y_pred)),
        })
    sweep = pd.DataFrame(rows).sort_values("f1", ascending=False)
    sweep_path = OUT_DIR / "val_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best = sweep.iloc[0].to_dict()
    print("Best threshold by F1 on VAL:", best)
    print("Saved:", fp_path, fn_path, sweep_path)

    # Log artifacts/metrics back into the SAME run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(base_metrics)
        mlflow.log_artifact(str(fp_path))
        mlflow.log_artifact(str(fn_path))
        mlflow.log_artifact(str(sweep_path))


if __name__ == "__main__":
    main()
