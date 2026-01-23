#!/usr/bin/env python3
from pathlib import Path
import shutil
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "sentiment_tweets_clean"
OUT_DIR = Path("models/best")

def get_best_run_id(experiment_name: str, metric: str = "val_auc") -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs([exp.experiment_id], order_by=[f"metrics.{metric} DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found.")
    return runs[0].info.run_id

def main():
    run_id = get_best_run_id(EXPERIMENT_NAME, "val_auc")
    model_uri = f"runs:/{run_id}/model"
    print("Exporting best model from:", model_uri)

    tmp = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Nettoie la destination et copie
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    shutil.copytree(tmp, OUT_DIR)

    print("Exported to:", OUT_DIR)

if __name__ == "__main__":
    main()
