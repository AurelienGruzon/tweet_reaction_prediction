#!/usr/bin/env python3
from pathlib import Path
import shutil

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "sentiment_tweets_clean"
OUT_DIR = Path("models/best_tokenizer")


def get_best_run_id(experiment_name: str, metric: str = "val_auc") -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs([exp.experiment_id], order_by=[f"metrics.{metric} DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found.")
    return runs[0].info.run_id, runs[0].data.tags.get("mlflow.runName", "")


def main():
    client = MlflowClient()
    run_id, run_name = get_best_run_id(EXPERIMENT_NAME, "val_auc")
    print("BEST RUN:", run_id, run_name)

    # Trouver le dossier tokenizer dans les artefacts
    tokenizer_dir = None
    for a in client.list_artifacts(run_id):
        if a.is_dir and "tokenizer" in a.path:
            tokenizer_dir = a.path
            break
    if tokenizer_dir is None:
        raise RuntimeError("Tokenizer dir not found in run artifacts.")

    print("Tokenizer artifact dir:", tokenizer_dir)

    local_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=tokenizer_dir))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_dir, OUT_DIR)

    print("Exported tokenizer to:", OUT_DIR)


if __name__ == "__main__":
    main()
