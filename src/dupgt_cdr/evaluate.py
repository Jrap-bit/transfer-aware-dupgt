"""Baselines, error metrics and the two-dataset comparison."""
from __future__ import annotations

import csv
import json

import numpy as np

from .common import resolve, write_json

# Predict each Movie's training average.
def mean_predictions(train_rows: np.ndarray, query: np.ndarray, n_items: int, smoothing: float = 0.0) -> np.ndarray:
    """Target training ratings only; unseen items fall back to training global mean."""
    global_mean = float(train_rows[:, 2].mean())
    count = np.bincount(train_rows[:, 1], minlength=n_items+1)
    sums = np.bincount(train_rows[:, 1], weights=train_rows[:, 2], minlength=n_items+1)
    average = np.full(n_items+1, global_mean, dtype=np.float64)
    observed = count > 0
    average[observed] = (sums[observed] + smoothing*global_mean) / (count[observed] + smoothing)
    return average[query[:, 1]]

# Measure rating error with MAE and RMSE.
def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("Metric arrays must have the same nonempty shape")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Non-finite predictions")
    residual = prediction - truth
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.square(residual).mean())), "n_ratings": len(truth)}



# Measure error for each user.
def user_metrics(rows: np.ndarray, prediction: np.ndarray) -> dict:
    ids, inverse = np.unique(rows[:, 0], return_inverse=True)
    absolute = np.abs(rows[:, 2] - prediction)
    counts = np.bincount(inverse)
    mae = np.bincount(inverse, weights=absolute) / counts
    rmse = np.sqrt(np.bincount(inverse, weights=absolute**2) / counts)
    return {"ids": ids, "mae": mae, "rmse": rmse}



def stratified_results(rows: np.ndarray, data: dict, predictions: dict) -> dict:
    count = data["polarity_counts"][rows[:, 0]]
    definitions = {
        "source_has_no_low_ratings": count[:, 1] == 0,
        "source_has_both_polarities": (count[:, 0] > 0) & (count[:, 1] > 0),
        "source_1_to_5_ratings": count.sum(axis=1) <= 5,
        "source_6_to_10_ratings": (count.sum(axis=1) > 5) & (count.sum(axis=1) <= 10),
        "source_11_to_20_ratings": (count.sum(axis=1) > 10) & (count.sum(axis=1) <= 20),
        "source_over_20_ratings": count.sum(axis=1) > 20,
    }
    return {group: {"users": len(np.unique(rows[mask, 0])), "metrics": {
        name: metrics(rows[mask, 2], prediction[mask]) for name, prediction in predictions.items()}}
        for group, mask in definitions.items() if mask.any()}

MATCHED_KEYS = ("seed", "max_shared_users", "max_source_only_users", "max_target_only_users",
    "validation_fraction", "test_fraction", "embedding_dim", "factor_initialization_std",
    "pretrain_epochs", "mapping_epochs", "meta_epochs", "batch_size", "meta_batch_size",
    "evaluation_batch_size", "pretrain_learning_rate", "mapping_learning_rate",
    "meta_learning_rate", "weight_decay", "torch_threads", "device")


# Summarize two finished dataset runs.
def compare(run_paths: list[str], output_dir: str) -> dict:
    output = resolve(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty comparison directory: {output}")
    runs = []
    for path in run_paths:
        run = resolve(path)
        run_metrics = json.loads((run/"metrics.json").read_text())
        config = json.loads((run/"config.json").read_text())
        if run_metrics["status"] != "completed":
            raise ValueError(f"Incomplete run: {run}")
        runs.append({"directory": str(run), "config": config, "metrics": run_metrics})
    if len(runs) < 2:
        raise ValueError("Comparison requires two or more completed runs")
    for other in runs[1:]:
        differences = [key for key in MATCHED_KEYS if other["config"].get(key) != runs[0]["config"].get(key)]
        if differences:
            raise ValueError(f"Unmatched experiment settings: {differences}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        reference = run["metrics"]["results"]["movie_mean"]["raw"]
        for name, value in run["metrics"]["results"].items():
            score = value["raw"]
            rows.append({"release": run["config"]["dataset_release"], "model": name,
                "mae": score["mae"], "rmse": score["rmse"], "test_ratings": score["n_ratings"],
                "mae_improvement_vs_movie_mean_pct": 100*(reference["mae"]-score["mae"])/reference["mae"],
                "rmse_improvement_vs_movie_mean_pct": 100*(reference["rmse"]-score["rmse"])/reference["rmse"]})
    with (output/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"matched_settings": {key: runs[0]["config"].get(key) for key in MATCHED_KEYS},
        "results": rows, "runs": [run["directory"] for run in runs]}
    write_json(output/"comparison.json", result)
    for row in rows:
        print(f"{row['release']} {row['model']}: MAE {row['mae']:.4f}, RMSE {row['rmse']:.4f}")
    return result
