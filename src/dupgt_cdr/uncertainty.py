"""Uncertainty-gated transfer: an ensemble of bridges and a validation-fitted fallback.

Nothing here changes how a single run is trained or scored. It adds three steps on top of
finished `transfer-train` runs that share the same real users but use different bridge seeds:

1. `save_validation_predictions`: score each member on the validation split (never test).
2. `fit_blend`: combine the members on validation only. For every prediction,

       mu = mean of the K bridges          (ensemble prediction)
       d  = standard deviation across them (disagreement = uncertainty of the transfer)
       m  = the movie's training average   (target-only fallback)

       prediction = w * mu + (1 - w) * m,   w = s0 / (s0 + tau2 + c * d^2)

   This is inverse-variance weighting: s0 is the movie average's error variance, and the
   transfer's error variance is modelled as tau2 + c * d^2. tau2 and c are fitted on
   validation MSE. Setting c = 0 gives a constant blend (the same shrinkage for everyone),
   which is the control: if disagreement carries no information, c adds nothing.
3. `evaluate_blend`: after the members are locked (the lock reason cites the blend record's
   hash) and test-scored with `transfer-evaluate`, apply the frozen blend to their saved test
   predictions once.
"""
from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import file_sha256, resolve, write_json
from .evaluate import mean_predictions, metrics, user_metrics

VALIDATION_FILE = "validation_predictions.csv.gz"
TAU2_GRID = np.logspace(-3, 1, 41)
C_GRID = np.concatenate([[0.0], np.logspace(-2, 3, 51)])
MATCHED = ("dataset_sha256", "factor_sha256", "overlap_seed", "overlap_fraction", "real_rows_sha256",
           "users_sha256", "validation_rows_sha256", "model_variant", "semantic_cache", "teacher_kind")


# 1. VALIDATION PREDICTIONS FOR ONE FINISHED RUN

def save_validation_predictions(run_dir: str) -> dict:
    from .data import get_benchmark
    from .experiments import validated_run, parameter_hash, array_hash
    from .predict import predict
    from .runtime import choose_device, load_saved_model, set_seed
    directory = resolve(run_dir)
    config, experiment, _, artifacts = validated_run(directory)
    record = directory/"validation_predictions.json"
    if record.exists():
        saved = json.loads(record.read_text())
        if saved["predictions_sha256"] != file_sha256(directory/VALIDATION_FILE):
            raise ValueError("Saved validation predictions changed")
        return saved
    data, manifest = get_benchmark(config)
    if manifest["benchmark_sha256"] != experiment["dataset_sha256"]:
        raise ValueError("Benchmark differs from training")
    checkpoint = torch.load(directory/"checkpoints/dupgt_best.pt", map_location="cpu", weights_only=True)
    set_seed(experiment["transfer_seed"], config["torch_threads"])
    model = load_saved_model(checkpoint, choose_device(config["device"]))
    if parameter_hash(model, ("src_model.", "tgt_model.")) != artifacts["factor_sha256"]:
        raise ValueError("Checkpoint factors differ from training inputs")
    rows = data["validation"]
    values, gates = predict(model, rows, torch.as_tensor(data["histories"]), "test_meta",
                            choose_device(config["device"]), config["evaluation_batch_size"])
    reference = mean_predictions(data["target_train"], rows, checkpoint["model_spec"]["n_items"])
    with gzip.open(directory/VALIDATION_FILE, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user", "movie", "actual_rating", "movie_mean", "dupgt", "alpha_high"])
        writer.writerows(zip(rows[:, 0].tolist(), rows[:, 1].tolist(), rows[:, 2].tolist(),
                             reference.tolist(), values.tolist(), gates.tolist()))
    result = {"schema": 1, "split": "validation", "metrics": metrics(rows[:, 2], values),
              "movie_mean": metrics(rows[:, 2], reference), "validation_rows_sha256": array_hash(rows),
              "artifact_manifest_sha256": file_sha256(directory/"artifact_manifest.json"),
              "predictions_sha256": file_sha256(directory/VALIDATION_FILE)}
    write_json(record, result)
    return result


# 2. THE BLEND: INVERSE-VARIANCE WEIGHTING WITH A DISAGREEMENT TERM

def blend_weight(disagreement: np.ndarray, s0: float, tau2: float, c: float) -> np.ndarray:
    return s0 / (s0 + tau2 + c * np.square(disagreement))


def blended(members: np.ndarray, prior: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """members: K x N bridge predictions; prior: N movie averages. Returns prediction, weight, disagreement."""
    mu, d = members.mean(0), members.std(0)
    w = blend_weight(d, params["s0"], params["tau2"], params["c"])
    return w * mu + (1 - w) * prior, w, d


def fit_parameters(members: np.ndarray, prior: np.ndarray, truth: np.ndarray, *, constant: bool) -> dict:
    s0 = float(np.mean(np.square(prior - truth)))
    best = None
    for c in ([0.0] if constant else C_GRID):
        for tau2 in TAU2_GRID:
            params = {"s0": s0, "tau2": float(tau2), "c": float(c)}
            mse = float(np.mean(np.square(blended(members, prior, params)[0] - truth)))
            if best is None or mse < best[0]:
                best = (mse, params)
    return best[1]


def read_validation(directory: Path) -> dict:
    with gzip.open(directory/VALIDATION_FILE, "rt") as handle:
        columns = list(zip(*list(csv.reader(handle))[1:]))
    names = ["user", "movie", "actual_rating", "movie_mean", "dupgt", "alpha_high"]
    return {name: np.asarray(values, dtype=np.float64) for name, values in zip(names, columns)}


def check_members(paths: list[Path]) -> list[dict]:
    from .experiments import validated_run
    if len(paths) < 2:
        raise ValueError("An ensemble needs at least two bridges")
    experiments = [validated_run(path)[1] for path in paths]
    first = experiments[0]
    for experiment in experiments[1:]:
        if any(experiment.get(key) != first.get(key) for key in MATCHED):
            raise ValueError("Ensemble members must share data, factors, real users and model variant")
    seeds = [e["transfer_seed"] for e in experiments]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble members need different bridge seeds")
    return experiments


def fit_blend(run_paths: list[str], output_dir: str) -> dict:
    from .experiments import empty_destination
    paths = [resolve(path) for path in run_paths]
    experiments = check_members(paths)
    if any((path/"test_metrics.json").exists() for path in paths):
        raise ValueError("Fit the blend before any member is scored on test")
    records = [save_validation_predictions(str(path)) for path in paths]
    columns = [read_validation(path) for path in paths]
    truth, prior = columns[0]["actual_rating"], columns[0]["movie_mean"]
    if any(not np.array_equal(c["actual_rating"], truth) or not np.array_equal(c["movie_mean"], prior) for c in columns):
        raise ValueError("Members were scored on different validation rows")
    members = np.stack([c["dupgt"] for c in columns])
    uncertainty = fit_parameters(members, prior, truth, constant=False)
    constant = fit_parameters(members, prior, truth, constant=True)
    final, weight, disagreement = blended(members, prior, uncertainty)
    output = resolve(output_dir)
    empty_destination(output)
    result = {"schema": 1, "method": "uncertainty_gated_ensemble", "split_used_for_fitting": "validation",
              "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
              "formula": "w = s0 / (s0 + tau2 + c * d^2); prediction = w * mean(bridges) + (1 - w) * movie_mean",
              "members": [{"run_dir": str(p), "transfer_seed": e["transfer_seed"],
                           "artifact_manifest_sha256": file_sha256(p/"artifact_manifest.json"),
                           "validation_predictions_sha256": r["predictions_sha256"]}
                          for p, e, r in zip(paths, experiments, records)],
              "overlap_fraction": experiments[0]["overlap_fraction"], "overlap_seed": experiments[0]["overlap_seed"],
              "model_variant": experiments[0].get("model_variant", "dupgt"),
              "uncertainty_blend": uncertainty, "constant_blend": constant,
              "validation": {"single_bridges": [metrics(truth, m) for m in members],
                             "ensemble_mean": metrics(truth, members.mean(0)),
                             "constant_blend": metrics(truth, blended(members, prior, constant)[0]),
                             "uncertainty_blend": metrics(truth, final),
                             "movie_mean": metrics(truth, prior),
                             "mean_weight": float(weight.mean()), "mean_disagreement": float(disagreement.mean())}}
    write_json(output/"blend.json", result)
    return result


# 3. ONE FINAL TEST APPLICATION OF THE FROZEN BLEND

def read_test(directory: Path) -> dict:
    with gzip.open(directory/"test_predictions.csv.gz", "rt") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {"user_id": np.array([r["user_id"] for r in rows]),
            "actual_rating": np.array([float(r["actual_rating"]) for r in rows]),
            "movie_mean": np.array([float(r["movie_mean"]) for r in rows]),
            "dupgt": np.array([float(r["dupgt"]) for r in rows]),
            "alpha_high": np.array([float(r["alpha_high"]) for r in rows])}


def evaluate_blend(blend_dir: str, selection_record: str) -> dict:
    directory, record_path = resolve(blend_dir), resolve(selection_record)
    blend_file = directory/"blend.json"
    blend = json.loads(blend_file.read_text())
    if (directory/"test_metrics.json").exists():
        return json.loads((directory/"test_metrics.json").read_text())
    selection = json.loads(record_path.read_text())
    if file_sha256(blend_file) not in selection["reason"]:
        raise ValueError("The test lock must cite this blend record's hash (blend fitted before test)")
    locked = {entry["run_dir"] for entry in selection["runs"]}
    members = [resolve(m["run_dir"]) for m in blend["members"]]
    for member, saved in zip(members, blend["members"]):
        if str(member) not in locked:
            raise ValueError("Every ensemble member must be in the locked selection")
        if file_sha256(member/"artifact_manifest.json") != saved["artifact_manifest_sha256"]:
            raise ValueError("A member changed after the blend was fitted")
        scores = json.loads((member/"test_metrics.json").read_text())
        if scores["selection_sha256"] != file_sha256(record_path):
            raise ValueError("Members must be test-scored under this selection")
    tests = [read_test(member) for member in members]
    truth, prior, users = tests[0]["actual_rating"], tests[0]["movie_mean"], tests[0]["user_id"]
    if any(not np.array_equal(t["actual_rating"], truth) or not np.array_equal(t["user_id"], users) for t in tests):
        raise ValueError("Members were scored on different test rows")
    stacked = np.stack([t["dupgt"] for t in tests])
    final, weight, disagreement = blended(stacked, prior, blend["uncertainty_blend"])
    constant = blended(stacked, prior, blend["constant_blend"])[0]
    _, codes = np.unique(users, return_inverse=True)
    rows = np.stack([codes, np.zeros_like(codes), truth], axis=1)
    reference_users = user_metrics(rows, prior)["mae"]

    def summary(prediction):
        per_user = user_metrics(rows, prediction)["mae"]
        return {**metrics(truth, prediction), "negative_transfer_rate": float(np.mean(per_user > reference_users))}

    result = {"schema": 1, "split": "test", "blend_sha256": file_sha256(blend_file),
              "selection_sha256": file_sha256(record_path),
              "single_bridges": [summary(s) for s in stacked], "ensemble_mean": summary(stacked.mean(0)),
              "constant_blend": summary(constant), "uncertainty_blend": summary(final),
              "movie_mean": metrics(truth, prior), "mean_weight": float(weight.mean()),
              "mean_disagreement": float(disagreement.mean())}
    with gzip.open(directory/"test_predictions.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "actual_rating", "movie_mean", "ensemble_mean", "disagreement", "weight",
                         "constant_blend", "uncertainty_blend"])
        writer.writerows(zip(users.tolist(), truth.tolist(), prior.tolist(), stacked.mean(0).tolist(),
                             disagreement.tolist(), weight.tolist(), constant.tolist(), final.tolist()))
    result["predictions_sha256"] = file_sha256(directory/"test_predictions.csv.gz")
    write_json(directory/"test_metrics.json", result)
    return result
