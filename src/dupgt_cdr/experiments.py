"""Frozen-factor bridge experiments and explicit final evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import file_sha256, resolve, write_json
from .data import get_benchmark
from .evaluate import mean_predictions, metrics, user_metrics
from .model import DUPGTModel
from .predict import predict
from .runtime import choose_device, model_spec, set_seed
from .settings import TRANSFER
from .train import fit_transfer, parameter_hash, save_predictions


# 1. TRAINING VIEWS; THE ORIGINAL BENCHMARK IS NEVER REPARTITIONED

def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_only_user_ids(data: dict) -> np.ndarray:
    source = np.unique(data["source_train"][:, 0])
    target = np.unique(np.concatenate([data[key][:, 0] for key in ("target_train", "validation", "test")]))
    # Original held-out partitions remain excluded even after an overlap restriction.
    target = np.union1d(target, np.concatenate([data[key] for key in
                         ("shared_train_users", "validation_users", "test_users")]))
    return np.setdiff1d(source, target)


def select_overlap_rows(data: dict, fraction: float, overlap_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("overlap_fraction must be in (0, 1]")
    permutation = np.random.default_rng(overlap_seed).permutation(data["shared_train_users"])
    count = max(1, math.floor(fraction * len(permutation)))
    users = np.sort(permutation[:count])
    rows = data["meta_train"][np.isin(data["meta_train"][:, 0], users)]
    if not len(users) or not len(rows):
        raise ValueError("No selected real supervision")
    return users, rows


# 2. LOAD INDEPENDENT FACTORS AND KEEP A FRESH BRIDGE

def load_factor_initialized_model(checkpoint_path: Path, data: dict, manifest: dict,
                                  config: dict, transfer_seed: int, semantic_dir=None) -> DUPGTModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_spec = model_spec(data, config)
    if checkpoint.get("stage") != "pretrained":
        raise ValueError("Use pretrained_factors.pt, not a bridge already trained on full overlap")
    if checkpoint.get("dataset_sha256") != manifest["benchmark_sha256"] or checkpoint.get("model_spec") != expected_spec:
        raise ValueError("Factor checkpoint and benchmark/model spec do not match")
    set_seed(transfer_seed, config["torch_threads"])
    if semantic_dir:
        from .semantics import SemanticStore,SemanticDUPGTModel
        store = SemanticStore(semantic_dir,manifest["benchmark_sha256"])
        if store.manifest["n_items"] != expected_spec["n_items"] or store.manifest["n_source"] != len(data["source_item_ids"]):
            raise ValueError("Semantic item mapping differs from factor checkpoint")
        model = SemanticDUPGTModel(**expected_spec,semantic_store=store)
    else:
        model = DUPGTModel(**expected_spec)
    state = checkpoint["model_state"]
    for name in ("src_model", "tgt_model"):
        module = getattr(model, name)
        expected = module.state_dict()
        values = {key.removeprefix(name+"."): value for key, value in state.items() if key.startswith(name+".")}
        if values.keys() != expected.keys() or any(values[key].shape != expected[key].shape for key in expected):
            raise ValueError(f"Missing or mismatched {name} embedding keys/shapes")
        module.load_state_dict(values, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.dupgt.parameters():
        parameter.requires_grad_(True)
    return model


def empty_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_history(path: Path, rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 3. FIT THE BRIDGE USING REAL RATINGS AND OPTIONAL PREFERENCE PAIRS

def transfer_train(config: dict, factor_checkpoint: str, output_dir: str, *,
                   overlap_fraction: float = 0.01, overlap_seed: int = 2020,
                   transfer_seed: int = 2020, teacher_kind: str = "none", teacher_dir: str | None = None,
                   pseudo_weight: float = 0.0, pseudo_batch_size: int = 64, semantic_dir=None) -> dict:
    from .teacher import load_training_pairs, seal_files
    if config["dataset_release"] not in {"2014", "2023"} or config["max_shared_users"] != 0:
        raise ValueError("Transfer pilot requires 2014/full or 2023/full, with the saved benchmark unchanged")
    if teacher_kind not in {"none", "text", "llm", "shuffled", "retrieval_llm", "retrieval_text", "retrieval_shuffled",
                            "retrieval_text_only", "retrieval_text_only_shuffled"}:
        raise ValueError("Unknown teacher kind")
    if not np.isfinite(pseudo_weight) or pseudo_weight < 0:
        raise ValueError("pseudo_weight must be finite and nonnegative")
    if (teacher_kind == "none" and pseudo_weight != 0) or (teacher_kind != "none" and not teacher_dir):
        raise ValueError("A teacher needs its directory; teacher none requires zero pseudo weight")
    output, factor_path = resolve(output_dir), resolve(factor_checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    started = time.monotonic()
    data, manifest = get_benchmark(config)
    users, rows = select_overlap_rows(data, overlap_fraction, overlap_seed)
    model = load_factor_initialized_model(factor_path, data, manifest, config, transfer_seed,semantic_dir)
    factors = parameter_hash(model, ("src_model.", "tgt_model."))
    initial_bridge = parameter_hash(model, ("dupgt.bridging_high.", "dupgt.bridging_low.", "dupgt.gating_mlp."))
    semantic_cache = model.semantic_store.reference if semantic_dir else None
    initial_semantic = parameter_hash(model, ("dupgt.semantic.",)) if semantic_dir else None
    pairs, teacher = None, None
    if teacher_kind != "none":
        if teacher_kind.startswith("retrieval_"):
            if not semantic_dir:
                raise ValueError("Retrieval teachers require --semantic-dir")
            from .retrieval import load_retrieval_pairs,load_text_only_pairs
            loader = load_text_only_pairs if teacher_kind.startswith("retrieval_text_only") else load_retrieval_pairs
            pairs,teacher = loader(teacher_dir,teacher_kind,data,manifest,semantic_cache)
        else:
            pairs, teacher = load_training_pairs(resolve(teacher_dir), teacher_kind, data, manifest, factors)
    empty_destination(output)
    write_json(output/"config.json", config)
    np.save(output/"real_training_users.npy", users, allow_pickle=False)
    experiment = {"schema": 1, "method": "limited_bridge_supervision", "overlap_fraction": overlap_fraction,
                  "model_variant":"semantics" if semantic_dir else "dupgt",
                  "semantic_cache":semantic_cache,"initial_semantic_sha256":initial_semantic,
                  "source_files": {name: file_sha256(Path(__file__).with_name(name)) for name in
                                   ("experiments.py", "train.py", "teacher.py", "metadata.py", "settings.py", "model.py", "upstream.py", "semantics.py", "retrieval.py", "runtime.py")},
                  "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
                  "overlap_seed": overlap_seed, "transfer_seed": transfer_seed, "teacher_kind": teacher_kind,
                  "teacher_dir": str(resolve(teacher_dir)) if teacher_dir else None,
                  "teacher": teacher, "pseudo_weight": pseudo_weight, "pseudo_batch_size": pseudo_batch_size,
                  "pseudo_pairs": len(pairs) if pairs is not None else 0,
                  "real_training_users": len(users), "real_training_rows": len(rows),
                  "optimizer_steps_per_epoch": math.ceil(len(rows)/config["meta_batch_size"]),
                  "factor_checkpoint": str(factor_path), "factor_checkpoint_sha256": file_sha256(factor_path),
                  "factor_sha256": factors, "initial_bridge_sha256": initial_bridge,
                  "real_rows_sha256": array_hash(rows), "users_sha256": array_hash(users),
                  "dataset_sha256": manifest["benchmark_sha256"],
                  "validation_rows_sha256": array_hash(data["validation"])}
    write_json(output/"experiment.json", experiment)
    device = choose_device(config["device"])
    model = model.to(device)
    log = []
    print(f"Bridge-only: {len(users):,} users, {len(rows):,} real ratings, teacher={teacher_kind}, device={device}", flush=True)
    best = fit_transfer(model, data, config, device, "dupgt", log, output, model_spec(data, config),
                        manifest["benchmark_sha256"], real_rows=rows, pseudo_pairs=pairs,
                        pseudo_weight=pseudo_weight, transfer_seed=transfer_seed, pseudo_batch_size=pseudo_batch_size)
    final_factors = parameter_hash(model, ("src_model.", "tgt_model."))
    if factors != final_factors:
        raise RuntimeError("Domain factors changed")
    if initial_bridge == parameter_hash(model, ("dupgt.bridging_high.", "dupgt.bridging_low.", "dupgt.gating_mlp.")):
        raise RuntimeError("Transfer parameters did not change")
    write_history(output/"training_history.csv", log)
    prediction, _ = predict(model, data["validation"], torch.as_tensor(data["histories"]), "test_meta", device,
                            config["evaluation_batch_size"])
    reference = mean_predictions(data["target_train"], data["validation"], model_spec(data, config)["n_items"])
    validation = {"status": "completed", "selected_checkpoint": best, "raw": metrics(data["validation"][:, 2], prediction),
                  "clipped_to_1_5": metrics(data["validation"][:, 2], prediction.clip(1, 5)),
                  "movie_mean": metrics(data["validation"][:, 2], reference),
                  "factor_sha256_after": final_factors, "factors_unchanged": True,
                  "pseudo_comparisons_processed": sum(row["pseudo_comparisons"] for row in log),
                  "pseudo_exposures_per_pair": sum(row["pseudo_comparisons"] for row in log)/max(1, experiment["pseudo_pairs"]),
                  "elapsed_seconds": time.monotonic()-started, "test_evaluated": False}
    write_json(output/"validation_metrics.json", validation)
    artifacts = {"schema": 1, "dataset_sha256": manifest["benchmark_sha256"], "factor_sha256": factors,
                 "files": seal_files(output, ["config.json", "experiment.json", "real_training_users.npy",
                                                "training_history.csv", "validation_metrics.json"]),
                 "checkpoint_sha256": file_sha256(output/"checkpoints/dupgt_best.pt")}
    write_json(output/"artifact_manifest.json", artifacts)
    print(f"Validation MAE={validation['raw']['mae']:.6f}, RMSE={validation['raw']['rmse']:.6f}; test not evaluated", flush=True)
    return validation


# 4. RUN THE PREDECLARED VALIDATION GRID FROM ONE CACHED TEACHER

def run_transfer_grid(config: dict, factor_checkpoint: str, teacher_dir: str, output_dir: str,
                      overlap_seed: int = 2020, transfer_seed: int = 2020) -> dict:
    from .teacher import build_controls, identity
    output = resolve(output_dir)
    if output.exists() and any(output.iterdir()) and not (output/"grid_summary.json").exists():
        raise FileExistsError("Existing directory is not a resumable transfer grid")
    output.mkdir(parents=True, exist_ok=True)
    controls = build_controls(resolve(teacher_dir), TRANSFER["teacher_seed"])
    if controls["status"] != "ready" or not controls["personalization_identifiable"]:
        write_json(output/"grid_summary.json", {"status": "blocked_teacher_support", "controls": controls})
        raise ValueError("Need sufficient matched pairs and varied per-user labels before bridge training")
    variants = [("none", 0.0)] + [(kind, weight) for kind in ("text", "llm", "shuffled")
                                 for weight in TRANSFER["pseudo_weights"]]
    summary = {"status": "running", "dataset_release": config["dataset_release"],
        "overlap_fractions": TRANSFER["overlap_fractions"], "variants": variants,
        "overlap_seed": overlap_seed, "transfer_seed": transfer_seed,
        "teacher_dir": str(resolve(teacher_dir)), "test_evaluated": False, "groups": []}
    if (output/"grid_summary.json").exists():
        previous = json.loads((output/"grid_summary.json").read_text())
        keys = ("dataset_release", "overlap_fractions", "variants", "overlap_seed", "transfer_seed", "teacher_dir", "test_evaluated")
        if any(identity(previous.get(key)) != identity(summary[key]) for key in keys):
            raise ValueError("Saved grid settings differ; choose a new output directory")
        for saved_config in output.glob("overlap_*/*/config.json"):
            if json.loads(saved_config.read_text()) != config:
                raise ValueError("Saved grid training configuration differs")
            if (saved_config.parent/"test_metrics.json").exists():
                raise ValueError("Cannot extend a validation grid after its test evaluation")
    factor_digest = file_sha256(resolve(factor_checkpoint))
    controls_digest = file_sha256(resolve(teacher_dir)/"controls.json")
    summary["reused_runs"] = []
    write_json(output/"grid_summary.json", summary)
    for fraction in TRANSFER["overlap_fractions"]:
        group = output/f"overlap_{fraction:.2f}"
        group.mkdir(exist_ok=True)
        runs, by_kind = [], {}
        for kind, weight in variants:
            run = group/f"{kind}_w{weight:g}"
            if (run/"artifact_manifest.json").exists():
                saved_config, experiment, _, _ = validated_run(run)
                expected = {"overlap_fraction": fraction, "overlap_seed": overlap_seed,
                    "transfer_seed": transfer_seed, "teacher_kind": kind, "pseudo_weight": weight,
                    "pseudo_batch_size": TRANSFER["pseudo_batch_size"],
                    "factor_checkpoint_sha256": factor_digest,
                    "teacher_dir": str(resolve(teacher_dir)) if kind != "none" else None}
                if saved_config != config or any(experiment.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Saved run inputs differ: {run}")
                if kind != "none" and experiment["teacher"]["controls_sha256"] != controls_digest:
                    raise ValueError(f"Saved run teacher controls differ: {run}")
                summary["reused_runs"].append(str(run))
                print(f"Reusing verified run: {run.name}", flush=True)
            else:
                transfer_train(config, factor_checkpoint, str(run), overlap_fraction=fraction,
                    overlap_seed=overlap_seed, transfer_seed=transfer_seed, teacher_kind=kind,
                    teacher_dir=teacher_dir if kind != "none" else None,
                    pseudo_weight=weight, pseudo_batch_size=TRANSFER["pseudo_batch_size"])
            runs.append(str(run))
            by_kind.setdefault(kind, []).append(str(run))
        baseline = by_kind["none"][0]
        comparisons = {}
        # Compare teacher methods at an equal weight; tune weights from their saved metrics.
        for weight in TRANSFER["pseudo_weights"]:
            matched = [baseline, *[str(group/f"{kind}_w{weight:g}") for kind in ("text", "llm", "shuffled")]]
            destination = group/f"validation_w{weight:g}"
            if (destination/"comparison.json").exists():
                saved = json.loads((destination/"comparison.json").read_text())
                inputs = {str(resolve(path)): file_sha256(resolve(path)/"artifact_manifest.json") for path in matched}
                content = {key: value for key, value in saved.items() if key != "comparison_sha256"}
                if saved["inputs"] != inputs or identity(content) != saved["comparison_sha256"]:
                    raise ValueError("Saved comparison changed or belongs to other runs")
            else:
                compare_experiments(matched, str(destination))
            comparisons[str(weight)] = str(destination)
        recommendations = {}
        for kind in ("text", "llm", "shuffled"):
            # The no-teacher run is the shared zero-weight candidate.
            candidates = [baseline, *by_kind[kind]]
            chosen = min(candidates, key=lambda path: json.loads((resolve(path)/"validation_metrics.json").read_text())["raw"]["rmse"])
            recommendations[kind] = {"run": chosen,
                "pseudo_weight": json.loads((resolve(chosen)/"experiment.json").read_text())["pseudo_weight"]}
        summary["groups"].append({"overlap_fraction": fraction, "runs": runs,
            "validation_comparisons": comparisons, "recommendations": recommendations})
        write_json(output/"grid_summary.json", summary)
    summary["status"] = "validation_complete_final_selection_pending"
    write_json(output/"grid_summary.json", summary)
    return summary


# 5. LOCK CHOSEN RUNS BEFORE EXPLICIT TEST EVALUATION

def validated_run(directory: Path) -> tuple[dict, dict, dict, dict]:
    from .teacher import check_files
    artifacts = json.loads((directory/"artifact_manifest.json").read_text())
    check_files(directory, artifacts["files"])
    if file_sha256(directory/"checkpoints/dupgt_best.pt") != artifacts["checkpoint_sha256"]:
        raise ValueError("Experiment checkpoint checksum changed")
    config = json.loads((directory/"config.json").read_text())
    experiment = json.loads((directory/"experiment.json").read_text())
    validation = json.loads((directory/"validation_metrics.json").read_text())
    if validation["status"] != "completed" or not validation["factors_unchanged"]:
        raise ValueError("Incomplete experiment or changed factors")
    return config, experiment, validation, artifacts


def lock_selection(run_paths: list[str], record_path: str, reason: str) -> dict:
    if not reason.strip() or not run_paths:
        raise ValueError("List chosen runs and a validation-based selection reason")
    destination = resolve(record_path)
    if destination.exists():
        raise FileExistsError("Selection records are immutable; choose a new path")
    entries = []
    for run_path in run_paths:
        directory = resolve(run_path)
        if (directory/"test_metrics.json").exists():
            raise ValueError("Cannot create a new selection after this run's test evaluation")
        _, experiment, validation, _ = validated_run(directory)
        entries.append({"run_dir": str(directory), "artifact_manifest_sha256": file_sha256(directory/"artifact_manifest.json"),
                        "dataset_sha256": experiment["dataset_sha256"], "validation_rmse": validation["raw"]["rmse"]})
    if len({entry["run_dir"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate selected run")
    result = {"schema": 1, "locked_at_utc": datetime.now(timezone.utc).isoformat(),
              "reason": reason, "metric": "validation_rmse", "runs": entries}
    write_json(destination, result)
    return result


def transfer_evaluate(run_dir: str, selection_record: str) -> dict:
    directory, record_path = resolve(run_dir), resolve(selection_record)
    config, experiment, _, artifacts = validated_run(directory)
    record = json.loads(record_path.read_text())
    selected = next((entry for entry in record["runs"] if entry["run_dir"] == str(directory)), None)
    if not selected or selected["artifact_manifest_sha256"] != file_sha256(directory/"artifact_manifest.json"):
        raise ValueError("Run is absent from the locked selection or has changed")
    if (directory/"test_metrics.json").exists():
        saved = json.loads((directory/"test_metrics.json").read_text())
        if saved["selection_sha256"] != file_sha256(record_path):
            raise ValueError("This run was already evaluated under a different selection")
        if saved["predictions_sha256"] != file_sha256(directory/"test_predictions.csv.gz"):
            raise ValueError("Saved final predictions changed")
        return saved
    data, manifest = get_benchmark(config)
    if manifest["benchmark_sha256"] != experiment["dataset_sha256"]:
        raise ValueError("Evaluation benchmark differs from training")
    checkpoint = torch.load(directory/"checkpoints/dupgt_best.pt", map_location="cpu", weights_only=True)
    if checkpoint["dataset_sha256"] != manifest["benchmark_sha256"] or checkpoint["model_spec"] != model_spec(data, config):
        raise ValueError("Evaluation checkpoint/data spec mismatch")
    set_seed(experiment["transfer_seed"], config["torch_threads"])
    device = choose_device(config["device"])
    from .runtime import load_saved_model
    model = load_saved_model(checkpoint,device)
    if parameter_hash(model, ("src_model.", "tgt_model.")) != artifacts["factor_sha256"]:
        raise ValueError("Final checkpoint factors differ from training inputs")
    test = data["test"]
    values, gates = predict(model, test, torch.as_tensor(data["histories"]), "test_meta", device, config["evaluation_batch_size"])
    reference = mean_predictions(data["target_train"], test, checkpoint["model_spec"]["n_items"])
    save_predictions(directory/"test_predictions.csv.gz", test, data, {"movie_mean": reference, "dupgt": values}, gates)
    by_user = user_metrics(test, values)
    ref_users = user_metrics(test, reference)
    with (directory/"test_user_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "dupgt_mae", "movie_mean_mae", "dupgt_minus_movie_mean_mae"])
        for user, error, baseline in zip(by_user["ids"], by_user["mae"], ref_users["mae"]):
            writer.writerow([int(user), float(error), float(baseline), float(error-baseline)])
    result = {"raw": metrics(test[:, 2], values), "clipped_to_1_5": metrics(test[:, 2], values.clip(1, 5)),
              "movie_mean": metrics(test[:, 2], reference), "test_rows_sha256": array_hash(test),
              "artifact_manifest_sha256": file_sha256(directory/"artifact_manifest.json"),
              "dataset_sha256": manifest["benchmark_sha256"], "selection_sha256": file_sha256(record_path),
              "predictions_sha256": file_sha256(directory/"test_predictions.csv.gz")}
    write_json(directory/"test_metrics.json", result)
    print(json.dumps(result, indent=2))
    return result


# 6. COMPARE MATCHED VARIANTS, OR AN EXPLICIT OVERLAP CURVE

def compare_experiments(run_paths: list[str], output_dir: str, *, final: bool = False,
                        across_overlap: bool = False) -> dict:
    from .teacher import identity
    loaded = [(resolve(path), *validated_run(resolve(path))) for path in run_paths]
    if len(loaded) < 2:
        raise ValueError("Comparison requires at least two runs")
    reference_config, reference_exp = loaded[0][1], loaded[0][2]
    budget_keys = ("meta_epochs", "meta_batch_size", "meta_learning_rate", "weight_decay",
                   "evaluation_batch_size", "device", "torch_threads")
    experiment_keys = ["dataset_sha256", "factor_sha256", "initial_bridge_sha256", "transfer_seed", "overlap_seed", "validation_rows_sha256"]
    if not across_overlap:
        experiment_keys += ["real_rows_sha256", "users_sha256", "optimizer_steps_per_epoch"]
    non_baselines, rows, test_hashes, selection_hashes = [], [], set(), set()
    for directory, config, experiment, validation, _ in loaded:
        if any(config[k] != reference_config[k] for k in budget_keys) or any(experiment[k] != reference_exp[k] for k in experiment_keys):
            raise ValueError("Unmatched benchmark, factors, initialization, seeds, real supervision or training budget")
        if across_overlap and experiment["teacher_kind"] != "none":
            raise ValueError("Across-overlap curves currently compare no-teacher baselines only")
        if experiment["teacher_kind"] != "none":
            non_baselines.append(experiment)
        scores = validation
        if final:
            scores = json.loads((directory/"test_metrics.json").read_text())
            if (scores["dataset_sha256"] != experiment["dataset_sha256"] or
                scores["artifact_manifest_sha256"] != file_sha256(directory/"artifact_manifest.json")):
                raise ValueError("Final scores belong to a different experiment")
            if scores["predictions_sha256"] != file_sha256(directory/"test_predictions.csv.gz"):
                raise ValueError("Final predictions changed")
            test_hashes.add(scores["test_rows_sha256"])
            selection_hashes.add(scores["selection_sha256"])
        teacher = experiment["teacher"]
        rows.append({"run": str(directory), "teacher": experiment["teacher_kind"],
                     "model_variant":experiment.get("model_variant","dupgt"),
                     "overlap_fraction": experiment["overlap_fraction"], "real_users": experiment["real_training_users"],
                     "real_ratings": experiment["real_training_rows"], "pseudo_weight": experiment["pseudo_weight"],
                     "mae": scores["raw"]["mae"], "rmse": scores["raw"]["rmse"],
                     "movie_mean_mae": scores["movie_mean"]["mae"], "movie_mean_rmse": scores["movie_mean"]["rmse"],
                     "requested_pairs": teacher["coverage"]["requested_pairs"] if teacher else 0,
                     "matched_pairs": teacher["coverage"]["matched_pairs"] if teacher else 0,
                     "changed_label_fraction": teacher["coverage"]["changed_label_fraction"] if teacher else None})
    if non_baselines:
        reference = non_baselines[0]
        for experiment in non_baselines[1:]:
            if (experiment["teacher"]["support_sha256"] != reference["teacher"]["support_sha256"] or
                experiment["teacher"]["controls_sha256"] != reference["teacher"]["controls_sha256"] or
                experiment["pseudo_batch_size"] != reference["pseudo_batch_size"] or
                experiment["pseudo_weight"] != reference["pseudo_weight"]):
                raise ValueError("Teacher comparison requires matched support, pseudo weight and exposure budget")
    semantic_runs = [e for _,_,e,_,_ in loaded if e.get("semantic_cache")]
    if semantic_runs and any(e["semantic_cache"] != semantic_runs[0]["semantic_cache"] or
                            e["initial_semantic_sha256"] != semantic_runs[0]["initial_semantic_sha256"]
                            for e in semantic_runs):
        raise ValueError("Semantic comparisons require identical caches and semantic initialization")
    if semantic_runs and any(e["teacher_kind"] != "none" and not e.get("semantic_cache") for _,_,e,_,_ in loaded):
        raise ValueError("Compare the new semantic experiment with matching semantic teacher controls")
    if final and (len(test_hashes) != 1 or len(selection_hashes) != 1):
        raise ValueError("Final comparisons require identical test pairs and the same locked selection")
    output = resolve(output_dir)
    empty_destination(output)
    with (output/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"axis": "overlap_fraction" if across_overlap else "teacher", "split": "test" if final else "validation",
              "rows": rows, "inputs": {str(d): file_sha256(d/"artifact_manifest.json") for d, *_ in loaded},
              "dataset_sha256": reference_exp["dataset_sha256"], "factor_sha256": reference_exp["factor_sha256"]}
    result["comparison_sha256"] = identity(result)
    write_json(output/"comparison.json", result)
    # Keep a small chart beside the numeric table for presentation.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"{row['overlap_fraction']:.0%}" if across_overlap else
              f"{row['model_variant']} + {row['teacher']}\nλ={row['pseudo_weight']:g}" for row in rows]
    for axis, metric in zip(axes, ("mae", "rmse")):
        axis.bar(np.arange(len(rows)), [row[metric] for row in rows], color="#356c9b")
        axis.axhline(rows[0][f"movie_mean_{metric}"], linestyle="--", color="#a35836", label="Movie mean")
        axis.set_xticks(np.arange(len(rows)), labels)
        axis.set(title=f"{result['split'].title()} {metric.upper()}", ylabel="Rating error")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output/"comparison.png", dpi=160)
    plt.close(fig)
    return result


# 7. AUTOMATIC FULL SEMANTIC STUDY; EACH HEAVY STAGE GETS ITS OWN PROCESS

def run_semantic_study(config,semantic_dir,teacher_dir,output_dir,*,factor_checkpoint,
                       pseudo_weight=.01,teacher_users=200,encoding_device="mps",
                       model="dupgt-qwen35-9b",runtime_file=None):
    import fcntl
    import shutil
    import subprocess
    import sys
    from .common import ROOT
    from .teacher import identity

    if config["max_shared_users"]!=0 or not 0<pseudo_weight<1 or teacher_users<2:
        raise ValueError("Use the full benchmark, at least two teacher users and an auxiliary weight in (0,1)")
    output=resolve(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/"batch.lock").open("w")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This semantic batch is already running")
    python=[sys.executable,"-m","dupgt_cdr"]
    lms=shutil.which("lms")
    if not lms or not runtime_file:
        lock.close()
        raise ValueError("The automatic batch needs LM Studio CLI and a verified runtime file")
    runtime=json.loads(resolve(runtime_file).read_text())
    if runtime["model_identifier"]!=model:
        lock.close()
        raise ValueError("Batch model differs from its verified runtime")
    source_hashes={p.name:file_sha256(p) for p in Path(__file__).parent.glob("*.py")}
    plan={"schema":1,"dataset":config["dataset_release"],"overlap_fraction":1.,
          "epochs":config["meta_epochs"],"seed":2020,"pseudo_weight":pseudo_weight,
          "teacher_users":teacher_users,"top_k":10,"encoding_device":encoding_device,
          "training_device":config["device"],"semantic_dir":str(resolve(semantic_dir)),
          "teacher_dir":str(resolve(teacher_dir)),"factor_checkpoint":str(resolve(factor_checkpoint)),
          "factor_checkpoint_sha256":file_sha256(resolve(factor_checkpoint)),
          "runtime_file_sha256":file_sha256(resolve(runtime_file)),"source_sha256":source_hashes,
          "variants":["semantics","semantics_llm","dupgt","semantics_text","semantics_shuffled"],
          "final_evaluation":"follow-up on the previously examined test split; validation selects checkpoints"}
    plan_path=output/"batch_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        lock.close()
        raise ValueError("Batch contract changed; choose a new output directory")
    write_json(plan_path,plan)
    status_path=output/"batch_status.json"
    status=json.loads(status_path.read_text()) if status_path.exists() else {"completed":{},"status":"running"}
    status.update(status="running",pid=__import__('os').getpid())
    write_json(status_path,status)

    def step(name,arguments,artifact,check=None):
        if any(file_sha256(Path(__file__).with_name(k))!=v for k,v in source_hashes.items()):
            raise ValueError("Source code changed during the batch")
        artifact=resolve(artifact)
        if name in status["completed"]:
            if (file_sha256(artifact)!=status["completed"][name]["artifact_sha256"] or
                status["completed"][name]["command"]!=arguments):
                raise ValueError(f"Completed batch artifact changed: {name}")
            if check:
                check()
            print(f"Reusing completed stage: {name}",flush=True)
            return
        status.update(stage=name,stage_started_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path,status)
        print(f"Starting stage: {name}",flush=True)
        subprocess.run(arguments,cwd=ROOT,check=True)
        if check:
            check()
        status["completed"][name]={"command":arguments,"artifact":str(artifact),"artifact_sha256":file_sha256(artifact),
                                    "finished_utc":datetime.now(timezone.utc).isoformat()}
        write_json(status_path,status)

    def generated():
        summary=json.loads((resolve(teacher_dir)/"generation_summary.json").read_text())
        if not summary["complete"]:
            raise ValueError("Teacher generation remains incomplete; resume cached responses before retrieval")

    def encoded():
        if json.loads((resolve(semantic_dir)/"manifest.json").read_text())["status"]!="completed":
            raise ValueError("Semantic encoding is incomplete")

    try:
        common=["--dataset",config["dataset_release"],"--size","full"]
        # Free Qwen before the embedding job; reload only while generating interests.
        subprocess.run([lms,"unload",model],cwd=ROOT,check=False)
        step("embeddings",python+["semantic-prepare",*common,"--device",encoding_device,
             "--output-dir",str(resolve(semantic_dir))],resolve(semantic_dir)/"manifest.json",encoded)
        step("prepare_teacher",python+["profile-prepare",*common,"--semantic-dir",str(resolve(semantic_dir)),
             "--teacher-users",str(teacher_users),"--output-dir",str(resolve(teacher_dir))],
             resolve(teacher_dir)/"manifest.json")
        if "generate_teacher" not in status["completed"]:
            subprocess.run([lms,"server","start","--port","1234","--bind","127.0.0.1"],cwd=ROOT,check=True)
            subprocess.run([lms,"load",runtime["model_key"],"--context-length","8192","--parallel","1",
                            "--gpu","max","--identifier",model,"--yes"],cwd=ROOT,check=True)
        try:
            step("generate_teacher",python+["profile-generate","--teacher-dir",str(resolve(teacher_dir)),
                 "--llm-model",model,"--runtime-file",str(resolve(runtime_file)),"--max-requests","220",
                 "--max-total-tokens","2000000"],resolve(teacher_dir)/"generation_summary.json",generated)
        finally:
            subprocess.run([lms,"unload",model],cwd=ROOT,check=False)
        step("retrieve",python+["profile-retrieve","--teacher-dir",str(resolve(teacher_dir)),
             "--device",encoding_device,"--top-k","10"],resolve(teacher_dir)/"controls.json")
        runs=[]
        for name,kind in [("semantics","none"),("semantics_llm","retrieval_llm"),("dupgt","none"),
                          ("semantics_text","retrieval_text"),("semantics_shuffled","retrieval_shuffled")]:
            path=output/name
            arguments=python+["transfer-train",*common,"--device",config["device"],"--overlap-fraction","1.0",
                "--epochs",str(config["meta_epochs"]),"--transfer-seed","2020","--overlap-seed","2020",
                "--factor-checkpoint",str(resolve(factor_checkpoint)),"--output-dir",str(path)]
            if name!="dupgt":
                arguments += ["--semantic-dir",str(resolve(semantic_dir))]
            if kind!="none":
                arguments += ["--teacher-dir",str(resolve(teacher_dir)),"--teacher-kind",kind,
                              "--pseudo-weight",str(pseudo_weight)]
            step(name,arguments,path/"artifact_manifest.json",lambda p=path:validated_run(p))
            runs.append(str(path))
        step("validation_comparison",python+["transfer-compare","--runs",*runs,
             "--output-dir",str(output/"validation_comparison")],output/"validation_comparison/comparison.json")
        selection=output/"final_selection.json"
        step("lock",python+["transfer-lock","--runs",*runs,"--selection-record",str(selection),"--reason",
             "Predeclared full-supervision variants and controls; fixed auxiliary weight; best checkpoints chosen by validation RMSE. Test scores are follow-up evidence."],selection)
        for run in runs:
            step("test_"+Path(run).name,python+["transfer-evaluate","--run-dir",run,
                 "--selection-record",str(selection)],Path(run)/"test_metrics.json")
        step("final_comparison",python+["transfer-compare","--runs",*runs,"--final",
             "--output-dir",str(output/"final_comparison")],output/"final_comparison/comparison.json")
        status.update(status="completed",stage="completed")
        write_json(status_path,status)
        return status
    except Exception as exc:
        status.update(status="failed",error=str(exc))
        write_json(status_path,status)
        raise
    finally:
        lock.close()
