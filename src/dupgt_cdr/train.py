"""Train factors and transfer models, then save test results."""
from __future__ import annotations

import csv
import gzip
import hashlib
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import DUPGTModel
from .common import resolve, write_json
from .data import get_benchmark
from .predict import predict, query_input
from .evaluate import mean_predictions, metrics, user_metrics, stratified_results
from .runtime import set_seed, choose_device, model_spec, initialize_factors, cpu_state, save_checkpoint


# Save the raw ratings and model predictions.
def save_predictions(path: Path, rows: np.ndarray, data: dict, predictions: dict, gates: np.ndarray) -> None:
    columns = ["user_id", "movie_asin", "actual_rating", *predictions,
               "alpha_high", "alpha_low", "source_high_count", "source_low_count"]
    n_source = len(data["source_item_ids"])
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, (user, movie, rating) in enumerate(rows):
            high, low = data["polarity_counts"][user]
            writer.writerow([data["user_ids"][user], data["target_item_ids"][movie - n_source - 1],
                             rating, *(values[i] for values in predictions.values()),
                             gates[i], 1 - gates[i], high, low])


# Save training and test score charts.
def save_plots(output: Path, history: list, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for stage in dict.fromkeys(row["stage"] for row in history):
        rows = [row for row in history if row["stage"] == stage]
        axes[0].plot([row["epoch"] for row in rows], [row["train_mse"] for row in rows], label=stage)
        if rows[0].get("validation_rmse") is not None:
            axes[1].plot([row["epoch"] for row in rows], [row["validation_rmse"] for row in rows], label=stage)
    for axis, title in zip(axes, ["Training MSE", "Validation RMSE"]):
        axis.set(title=title, xlabel="Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output/"training_curves.png", dpi=150)
    plt.close(fig)

    names = list(results)
    fig, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    axis.bar(x - 0.2, [results[name]["raw"]["mae"] for name in names], 0.4, label="MAE")
    axis.bar(x + 0.2, [results[name]["raw"]["rmse"] for name in names], 0.4, label="RMSE")
    axis.set_xticks(x, names)
    axis.set_ylabel("Rating error")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output/"test_metrics.png", dpi=150)
    plt.close(fig)


def learning_rate_schedule(optimizer, rate: float, total_steps: int):
    # OneCycleLR fails when its warm-up phase fits in one optimizer step.
    if total_steps <= 5:
        return None  # Keep the learning rate fixed for tiny smoke runs.
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=rate, total_steps=total_steps, pct_start=0.2
    )


# Learn Book or Movie user/item vectors.

def fit_lfm(model: DUPGTModel, rows: np.ndarray, stage: str, config: dict,
            device: torch.device, log: list, output: Path) -> None:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = model.src_model if stage == "source_pretrain" else model.tgt_model
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    epochs, batch_size = config["pretrain_epochs"], config["batch_size"]
    optimizer = torch.optim.Adam(module.parameters(), lr=config["pretrain_learning_rate"], weight_decay=config["weight_decay"])
    steps = math.ceil(len(rows)/batch_size)
    scheduler = learning_rate_schedule(optimizer, config["pretrain_learning_rate"], epochs * steps)
    tensor = torch.as_tensor(rows, dtype=torch.long)
    rng = np.random.default_rng(config["seed"] + (0 if stage == "source_pretrain" else 1))
    upstream_stage = "train_src" if stage == "source_pretrain" else "train_tgt"
    for epoch in range(1, epochs+1):
        started = time.monotonic()
        squared_error, seen = 0.0, 0
        for idx in np.array_split(rng.permutation(len(rows)), steps):
            batch = tensor[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[:, :2], upstream_stage)
            loss = torch.nn.functional.mse_loss(prediction, batch[:, 2].float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {stage}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            squared_error += loss.item()*len(batch)
            seen += len(batch)
        row = {"stage": stage, "epoch": epoch, "train_mse": squared_error/seen, "seconds": time.monotonic()-started}
        log.append(row)
        print(f"{stage:17s} {epoch:02d}/{epochs}: train MSE {row['train_mse']:.5f} ({row['seconds']:.1f}s)", flush=True)



# Train either the simple mapping or DUPGT.

def parameter_hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(prefixes):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pairwise_loss(preferred: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(preferred - other)).mean()

def fit_transfer(model: DUPGTModel, data: dict, config: dict, device: torch.device,
                 kind: str, log: list, output: Path, spec: dict, dataset_hash: str, *,
                 real_rows=None, pseudo_pairs=None, pseudo_weight: float = 0.0,
                 transfer_seed: int | None = None, pseudo_batch_size: int = 64) -> dict:
    if not np.isfinite(pseudo_weight) or pseudo_weight < 0 or pseudo_batch_size < 1:
        raise ValueError("Invalid pseudo weight or batch size")
    use_pairs = pseudo_weight > 0
    if use_pairs and (kind != "dupgt" or pseudo_pairs is None or len(pseudo_pairs) == 0):
        raise ValueError("Positive pseudo weight requires valid DUPGT preference pairs")
    if use_pairs:
        pseudo_pairs = np.asarray(pseudo_pairs)
        if pseudo_pairs.ndim != 2 or pseudo_pairs.shape[1] != 3 or not np.issubdtype(pseudo_pairs.dtype, np.integer):
            raise ValueError("Pseudo pairs must be integer (user, preferred Movie, other Movie) rows")
        target_users = np.unique(np.concatenate([data[key][:, 0] for key in ("target_train", "validation", "test")]))
        allowed_users = np.setdiff1d(np.unique(data["source_train"][:, 0]), target_users)
        warm = np.unique(data["target_train"][:, 1])
        if (not np.isin(pseudo_pairs[:, 0], allowed_users).all() or
            not np.isin(pseudo_pairs[:, 1:], warm).all() or
            np.any(pseudo_pairs[:, 1] == pseudo_pairs[:, 2])):
            raise ValueError("Pseudo pairs require genuine source-only users and distinct warm Movies")
    factors_before = parameter_hash(model, ("src_model.", "tgt_model.", "aug_model."))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    is_meta = kind == "dupgt"
    module = model.dupgt if is_meta else model.mapping
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    real_rows = data["meta_train"] if real_rows is None else real_rows
    rows = real_rows if is_meta else np.unique(real_rows[:, 0])[:, None]
    if not len(rows):
        raise ValueError("No real bridge-training ratings")
    tensor = torch.as_tensor(rows, dtype=torch.long)
    history = torch.as_tensor(data["histories"], dtype=torch.long)
    epochs = config["meta_epochs"] if is_meta else config["mapping_epochs"]
    batch_size = config["meta_batch_size"] if is_meta else config["batch_size"]
    lr = config["meta_learning_rate"] if is_meta else config["mapping_learning_rate"]
    optimizer = (torch.optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=config["weight_decay"])
        if is_meta else torch.optim.Adam(module.parameters(), lr=lr, weight_decay=config["weight_decay"]))
    steps = math.ceil(len(rows)/batch_size)
    scheduler = learning_rate_schedule(optimizer, lr, epochs * steps)
    best_value, best = float("inf"), None
    seed = config["seed"] if transfer_seed is None else transfer_seed
    rng = np.random.default_rng(seed + (3 if is_meta else 2))
    pair_rng = np.random.default_rng(seed + 100)
    pair_tensor = torch.as_tensor(pseudo_pairs, dtype=torch.long) if use_pairs else None
    for epoch in range(1, epochs+1):
        started = time.monotonic()
        model.eval()
        module.train()
        squared_error, seen = 0.0, 0
        pair_error, pair_seen, objective = 0.0, 0, 0.0
        for idx in np.array_split(rng.permutation(len(rows)), steps):
            batch = tensor[idx]
            optimizer.zero_grad(set_to_none=True)
            if is_meta:
                features = query_input(batch, history, device)
                prediction = model(features, "train_meta")
                loss = torch.nn.functional.mse_loss(prediction, batch[:, 2].to(device).float())
            else:
                users = batch[:, 0].to(device)
                source = model.src_model.uid_embedding(users)
                target = model.tgt_model.uid_embedding(users)
                loss = torch.nn.functional.mse_loss(model.mapping(source), target)
            real_loss = loss
            if use_pairs:
                # Separate sampling keeps the real batches identical across teachers.
                pairs = pair_tensor[pair_rng.integers(len(pair_tensor), size=pseudo_batch_size)]
                preferred = query_input(pairs[:, [0, 1]], history, device)
                other = query_input(pairs[:, [0, 2]], history, device)
                synthetic_loss = pairwise_loss(model(preferred, "train_meta"), model(other, "train_meta"))
                loss = real_loss + pseudo_weight * synthetic_loss
                pair_error += synthetic_loss.item() * len(pairs)
                pair_seen += len(pairs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {kind}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            squared_error += real_loss.item()*len(batch)
            objective += loss.item()
            seen += len(batch)
        prediction, _ = predict(model, data["validation"], history, "test_meta" if is_meta else "test_map", device, config["evaluation_batch_size"])
        validation = metrics(data["validation"][:, 2], prediction)
        row = {"stage": kind, "epoch": epoch, "train_mse": squared_error/seen,
            "validation_mae": validation["mae"], "validation_rmse": validation["rmse"], "seconds": time.monotonic()-started}
        if transfer_seed is not None or use_pairs or real_rows is not data["meta_train"]:
            row.update(pair_loss=pair_error/pair_seen if pair_seen else 0.0,
                       total_objective=objective/steps, pseudo_comparisons=pair_seen,
                       optimizer_steps=steps)
        log.append(row)
        if validation["rmse"] < best_value:
            best_value = validation["rmse"]
            best = {"epoch": epoch, "validation": validation}
            save_checkpoint(output/"checkpoints"/f"{kind}_best.pt", model, spec, config, dataset_hash, kind, epoch)
        print(f"{kind:17s} {epoch:02d}/{epochs}: train MSE {row['train_mse']:.5f}; validation MAE {validation['mae']:.5f}, RMSE {validation['rmse']:.5f} ({row['seconds']:.1f}s)", flush=True)
    if parameter_hash(model, ("src_model.", "tgt_model.", "aug_model.")) != factors_before:
        raise RuntimeError("Frozen domain factors changed during transfer training")
    checkpoint = torch.load(output/"checkpoints"/f"{kind}_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if parameter_hash(model, ("src_model.", "tgt_model.", "aug_model.")) != factors_before:
        raise RuntimeError("Frozen domain factors changed during transfer training")
    return best

# Run all stages and save their results.

def train(config: dict, output_override: str | None = None) -> dict:
    started = time.monotonic()
    output = resolve(output_override or config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}. Use --output-dir for a new run.")
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = get_benchmark(config)
    write_json(output/"config.json", config)
    write_json(output/"dataset_manifest.json", manifest)
    set_seed(config["seed"], config["torch_threads"])
    device = choose_device(config["device"])
    print(f"Training on {device}; {len(data['user_ids']):,} users; {len(data['meta_train']):,} bridge-training ratings", flush=True)
    spec = model_spec(data, config)
    model = DUPGTModel(**spec).to(device)
    initialize_factors(model, config.get("factor_initialization_std"))
    log = []

    # First learn Book and Movie factors.

    fit_lfm(model, data["source_train"], "source_pretrain", config, device, log, output)
    fit_lfm(model, data["target_train"], "target_pretrain", config, device, log, output)
    pretrained = cpu_state(model)
    save_checkpoint(output/"checkpoints"/"pretrained_factors.pt", model, spec, config, manifest["benchmark_sha256"], "pretrained", config["pretrain_epochs"])
    selected = {}

    # Mapping is a baseline; DUPGT starts from the same pretrained factors.

    selected["linear_mapping"] = fit_transfer(model, data, config, device, "linear_mapping", log, output, spec, manifest["benchmark_sha256"])
    model.load_state_dict(pretrained)
    selected["dupgt"] = fit_transfer(model, data, config, device, "dupgt", log, output, spec, manifest["benchmark_sha256"])

    # No test predictions have been made before checkpoint selection is complete.

    history = torch.as_tensor(data["histories"], dtype=torch.long)

    # Score Movie ratings hidden from training.
    
    test = data["test"]
    predictions = {
        "global_mean": np.full(len(test), data["target_train"][:, 2].mean()),
        "movie_mean": mean_predictions(data["target_train"], test, spec["n_items"]),
    }
    checkpoint = torch.load(output/"checkpoints"/"linear_mapping_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    predictions["linear_mapping"], _ = predict(model, test, history, "test_map", device, config["evaluation_batch_size"])
    checkpoint = torch.load(output/"checkpoints"/"dupgt_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    predictions["dupgt"], gates = predict(model, test, history, "test_meta", device, config["evaluation_batch_size"])
    results = {name: {"raw": metrics(test[:, 2], values), "clipped_to_1_5": metrics(test[:, 2], values.clip(1, 5))} for name, values in predictions.items()}
    by_user = {name: user_metrics(test, value) for name, value in predictions.items()}
    baseline = by_user["movie_mean"]
    harm = {}
    for name in ["linear_mapping", "dupgt"]:
        difference = by_user[name]["mae"]-baseline["mae"]
        harm[name] = {"reference": "movie_mean", "definition": "Fraction of evaluated users with higher user MAE than the Movie-mean baseline", "fraction_harmed": float((difference>1e-12).mean()), "mean_user_mae_difference": float(difference.mean())}
    with (output/"user_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", *[f"{n}_mae" for n in by_user], "dupgt_minus_movie_mean_mae"])
        for i, user in enumerate(baseline["ids"]):
            writer.writerow([data["user_ids"][user], *[float(v["mae"][i]) for v in by_user.values()], float(by_user["dupgt"]["mae"][i]-baseline["mae"][i])])
    save_predictions(output/"test_predictions.csv.gz", test, data, predictions, gates)
    sample_idx = np.random.default_rng(config["seed"]).choice(len(test), size=min(30, len(test)), replace=False)
    save_predictions(output/"example_predictions.csv", test[sample_idx], data,
                     {name: values[sample_idx] for name, values in predictions.items()}, gates[sample_idx])
    with (output/"training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "epoch", "train_mse", "validation_mae", "validation_rmse", "seconds"])
        writer.writeheader(); writer.writerows(log)
    run_metrics = {"schema": 1, "status": "completed", "label": config["label"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "seed": config["seed"],
        "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device), "torch_threads": torch.get_num_threads(),
            "mps_available": torch.backends.mps.is_available()},
        "dataset_sha256": manifest["benchmark_sha256"], "results": results,
        "factor_initialization": (f"normal with std {config['factor_initialization_std']}" if "factor_initialization_std" in config else "upstream default normal with std 1"),
        "selected_checkpoints": selected, "user_level_comparison": harm,
        "stratified_test_results": stratified_results(test, data, predictions),
        "gate_summary": {"mean_high_weight": float(gates.mean()), "min_high_weight": float(gates.min()), "max_high_weight": float(gates.max())},
        "elapsed_seconds": time.monotonic()-started}
    save_plots(output, log, results)
    write_json(output/"metrics.json", run_metrics)
    print("\nFINAL TEST RESULTS (checkpoint selected on validation RMSE)", flush=True)
    for name, value in results.items():
        print(f"  {name:16s} MAE={value['raw']['mae']:.6f}  RMSE={value['raw']['rmse']:.6f}", flush=True)
    print(f"Artifacts: {output}", flush=True)
    return run_metrics
