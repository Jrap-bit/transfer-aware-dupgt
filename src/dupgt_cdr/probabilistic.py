"""Probabilistic bridge: DUPGT + semantic gate that also predicts how wrong it is likely to be.

This is a new model variant ("probabilistic"); the original DUPGT and semantic models, their
training and their saved runs are untouched. For each (user, movie) the model produces

    mu      the bridge's rating, computed exactly as in the semantic DUPGT model
    sigma2  the bridge's predicted error variance (a small variance head)
    m       the movie's training-set average (target-only prior)
    s0      the prior's error variance (one learned number)

and predicts with inverse-variance weighting, the MSE-optimal mix of two noisy estimates:

    w = s0 / (s0 + sigma2),    r_hat = w * mu + (1 - w) * m

Training uses three losses, each updating its own parameters:

    1. MSE(mu, r)                         -> bridge and gates (identical to DUPGT)
    2. Gaussian NLL of r given (mu, sigma2), with mu held fixed
       = 0.5 * ((r - mu)^2 / sigma2 + log sigma2)   -> variance head only
    3. MSE(r_hat, r), with mu and sigma2 held fixed -> s0 only

plus a gate prior, lambda * mean(sqrt(1 + z^2) - 1) with z = logit(alpha_high): a smooth Laplace-type
prior on the gate's log-odds (MAP estimation). It stops the high/low gate collapsing onto one branch,
as observed at 1% overlap. Its pull never vanishes once the gate saturates (unlike an entropy bonus),
and its gradient is bounded by lambda (unlike a Gaussian prior, whose gradient grows with z and
diverged in a trial because DUPGT's gate log-odds drift into the hundreds). Checkpoints are chosen by validation
RMSE of r_hat, as for every other run.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import file_sha256, resolve, write_json
from .evaluate import mean_predictions, metrics
from .semantics import FEATURE_NAMES, SemanticDUPGTModel

LOG_VARIANCE_RANGE = (-5.0, 3.0)


class VarianceHead(nn.Module):
    """Predicts log sigma2 from the user, the movie, the gate and the semantic evidence."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        inputs = 2 * embedding_dim + 2 + len(FEATURE_NAMES)
        self.network = nn.Sequential(nn.Linear(inputs, 32), nn.ELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)   # start at sigma2 = 1 for everyone

    def forward(self, user, movie, alpha, stats, valid):
        features = torch.cat((user, movie, alpha[:, None], stats, valid.to(user.dtype)[:, None]), dim=1)
        return self.network(features).squeeze(1).clamp(*LOG_VARIANCE_RANGE)


class ProbabilisticDUPGTModel(SemanticDUPGTModel):
    def __init__(self, n_users, n_items, embedding_dim, semantic_store, item_prior=None, prior_variance=1.0):
        super().__init__(n_users, n_items, embedding_dim, semantic_store)
        self.model_variant = "probabilistic"
        self.dupgt.uncertainty = VarianceHead(embedding_dim)
        self.dupgt.log_prior_variance = nn.Parameter(torch.tensor(float(math.log(prior_variance))))
        prior = torch.zeros(n_items + 1) if item_prior is None else torch.as_tensor(item_prior, dtype=torch.float32)
        self.register_buffer("item_prior", prior)

    def gate_logit(self, high, low, user, movie):
        """The gating MLP's pre-sigmoid output (same layers as BlockWiseGatingMLP.forward)."""
        gate = self.dupgt.gating_mlp
        blocks = [gate.proj[name](tensor) for name, tensor in
                  zip(["H", "L", "U", "I", "D"], [high, low, user, movie, high - low])]
        return gate.net(gate.norm(torch.cat(blocks, dim=1))).squeeze(1)

    def forward_with_details(self, x):
        """Same computation as the semantic model, but kept in log-odds so the gate prior has an exact gradient."""
        source = self.src_model.uid_embedding(x[:, 0])
        target = self.tgt_model.iid_embedding(x[:, 1])
        sequence = x[:, 2:42]
        features = self.src_model.iid_embedding(sequence)
        high = self.dupgt.bridging_high(features[:, :20], sequence[:, :20])
        low = self.dupgt.bridging_low(features[:, 20:], sequence[:, 20:])
        base_logit = self.gate_logit(high, low, source, target)
        profiles, movie, stats, valid = self.semantic_store.features(x)
        tensors = [torch.as_tensor(a, device=x.device, dtype=target.dtype) for a in (profiles, movie, stats)]
        valid = torch.as_tensor(valid, device=x.device)
        delta = self.dupgt.semantic(*tensors).clamp(-10, 10)
        logit = base_logit + torch.where(valid, delta, torch.zeros_like(delta))
        alpha = torch.sigmoid(logit)
        bridge = (alpha[:, None] * high + (1 - alpha[:, None]) * low).reshape(-1, self.emb_dim, self.emb_dim)
        rating = (torch.bmm(source.unsqueeze(1), bridge).squeeze(1) * target).sum(1)
        return rating, alpha, {"gate_logit": logit, "semantic_features": tensors[2], "has_semantics": valid}

    def forward_parts(self, x):
        mu, alpha, details = self.forward_with_details(x)
        user = self.src_model.uid_embedding(x[:, 0])
        movie = self.tgt_model.iid_embedding(x[:, 1])
        log_sigma2 = self.dupgt.uncertainty(user.detach(), movie.detach(), alpha.detach(),
                                            details["semantic_features"], details["has_semantics"])
        prior = self.item_prior[x[:, 1]]
        return mu, log_sigma2, prior, alpha, details["gate_logit"]

    def combine(self, mu, log_sigma2, prior):
        s0 = self.dupgt.log_prior_variance.exp()
        weight = s0 / (s0 + log_sigma2.exp())
        return weight * mu + (1 - weight) * prior, weight

    def forward_with_gate(self, x):
        mu, log_sigma2, prior, alpha, _ = self.forward_parts(x)
        return self.combine(mu, log_sigma2, prior)[0], alpha


def binary_entropy(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.clamp(1e-6, 1 - 1e-6)
    return -(alpha * alpha.log() + (1 - alpha) * (1 - alpha).log())


def losses(model: ProbabilisticDUPGTModel, features: torch.Tensor, rating: torch.Tensor, gate_prior_weight: float):
    mu, log_sigma2, prior, alpha, logit = model.forward_parts(features)
    bridge = nn.functional.mse_loss(mu, rating)
    nll = 0.5 * ((rating - mu.detach()) ** 2 * torch.exp(-log_sigma2) + log_sigma2).mean()
    combined, _ = model.combine(mu.detach(), log_sigma2.detach(), prior)
    blend = nn.functional.mse_loss(combined, rating)
    gate_prior = (torch.sqrt(1 + logit.square()) - 1).mean()
    total = bridge + nll + blend + gate_prior_weight * gate_prior
    return total, {"bridge_mse": bridge.item(), "nll": nll.item(), "blend_mse": blend.item(),
                   "gate_prior": gate_prior.item(), "mean_abs_gate_logit": logit.abs().mean().item(), "gate_entropy": binary_entropy(alpha).mean().item()}


def save_probabilistic_checkpoint(path, model, spec, config, dataset_hash, epoch):
    from .runtime import cpu_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": cpu_state(model), "model_spec": spec, "config": config,
                "dataset_sha256": dataset_hash, "stage": "dupgt", "epoch": epoch,
                "upstream_commit": "e6c6eedf1161d2aa4f0ebd616bd36176601e5b35",
                "adapter": "padding ID reserved; zero summary for empty history",
                "model_variant": "probabilistic", "semantic_cache": model.semantic_store.reference}, path)


def fit_probabilistic(model, data, config, device, log, output, spec, dataset_hash, *, real_rows,
                      transfer_seed, gate_prior_weight):
    """Same schedule, optimiser, batches and checkpoint rule as `fit_transfer` for DUPGT; new loss."""
    from .predict import predict, query_input
    from .train import learning_rate_schedule, parameter_hash
    factors_before = parameter_hash(model, ("src_model.", "tgt_model."))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.dupgt.parameters():
        parameter.requires_grad_(True)
    tensor = torch.as_tensor(real_rows, dtype=torch.long)
    history = torch.as_tensor(data["histories"], dtype=torch.long)
    epochs, batch_size, lr = config["meta_epochs"], config["meta_batch_size"], config["meta_learning_rate"]
    optimizer = torch.optim.SGD(model.dupgt.parameters(), lr=lr, momentum=0.9, weight_decay=config["weight_decay"])
    steps = math.ceil(len(real_rows) / batch_size)
    scheduler = learning_rate_schedule(optimizer, lr, epochs * steps)
    rng = np.random.default_rng(transfer_seed + 3)   # same batch order as the DUPGT runs
    best_value, best = float("inf"), None
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.eval()
        model.dupgt.train()
        sums, seen = {}, 0
        for idx in np.array_split(rng.permutation(len(real_rows)), steps):
            batch = tensor[idx]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = losses(model, query_input(batch, history, device), batch[:, 2].to(device).float(),
                                 gate_prior_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite probabilistic loss at epoch {epoch}: {parts}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value * len(batch)
            seen += len(batch)
        prediction, _ = predict(model, data["validation"], history, "test_meta", device, config["evaluation_batch_size"])
        validation = metrics(data["validation"][:, 2], prediction)
        row = {"stage": "dupgt", "epoch": epoch, "train_mse": sums["blend_mse"] / seen,
               "validation_mae": validation["mae"], "validation_rmse": validation["rmse"],
               "seconds": time.monotonic() - started, **{k: v / seen for k, v in sums.items()},
               "prior_variance": float(model.dupgt.log_prior_variance.detach().exp()), "optimizer_steps": steps}
        log.append(row)
        if validation["rmse"] < best_value:
            best_value, best = validation["rmse"], {"epoch": epoch, "validation": validation}
            save_probabilistic_checkpoint(output/"checkpoints/dupgt_best.pt", model, spec, config, dataset_hash, epoch)
        print(f"probabilistic {epoch:02d}/{epochs}: bridge MSE {row['bridge_mse']:.5f}, NLL {row['nll']:.4f}, "
              f"gate entropy {row['gate_entropy']:.3f}; validation MAE {validation['mae']:.5f}, "
              f"RMSE {validation['rmse']:.5f} ({row['seconds']:.1f}s)", flush=True)
    checkpoint = torch.load(output/"checkpoints/dupgt_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if parameter_hash(model, ("src_model.", "tgt_model.")) != factors_before:
        raise RuntimeError("Frozen domain factors changed during transfer training")
    return best


def probabilistic_train(config: dict, factor_checkpoint: str, output_dir: str, *, semantic_dir: str,
                        overlap_fraction: float = 0.01, overlap_seed: int = 2020, transfer_seed: int = 2020,
                        gate_prior_weight: float = 0.01) -> dict:
    """Mirror of `transfer_train` for the probabilistic variant; writes the same run files."""
    from .data import get_benchmark
    from .experiments import (array_hash, empty_destination, load_factor_initialized_model, parameter_hash,
                              select_overlap_rows, write_history)
    from .predict import predict
    from .runtime import choose_device, model_spec
    from .teacher import seal_files
    if not semantic_dir:
        raise ValueError("The probabilistic bridge builds on the semantic model; pass --semantic-dir")
    if not np.isfinite(gate_prior_weight) or gate_prior_weight < 0:
        raise ValueError("gate_prior_weight must be finite and nonnegative")
    output, factor_path = resolve(output_dir), resolve(factor_checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    started = time.monotonic()
    data, manifest = get_benchmark(config)
    users, rows = select_overlap_rows(data, overlap_fraction, overlap_seed)
    # Same seed and construction order as the semantic run, so the starting bridge is identical.
    base = load_factor_initialized_model(factor_path, data, manifest, config, transfer_seed, semantic_dir)
    spec = model_spec(data, config)
    prior = mean_predictions(data["target_train"], np.stack([np.zeros(spec["n_items"] + 1, dtype=np.int64),
                                                             np.arange(spec["n_items"] + 1)], axis=1), spec["n_items"])
    residual = data["target_train"][:, 2] - prior[data["target_train"][:, 1]]
    torch.manual_seed(transfer_seed + 7)
    model = ProbabilisticDUPGTModel(**spec, semantic_store=base.semantic_store, item_prior=prior,
                                    prior_variance=float(np.mean(residual ** 2)))
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if unexpected or any(not (k.startswith("dupgt.uncertainty.") or k in {"dupgt.log_prior_variance", "item_prior"})
                         for k in missing):
        raise ValueError("Unexpected parameters when extending the semantic model")
    factors = parameter_hash(model, ("src_model.", "tgt_model."))
    initial_bridge = parameter_hash(model, ("dupgt.bridging_high.", "dupgt.bridging_low.", "dupgt.gating_mlp."))
    empty_destination(output)
    write_json(output/"config.json", config)
    np.save(output/"real_training_users.npy", users, allow_pickle=False)
    experiment = {"schema": 1, "method": "probabilistic_bridge", "overlap_fraction": overlap_fraction,
                  "model_variant": "probabilistic", "semantic_cache": model.semantic_store.reference,
                  "initial_semantic_sha256": parameter_hash(model, ("dupgt.semantic.",)),
                  "gate_prior_weight": gate_prior_weight,
                  "source_files": {name: file_sha256(Path(__file__).with_name(name)) for name in
                                   ("probabilistic.py", "experiments.py", "train.py", "model.py", "semantics.py", "runtime.py")},
                  "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
                  "overlap_seed": overlap_seed, "transfer_seed": transfer_seed, "teacher_kind": "none",
                  "teacher_dir": None, "teacher": None, "pseudo_weight": 0.0, "pseudo_batch_size": 0, "pseudo_pairs": 0,
                  "real_training_users": len(users), "real_training_rows": len(rows),
                  "optimizer_steps_per_epoch": math.ceil(len(rows)/config["meta_batch_size"]),
                  "factor_checkpoint": str(factor_path), "factor_checkpoint_sha256": file_sha256(factor_path),
                  "factor_sha256": factors, "initial_bridge_sha256": initial_bridge,
                  "real_rows_sha256": array_hash(rows), "users_sha256": array_hash(users),
                  "dataset_sha256": manifest["benchmark_sha256"], "validation_rows_sha256": array_hash(data["validation"])}
    write_json(output/"experiment.json", experiment)
    device = choose_device(config["device"])
    model = model.to(device)
    log = []
    print(f"Probabilistic bridge: {len(users):,} users, {len(rows):,} real ratings, "
          f"gate prior weight {gate_prior_weight}, device={device}", flush=True)
    best = fit_probabilistic(model, data, config, device, log, output, spec, manifest["benchmark_sha256"],
                             real_rows=rows, transfer_seed=transfer_seed, gate_prior_weight=gate_prior_weight)
    if parameter_hash(model, ("src_model.", "tgt_model.")) != factors:
        raise RuntimeError("Domain factors changed")
    write_history(output/"training_history.csv", log)
    history = torch.as_tensor(data["histories"])
    prediction, gates = predict(model, data["validation"], history, "test_meta", device, config["evaluation_batch_size"])
    reference = mean_predictions(data["target_train"], data["validation"], spec["n_items"])
    validation = {"status": "completed", "selected_checkpoint": best, "raw": metrics(data["validation"][:, 2], prediction),
                  "clipped_to_1_5": metrics(data["validation"][:, 2], prediction.clip(1, 5)),
                  "movie_mean": metrics(data["validation"][:, 2], reference),
                  "mean_alpha_high": float(gates.mean()), "prior_variance": float(model.dupgt.log_prior_variance.detach().exp()),
                  "factor_sha256_after": factors, "factors_unchanged": True,
                  "pseudo_comparisons_processed": 0, "pseudo_exposures_per_pair": 0.0,
                  "elapsed_seconds": time.monotonic()-started, "test_evaluated": False}
    write_json(output/"validation_metrics.json", validation)
    write_json(output/"artifact_manifest.json", {
        "schema": 1, "dataset_sha256": manifest["benchmark_sha256"], "factor_sha256": factors,
        "files": seal_files(output, ["config.json", "experiment.json", "real_training_users.npy",
                                     "training_history.csv", "validation_metrics.json"]),
        "checkpoint_sha256": file_sha256(output/"checkpoints/dupgt_best.pt")})
    print(f"Validation MAE={validation['raw']['mae']:.6f}, RMSE={validation['raw']['rmse']:.6f}; test not evaluated", flush=True)
    return validation
