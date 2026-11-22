"""Predict ratings and recommend Movie IDs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from .model import DUPGTModel
from .common import resolve, write_json
from .data import get_benchmark
from .runtime import choose_device, set_seed, load_saved_model

# Attach each user's Book history to a query.
def query_input(rows: torch.Tensor, history: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    ids = rows[:, :2]
    if history is not None:
        ids = torch.cat((ids, history[ids[:, 0]]), dim=1)
    return ids.to(device)



# Score rows in batches without updating weights.
@torch.inference_mode()
def predict(model: DUPGTModel, rows: np.ndarray, history: torch.Tensor, stage: str,
            device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    values, gates = [], []
    tensor = torch.as_tensor(rows, dtype=torch.long)
    for batch in tensor.split(batch_size):
        features = query_input(batch, history if stage == "test_meta" else None, device)
        if stage == "test_meta":
            score, alpha = model.forward_with_gate(features)
            gates.append(alpha.cpu().numpy())
        else:
            score = model(features, stage)
        values.append(score.cpu().numpy())
    prediction = np.concatenate(values).astype(np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Model produced a non-finite rating")
    return prediction, np.concatenate(gates) if gates else None



# Rank Movie IDs using a saved checkpoint.
def infer(config: dict, run_dir: str | None = None, user_id: str | None = None,
          top_k: int = 10) -> list[dict]:
    """Rank warm Movie IDs for a known Book user; never read their Movie ratings."""
    directory = resolve(run_dir or config["output_dir"])
    data, manifest = get_benchmark(config)
    checkpoint = torch.load(directory/"checkpoints"/"dupgt_best.pt", map_location="cpu", weights_only=True)
    if checkpoint["dataset_sha256"] != manifest["benchmark_sha256"]:
        raise ValueError("Checkpoint and benchmark do not match")
    set_seed(config["seed"], config["torch_threads"])
    device = choose_device(config["device"])
    model = load_saved_model(checkpoint,device)
    if user_id is None:
        user = int(data["test_users"][0])
    else:
        matches = np.flatnonzero(data["user_ids"] == user_id)
        if not len(matches):
            raise ValueError("User not in this benchmark. A new user requires source-embedding inference, not a random ID lookup.")
        user = int(matches[0])
    if not data["polarity_counts"][user].sum():
        raise ValueError("This user has no Book history; cross-domain personalization is unavailable")
    candidates = np.unique(data["target_train"][:, 1])
    # Exclude known target training interactions for warm users, if any.
    seen = data["target_train"][data["target_train"][:, 0] == user, 1]
    candidates = candidates[~np.isin(candidates, seen)]
    rows = np.column_stack((np.full(len(candidates), user), candidates, np.zeros(len(candidates), dtype=np.int64)))
    prediction, gates = predict(model, rows, torch.as_tensor(data["histories"]), "test_meta", device, config["evaluation_batch_size"])
    order = np.argsort(-prediction, kind="stable")[:top_k]
    n_source = len(data["source_item_ids"])
    result = [{"user_id": str(data["user_ids"][user]), "rank": rank+1,
        "movie_asin": str(data["target_item_ids"][candidates[idx]-n_source-1]),
        "predicted_rating_raw": float(prediction[idx]), "predicted_rating_display": float(prediction[idx].clip(1,5)),
        "alpha_high": float(gates[idx]), "alpha_low": float(1-gates[idx])} for rank, idx in enumerate(order)]
    destination = directory/f"recommendations_{user}.json"
    write_json(destination, {"note": "Ranked by predicted rating. Movie IDs only; metadata titles are not available in the rating archives. This list is a demo, not a ranking-metric evaluation.", "items": result})
    print(json.dumps(result, indent=2), flush=True)
    return result
