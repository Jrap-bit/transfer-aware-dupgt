"""Seeds, device selection and model checkpoints."""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch
from .model import DUPGTModel
from .common import resolve,file_sha256

# Make repeated runs predictable.
def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)



# Select CPU or an available GPU.
def choose_device(request: str) -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            request = "cuda"
        elif torch.backends.mps.is_available():
            request = "mps"
        else:
            request = "cpu"
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; use device=cpu")
    return torch.device(request)



# Size the model from this dataset.
def model_spec(data: dict, config: dict) -> dict:
    return {"n_users": len(data["user_ids"]),
        "n_items": len(data["source_item_ids"]) + len(data["target_item_ids"]),
        "embedding_dim": config["embedding_dim"]}



def initialize_factors(model: DUPGTModel, std: float | None) -> None:
    """Optional standard small-normal MF initialization; no change to architecture."""
    if std is None:
        return
    if std <= 0:
        raise ValueError("factor_initialization_std must be positive")
    with torch.no_grad():
        for module in [model.src_model, model.tgt_model, model.aug_model]:
            torch.nn.init.normal_(module.uid_embedding.weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.iid_embedding.weight, mean=0.0, std=std)
            module.iid_embedding.weight[0].zero_()



def cpu_state(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}



# Save weights and the data identity together.
def save_checkpoint(path: Path, model: DUPGTModel, spec: dict, config: dict, dataset_hash: str, stage: str, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": cpu_state(model), "model_spec": spec,
        "config": config, "dataset_sha256": dataset_hash, "stage": stage,
        "epoch": epoch, "upstream_commit": "e6c6eedf1161d2aa4f0ebd616bd36176601e5b35",
        "adapter": "padding ID reserved; zero summary for empty history"}
    if getattr(model,"model_variant",None) == "semantics":
        payload.update(model_variant="semantics",semantic_cache=model.semantic_store.reference)
    torch.save(payload,path)


# Reload either architecture; text encoding and LLM calls are not needed here.
def load_saved_model(checkpoint: dict, device="cpu"):
    variant = checkpoint.get("model_variant","dupgt")
    if variant == "semantics":
        from .semantics import SemanticStore,SemanticDUPGTModel
        reference = checkpoint["semantic_cache"]
        path = resolve(reference["directory"])
        if file_sha256(path/"manifest.json") != reference["manifest_sha256"]:
            raise ValueError("Saved model's semantic cache changed")
        store = SemanticStore(path,checkpoint["dataset_sha256"])
        if store.manifest["n_items"] != checkpoint["model_spec"]["n_items"]:
            raise ValueError("Saved semantic item mapping differs from model")
        model = SemanticDUPGTModel(**checkpoint["model_spec"],semantic_store=store)
    elif variant == "probabilistic":
        # Added variant; its movie-average prior is restored from the saved state.
        from .semantics import SemanticStore
        from .probabilistic import ProbabilisticDUPGTModel
        reference = checkpoint["semantic_cache"]
        path = resolve(reference["directory"])
        if file_sha256(path/"manifest.json") != reference["manifest_sha256"]:
            raise ValueError("Saved model's semantic cache changed")
        model = ProbabilisticDUPGTModel(**checkpoint["model_spec"],
                                        semantic_store=SemanticStore(path,checkpoint["dataset_sha256"]))
    elif variant == "dupgt":
        model = DUPGTModel(**checkpoint["model_spec"])
    else:
        raise ValueError("Unknown saved model architecture")
    model.load_state_dict(checkpoint["model_state"],strict=True)
    return model.to(device)
