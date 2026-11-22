"""Check the demo's new-user path against real held-out test users.

For sampled test users it predicts their held-out Movie ratings twice with the same DUPGT checkpoint:
  learned  - the Book-taste vector learned during pretraining (what the reported results used)
  fold-in  - the vector re-derived from their Book ratings alone (what the demo does for a new person)
and reports MAE of each against the true ratings, plus how close the two predictions are.

    .venv/bin/python demo/verify_foldin.py [run_dir ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupgt_cdr.runtime import load_saved_model  # noqa: E402

LAMBDAS = [0.1, 0.3, 1.0]
N_USERS = 3000


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]] or [ROOT / "outputs/amazon2023/demo_models/overlap/overlap_0.10/seed_2020/dupgt"]
    a = np.load(ROOT / "data/processed/amazon2023/full/benchmark.npz")
    src, test, hist = a["source_train"], a["test"], a["histories"]
    rng = np.random.default_rng(7)
    users = rng.choice(a["test_users"], N_USERS, replace=False)
    rows = test[np.isin(test[:, 0], users)]
    order = np.argsort(src[:, 0], kind="stable")
    s = src[order]
    lo, hi = np.searchsorted(s[:, 0], users), np.searchsorted(s[:, 0], users, side="right")
    report = {}
    for run in runs:
        ck = torch.load(run / "checkpoints/dupgt_best.pt", map_location="cpu", weights_only=True)
        model = load_saved_model(ck, "cpu").eval()
        U = model.src_model.uid_embedding.weight.detach().clone()
        V = model.src_model.iid_embedding.weight.detach().numpy().astype(np.float64)
        x = torch.as_tensor(np.column_stack((rows[:, :2], hist[rows[:, 0]])), dtype=torch.long)
        truth = rows[:, 2].astype(float)
        with torch.inference_mode():
            learned = model.forward_with_gate(x)[0].numpy()
        result = {"n_users": int(len(users)), "n_ratings": int(len(rows)),
                  "mae_learned": float(np.abs(learned.clip(1, 5) - truth).mean())}
        for lam in LAMBDAS:
            W = U.clone()
            for u, a0, b0 in zip(users, lo, hi):
                Vi, r = V[s[a0:b0, 1]], s[a0:b0, 2].astype(float)
                W[u] = torch.as_tensor(np.linalg.solve(Vi.T @ Vi + lam * np.eye(Vi.shape[1]), Vi.T @ r), dtype=W.dtype)
            model.src_model.uid_embedding.weight.data.copy_(W)
            with torch.inference_mode():
                folded = model.forward_with_gate(x)[0].numpy()
            model.src_model.uid_embedding.weight.data.copy_(U)
            result[f"lambda_{lam:g}"] = {
                "mae_fold_in": float(np.abs(folded.clip(1, 5) - truth).mean()),
                "mean_abs_gap_to_learned": float(np.abs(folded - learned).mean()),
                "correlation_with_learned": float(np.corrcoef(folded, learned)[0, 1])}
        report[run.name] = result
        print(run.name, json.dumps(result, indent=1), flush=True)
    out = ROOT / "demo/bundle/fold_in_check.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
