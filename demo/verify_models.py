"""Compare every retrained demo model with the run behind the reported numbers.

    .venv/bin/python demo/verify_models.py

For each model it compares the checkpoint SHA-256 recorded in artifact_manifest.json, the validation MAE,
and (when the demo run has been scored) the test MAE. Writes demo/bundle/model_check.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/amazon2023"
NEW = OLD / "demo_models"
def pairs(G: str) -> dict:
    return {
        "DUPGT": ("overlap_seeds/" + G + "/dupgt", "overlap/" + G + "/dupgt"),
        "+ Semantic gate": ("overlap_seeds/" + G + "/semantics", "overlap/" + G + "/semantics"),
        "+ LLM teacher": ("overlap_seeds/" + G + "/semantics_llm_w0.01", "overlap/" + G + "/semantics_llm_w0.01"),
        "+ Shuffled teacher": ("overlap_seeds/" + G + "/semantics_shuffled_w0.01", "overlap/" + G + "/semantics_shuffled_w0.01"),
        "+ Text teacher": ("overlap_seeds/" + G + "/semantics_text_w0.01", "overlap/" + G + "/semantics_text_w0.01"),
        "Probabilistic bridge": ("probabilistic_bridge/" + G + "/probabilistic_g0.1", "probabilistic/" + G + "/probabilistic_g0.1"),
        "Probabilistic (no prior)": ("probabilistic_bridge/" + G + "/probabilistic_g0", "probabilistic/" + G + "/probabilistic_g0"),
        **{f"Blend bridge {s}": (f"uncertainty_ensemble/{G}/bridge_{s}", f"uncertainty/{G}/bridge_{s}") for s in (2020, 3020, 4020, 5020, 6020)},
    }


GROUPS = {"10%": "overlap_0.10/seed_2020", "1%": "overlap_0.01/seed_2020"}


def read(path: Path, name: str):
    f = path / name
    return json.loads(f.read_text()) if f.exists() else None


def mae(record):
    if not record:
        return None
    for key in ("raw", "validation", "metrics"):
        if isinstance(record.get(key), dict) and "mae" in record[key]:
            return record[key]["mae"]
    return record.get("mae")


def main() -> None:
    report = {}
    for name, G in GROUPS.items():
        print(f"== {name} overlap")
        rows = {}
        for label, (old, new) in pairs(G).items():
            o, n = OLD / old, NEW / new
            om, nm = read(o, "artifact_manifest.json"), read(n, "artifact_manifest.json")
            row = {"finished": nm is not None}
            if nm and om:
                row["identical_checkpoint"] = om["checkpoint_sha256"] == nm["checkpoint_sha256"]
                row["validation_mae"] = [mae(read(o, "validation_metrics.json")), mae(read(n, "validation_metrics.json"))]
            rows[label] = row
            print(f"  {label:26s} {row}")
        ob, nb = read(OLD / "uncertainty_ensemble" / G / "blend", "blend.json"), read(NEW / "uncertainty" / G / "blend", "blend.json")
        if ob and nb:
            rows["blend parameters"] = {"identical": ob["uncertainty_blend"] == nb["uncertainty_blend"] and ob["constant_blend"] == nb["constant_blend"]}
            print("  blend parameters", rows["blend parameters"])
        report[name] = rows
    (ROOT / "demo/bundle/model_check.json").write_text(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
