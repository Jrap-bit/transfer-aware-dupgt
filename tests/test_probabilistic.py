"""Check the probabilistic bridge: its math, that it starts from the same bridge as the semantic run,
and that it goes through the existing lock / evaluate / compare workflow unchanged."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_semantics import cache_fixture
from dupgt_cdr.model import DUPGTModel
from dupgt_cdr.runtime import model_spec, save_checkpoint, load_saved_model
from dupgt_cdr.predict import predict
from dupgt_cdr.experiments import transfer_train, lock_selection, transfer_evaluate, compare_experiments
from dupgt_cdr.probabilistic import binary_entropy, probabilistic_train, ProbabilisticDUPGTModel
from dupgt_cdr.semantics import SemanticDUPGTModel, SemanticStore


class MathTests(unittest.TestCase):
    def test_entropy_peaks_at_half_and_vanishes_at_the_ends(self):
        values = binary_entropy(torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
        self.assertAlmostEqual(values[2].item(), np.log(2), places=5)
        self.assertLess(values[0].item(), 1e-4)
        self.assertAlmostEqual(values[1].item(), values[3].item(), places=6)



class ProbabilisticWorkflowTests(unittest.TestCase):
    def test_trains_reloads_and_uses_existing_lock_evaluate_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data, manifest, config, cache = cache_fixture(folder)
            spec = model_spec(data, config)
            factor = folder / "factors.pt"
            save_checkpoint(factor, DUPGTModel(**spec), spec, config, manifest["benchmark_sha256"], "pretrained", 1)
            semantic, probabilistic = folder / "run_semantics", folder / "run_probabilistic"
            with patch("dupgt_cdr.experiments.get_benchmark", return_value=(data, manifest)), \
                 patch("dupgt_cdr.data.get_benchmark", return_value=(data, manifest)):
                transfer_train(config, str(factor), str(semantic), overlap_fraction=.5, overlap_seed=7,
                               transfer_seed=3, semantic_dir=cache)
                result = probabilistic_train(config, str(factor), str(probabilistic), semantic_dir=cache,
                                             overlap_fraction=.5, overlap_seed=7, transfer_seed=3)
                a = json.loads((semantic / "experiment.json").read_text())
                b = json.loads((probabilistic / "experiment.json").read_text())
                # Paired with the semantic run: same users, factors and starting bridge.
                for key in ("initial_bridge_sha256", "initial_semantic_sha256", "real_rows_sha256", "factor_sha256"):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(b["model_variant"], "probabilistic")
                self.assertTrue(result["factors_unchanged"])
                selection = folder / "selection.json"
                lock_selection([str(semantic), str(probabilistic)], str(selection), "unit-test selection")
                for path in (semantic, probabilistic):
                    transfer_evaluate(str(path), str(selection))
                comparison = compare_experiments([str(semantic), str(probabilistic)], str(folder / "compare"), final=True)
                self.assertEqual([r["model_variant"] for r in comparison["rows"]], ["semantics", "probabilistic"])
            checkpoint = torch.load(probabilistic / "checkpoints/dupgt_best.pt", weights_only=True)
            self.assertEqual(checkpoint["model_variant"], "probabilistic")
            model = load_saved_model(checkpoint).eval()
            self.assertGreater(float(model.item_prior.abs().sum()), 0)   # prior restored from the checkpoint
            values, gates = predict(model, data["validation"], torch.as_tensor(data["histories"]), "test_meta",
                                    torch.device("cpu"), 64)
            self.assertTrue(np.isfinite(values).all())
            self.assertAlmostEqual(float(np.sqrt(np.mean((values - data["validation"][:, 2]) ** 2))),
                                   result["raw"]["rmse"], places=5)
            mu, log_sigma2, prior, _, _ = model.forward_parts(torch.as_tensor(
                np.concatenate([data["validation"][:4, :2], data["histories"][data["validation"][:4, 0]]], axis=1)))
            final, weight = model.combine(mu, log_sigma2, prior)
            torch.testing.assert_close(final, weight * mu + (1 - weight) * prior)
            self.assertTrue(bool(((weight > 0) & (weight < 1)).all()))

    def test_log_odds_path_matches_the_semantic_model_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, manifest, config, cache = cache_fixture(Path(tmp))
            spec = model_spec(data, config)
            store = SemanticStore(cache)
            torch.manual_seed(0)
            semantic = SemanticDUPGTModel(**spec, semantic_store=store)
            torch.nn.init.normal_(semantic.dupgt.semantic.network[-1].weight)   # make the semantic shift non-zero
            probabilistic = ProbabilisticDUPGTModel(**spec, semantic_store=store)
            probabilistic.load_state_dict(semantic.state_dict(), strict=False)
            x = torch.as_tensor(np.concatenate([data["validation"][:, :2], data["histories"][data["validation"][:, 0]]], axis=1))
            semantic.eval(); probabilistic.eval()
            a_rating, a_alpha, _ = semantic.forward_with_details(x)
            b_rating, b_alpha, details = probabilistic.forward_with_details(x)
            torch.testing.assert_close(a_alpha, b_alpha)
            torch.testing.assert_close(a_rating, b_rating)
            torch.testing.assert_close(torch.sigmoid(details["gate_logit"]), b_alpha)

    def test_requires_semantics_and_valid_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "semantic"):
                probabilistic_train({}, "x", str(Path(tmp) / "a"), semantic_dir=None)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                probabilistic_train({}, "x", str(Path(tmp) / "b"), semantic_dir="cache", gate_prior_weight=-1)


if __name__ == "__main__":
    unittest.main()
