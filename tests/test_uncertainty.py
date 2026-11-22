"""Check the uncertainty-gated ensemble: validation-only fitting, the test lock chain, and the blend math."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_semantics import cache_fixture
from dupgt_cdr.common import file_sha256
from dupgt_cdr.model import DUPGTModel
from dupgt_cdr.runtime import model_spec, save_checkpoint
from dupgt_cdr.experiments import transfer_train, lock_selection, transfer_evaluate
from dupgt_cdr.uncertainty import blend_weight, blended, fit_parameters, fit_blend, evaluate_blend


class BlendMathTests(unittest.TestCase):
    def test_weight_is_inverse_variance_and_falls_with_disagreement(self):
        d = np.array([0.0, 0.1, 0.5, 2.0])
        w = blend_weight(d, s0=1.0, tau2=0.25, c=4.0)
        self.assertAlmostEqual(w[0], 1 / 1.25)
        self.assertTrue(np.all(np.diff(w) < 0))
        self.assertTrue(np.all((w > 0) & (w <= 1)))

    def test_fit_uses_disagreement_only_when_it_predicts_error(self):
        rng = np.random.default_rng(0)
        n = 4000
        truth = rng.normal(3.5, 1.0, n)
        prior = np.full(n, 3.5)
        noise = rng.uniform(0.05, 1.5, n)              # per-row uncertainty
        centre = truth + rng.normal(0, 1, n) * noise   # bridges are wrong where noise is high
        members = centre + rng.normal(0, 1, (5, n)) * noise[None, :] * 0.5
        params = fit_parameters(members, prior, truth, constant=False)
        constant = fit_parameters(members, prior, truth, constant=True)
        self.assertGreater(params["c"], 0)
        mse = lambda p: np.mean((blended(members, prior, p)[0] - truth) ** 2)
        self.assertLess(mse(params), mse(constant))
        self.assertLessEqual(mse(constant), np.mean((members.mean(0) - truth) ** 2) + 1e-12)


class EnsembleWorkflowTests(unittest.TestCase):
    def test_fit_on_validation_then_lock_then_score_test_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data, manifest, config, cache = cache_fixture(folder)
            factor = folder / "factors.pt"
            save_checkpoint(factor, DUPGTModel(**model_spec(data, config)), model_spec(data, config), config,
                            manifest["benchmark_sha256"], "pretrained", 1)
            members = [folder / f"bridge_{seed}" for seed in (1, 2, 3)]
            with patch("dupgt_cdr.experiments.get_benchmark", return_value=(data, manifest)), \
                 patch("dupgt_cdr.data.get_benchmark", return_value=(data, manifest)):
                for seed, path in zip((1, 2, 3), members):
                    transfer_train(config, str(factor), str(path), overlap_fraction=.5, overlap_seed=7,
                                   transfer_seed=seed, semantic_dir=cache)
                blend = fit_blend([str(p) for p in members], str(folder / "blend"))
                self.assertEqual(blend["split_used_for_fitting"], "validation")
                self.assertTrue(all((p / "validation_predictions.csv.gz").exists() for p in members))
                self.assertFalse(any((p / "test_metrics.json").exists() for p in members))
                # A lock that does not cite the blend cannot be used to score it.
                bad = folder / "bad_selection.json"
                lock_selection([str(p) for p in members], str(bad), "no blend hash here")
                for path in members:
                    transfer_evaluate(str(path), str(bad))
                with self.assertRaisesRegex(ValueError, "cite this blend"):
                    evaluate_blend(str(folder / "blend"), str(bad))
                with self.assertRaisesRegex(ValueError, "before any member is scored"):
                    fit_blend([str(p) for p in members], str(folder / "late_blend"))

    def test_members_must_share_users_and_differ_in_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data, manifest, config, cache = cache_fixture(folder)
            factor = folder / "factors.pt"
            save_checkpoint(factor, DUPGTModel(**model_spec(data, config)), model_spec(data, config), config,
                            manifest["benchmark_sha256"], "pretrained", 1)
            with patch("dupgt_cdr.experiments.get_benchmark", return_value=(data, manifest)), \
                 patch("dupgt_cdr.data.get_benchmark", return_value=(data, manifest)):
                transfer_train(config, str(factor), str(folder / "a"), overlap_fraction=.5, overlap_seed=7,
                               transfer_seed=1, semantic_dir=cache)
                transfer_train(config, str(factor), str(folder / "b"), overlap_fraction=.5, overlap_seed=7,
                               transfer_seed=1, semantic_dir=cache)
                transfer_train(config, str(factor), str(folder / "c"), overlap_fraction=.5, overlap_seed=8,
                               transfer_seed=2, semantic_dir=cache)
                with self.assertRaisesRegex(ValueError, "different bridge seeds"):
                    fit_blend([str(folder / "a"), str(folder / "b")], str(folder / "x"))
                with self.assertRaisesRegex(ValueError, "share data"):
                    fit_blend([str(folder / "a"), str(folder / "c")], str(folder / "y"))

    def test_full_chain_with_cited_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data, manifest, config, cache = cache_fixture(folder)
            factor = folder / "factors.pt"
            save_checkpoint(factor, DUPGTModel(**model_spec(data, config)), model_spec(data, config), config,
                            manifest["benchmark_sha256"], "pretrained", 1)
            members = [folder / f"bridge_{seed}" for seed in (1, 2)]
            with patch("dupgt_cdr.experiments.get_benchmark", return_value=(data, manifest)), \
                 patch("dupgt_cdr.data.get_benchmark", return_value=(data, manifest)):
                for seed, path in zip((1, 2), members):
                    transfer_train(config, str(factor), str(path), overlap_fraction=.5, overlap_seed=7,
                                   transfer_seed=seed, semantic_dir=cache)
                fit_blend([str(p) for p in members], str(folder / "blend"))
                selection = folder / "selection.json"
                lock_selection([str(p) for p in members], str(selection),
                               f"ensemble; blend sha256={file_sha256(folder / 'blend' / 'blend.json')}")
                for path in members:
                    transfer_evaluate(str(path), str(selection))
                result = evaluate_blend(str(folder / "blend"), str(selection))
            for key in ("ensemble_mean", "constant_blend", "uncertainty_blend"):
                self.assertTrue(np.isfinite(result[key]["rmse"]))
                self.assertTrue(0 <= result[key]["negative_transfer_rate"] <= 1)
            self.assertEqual(len(result["single_bridges"]), 2)
            self.assertTrue((folder / "blend" / "test_predictions.csv.gz").exists())


if __name__ == "__main__":
    unittest.main()
