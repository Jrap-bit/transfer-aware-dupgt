# DUPGT-CDR baseline: Amazon 2023 Books to Movies: full, small-factor initialization

Run completed: 2026-10-30T21:59:41.623328+00:00

This is an executed baseline, not a proposed model or a reproduction of the published numerical scores.

- Dataset: Amazon Reviews 2023 5-core; direction: Books -> Movies & TV.
- Scope: full user pool.
- Shared users: 127,188; target-only training users: 530,015.
- Target user split: {'shared_train': 89031, 'validation': 12719, 'test': 25438}; no validation/test Movie ratings entered training.
- Device: cpu; PyTorch 2.8.0; seed 2020.
- Factor initialization: normal with std 0.1.
- End-to-end training/evaluation time: 5098.2 seconds (excludes raw-data preparation).

## Final test results

One checkpoint per trained model was selected by validation RMSE; its MAE and RMSE below are from the same checkpoint. No test-set epoch selection or clipping was used for the primary results.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| global_mean | 0.910987 | 1.143215 |
| movie_mean | 0.762019 | 1.050694 |
| linear_mapping | 0.833627 | 1.105525 |
| dupgt | 0.723044 | 1.011914 |

Global mean and Movie mean use target training ratings only. The linear mapping is an EMCDR-style embedding-MSE baseline; it is not claimed to reproduce the original EMCDR paper protocol.

## What was actually trained

1. Source-domain latent factor model on available Book ratings.
2. Target-domain latent factor model on Movie ratings from training users only.
3. A linear Book-user to Movie-user mapping on shared training users.
4. DUPGT's two attention encoders, two transformation meta-networks, candidate-aware GLU gate, bridge matrix multiplication, and latent dot-product rating predictor.

The pretrained source and target embeddings are frozen during transfer training, following the optimizer scope in the upstream runner. The model uses high ratings 4-5 and low ratings 1-3. It does not include semantics, evidence-confidence formulas, a neutral branch, or an added no-transfer gate.

## Dataset split sizes

| Split | Ratings | Users | Items |
|---|---:|---:|---:|
| source_train | 9,488,297 | 776,370 | 495,063 |
| target_train | 6,797,056 | 619,046 | 197,943 |
| meta_train | 1,497,886 | 89,031 | 176,658 |
| validation | 210,455 | 12,719 | 82,782 |
| test | 433,618 | 25,438 | 118,971 |

## Integrity and interpretation

- Test and validation users are disjoint from target training users. Their source histories are deliberately available: this is target-domain user cold start, not a new user with no information anywhere.
- Evaluation items must have a Movie training rating. Cold-item exclusions are recorded in dataset_manifest.json; this run does not test new Movie IDs.
- This is a static, user-disjoint benchmark, not a chronological simulation. Some source events can be later than a withheld target event.
- IDs are deterministically remapped; Book and Movie item IDs are disjoint, and 0 is reserved exclusively for padding.
- Attention on a completely padded branch produces a zero history summary rather than averaging random padding embeddings. This is a documented implementation correction, not a research contribution.
- This run uses one seed. It establishes that the pipeline works; it does not establish a statistically significant improvement or state-of-the-art performance.
- Full-pool execution is available separately and has not been run unless its own output folder contains completed metrics.

## Available evidence

- metrics.json: exact raw and clipped metrics, source-history strata, user-level comparisons, environment, and checkpoint selection.
- training_history.csv: loss and validation curves for every epoch.
- test_predictions.csv.gz: actual held-out ratings, model predictions, and learned high/low gate weights.
- example_predictions.csv: a short human-readable sample of those same predictions.
- user_metrics.csv: per-user errors and comparisons with the Movie-mean baseline.
- checkpoints/: frozen-factor and best-transfer model weights.
- training_curves.png / test_metrics.png: figures generated from this run.

Upstream code: https://github.com/isdlyusukeshimizu/DUPGT-CDR (pinned commit e6c6eedf1161d2aa4f0ebd616bd36176601e5b35).
