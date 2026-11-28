# DUPGT-CDR baseline: Amazon 2014 Books to Movies: full, small-factor initialization

Run completed: 2026-10-30T21:06:14.689667+00:00

This is an executed baseline, not a proposed model or a reproduction of the published numerical scores.

- Dataset: Amazon Reviews 2014 5-core; direction: Books -> Movies & TV.
- Scope: full user pool.
- Shared users: 37,388; target-only training users: 86,572.
- Target user split: {'shared_train': 26171, 'validation': 3739, 'test': 7478}; no validation/test Movie ratings entered training.
- Device: cpu; PyTorch 2.8.0; seed 2020.
- Factor initialization: normal with std 0.1.
- End-to-end training/evaluation time: 2509.5 seconds (excludes raw-data preparation).

## Final test results

One checkpoint per trained model was selected by validation RMSE; its MAE and RMSE below are from the same checkpoint. No test-set epoch selection or clipping was used for the primary results.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| global_mean | 0.934227 | 1.181253 |
| movie_mean | 0.826783 | 1.075740 |
| linear_mapping | 0.862957 | 1.103667 |
| dupgt | 0.787939 | 1.041991 |

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
| source_train | 8,898,041 | 603,668 | 367,982 |
| target_train | 1,446,273 | 112,743 | 50,051 |
| meta_train | 541,059 | 26,171 | 47,576 |
| validation | 87,463 | 3,739 | 27,532 |
| test | 163,789 | 7,478 | 35,724 |

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
