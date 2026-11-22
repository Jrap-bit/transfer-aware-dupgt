# DUPGT-CDR: Books to Movies

Cold-start rating prediction across domains (Amazon 2023: Books → Movies & TV), extending DUPGT-CDR with:

- a **semantic gate**: Jina text embeddings of the liked/disliked books adjust DUPGT's gate per movie
- an **LLM teacher**: Qwen3.5-9B pairwise preferences used as a weak auxiliary loss (with shuffled and text-only controls)
- an **uncertainty-aware fallback**: a 5-bridge uncertainty blend and a probabilistic bridge with a gate prior, which fall back to the movie average when the transfer is unreliable

## Layout

| Folder | Contents |
|---|---|
| `src/dupgt_cdr/` | data loading, model, training, evaluation, semantics, LLM teacher, fallbacks |
| `scripts/` | limited-overlap (1%, 10%) experiment runners and summaries |
| `notebooks/` | preprocessing and results |
| `tests/` | unit tests |
| `demo/` | interactive demo (local server + web page) |
| `data/`, `outputs/` | datasets, caches, trained runs and logs |

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[notebook,semantics]" pandas
.venv/bin/python -m unittest discover -s tests -v
```

## Main commands

```bash
.venv/bin/python -m dupgt_cdr pipeline --dataset 2023 --size full --device cpu      # DUPGT baseline
.venv/bin/python -m dupgt_cdr semantic-run --dataset 2023 --size full                # semantic gate variants
.venv/bin/python scripts/run_overlap_seeds.py --fractions 0.10 0.01                  # limited overlap
.venv/bin/python scripts/run_uncertainty.py                                          # uncertainty blend
.venv/bin/python scripts/run_probabilistic.py                                        # probabilistic bridge
.venv/bin/python -m dupgt_cdr --help                                                 # all commands
```

## Demo

```bash
zsh demo/start_demo.sh            # loads the LLM in LM Studio, starts the server, opens http://127.0.0.1:8765
zsh demo/start_demo.sh --no-llm   # without the LLM
```

The demo loads the trained checkpoints from `outputs/amazon2023/demo_models` (1% and 10% overlap), folds a new reader in from their rated books, and shows every model's predictions side by side. `demo/verify_models.py` checks these checkpoints against the original runs.
