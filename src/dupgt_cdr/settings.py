"""Settings for the two datasets and three experiment sizes.

Edit BASE for training settings. The dataset paths and cohort sizes are below.
Each run still saves its complete settings in outputs/<run>/config.json.
"""

# Settings shared by both datasets.
BASE = {
    "seed": 2020,
    "validation_fraction": 0.1,
    "test_fraction": 0.2,
    "embedding_dim": 10,
    "factor_initialization_std": 0.1,
    "pretrain_epochs": 15,
    "mapping_epochs": 30,
    "meta_epochs": 15,
    "batch_size": 1024,
    "meta_batch_size": 256,
    "evaluation_batch_size": 1024,
    "pretrain_learning_rate": 0.02,
    "mapping_learning_rate": 0.01,
    "meta_learning_rate": 0.02,
    "weight_decay": 0.0,
    "torch_threads": 4,
    "device": "auto",
}

# Raw input filenames.
DATASETS = {
    "2014": ("reviews_Books_5.json.gz", "reviews_Movies_and_TV_5.json.gz"),
    "2023": ("Books.csv", "Movies_and_TV.csv.gz"),
}

# Full and smaller trial runs.
SIZES = {
    "smoke": {"max_shared_users": 100, "max_source_only_users": 0,
              "max_target_only_users": 100, "pretrain_epochs": 1,
              "mapping_epochs": 1, "meta_epochs": 1},
    "progress": {"max_shared_users": 6000, "max_source_only_users": 0,
                 "max_target_only_users": 3000},
    "full": {"max_shared_users": 0, "max_source_only_users": -1,
             "max_target_only_users": -1},
}


# Combine settings for one run.
def get_config(dataset: str = "2014", size: str = "full") -> dict:
    """Build one independent settings dictionary for a run."""
    if dataset not in DATASETS or size not in SIZES:
        raise ValueError("Choose dataset 2014/2023 and size smoke/progress/full")
    source, target = DATASETS[dataset]
    folder = f"amazon{dataset}"
    config = {
        **BASE,
        **SIZES[size],
        "dataset_release": dataset,
        "label": f"Amazon {dataset} Books to Movies: {size}, small-factor initialization",
        "source_path": f"data/raw/{folder}/{source}",
        "target_path": f"data/raw/{folder}/{target}",
        "cache_dir": f"data/cache/{folder}",
        "prepared_dir": f"data/processed/{folder}/{size}",
        "output_dir": f"outputs/{folder}/{size}_seed{BASE['seed']}",
    }
    if not 0 < config["validation_fraction"] < 1 or not 0 < config["test_fraction"] < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if config["validation_fraction"] + config["test_fraction"] >= 1:
        raise ValueError("Validation and test must leave training users")
    for name in ("pretrain_epochs", "mapping_epochs", "meta_epochs", "batch_size",
                 "meta_batch_size", "evaluation_batch_size", "embedding_dim", "torch_threads"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("pretrain_learning_rate", "mapping_learning_rate", "meta_learning_rate",
                 "factor_initialization_std"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay cannot be negative")
    return config


# Offline teacher and bridge-only experiments; benchmark seed stays unchanged.
TRANSFER = {
    "overlap_fractions": [0.01, 0.10],
    "pseudo_weights": [0.01, 0.10],
    "overlap_seed": 2020,
    "teacher_seed": 2020,
    "transfer_seed": 2020,
    "teacher_users": 200,
    "candidate_movies": 10,
    "pseudo_batch_size": 64,
    "llm_model": "dupgt-qwen35-9b",
    "runtime_file": "outputs/amazon2014/llm_transfer/qwen35_runtime.json",
    "max_output_tokens": 512,
    "context_tokens": 8192,
    "max_total_tokens": 1_000_000,
    "text_tie_tolerance": 1e-8,
}
