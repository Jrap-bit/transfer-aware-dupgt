"""Read and prepare the Book/Movie rating data."""
from __future__ import annotations

import array
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import orjson

from .common import data_sha256, file_sha256, resolve, save_arrays, write_json

CACHE_SCHEMA = 2
SCHEMA = 2
HISTORY_LENGTH = 20


# 1. STREAM BOTH DATASET FORMATS
# 2014 is JSON Lines; 2023 is CSV. Each may be gzip-compressed.
def open_text(path: Path):
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8", newline="") if compressed else path.open("r", encoding="utf-8", newline="")


def raw_rows(path: Path, release: str):
    """Read one row at a time, without loading the raw file into memory."""
    if release not in {"2014", "2023"}:
        raise ValueError("Unknown dataset release")
    with open_text(path) as handle:
        if release == "2023":
            reader = csv.DictReader(handle)
            required = {"user_id", "parent_asin", "rating", "timestamp"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing 2023 CSV columns in {path}: {reader.fieldnames}")
        else:
            reader = (orjson.loads(line) for line in handle)
        yield from enumerate(reader, 2 if release == "2023" else 1)


# 2. VALIDATE IDs AND RATINGS
# Bad IDs or ratings fail early, with the raw file's line number.
def validate_ids_and_rating(user, item, rating) -> int:
    if not isinstance(user, str) or not isinstance(item, str) or not user or not item:
        raise ValueError("Missing string user/item ID")
    rating = float(rating)
    if not rating.is_integer() or not 1 <= rating <= 5:
        raise ValueError("Expected rating 1..5")
    return int(rating)


# 3. NORMALIZE TIMESTAMP UNITS
# 2014 stores seconds; 2023 stores milliseconds. Internally, both use ms.
def timestamp_milliseconds(value, release: str) -> int:
    timestamp = int(value)
    if timestamp < 0:
        raise ValueError("Expected a nonnegative timestamp")
    return timestamp * 1000 if release == "2014" else timestamp


# Combine the format, validation, and time rules into one common record.
def records(path: Path, release: str):
    """Yield validated (user ID, item ID, rating, Unix milliseconds)."""
    for number, row in raw_rows(path, release):
        try:
            if release == "2023":
                user, item = row["user_id"], row["parent_asin"]
                rating, timestamp = row["rating"], row["timestamp"]
            else:
                user, item = row["reviewerID"], row["asin"]
                rating, timestamp = row["overall"], row["unixReviewTime"]
            yield user, item, validate_ids_and_rating(user, item, rating), timestamp_milliseconds(timestamp, release)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {release} record at line {number} in {path.name}: {exc}") from exc


# 4. MAP IDENTIFIERS TO COMPACT INTEGERS
# Store raw IDs once, then use small integer IDs in the rating arrays.
def load_domain(path: Path, cache_dir: Path, label: str, release: str) -> dict:
    stat = path.stat()
    identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache, manifest = cache_dir / f"{label}.npz", cache_dir / f"{label}.json"
    if cache.exists() and manifest.exists():
        old = orjson.loads(manifest.read_bytes())
        if old.get("schema") == CACHE_SCHEMA and old.get("release") == release and old.get("raw_file") == identity:
            if file_sha256(cache) != old["cache_sha256"]:
                raise ValueError(f"Cache checksum mismatch: {cache}")
            print(f"Using verified {release}/{label} cache: {old['rows']:,} ratings", flush=True)
            with np.load(cache, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
    started = time.monotonic()
    users, items = {}, {}
    user_values, item_values = array.array("I"), array.array("I")
    ratings, timestamps = array.array("B"), array.array("q")
    print(f"Streaming {release}/{label}: {path.name}", flush=True)
    for count, (user, item, rating, timestamp) in enumerate(records(path, release), 1):
        user_values.append(users.setdefault(user, len(users)))
        item_values.append(items.setdefault(item, len(items)))
        ratings.append(rating)
        timestamps.append(timestamp)
        if count % 500_000 == 0:
            print(f"  {label}: {count:,} ratings ({time.monotonic()-started:.1f}s)", flush=True)
    if not ratings:
        raise ValueError(f"Empty dataset: {path}")
    result = {"users": np.asarray(list(users)), "items": np.asarray(list(items)),
        "u": np.frombuffer(user_values, dtype=np.uint32).copy(),
        "i": np.frombuffer(item_values, dtype=np.uint32).copy(),
        "r": np.frombuffer(ratings, dtype=np.uint8).copy(),
        "t": np.frombuffer(timestamps, dtype=np.int64).copy()}
    save_arrays(cache, **result)
    write_json(manifest, {"schema": CACHE_SCHEMA, "release": release, "raw_file": identity,
        "timestamp_unit": "Unix milliseconds", "rows": len(ratings), "users": len(users), "items": len(items),
        "cache_sha256": file_sha256(cache), "elapsed_seconds": time.monotonic()-started})
    print(f"Cached {release}/{label}: {len(ratings):,} ratings; {len(users):,} users", flush=True)
    return result


def project_domain(domain: dict, global_users: dict[str, int], allowed_users: set[str]) -> tuple:
    """Map selected users and items to the model's integer IDs."""
    translation = np.full(len(domain["users"]), -1, dtype=np.int64)
    for local, raw in enumerate(domain["users"]):
        if raw in allowed_users:
            translation[local] = global_users[str(raw)]
    mask = translation[domain["u"]] >= 0
    rows = np.column_stack((translation[domain["u"][mask]], domain["i"][mask], domain["r"][mask])).astype(np.int64)
    times = domain["t"][mask]
    rows, duplicates = resolve_duplicates(rows, times)
    present = np.unique(rows[:, 1])
    rows[:, 1] = np.searchsorted(present, rows[:, 1]) + 1  # 0 is padding only.
    return rows, domain["items"][present], duplicates


# 5. RESOLVE DUPLICATE USER-ITEM RATINGS
# Keep the newest rating; ties use the later file row.
def resolve_duplicates(rows: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, int]:
    if len(rows):
        order = np.lexsort((np.arange(len(rows)), times, rows[:, 1], rows[:, 0]))
        sorted_rows = rows[order]
        last = np.r_[np.any(sorted_rows[1:, :2] != sorted_rows[:-1, :2], axis=1), True]
        keep = np.sort(order[last])
        duplicates = len(rows) - len(keep)
        rows = rows[keep]
    else:
        duplicates = 0
    # Restore file order because the Book-history rule uses that order.
    return rows, duplicates


# 6. IDENTIFY SHARED USERS
# Shared users have ratings in both domains and can train the transfer model.
def identify_shared_users(source: dict, target: dict) -> tuple[set[str], set[str], set[str]]:
    source_users = set(source["users"].tolist())
    target_users = set(target["users"].tolist())
    return source_users, target_users, source_users & target_users


def choose_users(values: set[str], cap: int, rng: np.random.Generator) -> list[str]:
    ordered = np.asarray(sorted(values))
    if cap < 0 or cap >= len(ordered):
        return ordered.tolist()
    if cap == 0:
        return []
    return sorted(rng.choice(ordered, size=cap, replace=False).tolist())


# 7. MAKE USER-DISJOINT TARGET-COLD SPLITS
# Validation and test users keep Book histories but no training Movie ratings.
def split_shared_users(shared: list[str], global_users: dict[str, int], config: dict,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    permuted = rng.permutation(shared)
    n_valid = max(1, round(len(shared) * config["validation_fraction"]))
    n_test = max(1, round(len(shared) * config["test_fraction"]))
    validation_users = np.sort([global_users[str(u)] for u in permuted[:n_valid]])
    test_users = np.sort([global_users[str(u)] for u in permuted[n_valid:n_valid+n_test]])
    shared_train_users = np.sort([global_users[str(u)] for u in permuted[n_valid+n_test:]])
    return shared_train_users, validation_users, test_users


def split_target_ratings(target: np.ndarray, shared_train_users: np.ndarray,
                         validation_users: np.ndarray, test_users: np.ndarray) -> tuple:
    valid_mask = np.isin(target[:, 0], validation_users)
    test_mask = np.isin(target[:, 0], test_users)
    target_train = target[~(valid_mask | test_mask)]
    meta_train = target_train[np.isin(target_train[:, 0], shared_train_users)]
    return target_train, meta_train, target[valid_mask], target[test_mask]


# 8. BUILD FIXED-LENGTH HIGH/LOW BOOK HISTORIES
# High means 4-5 stars; low means 1-3. Zero pads missing items.
def build_histories(source: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """Match upstream: retain last 20 file-order items, post-pad with zero."""
    buckets = [[[], []] for _ in range(n_users)]
    for user, item, rating in source:
        buckets[int(user)][0 if rating >= 4 else 1].append(int(item))
    histories = np.zeros((n_users, 2 * HISTORY_LENGTH), dtype=np.int64)
    counts = np.zeros((n_users, 2), dtype=np.int64)
    for user, parts in enumerate(buckets):
        for branch, values in enumerate(parts):
            counts[user, branch] = len(values)
            retained = values[-HISTORY_LENGTH:]
            offset = branch * HISTORY_LENGTH
            histories[user, offset:offset + len(retained)] = retained
    return histories, counts


# 9. EXCLUDE UNSEEN MOVIE ITEMS FROM EVALUATION
# The model cannot score a Movie item ID it never saw during training.
def keep_warm_items(rows: np.ndarray, target_train: np.ndarray) -> np.ndarray:
    warm_items = np.unique(target_train[:, 1])
    return rows[np.isin(rows[:, 1], warm_items)]


# 10. CHECK INTEGRITY AND SAVE CHECKSUMMED ARRAYS
# Check that the split, padding, and histories match the intended experiment.
def assert_prepared_integrity(arrays: dict) -> dict:
    groups = {k: set(arrays[k].tolist()) for k in ["shared_train_users", "validation_users", "test_users"]}
    if groups["shared_train_users"] & groups["validation_users"] or groups["shared_train_users"] & groups["test_users"] or groups["validation_users"] & groups["test_users"]:
        raise ValueError("Shared-user partitions overlap")
    train_users = set(arrays["target_train"][:, 0].tolist())
    if train_users & (groups["validation_users"] | groups["test_users"]):
        raise ValueError("Held-out target user leaked into target training")
    if set(arrays["meta_train"][:, 0]) != groups["shared_train_users"]:
        raise ValueError("Meta training must contain exactly the shared training users")
    n_source = len(arrays["source_item_ids"])
    source_ids = set(arrays["source_train"][:, 1].tolist())
    target_ids = set(arrays["target_train"][:, 1].tolist())
    if 0 in source_ids or 0 in target_ids or source_ids & target_ids:
        raise ValueError("Padding collision or intersecting source/target item IDs")
    for name in ["validation", "test"]:
        if not len(arrays[name]):
            raise ValueError(f"No warm-item {name} ratings remain; increase cohort size")
        if not set(arrays[name][:, 1]).issubset(target_ids):
            raise ValueError(f"Cold target items leaked into user-cold {name} benchmark")
    histories = arrays["histories"]
    if histories.min() < 0 or histories.max() > n_source:
        raise ValueError("Source history contains a non-source item")
    # Validate that every non-padding history entry actually has the right sign.
    expected_histories, expected_counts = build_histories(arrays["source_train"], len(arrays["user_ids"]))
    if not np.array_equal(histories, expected_histories) or not np.array_equal(arrays["polarity_counts"], expected_counts):
        raise ValueError("History/polarity construction mismatch")
    return {"disjoint_target_user_splits": True, "heldout_target_ratings_used_for_training": False,
        "disjoint_domain_item_ids": True, "zero_reserved_for_padding": True,
        "evaluation_target_items_seen_in_training": True, "histories_use_source_ratings_only": True}



# Assemble the benchmark from the preparation steps above.
def build_benchmark(source: dict, target: dict, config: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(config["seed"])
    source_users, target_users, all_shared = identify_shared_users(source, target)
    max_shared = config["max_shared_users"]
    shared = choose_users(all_shared, -1 if max_shared == 0 else max_shared, rng)
    if len(shared) < 10:
        raise ValueError("At least ten shared users are required")
    source_only = choose_users(source_users - target_users, config["max_source_only_users"], rng)
    target_only = choose_users(target_users - source_users, config["max_target_only_users"], rng)
    user_ids = sorted(set(shared) | set(source_only) | set(target_only))
    global_users = {raw: index for index, raw in enumerate(user_ids)}
    shared_train_users, validation_users, test_users = split_shared_users(shared, global_users, config, rng)
    src, source_items, src_duplicates = project_domain(source, global_users, set(shared) | set(source_only))
    tgt, target_items, tgt_duplicates = project_domain(target, global_users, set(shared) | set(target_only))
    tgt[:, 1] += len(source_items)
    target_train, meta_train, original_valid, original_test = split_target_ratings(
        tgt, shared_train_users, validation_users, test_users)
    valid = keep_warm_items(original_valid, target_train)
    test = keep_warm_items(original_test, target_train)
    histories, counts = build_histories(src, len(user_ids))
    arrays = {"source_train": src, "target_train": target_train, "meta_train": meta_train,
        "validation": valid, "test": test, "histories": histories, "polarity_counts": counts,
        "user_ids": np.asarray(user_ids), "source_item_ids": source_items,
        "target_item_ids": target_items, "shared_train_users": shared_train_users,
        "validation_users": validation_users, "test_users": test_users}
    audit = assert_prepared_integrity(arrays)
    manifest = {
        "schema": SCHEMA, "label": config["label"], "data_config_sha256": data_sha256(config),
        "dataset": f"Amazon Reviews {config.get('dataset_release', '2014')} 5-core", "direction": "Books -> Movies & TV",
        "scope": "full user pool" if max_shared == 0 and config["max_source_only_users"] < 0 and config["max_target_only_users"] < 0 else "deterministic user-cohort subset, not full paper reproduction",
        "raw": {"source_ratings": len(source["r"]), "target_ratings": len(target["r"]),
            "source_users": len(source_users), "target_users": len(target_users), "shared_users": len(all_shared)},
        "cohort": {"shared_users": len(shared), "source_only_users": len(source_only),
            "target_only_users": len(target_only), "all_users": len(user_ids),
            "source_items": len(source_items), "target_items": len(target_items)},
        "splits": {name: {"ratings": len(arrays[name]), "users": len(np.unique(arrays[name][:, 0])),
            "items": len(np.unique(arrays[name][:, 1]))} for name in ["source_train", "target_train", "meta_train", "validation", "test"]},
        "user_split_counts": {"shared_train": len(shared_train_users), "validation": len(validation_users), "test": len(test_users)},
        "evaluation_filter": {"validation_cold_item_rows_excluded": len(original_valid)-len(valid),
            "test_cold_item_rows_excluded": len(original_test)-len(test),
            "reason": "Isolate user cold start; an unseen target item has no trained ID embedding"},
        "duplicates_removed": {"source": src_duplicates, "target": tgt_duplicates},
        "source_polarity": {"high_threshold": "ratings 4-5", "low_threshold": "ratings 1-3",
            "shared_with_empty_high": int(np.sum(counts[[global_users[u] for u in shared], 0] == 0)),
            "shared_with_empty_low": int(np.sum(counts[[global_users[u] for u in shared], 1] == 0))},
        "protocol": "User-disjoint target cold start, 70/10/20 by default. Source histories of held-out target users are allowed. This is not a chronological deployment simulation.",
        "history_policy": "Last 20 high and last 20 low source items in archive file order, post-padding with 0; all source ratings retained for source pretraining",
        "integrity": audit,
    }
    return arrays, manifest



# Save the benchmark arrays and their checksums, or reuse a verified copy.
def prepare(config: dict) -> Path:
    destination = resolve(config["prepared_dir"])
    archive = destination / "benchmark.npz"
    manifest = destination / "manifest.json"
    if archive.exists() and manifest.exists():
        existing = orjson.loads(manifest.read_bytes())
        if existing.get("data_config_sha256") != data_sha256(config):
            raise ValueError("Prepared data configuration changed. Choose a new prepared_dir to preserve the old benchmark.")
        for label, key in [("source", "source_path"), ("target", "target_path")]:
            raw = resolve(config[key])
            st = raw.stat()
            if existing["raw_file_identity"][label] != {"path": str(raw.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}:
                raise ValueError("Raw file changed; choose a new prepared_dir")
        if file_sha256(archive) != existing["benchmark_sha256"]:
            raise ValueError("Prepared benchmark checksum mismatch")
        print(f"Using prepared benchmark: {archive}", flush=True)
        return archive
    source = load_domain(resolve(config["source_path"]), resolve(config["cache_dir"]), "books", config["dataset_release"])
    target = load_domain(resolve(config["target_path"]), resolve(config["cache_dir"]), "movies", config["dataset_release"])
    arrays, dataset_manifest = build_benchmark(source, target, config)
    save_arrays(archive, **arrays)
    dataset_manifest["benchmark_sha256"] = file_sha256(archive)
    dataset_manifest["raw_file_identity"] = {}
    for label, key in [("source", "source_path"), ("target", "target_path")]:
        raw = resolve(config[key]); st = raw.stat()
        dataset_manifest["raw_file_identity"][label] = {"path": str(raw.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    write_json(manifest, dataset_manifest)
    print(orjson.dumps(dataset_manifest, option=orjson.OPT_INDENT_2).decode(), flush=True)
    return archive



# Load a prepared benchmark only after checking its checksum.
def get_benchmark(config: dict) -> tuple[dict, dict]:
    directory = resolve(config["prepared_dir"])
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["data_config_sha256"] != data_sha256(config):
        raise ValueError("Configuration differs from prepared benchmark; run prepare with a new prepared_dir")
    path = directory / "benchmark.npz"
    if file_sha256(path) != manifest["benchmark_sha256"]:
        raise ValueError("Prepared dataset checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert_prepared_integrity(data)
    return data, manifest
