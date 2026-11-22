"""Compare local teachers on validation users without producing training labels."""
from __future__ import annotations

import collections
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .common import file_sha256, resolve, write_json
from .teacher import (PROMPT, PROMPT_VERSION, canonical, check_files, extract_metadata,
    identity, item_asin, load_local_teacher, local_json, normalize_response,
    preference_payload, public_prompt, read_jsonl, retained_books, sample_books,
    seal_files, text_teacher, write_jsonl)


# 1. FIX A SMALL VALIDATION COHORT BEFORE ASKING EITHER MODEL

def audit_candidates(data: dict, users: int, seed: int) -> np.ndarray:
    if users < 2:
        raise ValueError("Use at least two users for the shuffled-profile control")
    rng = np.random.default_rng(seed)
    return rng.permutation(data["validation_users"])[:min(len(data["validation_users"]), users*5)]


def audit_request(content: dict, mapping: dict, mode: str, profile_user: int) -> dict:
    key = identity({"input": content, "prompt": PROMPT, "version": PROMPT_VERSION})
    return {"request_id": key, "input": {"request_id": key, **content},
            "mapping": mapping, "mode": mode, "profile_user_id": profile_user}


def prepare_audit(config: dict, book_source: str, movie_source: str,
                  output_dir: str, users: int = 8, seed: int = 2020, minimum_books: int = 1) -> dict:
    from .data import get_benchmark
    if config["dataset_release"] not in {"2014", "2023"} or config["max_shared_users"] != 0:
        raise ValueError("This audit uses a saved 2014/full or 2023/full validation split")
    if not 1 <= minimum_books <= 6:
        raise ValueError("minimum_books must be between one and six")
    output = resolve(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new audit directory; saved inputs stay fixed")
    data, benchmark = get_benchmark(config)
    candidates = audit_candidates(data, users, seed)
    profiles = retained_books(data, candidates)
    validation = data["validation"][np.isin(data["validation"][:, 0], candidates)]
    required_books = {item_asin(data, item, "book") for parts in profiles.values()
                      for branch in parts for item, _ in branch}
    required_movies = {item_asin(data, int(item), "movie") for item in validation[:, 1]}
    books, book_audit = extract_metadata(book_source, required_books, config["dataset_release"])
    movies, movie_audit = extract_metadata(movie_source, required_movies, config["dataset_release"])
    rng = np.random.default_rng(seed)
    requests, outcomes, exclusions = [], [], []
    warm = set(data["target_train"][:, 1].tolist())
    counts = collections.Counter(int(item) for item in data["target_train"][:, 1])

    def compact(row):
        # A fixed, shared text budget keeps the two teachers' prompts identical.
        return {"title": row["title"][:120], "description": row["description"][:240],
                "categories": row["categories"][:80]}

    for user in candidates:
        chosen = sample_books(profiles[int(user)], books, data, rng)
        chosen = ([p for p in chosen if p[1] >= 4][:3] +
                  [p for p in chosen if p[1] < 4][:3])
        rows = [(int(item), int(rating)) for _, item, rating in validation[validation[:, 0] == user]
                if int(item) in warm and item_asin(data, int(item), "movie") in movies]
        rows = [rows[i] for i in rng.permutation(len(rows))][:6]
        # Diagnostic sample: only unequal observed ratings provide an A/B answer.
        pairs = [(a, b) for a, b in itertools.combinations(rows, 2) if a[1] != b[1]][:3]
        if len(chosen) < minimum_books or not pairs:
            exclusions.append(int(user))
            continue
        items = list(dict.fromkeys(item for a, b in pairs for item, _ in (a, b)))
        aliases = {item: f"m{i}" for i, item in enumerate(items)}
        content = {"profile_token": f"v{len(requests):03d}",
            "books": [{"alias": f"b{i}", "rating": rating,
                       **compact(books[item_asin(data, item, "book")])}
                      for i, (item, rating) in enumerate(chosen)],
            "movies": [{"alias": aliases[item], **compact(movies[item_asin(data, item, "movie")])}
                       for item in items],
            "pairs": [{"pair_id": f"p{i}", "a": aliases[a[0]], "b": aliases[b[0]]}
                      for i, (a, b) in enumerate(pairs)]}
        mapping = {"user_id": int(user), "movies": {v: k for k, v in aliases.items()},
                   "books": {f"b{i}": item for i, (item, _) in enumerate(chosen)}}
        request = audit_request(content, mapping, "original", int(user))
        requests.append(request)
        for i, (a, b) in enumerate(pairs):
            outcomes.append({"profile_token": content["profile_token"], "pair_id": f"p{i}",
                "rating_a": a[1], "rating_b": b[1], "correct_choice": "A" if a[1] > b[1] else "B",
                "popularity_choice": "A" if counts[a[0]] > counts[b[0]] else "B" if counts[a[0]] < counts[b[0]] else "tie"})
        if len(requests) == users:
            break
    if len(requests) < 2:
        raise ValueError("Not enough metadata-covered validation users; no LLM calls made")
    original = list(requests)
    for i, recipient in enumerate(original):
        donor = original[(i+1) % len(original)]
        content = {k: v for k, v in recipient["input"].items() if k != "request_id"}
        content["books"] = donor["input"]["books"]
        mapping = {**recipient["mapping"], "books": donor["mapping"]["books"]}
        requests.append(audit_request(content, mapping, "shuffled", donor["mapping"]["user_id"]))
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output/"requests.jsonl", requests)
    write_jsonl(output/"hidden_outcomes.jsonl", outcomes)
    write_jsonl(output/"metadata.jsonl", [{"domain": domain, **row} for domain, lookup in
        (("book", books), ("movie", movies)) for _, row in sorted(lookup.items())])
    selection = {"seed": seed, "requested_users": users, "candidate_users": candidates.tolist(),
        "minimum_books": minimum_books,
        "selected_users": [r["mapping"]["user_id"] for r in original], "excluded_users": exclusions,
        "book_metadata": book_audit, "movie_metadata": movie_audit,
        "selection_rule": "Seeded validation users; up to six metadata-covered warm rated Movies per user; first three unequal-rating combinations. Conditional diagnostic, not an unbiased catalogue sample.",
        "text_limits": {"books": 6, "movies": 6, "title_chars": 120, "description_chars": 240, "category_chars": 80},
        "control": "Cyclically assign the next selected user's rated Books to each recipient; keep Movie pairs fixed."}
    write_json(output/"selection.json", selection)
    manifest = {"purpose": "validation_teacher_audit_only", "never_training_labels": True,
        "dataset_sha256": benchmark["benchmark_sha256"], "prompt_version": PROMPT_VERSION,
        "prompt": PROMPT, "users": len(original), "requests": len(requests), "pairs_per_mode": len(outcomes),
        "model_selection": "Assess accuracy with coverage and matched-support results; inspect original versus shuffled profiles for personalisation. Use latency/memory to break a close result. This small pilot provides a practical choice, not statistical proof of superiority.",
        "files": seal_files(output, ["requests.jsonl", "hidden_outcomes.jsonl", "metadata.jsonl", "selection.json"])}
    write_json(output/"manifest.json", manifest)
    return manifest


# 2. CACHE ANSWERS WITHOUT READING THE HIDDEN MOVIE RATINGS

def generate_audit(directory: Path, runtime_file: Path, max_requests: int = 16,
                   context_tokens: int = 8192, max_output_tokens: int = 512) -> dict:
    if max_requests < 1 or max_output_tokens < 1 or context_tokens < max_output_tokens+256:
        raise ValueError("Set positive request/output limits and a sufficient context")
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("purpose") != "validation_teacher_audit_only" or not manifest.get("never_training_labels"):
        raise ValueError("Expected a separate validation-only audit")
    if manifest.get("prompt") != PROMPT or manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Audit prompt changed; prepare a separate audit before generating answers")
    # Hash outcomes for integrity; do not decode their values during generation.
    check_files(directory, manifest["files"])
    runtime = json.loads(runtime_file.read_text())
    model, base = runtime["model_identifier"], runtime["server_url"]
    details = load_local_teacher(runtime_file, model, base, context_tokens)
    output = directory/model
    output.mkdir(exist_ok=True)
    config = {**details, "model": model, "context_tokens": context_tokens,
        "max_output_tokens": max_output_tokens, "temperature": 0, "seed": 2020,
        "reasoning_effort": "none", "audit_manifest_sha256": file_sha256(directory/"manifest.json")}
    config_path = output/"generation.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Saved audit model/settings differ; do not overwrite its cache")
    write_json(config_path, config)
    requests = read_jsonl(directory/"requests.jsonl")
    if (output/"summary.json").exists():
        saved_summary = json.loads((output/"summary.json").read_text())
        if saved_summary["complete"]:
            check_files(output, saved_summary["files"])
    completed = read_jsonl(output/"responses.jsonl") if (output/"responses.jsonl").exists() else []
    seen = {r["request_id"] for r in completed}
    if len(seen) != len(completed) or not seen <= {r["request_id"] for r in requests}:
        raise ValueError("Duplicate or foreign cached audit response")
    for row in completed:
        request = next(r for r in requests if r["request_id"] == row["request_id"])
        if row["mode"] != request["mode"]:
            raise ValueError("Cached audit profile mode changed")
        if row["valid"]:
            normalize_response(row["response"], request)
    calls = 0
    for request in requests:
        if request["request_id"] in seen:
            continue
        if calls >= max_requests:
            break
        row = {"request_id": request["request_id"], "mode": request["mode"], "valid": False}
        started = time.monotonic()
        calls += 1
        try:
            bound = len(public_prompt(request).encode()) + max_output_tokens + 256
            if bound > context_tokens:
                raise ValueError("Prompt exceeds conservative context admission bound")
            raw = local_json(base, "/v1/chat/completions", preference_payload(request, model, max_output_tokens))
            row["raw_response"] = raw
            choice = raw["choices"][0]
            if raw.get("model") != model or choice["finish_reason"] != "stop":
                raise ValueError("Wrong model identifier or incomplete generation")
            response = json.loads(choice["message"]["content"])
            normalize_response(response, request)
            row.update(valid=True, response=response, usage=raw.get("usage", {}))
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            row["error"] = str(exc)
        row["elapsed_seconds"] = round(time.monotonic()-started, 3)
        completed.append(row)
        write_jsonl(output/"responses.jsonl", completed)
        print(f"{model}: {len(completed)}/{len(requests)} responses, valid={row['valid']}, {row['elapsed_seconds']}s", flush=True)
    summary = {"model": model, "complete": len(completed) == len(requests), "requests": len(requests),
        "responses": len(completed), "valid": sum(r["valid"] for r in completed),
        "files": seal_files(output, ["generation.json", "responses.jsonl"])}
    write_json(output/"summary.json", summary)
    return summary


# 3. LOCK BOTH MODELS' ANSWERS, THEN SCORE THE SAME VALIDATION PAIRS

def score_choices(requests: list[dict], responses: list[dict], outcomes: list[dict], mode: str) -> dict:
    truth = {(r["profile_token"], r["pair_id"]): r["correct_choice"] for r in outcomes}
    answers = {r["request_id"]: r for r in responses if r["valid"]}
    choices, correct, covered, total, valid = collections.Counter(), 0, 0, 0, 0
    for request in requests:
        if request["mode"] != mode:
            continue
        total += len(request["input"]["pairs"])
        response = answers.get(request["request_id"])
        if response is None:
            continue
        valid += 1
        for pair in response["response"]["preferences"]:
            choice = pair["choice"]
            choices[choice] += 1
            if choice in {"A", "B"}:
                covered += 1
                correct += int(choice == truth[(request["input"]["profile_token"], pair["pair_id"])])
    return {"pairs": total, "valid_profiles": valid, "answered_pairs": covered, "correct_pairs": correct,
        "coverage": covered/total if total else 0, "accuracy_when_answered": correct/covered if covered else None,
        "correct_fraction_of_all_pairs": correct/total if total else 0, "choices": dict(choices)}


def compare_audits(directory: Path, models: list[str]) -> dict:
    if not models or len(models) != len(set(models)):
        raise ValueError("Provide one or more distinct local model identifiers")
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("purpose") != "validation_teacher_audit_only" or not manifest.get("never_training_labels"):
        raise ValueError("Expected a separate validation-only audit")
    check_files(directory, manifest["files"])
    locked = {"audit_manifest_sha256": file_sha256(directory/"manifest.json"), "models": {}}
    all_responses = {}
    for model in models:
        output = directory/model
        summary = json.loads((output/"summary.json").read_text())
        check_files(output, summary["files"])
        if not summary["complete"]:
            raise ValueError(f"Finish {model}'s audit before scoring")
        locked["models"][model] = summary["files"]
        all_responses[model] = read_jsonl(output/"responses.jsonl")
    lock_path = directory/"prediction_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != locked:
        raise ValueError("Predictions changed after locking")
    write_json(lock_path, locked)
    # Outcome values are opened only after every selected prediction file is fixed.
    requests = read_jsonl(directory/"requests.jsonl")
    outcomes = read_jsonl(directory/"hidden_outcomes.jsonl")
    results = []
    for model, responses in all_responses.items():
        for mode in ("original", "shuffled"):
            subset = [r for r in responses if r["mode"] == mode]
            results.append({"model": model, "profile": mode,
                **score_choices(requests, responses, outcomes, mode),
                "median_seconds_per_profile": float(np.median([r["elapsed_seconds"] for r in subset])) if subset else None,
                "valid_json_fraction": sum(r["valid"] for r in subset)/len(subset) if subset else None})
    truth = {(r["profile_token"], r["pair_id"]): r["correct_choice"] for r in outcomes}
    choices_by_model, personalization = {}, {}
    for model, responses in all_responses.items():
        lookup = {r["request_id"]: r for r in responses if r["valid"]}
        choices_by_model[model] = {(request["input"]["profile_token"], pair["pair_id"]): pair["choice"]
            for request in requests if request["mode"] == "original" and request["request_id"] in lookup
            for pair in lookup[request["request_id"]]["response"]["preferences"] if pair["choice"] in {"A", "B"}}
        shuffled = {(request["input"]["profile_token"], pair["pair_id"]): pair["choice"]
            for request in requests if request["mode"] == "shuffled" and request["request_id"] in lookup
            for pair in lookup[request["request_id"]]["response"]["preferences"] if pair["choice"] in {"A", "B"}}
        original = choices_by_model[model]
        shared = set(original) & set(shuffled)
        personalization[model] = {"common_answered_pairs": len(shared),
            "changed_choices": sum(original[key] != shuffled[key] for key in shared),
            "original_accuracy": sum(original[key] == truth[key] for key in shared)/len(shared) if shared else None,
            "shuffled_accuracy": sum(shuffled[key] == truth[key] for key in shared)/len(shared) if shared else None}
        all_choices = {mode: {(request["input"]["profile_token"], pair["pair_id"]): pair["choice"]
            for request in requests if request["mode"] == mode and request["request_id"] in lookup
            for pair in lookup[request["request_id"]]["response"]["preferences"]}
            for mode in ("original", "shuffled")}
        common_valid = set(all_choices["original"]) & set(all_choices["shuffled"])
        personalization[model].update(valid_pairs_in_both_profiles=len(common_valid),
            changed_choices_including_abstentions=sum(all_choices["original"][key] != all_choices["shuffled"][key] for key in common_valid),
            original_only_answered=len(set(original)-set(shuffled)),
            shuffled_only_answered=len(set(shuffled)-set(original)))
    common = set.intersection(*(set(rows) for rows in choices_by_model.values()))
    common_support = {"pairs": len(common), "models": {model: {
        "correct_pairs": sum(rows[key] == truth[key] for key in common),
        "accuracy": sum(rows[key] == truth[key] for key in common)/len(common) if common else None}
        for model, rows in choices_by_model.items()}}
    # Cheap controls see the same supplied text and the same target pairs.
    originals = [r for r in requests if r["mode"] == "original"]
    text_pairs = {(r["request_id"], r["pair_id"]): r for r in text_teacher(originals)}
    text_responses, popularity_responses = [], []
    for request in originals:
        preferences, popular = [], []
        for pair in request["input"]["pairs"]:
            label = text_pairs.get((request["request_id"], pair["pair_id"]))
            choice = "tie" if label is None else "A" if label["preferred_movie_id"] == request["mapping"]["movies"][pair["a"]] else "B"
            preferences.append({"pair_id": pair["pair_id"], "choice": choice})
            truth = next(r for r in outcomes if r["profile_token"] == request["input"]["profile_token"] and r["pair_id"] == pair["pair_id"])
            popular.append({"pair_id": pair["pair_id"], "choice": truth["popularity_choice"]})
        for destination, values in ((text_responses, preferences), (popularity_responses, popular)):
            destination.append({"request_id": request["request_id"], "valid": True,
                "response": {"request_id": request["request_id"], "preferences": values}})
    for name, responses in (("tfidf_text", text_responses), ("training_popularity", popularity_responses)):
        results.append({"model": name, "profile": "original", **score_choices(requests, responses, outcomes, "original")})
    report = {"purpose": "validation_teacher_audit_only", "users": manifest["users"],
        "comparison_scope": "local teachers versus text, popularity and shuffled-profile controls",
        "prediction_lock_sha256": file_sha256(lock_path), "results": results,
        "common_decisive_support": common_support, "profile_shuffle_control": personalization,
        "label_balance": dict(collections.Counter(row["correct_choice"] for row in outcomes)),
        "position_baselines": {f"always_{choice}": sum(row["correct_choice"] == choice for row in outcomes)/len(outcomes)
                               for choice in ("A", "B")},
        "limitations": ["Small development pilot; pairs within a user are correlated.",
            "Pairs are conditioned on unequal observed ratings, not randomly sampled catalogue choices.",
            "Candidates are Movies/TV products the validation user actually rated; this is not full-catalogue retrieval.",
            "Prompt text is shortened identically for both models; no reasoning mode, one seed.",
            "No test outcomes were used and no audit label may enter bridge training.",
            "Teacher pair accuracy does not establish an improvement in DUPGT MAE/RMSE."]}
    write_json(directory/"comparison.json", report)
    import csv
    fields = ["model", "profile", "pairs", "answered_pairs", "correct_pairs", "coverage", "accuracy_when_answered", "correct_fraction_of_all_pairs", "median_seconds_per_profile", "valid_json_fraction"]
    with (directory/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return report
