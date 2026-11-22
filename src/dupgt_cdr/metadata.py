"""Clean catalogue text and measure its coverage of the saved rating datasets."""
from __future__ import annotations

import ast
import collections
import csv
import gzip
import hashlib
import html
import json
import re
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import orjson

from .common import file_sha256, resolve, write_json

CLEANING_VERSION = "catalogue-content-v2"
ROOT_CATEGORIES = {"books", "movies & tv", "movies and tv", "movies", "tv", "dvd", "vhs", "blu-ray", "formats"}
BIOGRAPHY = re.compile(r"\b(?:is|was|serves as) (?:an? |the )?(?:[\w-]+ ){0,3}(?:professor|founder|publisher|journalist|illustrator|author)\b|\b(?:he|she|they) (?:lives?|resides?)\b|\bborn (?:in|on)\b", re.I)
SALES = re.compile(r"\b(?:ships?|shipping|seller|condition|factory sealed|customer review|five stars?|isbn|new york times|bestsell\w*|highly recommend|excellent quality)\b", re.I)
CONTENT = re.compile(r"\b(?:story|novel|follows|explores|exploring|adventure|journey|mystery|plot|narrative|characters?|set in|tells|chronicles|guide|learn|techniques|history|account|examines|investigates|introduces|explains|covers|topics|science|theory)\b", re.I)
SUBJECT_KEYS = {"genre", "genres", "subject", "subjects", "theme", "themes"}
BIOGRAPHY_HINT = re.compile(r"\b(?:professor|cofounder|co-founder|presentation coach|journalist|medical editor|health writer|illustrator|author of|has written|has authored|university|ph\.?d)\b", re.I)
NARRATIVE = re.compile(r"\b(?:this|the|a|an) (?:book|novel|film|movie|story|guide|series|volume|collection|documentary)\b|\b(?:follows|explores|tells the|set in|chronicles|examines|investigates|introduces|explains|covers|teaches|learn how|step-by-step|plot|protagonist|discovers?|struggles?|travels?|confronts?|traces)\b", re.I)
TESTIMONIAL = re.compile(r"^\W*(?:as one of|i |we |my |our |highly recommended)|[—–]\s*[A-Z][a-z]+", re.I)
AGGREGATE = re.compile(r"\b\d(?:\.\d+)?\s*(?:out\s+of\s+(?:5|five)|stars?)\b|\b[\d,]+\s+(?:customer\s+)?(?:ratings|reviews)\b", re.I)


# 1. READ THE ORIGINAL METADATA WITHOUT KEEPING THE HUGE DOWNLOAD

def source_urls(release: str) -> dict:
    if release == "2014":
        base = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_"
        return {name: base+category+".json.gz" for name, category in (("books", "Books"), ("movies", "Movies_and_TV"))}
    if release == "2023":
        base = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_"
        return {name: base+category+".jsonl" for name, category in (("books", "Books"), ("movies", "Movies_and_TV"))}
    raise ValueError("Choose metadata release 2014 or 2023")


@contextmanager
def source_stream(source: str, doh: bool = False):
    if not source.startswith("https://"):
        path = resolve(source)
        with (gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")) as handle:
            yield handle
        return
    # curl streams to Python; no full archive or network cache is saved.
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "30",
               "--speed-time", "90", "--speed-limit", "1024"]
    if doh:
        command += ["--doh-url", "https://cloudflare-dns.com/dns-query", "--resolve", "cloudflare-dns.com:443:1.1.1.1"]
    with subprocess.Popen(command+[source], stdout=subprocess.PIPE) as process:
        try:
            if source.endswith(".gz"):
                with gzip.GzipFile(fileobj=process.stdout) as handle:
                    yield handle
            else:
                yield process.stdout
            if process.wait() != 0:
                raise OSError("Metadata download failed; the incomplete output was not accepted")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def text_parts(value) -> list[str]:
    if isinstance(value, list):
        return [part for entry in value for part in text_parts(entry)]
    if not isinstance(value, str):
        return []
    value = re.sub(r"</p>|</div>|<br\s*/?>|\n+", "\n", value, flags=re.I)
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return [" ".join(part.split()) for part in value.splitlines() if part.strip()]


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")


def item_key(row: dict, release: str) -> str | None:
    declared = row.get("dataset_release")
    if declared is not None and declared != release:
        raise ValueError("Metadata release differs from the benchmark")
    # Canonical extracts declare their release; raw 2023 records require parent_asin.
    value = row.get("asin") if declared == release or release == "2014" else row.get("parent_asin")
    return value if isinstance(value, str) and value else None


# 2. SELECT CONTENT TEXT; NEVER GENERATE A SYNOPSIS OR USE RATING AGGREGATES

def clean_record(row: dict, release: str, domain: str) -> tuple[dict | None, dict]:
    asin = item_key(row, release)
    title = " ".join(text_parts(row.get("title")))
    passages = [("description", part) for part in text_parts(row.get("description"))]
    if release == "2023":
        passages += [("features", part) for part in text_parts(row.get("features"))]
    rejected = collections.Counter()
    candidates = []
    for position, (field, passage) in enumerate(passages):
        # Keep sentence boundaries so a later synopsis can survive an author biography.
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", passage):
            words = re.findall(r"\b[a-zA-Z]+\b", sentence)
            cues = len(CONTENT.findall(sentence))
            if len(words) < 12:
                rejected["short_or_format_only"] += 1
            elif AGGREGATE.search(sentence):
                rejected["rating_or_review_aggregate"] += 1
            elif BIOGRAPHY.search(sentence) or (BIOGRAPHY_HINT.search(sentence) and not NARRATIVE.search(sentence)):
                rejected["author_biography"] += 1
            elif (SALES.search(sentence) or TESTIMONIAL.search(sentence)) and not NARRATIVE.search(sentence):
                rejected["sales_or_review_text"] += 1
            elif sentence.count("!") > 3:
                rejected["promotional_text"] += 1
            else:
                score = 3*cues + min(len(words), 80)/80
                candidates.append((score, position, field, sentence))
    categories = list(dict.fromkeys(part for part in text_parts(row.get("categories"))
                                   if part.lower() not in ROOT_CATEGORIES))
    details = row.get("details")
    if isinstance(details, dict):
        categories += [f"{key}: {' '.join(text_parts(value))}" for key, value in details.items()
                       if key.strip().lower() in SUBJECT_KEYS and text_parts(value)]
    diagnostic = {"asin": asin, "domain": domain, "title": shorten(title, 180),
        "original_description_chars": sum(len(part) for part in text_parts(row.get("description"))),
        "discarded_passages": dict(rejected), "before_excerpt": shorten(" ".join(text_parts(row.get("description"))), 220)}
    if not asin or not title or not candidates:
        diagnostic.update(accepted=False, reason="missing_id_or_title" if not asin or not title else "no_content_passage")
        return None, diagnostic
    # Rank content before shortening; retain source wording without an LLM rewrite.
    candidates.sort(key=lambda value: (-value[0], value[1]))
    selected, seen = [], set()
    for _, position, field, sentence in candidates:
        if sentence.lower() not in seen:
            selected.append((position, field, sentence))
            seen.add(sentence.lower())
        if sum(len(part[2]) for part in selected) >= 600:
            break
    selected.sort(key=lambda part: part[0])
    description = shorten(" ".join(part[2] for part in selected), 600)
    cleaned = {"dataset_release": release, "asin": asin, "title": shorten(title, 180),
        "description": description, "categories": shorten(" ".join(dict.fromkeys(categories)), 160)}
    diagnostic.update(accepted=True, reason="content_passage", text_fields=sorted({part[1] for part in selected}),
                      cleaned_description_chars=len(description), after_excerpt=shorten(description, 220))
    return cleaned, diagnostic


# 3. KEEP MATCHING ITEM IDS, RESOLVE DUPLICATES, AND SAVE BEFORE/AFTER COUNTS

def clean_domain(source: str, required: set[str], release: str, domain: str,
                 output: Path, doh: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    destination = output/f"{domain}.jsonl"
    if destination.exists():
        report = json.loads((output/f"{domain}_summary.json").read_text())
        if (report["cleaning_version"] != CLEANING_VERSION or report["source"] != source or
                report["dataset_release"] != release or report["domain"] != domain):
            raise ValueError("Existing cleaning settings/source differ; choose another directory")
        if not source.startswith("https://") and report.get("input_sha256") != file_sha256(resolve(source)):
            raise ValueError("Local metadata source changed")
        for name, digest in report["files"].items():
            if file_sha256(output/name) != digest:
                raise ValueError("Saved cleaned metadata changed")
        if report["required_ids_sha256"] != hashlib.sha256("\n".join(sorted(required)).encode()).hexdigest():
            raise ValueError("Requested metadata IDs differ; choose another directory")
        print(f"Using verified {domain} metadata: {report['accepted_items']:,} items", flush=True)
        return report
    accepted, seen, ambiguous, diagnostics = {}, {}, set(), {}
    counts = collections.Counter()
    digest = hashlib.sha256()
    source_path = output/f"{domain}_source.jsonl.gz"
    partial = source_path.with_suffix(source_path.suffix+".partial")
    retain_snapshot = source.startswith("https://")
    input_sha256 = None if retain_snapshot else file_sha256(resolve(source))
    print(f"Reading {release}/{domain}: {len(required):,} benchmark IDs", flush=True)
    snapshot = gzip.open(partial, "wt", encoding="utf-8", compresslevel=1) if retain_snapshot else nullcontext(None)
    with snapshot as retained, source_stream(source, doh) as handle:
        for count, line in enumerate(handle, 1):
            digest.update(line)
            counts["rows_scanned"] += 1
            if count % 250_000 == 0:
                print(f"{release}/{domain}: {count:,} rows; {len(accepted):,} accepted", flush=True)
            try:
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError:
                    row = ast.literal_eval(line.decode("utf-8")) if release == "2014" else None
                if not isinstance(row, dict):
                    raise ValueError("Expected a metadata dictionary")
                asin = item_key(row, release)
                if asin not in required or asin in ambiguous:
                    continue
                # Save only descriptive source fields; no prices/reviews/aggregate outcomes.
                details = row.get("details")
                raw = {"dataset_release": release, "asin": asin, "title": row.get("title"),
                    "description": row.get("description"), "categories": row.get("categories"),
                    "features": row.get("features", []),
                    "details": {k: v for k, v in details.items() if k.strip().lower() in SUBJECT_KEYS} if isinstance(details, dict) else {}}
                signature = hashlib.sha256(orjson.dumps(raw, option=orjson.OPT_SORT_KEYS)).hexdigest()
                if retained is not None:
                    retained.write(orjson.dumps(raw).decode()+"\n")
                if asin in seen:
                    if signature != seen[asin]:
                        ambiguous.add(asin)
                        accepted.pop(asin, None)
                        diagnostics[asin] = {"asin": asin, "domain": domain, "accepted": False, "reason": "conflicting_metadata"}
                    else:
                        counts["identical_duplicates"] += 1
                    continue
                seen[asin] = signature
                cleaned, diagnostic = clean_record(raw, release, domain)
                diagnostics[asin] = diagnostic
                if cleaned:
                    accepted[asin] = cleaned
            except (ValueError, SyntaxError, TypeError, UnicodeError, RecursionError):
                counts["malformed_rows"] += 1
    if retain_snapshot:
        partial.replace(source_path)
    # A duplicate-conflict list also prevents reuse of rejected source rows.
    names = [f"{domain}.jsonl", f"{domain}_quality.jsonl"]
    if retain_snapshot:
        names.append(source_path.name)
    for name, rows in ((names[0], accepted), (names[1], diagnostics)):
        with (output/name).open("w") as handle:
            for _, row in sorted(rows.items()):
                handle.write(orjson.dumps(row).decode()+"\n")
    report = {"dataset_release": release, "domain": domain, "source": source,
        "cleaning_version": CLEANING_VERSION, "code_sha256": file_sha256(Path(__file__)),
        "input_sha256": input_sha256,
        "source_snapshot": str(source_path if retain_snapshot else resolve(source)),
        "required_ids": len(required), "required_ids_sha256": hashlib.sha256("\n".join(sorted(required)).encode()).hexdigest(),
        "matched_items": len(seen), "accepted_items": len(accepted), "rejected_items": len(seen)-len(accepted),
        "missing_items": len(required)-len(seen), "ambiguous_asins": sorted(ambiguous),
        "decoded_stream_sha256": digest.hexdigest(), "counts": dict(counts),
        "reasons": dict(collections.Counter(row["reason"] for row in diagnostics.values())),
        "files": {name: file_sha256(output/name) for name in names},
        "limitations": "Deterministic text-quality heuristics, not human-verified plot labels or proof of teacher usefulness."}
    write_json(output/f"{domain}_summary.json", report)
    print(f"{release}/{domain}: {len(seen):,} matched, {len(accepted):,} content-covered", flush=True)
    return report


# 4. COUNT USABLE BOOK HISTORIES WITHOUT LOOKING AT HIDDEN MOVIE RATINGS

def coverage_report(data: dict, books: set[str], movies: set[str], minimum_books: int = 3) -> dict:
    from .experiments import source_only_user_ids
    book_mask = np.zeros(len(data["source_item_ids"])+1, dtype=bool)
    book_mask[1:] = np.isin(data["source_item_ids"], list(books))
    source_only = source_only_user_ids(data)
    populations = {"source_only_training": source_only, "shared_training": data["shared_train_users"],
                   "validation": data["validation_users"], "test": data["test_users"]}
    groups = {}
    for name, users in populations.items():
        # Source histories are permitted; no validation/test Movie outcomes are read.
        covered = book_mask[data["histories"][users]]
        totals = covered.sum(axis=1)
        groups[name] = {"users": len(users), "with_any_clean_book": int((totals >= 1).sum()),
            "with_at_least_three_clean_books": int((totals >= 3).sum()),
            "with_at_least_five_clean_books": int((totals >= 5).sum()),
            "eligible_at_minimum": int((totals >= minimum_books).sum()),
            "eligible_with_both_branches": int(((totals >= minimum_books) & covered[:, :20].any(axis=1) & covered[:, 20:].any(axis=1)).sum())}
    warm = np.unique(data["target_train"][:, 1])
    target_ids = data["target_item_ids"][warm-len(data["source_item_ids"])-1]
    warm_covered = int(np.isin(target_ids, list(movies)).sum())
    return {"minimum_clean_books_per_profile": minimum_books, "populations": groups,
        "warm_movie_items": len(warm), "warm_movies_with_content": warm_covered,
        "enough_for_200_user_10_movie_pilot": groups["source_only_training"]["eligible_at_minimum"] >= 200 and warm_covered >= 10,
        "scope": "Metadata and source-history availability only. Does not establish useful LLM labels or improved rating prediction."}


# 5. RUN THE SAME PREPROCESSING FOR EITHER SAVED BENCHMARK

def prepare_metadata(config: dict, output_dir: str, book_source: str | None = None,
                     movie_source: str | None = None, doh: bool = False, minimum_books: int = 3) -> dict:
    from .data import get_benchmark
    if minimum_books < 1:
        raise ValueError("minimum_books must be positive")
    data, benchmark = get_benchmark(config)
    release = config["dataset_release"]
    defaults = source_urls(release)
    output = resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    before = {"dataset_sha256": benchmark["benchmark_sha256"], "dataset_release": release,
              "cleaning_version": CLEANING_VERSION, "minimum_books": minimum_books}
    if (output/"metadata_manifest.json").exists():
        saved = json.loads((output/"metadata_manifest.json").read_text())
        if any(saved[key] != value for key, value in before.items()):
            raise ValueError("Existing metadata belongs to different inputs/settings")
    # All benchmark Book IDs and warm Movie IDs are eligible; no outcome-based filtering.
    warm = np.unique(data["target_train"][:, 1])
    movie_ids = data["target_item_ids"][warm-len(data["source_item_ids"])-1]
    book = clean_domain(book_source or defaults["books"], set(data["source_item_ids"].tolist()), release, "books", output, doh)
    movie = clean_domain(movie_source or defaults["movies"], set(movie_ids.tolist()), release, "movies", output, doh)
    def keys(name):
        with (output/f"{name}.jsonl").open() as handle:
            return {json.loads(line)["asin"] for line in handle}
    coverage = coverage_report(data, keys("books"), keys("movies"), minimum_books)
    write_json(output/"coverage.json", coverage)
    manifest = {**before, "status": "ready_for_teacher_check" if coverage["enough_for_200_user_10_movie_pilot"] else "insufficient_metadata",
        "books": book, "movies": movie, "coverage": coverage,
        "files": {name: file_sha256(output/name) for name in ("books.jsonl", "movies.jsonl", "coverage.json", "books_summary.json", "movies_summary.json")},
        "benchmark_unchanged": file_sha256(resolve(config["prepared_dir"])/"benchmark.npz") == benchmark["benchmark_sha256"]}
    write_json(output/"metadata_manifest.json", manifest)
    with (output/"coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["population", *next(iter(coverage["populations"].values()))])
        writer.writeheader()
        for name, row in coverage["populations"].items():
            writer.writerow({"population": name, **row})
    print(json.dumps(coverage, indent=2), flush=True)
    return manifest
