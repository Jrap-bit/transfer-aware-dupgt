"""Build the small data bundle the live demo reads (adds files under demo/bundle; changes nothing else).

    .venv/bin/python demo/build_bundle.py

Writes:
  books.csv.gz     book item id, asin, title, category, rating count (searchable catalogue)
  movies.csv.gz    movie item id, asin, title, category, rating count, movie average
  personas.json     one-click example users built from popular books in a category
  replay.json       real held-out test users with the ORIGINAL saved predictions of every method
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo/bundle"
BENCH = ROOT / "data/processed/amazon2023/full/benchmark.npz"
META = ROOT / "data/metadata/amazon2023/clean"
SAVED = ROOT / "outputs/amazon2023"
MIN_MOVIE_RATINGS = 100     # only recommend movies with a reasonable number of ratings
MIN_BOOK_RATINGS = 5        # searchable books
REPLAY_USERS = 900


def titles(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as handle:
        for line in handle:
            d = json.loads(line)
            if d.get("title"):
                rows.append((d["asin"], d["title"].strip(), (d.get("categories") or "").strip()))
    return pd.DataFrame(rows, columns=["asin", "title", "category"]).drop_duplicates("asin")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(BENCH)
    source_ids, target_ids = a["source_item_ids"], a["target_item_ids"]
    n_source = len(source_ids)
    src, tgt = a["source_train"], a["target_train"]

    # Books: internal id i (1..n_source) <-> source_item_ids[i-1]
    book_counts = np.bincount(src[:, 1], minlength=n_source + 1)
    book_mean = np.bincount(src[:, 1], weights=src[:, 2], minlength=n_source + 1) / np.maximum(book_counts, 1)
    books = pd.DataFrame({"item": np.arange(1, n_source + 1), "asin": source_ids,
                          "n_ratings": book_counts[1:], "avg_rating": book_mean[1:].round(2)})
    books = books.merge(titles(META / "books.jsonl"), on="asin", how="inner")
    books = books[books.n_ratings >= MIN_BOOK_RATINGS].sort_values("n_ratings", ascending=False)
    books.to_csv(OUT / "books.csv.gz", index=False)

    # Movies: internal id n_source+1+j <-> target_item_ids[j]
    size = n_source + len(target_ids) + 1
    movie_counts = np.bincount(tgt[:, 1], minlength=size)
    movie_mean = np.bincount(tgt[:, 1], weights=tgt[:, 2], minlength=size) / np.maximum(movie_counts, 1)
    ids = np.arange(n_source + 1, size)
    movies = pd.DataFrame({"item": ids, "asin": target_ids, "n_ratings": movie_counts[ids],
                           "movie_mean": movie_mean[ids]})
    movies = movies.merge(titles(META / "movies.jsonl"), on="asin", how="left")
    movies["title"] = movies["title"].fillna("(untitled Movies & TV item " + movies["asin"] + ")")
    movies["category"] = movies["category"].fillna("")
    movies["candidate"] = (movies.n_ratings >= MIN_MOVIE_RATINGS) & ~movies.title.str.startswith("(untitled")
    movies.to_csv(OUT / "movies.csv.gz", index=False)
    print(f"books searchable: {len(books):,}; movies: {len(movies):,}, candidates: {int(movies.candidate.sum()):,}")

    personas = make_personas(books)
    (OUT / "personas.json").write_text(json.dumps(personas, indent=1))
    replay = make_replay(a, books, movies)
    (OUT / "replay.json").write_text(json.dumps(replay))
    print(f"personas: {len(personas)}, replay users: {len(replay['users'])}")


# Hand-picked, recognisable books (matched to the most-rated catalogue title starting with each string).
# (name, loved, disliked, note). Order = shown order; readers with a note are flagged as good showcases.
# Notes come from measuring how differently the 10% models treat each reader (demo/bundle/persona_spread.json).
PERSONAS = [
    ("Comedy memoirs",
     ["Born a Crime", "Bossypants", "Yes Please", "Is Everyone Hanging Out", "Me Talk Pretty", "Let's Pretend This Never"],
     ["The Selfish Gene", "Mastering the Art"], "Models disagree most: DUPGT −0.47★, semantic −0.60★, blend −0.24★"),
    ("Progressive politics",
     ["Becoming", "Caste", "The Audacity of Hope", "Fear: Trump", "Fire and Fury", "How to Be an Antiracist"],
     ["Liars, Leakers", "The Case for Trump"], "Fallback trusts the bridge least (w ≈ 0.58)"),
    ("Conservative politics",
     ["Liars, Leakers", "The Case for Trump", "Killing Lincoln", "Hillbilly Elegy", "American Marxism", "Unbroken"],
     ["Fear: Trump", "Fire and Fury"], "Compare with progressive: different gate and shift"),
    ("Popular science",
     ["A Brief History of Time", "Astrophysics for People", "Sapiens", "The Gene", "The Selfish Gene", "The Immortal Life of Henrietta"],
     ["Twilight", "Fifty Shades of Grey"], ""),
    ("Comics & graphic novels",
     ["Watchmen", "Maus", "The Walking Dead", "Batman: The Killing", "Saga", "V for Vendetta"],
     ["Redeeming Love", "The Purpose Driven Life"], "Most different top-5 lists between models"),
    ("Hard sci-fi",
     ["Dune", "Foundation", "Neuromancer", "The Left Hand of Darkness", "The Three-Body", "Project Hail Mary"],
     ["Redeeming Love", "Magnolia Table"], "Large model differences on famous movies"),
    ("Parent reading picture books",
     ["The Very Hungry Caterpillar", "Giraffes Can't Dance", "The Wonky Donkey", "The Day the Crayons Quit", "Goodnight Moon", "Where the Wild Things Are"],
     ["Gone Girl", "The Silent Patient"], "Best for the live LLM step"),
    ("True crime",
     ["In Cold Blood", "I'll Be Gone in the Dark", "Helter Skelter", "Killers of the Flower Moon", "Mindhunter", "The Stranger Beside Me"],
     ["The Very Hungry Caterpillar", "Redeeming Love"], ""),
    ("Young adult",
     ["The Hunger Games", "Divergent", "Twilight", "The Fault in Our Stars", "Percy Jackson", "Harry Potter and the Deathly"],
     ["Good to Great", "In Cold Blood"], ""),
    ("Stephen King horror",
     ["It: A Novel", "The Shining", "Pet Sematary", "Misery", "Carrie", "The Stand"],
     ["Redeeming Love", "The Notebook"], ""),
    ("Thriller & mystery fan",
     ["Gone Girl", "The Girl on the Train", "The Silent Patient", "The Girl with the Dragon Tattoo", "Big Little Lies", "The Da Vinci Code"],
     ["Fifty Shades of Grey", "Twilight"], ""),
    ("Business & self-improvement",
     ["The 7 Habits", "Rich Dad Poor Dad", "Good to Great", "The Lean Startup", "Outliers", "Thinking, Fast and Slow"],
     ["Twilight", "The Notebook"], ""),
    ("Fantasy & sci-fi reader",
     ["Harry Potter and the Sorcerer", "The Hobbit", "Ender's Game", "Dune", "The Name of the Wind", "The Martian"],
     ["Fifty Shades of Grey", "Redeeming Love"], ""),
    ("History buff",
     ["Unbroken", "The Devil in the White City", "Killing Lincoln", "1776", "The Boys in the Boat", "Team of Rivals"],
     ["Twilight", "Fifty Shades of Grey"], ""),
    ("Romance reader",
     ["The Notebook", "Me Before You", "Outlander", "The Fault in Our Stars", "Twilight", "Redeeming Love"],
     ["Killing Lincoln", "Dune"], ""),
    ("Faith & inspiration reader",
     ["The Shack", "Heaven is for Real", "The Purpose Driven Life", "Jesus Calling", "Redeeming Love"],
     ["Fifty Shades of Grey", "Gone Girl"], ""),
]


def make_personas(books: pd.DataFrame) -> list[dict]:
    lower = books.title.str.lower()
    def find(prefix):
        hit = books[lower.str.startswith(prefix.lower())].head(1)   # books is sorted by rating count
        return None if hit.empty else hit.iloc[0]
    result = []
    for name, loved, disliked, note in PERSONAS:
        picks = []
        for titles, stars in ((loved, 5), (disliked, 2)):
            for t in titles:
                row = find(t)
                if row is not None:
                    picks.append({"item": int(row["item"]), "title": row["title"], "stars": stars})
        result.append({"name": name, "note": note, "picks": picks})
    return result


# label -> (folder, column) per overlap; all are the original saved runs behind the slides (seed 2020).
def overlap_methods(g: str) -> dict:
    return {
        "Movie average": (f"overlap_seeds/{g}/seed_2020/dupgt", "movie_mean"),
        "DUPGT": (f"overlap_seeds/{g}/seed_2020/dupgt", "dupgt"),
        "+ Semantic gate": (f"overlap_seeds/{g}/seed_2020/semantics", "dupgt"),
        "+ LLM teacher": (f"overlap_seeds/{g}/seed_2020/semantics_llm_w0.01", "dupgt"),
        "+ Shuffled teacher": (f"overlap_seeds/{g}/seed_2020/semantics_shuffled_w0.01", "dupgt"),
        "Probabilistic bridge": (f"probabilistic_bridge/{g}/seed_2020/probabilistic_g0.1", "dupgt"),
        "Ensemble mean": (f"uncertainty_ensemble/{g}/seed_2020/blend", "ensemble_mean"),
        "Constant blend": (f"uncertainty_ensemble/{g}/seed_2020/blend", "constant_blend"),
        "Uncertainty blend": (f"uncertainty_ensemble/{g}/seed_2020/blend", "uncertainty_blend"),
    }


OVERLAPS = {
    "1%": overlap_methods("overlap_0.01"),
    "10%": overlap_methods("overlap_0.10"),
    "100%": {  # full overlap: single seed, no fallback runs (as in the slides' all-users table)
        "Movie average": ("semantic_transfer/full_seed2020/dupgt", "movie_mean"),
        "DUPGT": ("semantic_transfer/full_seed2020/dupgt", "dupgt"),
        "+ Semantic gate": ("semantic_transfer/full_seed2020/semantics", "dupgt"),
        "+ LLM teacher": ("semantic_transfer/full_seed2020/semantics_llm", "dupgt"),
        "+ Shuffled teacher": ("semantic_transfer/full_seed2020/semantics_shuffled", "dupgt"),
    },
}


def load_overlap(methods: dict, base: pd.DataFrame) -> pd.DataFrame:
    frame, cache = pd.DataFrame(index=base.index), {}
    for label, (folder, column) in methods.items():
        if folder not in cache:
            cache[folder] = pd.read_csv(SAVED / folder / "test_predictions.csv.gz")
            other = cache[folder]
            assert (other.user_id.values == base.user_id.values).all() and np.allclose(other.actual_rating.values, base.actual_rating.values)
        frame[label] = cache[folder][column].values
    first = cache[next(iter(methods.values()))[0]]
    frame["alpha_high"] = first.alpha_high.values
    blend = next((f for f, c in methods.values() if f.endswith("/blend")), None)
    frame["trust_w"] = cache[blend].weight.values if blend else np.nan
    return frame


def make_replay(a, books: pd.DataFrame, movies: pd.DataFrame) -> dict:
    base = pd.read_csv(SAVED / "overlap_seeds/overlap_0.10/seed_2020/dupgt/test_predictions.csv.gz")[["user_id", "movie_asin", "actual_rating"]]
    frames = {o: load_overlap(m, base) for o, m in OVERLAPS.items()}
    truth = base.actual_rating.values

    # Headline numbers per overlap over ALL held-out ratings (raw predictions, as in the slides).
    overall = {o: {l: {"rmse": round(float(np.sqrt(np.mean((f[l].values - truth) ** 2))), 4),
                       "mae": round(float(np.mean(np.abs(f[l].values - truth))), 4)} for l in OVERLAPS[o]}
               for o, f in frames.items()}

    # The same real people under every overlap: 5-15 hidden movies, all titled, at least 3 titled books.
    movie_title = dict(zip(movies.asin, movies.title))
    counts = base.groupby("user_id").size()
    keep = set(counts[(counts >= 5) & (counts <= 15)].index)
    titled = base.movie_asin.map(lambda m: m in movie_title and not movie_title[m].startswith("(untitled"))
    all_titled = titled.groupby(base.user_id).all()
    keep &= set(all_titled[all_titled].index)
    pool = sorted(keep)
    rng = np.random.default_rng(2020)
    f10 = frames["10%"]
    per_user = (f10.assign(user_id=base.user_id, y=truth)
                .assign(d=lambda t: np.abs(t["DUPGT"] - t.y) - np.abs(t["Movie average"] - t.y))
                .groupby("user_id").d.mean())
    chosen = []
    for bucket in [per_user < -0.3, per_user.abs() <= 0.05, per_user > 0.3]:
        ids = [u for u in per_user.index[bucket.values] if u in keep]
        chosen += list(rng.choice(ids, size=min(70, len(ids)), replace=False))

    user_index = {u: i for i, u in enumerate(a["user_ids"])}
    book_title = dict(zip(books.item, books.title))
    src = a["source_train"]
    order = np.argsort(src[:, 0], kind="stable")
    sorted_users = src[order, 0]
    users = []
    rows_by_user = base.groupby("user_id").indices
    for u in chosen:
        uid = user_index[u]
        lo, hi = np.searchsorted(sorted_users, uid), np.searchsorted(sorted_users, uid, side="right")
        rows = src[order[lo:hi]]
        shelf = [{"title": book_title[int(i)], "stars": int(r)} for _, i, r in rows if int(i) in book_title]
        if len(shelf) < 3:
            continue
        idx = rows_by_user[u]
        record = {"user": u[-6:], "n_books": int(len(rows)), "books": shelf[:30],
                  "movies": [{"title": movie_title[base.movie_asin.iat[k]], "actual": float(truth[k])} for k in idx],
                  "by_overlap": {}}
        for o, f in frames.items():
            labels = list(OVERLAPS[o])
            sub = f.iloc[idx]
            record["by_overlap"][o] = {
                "pred": {l: [round(float(v), 3) for v in sub[l].values] for l in labels},
                "alpha_high": [round(float(v), 3) for v in sub.alpha_high.values],
                "trust_w": [None if np.isnan(v) else round(float(v), 3) for v in sub.trust_w.values],
                "mae": {l: round(float(np.mean(np.abs(sub[l].values - truth[idx]))), 3) for l in labels},
            }
        users.append(record)
    return {"overlaps": {o: list(m) for o, m in OVERLAPS.items()}, "overall": overall, "users": users,
            "shared_users": {"1%": 890, "10%": 8903, "100%": 89031}}


if __name__ == "__main__":
    main()
