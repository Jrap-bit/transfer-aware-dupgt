"""Live scoring for the demo: a new person's Book picks -> each model's Movie recommendations.

Reuses the project's own model classes and checkpoints (retrained into outputs/amazon2023/demo_models,
same seeds, same 10% overlap as the reported runs). Nothing in src/ is changed.

How a brand-new person is scored
  1. Book-taste vector u: ridge fold-in against the frozen Book item factors,
         u = argmin  sum_i (stars_i - v_i . u)^2 + LAMBDA * |u|^2
  2. Book history: 4-5 stars -> liked branch, 1-3 stars -> disliked branch (same rule as data.py).
  3. Each model runs its normal forward pass with that user vector in slot 0.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupgt_cdr.runtime import load_saved_model  # noqa: E402
from dupgt_cdr.semantics import SemanticStore  # noqa: E402

BUNDLE = ROOT / "demo/bundle"
MODELS = ROOT / "outputs/amazon2023/demo_models"
LAMBDA = 0.3
HISTORY = 20
# Overlap setting -> run group (share of shared training users; seed 2020 as in the reported runs).
GROUPS = {"10%": "overlap_0.10/seed_2020", "1%": "overlap_0.01/seed_2020"}


def single_runs(group: str) -> dict:
    """label -> run folder (single models). The uncertainty blend is handled separately."""
    return {
        "DUPGT": MODELS / "overlap" / group / "dupgt",
        "+ Semantic gate": MODELS / "overlap" / group / "semantics",
        "+ LLM teacher": MODELS / "overlap" / group / "semantics_llm_w0.01",
        "+ Shuffled teacher": MODELS / "overlap" / group / "semantics_shuffled_w0.01",
        "Probabilistic bridge": MODELS / "probabilistic" / group / "probabilistic_g0.1",
    }
PROBES = ["Frozen", "Toy Story 3", "Finding Nemo", "The Notebook (2004)", "Pride and Prejudice", "Downton Abbey",
          "Band of Brothers", "Saving Private Ryan", "Schindler's List", "The Sixth Sense", "The Conjuring",
          "Star Wars: The Force Awakens", "Interstellar", "Blade Runner 2049", "War Room"]
ORDER = ["Movie average", "DUPGT", "+ Semantic gate", "+ LLM teacher", "+ Shuffled teacher",
         "Probabilistic bridge", "Uncertainty blend"]
DESCRIPTION = {
    "Movie average": "No personalisation: every movie's average rating. Same list for everyone.",
    "DUPGT": "The paper's model: liked and disliked Book branches build a personal bridge into Movies.",
    "+ Semantic gate": "DUPGT plus Jina text similarity between your Books and each Movie in the gate.",
    "+ LLM teacher": "Semantic gate, trained with extra LLM preference pairs (the LLM is not needed now).",
    "+ Shuffled teacher": "Control: same pairs with the LLM's answers shuffled. Should look like the LLM one.",
    "Probabilistic bridge": "Predicts its own uncertainty and falls back to the movie average when unsure.",
    "Uncertainty blend": "5 bridges vote; the more they disagree about you, the more it trusts the movie average.",
}


class OneShelfStore:
    """Same features as SemanticStore.features, but the Book profile is computed once per request.

    Every row in a demo request shares one bookshelf, so profiles/counts/coherence are identical across
    rows; only the Movie vector changes. The formula below is copied from semantics.SemanticStore.features.
    """

    def __init__(self, store):
        self.store = store
        self.vectors = store.vectors
        self.reference = store.reference
        self.manifest = store.manifest

    def lookup(self, ids):
        return self.store.lookup(ids)

    def features(self, x):
        values = x.detach().cpu().numpy()
        n = len(values)
        profiles, count, total, coherence = self.store.profiles(values[:1, 2:42])
        if not (values[:, 2:42] == values[:1, 2:42]).all():
            return self.store.features(x)          # mixed shelves: fall back to the original
        movie = self.store.lookup(values[:, 1])
        present = np.linalg.norm(movie, axis=-1) > 0
        similarities = movie @ profiles[0].T        # (n, 2): liked and disliked profile
        count, total, coherence = (np.repeat(a, n, axis=0) for a in (count, total, coherence))
        profiles = np.repeat(profiles, n, axis=0)
        stats = np.column_stack((similarities, np.log1p(count) / np.log(21),
            count / np.maximum(total, 1), coherence, count > 0, present, similarities[:, 0] - similarities[:, 1]))
        valid = present & (count.sum(1) > 0)
        return profiles, movie, stats.astype(np.float32), valid


def shrink(model: torch.nn.Module) -> torch.nn.Module:
    """Drop the 1.3M-user tables (not needed for a new person) to keep memory small."""
    k = model.emb_dim
    for part in (model.src_model, model.tgt_model, model.aug_model):
        part.uid_embedding = torch.nn.Embedding(1, k)
    model.aug_model.iid_embedding = torch.nn.Embedding(1, k)
    if hasattr(model, "semantic_store"):
        model.semantic_store = OneShelfStore(model.semantic_store)
    return model.eval()


def load_run(folder: Path) -> torch.nn.Module:
    checkpoint = torch.load(folder / "checkpoints/dupgt_best.pt", map_location="cpu", weights_only=True)
    return shrink(load_saved_model(checkpoint, "cpu"))


class Engine:
    def __init__(self) -> None:
        torch.set_num_threads(4)
        self.lock = threading.Lock()
        self.rank = "rating"
        self.books = pd.read_csv(BUNDLE / "books.csv.gz", keep_default_na=False)
        self.books["lower"] = self.books.title.str.lower()
        movies = pd.read_csv(BUNDLE / "movies.csv.gz", keep_default_na=False)
        self.candidates = movies[movies.candidate].reset_index(drop=True)
        # Any titled movie with at least 20 training ratings can be rated on request (wider than the 100+ recommendation pool).
        self.rateable = movies[(movies.n_ratings >= 20) & ~movies.title.str.startswith("(untitled")].sort_values("n_ratings", ascending=False).reset_index(drop=True)
        self.rateable["lower"] = self.rateable.title.str.lower()
        self.cand_ids = torch.as_tensor(self.candidates.item.values, dtype=torch.long)
        self.prior = self.candidates.movie_mean.values
        self.book_title = dict(zip(self.books.item, self.books.title))
        self.probe = []
        for name in PROBES:
            hit = self.candidates[self.candidates.title.str.startswith(name)].sort_values("n_ratings", ascending=False)
            if len(hit):
                self.probe.append(int(hit.index[0]))

        self.groups = {g: {"models": {}, "blend": None, "missing": []} for g in GROUPS}
        self.refresh()
        any_model = next((m for g in self.groups.values() for m in g["models"].values()), None)
        if any_model is None:
            raise SystemExit("No demo models found yet; wait for training to finish.")
        self.book_factors = any_model.src_model.iid_embedding.weight.detach().numpy().astype(np.float64)
        self.semantic = SemanticStore(ROOT / "data/semantics/amazon2023/jina_v2")

    @property
    def missing(self) -> list[str]:
        return [f"{g}: {m}" for g, rec in self.groups.items() for m in rec["missing"]]

    def refresh(self) -> None:
        """Load any model whose training has finished since the last check (cheap when nothing changed)."""
        for name, group in GROUPS.items():
            rec, runs = self.groups[name], single_runs(group)
            loaded = {label: load_run(folder) for label, folder in runs.items()
                      if label not in rec["models"] and (folder / "artifact_manifest.json").exists()}  # only finished runs
            blend = None
            blend_file = MODELS / "uncertainty" / group / "blend/blend.json"
            if rec["blend"] is None and blend_file.exists():  # written only after all 5 bridges finish
                record = json.loads(blend_file.read_text())
                blend = {"params": {k: float(v) for k, v in record["uncertainty_blend"].items()},
                         "constant": {k: float(v) for k, v in record["constant_blend"].items()},
                         "members": [load_run(Path(m["run_dir"])) for m in record["members"]]}
            with self.lock:
                merged = {**rec["models"], **loaded}
                rec["models"] = {k: merged[k] for k in runs if k in merged}   # keep the display order
                if blend:
                    rec["blend"] = blend
                rec["missing"] = [k for k in runs if k not in rec["models"]] + ([] if rec["blend"] else ["Uncertainty blend"])

    def available(self) -> list[str]:
        return [g for g, rec in self.groups.items() if not rec["missing"]]   # only complete settings

    # ---------- catalogue ----------
    def search(self, query: str, limit: int = 12) -> list[dict]:
        q = query.strip().lower()
        if len(q) < 2:
            return []
        hit = self.books[self.books.lower.str.contains(q, regex=False)].head(limit)
        return [{"item": int(r.item), "title": r.title, "category": r.category, "n_ratings": int(r.n_ratings)}
                for r in hit.itertuples()]

    def suggest(self, n: int = 8) -> list[dict]:
        """Random well-known books (top 1,500 by rating count, short titles) to offer as one-click adds."""
        if not hasattr(self, "_popular"):
            top = self.books.head(1500)
            self._popular = top[top.title.str.len() <= 60].reset_index(drop=True)
        pick = self._popular.sample(n=min(n, len(self._popular)))
        return [{"item": int(r.item), "title": r.title, "category": r.category, "n_ratings": int(r.n_ratings)} for r in pick.itertuples()]

    # ---------- a new person ----------
    def user_inputs(self, picks: list[dict]):
        items = np.array([int(p["item"]) for p in picks], dtype=np.int64)
        stars = np.array([float(p["stars"]) for p in picks])
        V = self.book_factors[items]
        u = np.linalg.solve(V.T @ V + LAMBDA * np.eye(V.shape[1]), V.T @ stars)
        high = [int(i) for i, s in zip(items, stars) if s >= 4][-HISTORY:]
        low = [int(i) for i, s in zip(items, stars) if s < 4][-HISTORY:]
        history = high + [0] * (HISTORY - len(high)) + low + [0] * (HISTORY - len(low))
        return torch.as_tensor(u, dtype=torch.float32), torch.as_tensor(history, dtype=torch.long)

    @torch.inference_mode()
    def run_model(self, model, u, history, movies: torch.Tensor):
        model.src_model.uid_embedding.weight.data[0] = u
        n = len(movies)
        x = torch.cat((torch.zeros(n, 1, dtype=torch.long), movies[:, None], history.expand(n, -1)), dim=1)
        if getattr(model, "model_variant", "") == "probabilistic":
            mu, log_s2, prior, alpha, _ = model.forward_parts(x)
            pred, weight = model.combine(mu, log_s2, prior)
            return pred.numpy(), alpha.numpy(), weight.numpy(), mu.numpy()
        pred, alpha = model.forward_with_gate(x)
        return pred.numpy(), alpha.numpy(), None, pred.numpy()

    def search_movies(self, query: str, limit: int = 10) -> list[dict]:
        q = query.strip().lower()
        if len(q) < 2:
            return []
        hit = self.rateable[self.rateable.lower.str.contains(q, regex=False)].head(limit)
        return [{"item": int(r.item), "title": r.title, "category": r.category, "n_ratings": int(r.n_ratings),
                 "movie_avg": round(float(r.movie_mean), 2)} for r in hit.itertuples()]

    def topical(self, picks: list[dict], n: int = 8, pool: int = 40, exclude: list[int] | None = None) -> list[dict]:
        """Movies whose text is most similar to the loved books (content match, not a model ranking); a random n of the top pool."""
        liked = [int(p["item"]) for p in picks if float(p["stars"]) >= 4]
        if not liked:
            return []
        if not hasattr(self, "_topical_pool"):
            known = self.rateable[self.rateable.n_ratings >= 50].reset_index(drop=True)
            vectors = self.semantic.lookup(known.item.values)
            keep = np.linalg.norm(vectors, axis=1) > 0
            self._topical_pool, self._topical_vectors = known[keep].reset_index(drop=True), vectors[keep]
        books = self.semantic.lookup(np.array(liked))
        books = books[np.linalg.norm(books, axis=1) > 0]
        if not len(books):
            return []
        profile = books.mean(0)
        profile /= np.linalg.norm(profile)
        sims = self._topical_vectors @ profile
        movies = self._topical_pool
        order = [i for i in np.argsort(-sims) if int(movies.item[i]) not in set(exclude or [])]
        seen, top = set(), []
        for i in order:   # one edition per title
            key = movies.title[i].split("[")[0].split("(")[0].strip().lower()
            if key not in seen:
                seen.add(key); top.append(i)
            if len(top) >= pool:
                break
        chosen = np.random.default_rng().choice(top, size=min(n, len(top)), replace=False)
        return [{"item": int(movies.item[i]), "title": movies.title[i], "category": movies.category[i], "n_ratings": int(movies.n_ratings[i]),
                 "movie_avg": round(float(movies.movie_mean[i]), 2), "similarity": round(float(sims[i]), 3)}
                for i in sorted(chosen, key=lambda i: -sims[i])]

    def rate(self, picks: list[dict], items: list[int], overlap: str = "10%") -> dict:
        """Each model's predicted stars for specific movies, for this person (the task the models are trained for)."""
        if not picks or not items:
            return {"movies": [], "overlap": overlap}
        meta = self.rateable.set_index("item")
        items = [int(i) for i in items if int(i) in meta.index][:8]
        ids = torch.as_tensor(items, dtype=torch.long)
        prior = meta.loc[items, "movie_mean"].to_numpy(dtype=float)
        with self.lock:
            u, history = self.user_inputs(picks)
            rec = self.groups[overlap] if overlap in self.available() else self.groups["10%"]
            preds = {"Movie average": prior}
            extra = [{} for _ in items]
            for label, model in rec["models"].items():
                pred, alpha, weight, mu = self.run_model(model, u, history, ids)
                preds[label] = pred
                for e, a, w in zip(extra, alpha, weight if weight is not None else [None] * len(items)):
                    e.setdefault("alpha", {})[label] = round(float(a), 3)
                    if w is not None:
                        e["prob_w"] = round(float(w), 3)
            if rec["blend"]:
                outs = [self.run_model(m, u, history, ids) for m in rec["blend"]["members"]]
                members = np.stack([o[0] for o in outs])
                p = rec["blend"]["params"]
                mu, d = members.mean(0), members.std(0)
                w = p["s0"] / (p["s0"] + p["tau2"] + p["c"] * d ** 2)
                preds["Uncertainty blend"] = w * mu + (1 - w) * prior
                for e, k in zip(extra, range(len(items))):
                    e.update(bridges=[round(float(v), 3) for v in members[:, k]], d=round(float(d[k]), 3), w=round(float(w[k]), 3))
        order = [m for m in ORDER if m in preds]
        movies = [{"item": it, "title": meta.at[it, "title"], "movie_avg": round(float(prior[k]), 3), "n": int(meta.at[it, "n_ratings"]),
                   "predicted": {m: round(float(np.clip(preds[m][k], 1, 5)), 3) for m in order}, **extra[k]} for k, it in enumerate(items)]
        return {"movies": movies, "models": order, "overlap": overlap if overlap in self.available() else "10%"}

    def recommend(self, picks: list[dict], top_k: int = 5, rank: str = "rating", overlap: str = "10%") -> dict:
        """rank="rating": highest predicted stars; rank="boost": largest gain over the movie's average."""
        if not picks:
            raise ValueError("Pick at least one book")
        with self.lock:
            u, history = self.user_inputs(picks)
            liked = [int(p["item"]) for p in picks if float(p["stars"]) >= 4]
            self.rank = rank
            rec = self.groups[overlap] if overlap in self.available() else self.groups["10%"]
            columns = {"Movie average": self.column(self.prior, None, None, top_k, liked, baseline=True)}
            fallback = [{"title": self.candidates.title[i], "movie_avg": round(float(self.prior[i]), 3)} for i in self.probe]
            for label, model in rec["models"].items():
                pred, alpha, weight, mu = self.run_model(model, u, history, self.cand_ids)
                columns[label] = self.column(pred, alpha, weight, top_k, liked)
                if weight is not None:  # probabilistic bridge: its own mu, sigma-based w, final
                    for f, i in zip(fallback, self.probe):
                        f["probabilistic"] = {"mu": round(float(mu[i]), 3), "w": round(float(weight[i]), 3), "final": round(float(pred[i]), 3)}
            if rec["blend"]:
                outs = [self.run_model(m, u, history, self.cand_ids) for m in rec["blend"]["members"]]
                members = np.stack([o[0] for o in outs])
                p, pc = rec["blend"]["params"], rec["blend"]["constant"]
                mu, d = members.mean(0), members.std(0)
                w = p["s0"] / (p["s0"] + p["tau2"] + p["c"] * d ** 2)
                w_const = pc["s0"] / (pc["s0"] + pc["tau2"])
                pred = w * mu + (1 - w) * self.prior
                alpha = np.mean([o[1] for o in outs], axis=0)
                columns["Uncertainty blend"] = self.column(pred, alpha, w, top_k, liked, disagreement=d)
                for f, i in zip(fallback, self.probe):
                    f.update(bridges=[round(float(v), 3) for v in members[:, i]], mu=round(float(mu[i]), 3), d=round(float(d[i]), 3),
                             w=round(float(w[i]), 3), final=round(float(pred[i]), 3), w_const=round(float(w_const), 3),
                             final_const=round(float(w_const * mu[i] + (1 - w_const) * self.prior[i]), 3))
                blend_params = {"s0": p["s0"], "tau2": p["tau2"], "c": p["c"], "tau2_const": pc["tau2"]}
        ordered = [{"model": k, "about": DESCRIPTION[k], **columns[k]} for k in ORDER if k in columns]
        probe = [{"title": self.candidates.title[i], "movie_avg": round(float(self.prior[i]), 2),
                  "n": int(self.candidates.n_ratings[i]),
                  "predicted": {c["model"]: c["probe"][j] for c in ordered}} for j, i in enumerate(self.probe)]
        for c in ordered:
            c.pop("probe")
        agree = self.agreement(ordered)
        return {"columns": ordered, "missing": self.missing, "user_vector_norm": float(u.norm()),
                "n_liked": len(liked), "n_disliked": len(picks) - len(liked), "agreement": agree, "n_candidates": int(len(self.candidates)),
                "probe": probe, "fallback": fallback, "blend_params": blend_params if rec["blend"] else None,
                "overlap": overlap if overlap in self.available() else "10%", "overlaps": self.available()}

    def column(self, pred, alpha, weight, top_k, liked, disagreement=None, baseline=False) -> dict:
        score = pred if (baseline or self.rank == "rating") else pred - self.prior
        order = np.argsort(-score, kind="stable")[:top_k]
        rows = []
        for i in order:
            m = self.candidates.iloc[int(i)]
            row = {"title": m.title, "category": m.category, "predicted": round(float(np.clip(pred[i], 1, 5)), 2),
                   "raw": round(float(pred[i]), 3), "movie_avg": round(float(m.movie_mean), 2), "n": int(m.n_ratings),
                   "boost": round(float(pred[i] - m.movie_mean), 2)}
            if alpha is not None:
                row["alpha_high"] = round(float(alpha[i]), 3)
            if weight is not None:
                row["trust"] = round(float(weight[i]), 3)
            if disagreement is not None:
                row["disagreement"] = round(float(disagreement[i]), 3)
            row["because"] = self.because(int(m["item"]), liked)
            rows.append(row)
        shift = pred - self.prior
        summary = {"mean_alpha_high": None if alpha is None else round(float(np.mean(alpha)), 3),
                   "mean_shift": round(float(np.mean(shift)), 3),
                   "mean_predicted": round(float(np.mean(np.clip(pred, 1, 5))), 3),
                   "mean_trust": None if weight is None else round(float(np.mean(weight)), 3),
                   "spread": round(float(np.std(pred)), 3)}
        probe = [{"stars": round(float(np.clip(pred[i], 1, 5)), 2),
                  "alpha_high": None if alpha is None else round(float(alpha[i]), 3),
                  "trust": None if weight is None else round(float(weight[i]), 3)} for i in self.probe]
        return {"top": rows, "summary": summary, "probe": probe}

    def because(self, movie_item: int, liked: list[int]):
        if not liked:
            return None
        vectors = self.semantic.lookup(np.array(liked))
        movie = self.semantic.lookup(np.array([movie_item]))[0]
        if not np.linalg.norm(movie):
            return None
        sims = vectors @ movie
        j = int(np.argmax(sims))
        return {"title": self.book_title.get(liked[j], "?"), "similarity": round(float(sims[j]), 2)}

    @staticmethod
    def agreement(columns: list[dict]) -> dict:
        sets = {c["model"]: {r["title"] for r in c["top"]} for c in columns}
        base = sets.get("+ Semantic gate") or sets.get("DUPGT")
        if not base:
            return {}
        return {k: len(v & base) for k, v in sets.items()}

