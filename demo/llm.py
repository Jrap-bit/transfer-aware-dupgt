"""Live LLM teacher call for the demo, using the project's exact teacher prompt and JSON schema.

Needs LM Studio serving Qwen3.5-9B on 127.0.0.1:1234. Load it before the talk with:
    lms load qwen3.5-9b --identifier dupgt-qwen35-9b --gpu max -c 8192 -y
If the server or model is unavailable, the demo says so and keeps working without it.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupgt_cdr.metadata import shorten  # noqa: E402
from dupgt_cdr.teacher import canonical, identity, preference_payload, whitelist_metadata  # noqa: E402

BASE = "http://127.0.0.1:1234"
MODEL = "dupgt-qwen35-9b"
LIMITS = {"title": 120, "description": 240, "categories": 80}   # same limits as the real teacher requests
META = ROOT / "data/metadata/amazon2023/clean"
# Five contrasting movie pairs (titles must match the engine's probe list).
PAIRS = [("Frozen", "The Conjuring"), ("The Notebook (2004)", "Saving Private Ryan"),
         ("Downton Abbey", "Interstellar"), ("Toy Story 3", "The Sixth Sense"),
         ("Pride and Prejudice", "Star Wars: The Force Awakens")]

_book_text: dict[str, dict] = {}
_movie_text: dict[str, dict] = {}
_ready = threading.Event()


def _load_text() -> None:
    for name, store in (("movies.jsonl", _movie_text), ("books.jsonl", _book_text)):
        with open(META / name) as handle:
            for line in handle:
                row = json.loads(line)
                fields = whitelist_metadata(row)
                store[row["asin"]] = {k: shorten(fields[k], LIMITS[k]) for k in ("title", "description", "categories")}
    _ready.set()


threading.Thread(target=_load_text, daemon=True).start()


def status() -> dict:
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=1.5) as r:
            models = [m["id"] for m in json.load(r).get("data", [])]
        return {"server": True, "model_loaded": MODEL in models, "models": models, "text_ready": _ready.is_set()}
    except Exception:
        return {"server": False, "model_loaded": False, "models": [], "text_ready": _ready.is_set()}


def build_request(engine, picks: list[dict]) -> tuple[dict, list[dict]]:
    _ready.wait(timeout=60)
    books_frame = engine.books.set_index("item")
    books = []
    for k, p in enumerate(picks[:12]):
        asin = books_frame.asin.get(int(p["item"]))
        text = _book_text.get(asin, {"title": engine.book_title.get(int(p["item"]), ""), "description": "", "categories": ""})
        books.append({"alias": f"b{k}", **text, "rating": int(p["stars"])})
    by_title = {engine.candidates.title[i]: i for i in engine.probe}
    movies, pairs, shown = [], [], []
    for j, (a, b) in enumerate(PAIRS):
        ia = next((i for t, i in by_title.items() if t.startswith(a)), None)
        ib = next((i for t, i in by_title.items() if t.startswith(b)), None)
        if ia is None or ib is None:
            continue
        for alias, i in ((f"m{2 * j}", ia), (f"m{2 * j + 1}", ib)):
            asin = engine.candidates.asin[i]
            movies.append({"alias": alias, **_movie_text.get(asin, {"title": engine.candidates.title[i], "description": "", "categories": ""})})
        pairs.append({"pair_id": f"p{j}", "a": f"m{2 * j}", "b": f"m{2 * j + 1}"})
        shown.append({"pair_id": f"p{j}", "a": engine.candidates.title[ia], "b": engine.candidates.title[ib]})
    body = {"books": books, "movies": movies, "pairs": pairs, "profile_token": "u0000"}
    return {"input": body, "request_id": identity(body)}, shown


def ask(engine, picks: list[dict], _titles=None, overlap: str = "10%") -> dict:
    if not picks:
        raise ValueError("Pick at least one book")
    request, shown = build_request(engine, picks)
    payload = preference_payload(request, MODEL, 512)
    started = time.monotonic()
    try:
        rq = urllib.request.Request(BASE + "/v1/chat/completions", data=canonical(payload).encode(),
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=120) as resp:
            raw = json.load(resp)
    except Exception as exc:
        return {"ok": False, "error": f"LM Studio not reachable or model not loaded ({exc.__class__.__name__}). "
                                      f"Run: lms load qwen3.5-9b --identifier {MODEL} --gpu max -c 8192 -y"}
    seconds = time.monotonic() - started
    answer = json.loads(raw["choices"][0]["message"]["content"])
    choice = {p["pair_id"]: p["choice"] for p in answer["preferences"]}
    # What each model would say for the same pairs: the movie it predicts higher for this person.
    probes = {row["title"]: row["predicted"] for row in engine.recommend(picks, overlap=overlap)["probe"]}
    results = []
    for pair in shown:
        pa, pb = probes.get(pair["a"], {}), probes.get(pair["b"], {})
        models = {}
        for name in pa:
            if name in pb:
                diff = pa[name]["stars"] - pb[name]["stars"]
                models[name] = "A" if diff > 0.02 else "B" if diff < -0.02 else "tie"
        results.append({**pair, "llm": choice.get(pair["pair_id"], "?"), "models": models})
    usage = raw.get("usage", {})
    return {"ok": True, "seconds": round(seconds, 1), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "pairs": results,
            "raw_json": answer, "prompt_preview": payload["messages"][1]["content"][:1200]}
