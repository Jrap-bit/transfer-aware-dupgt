"""Local demo server (standard library only).

    .venv/bin/python demo/server.py            # then open http://127.0.0.1:8765
    .venv/bin/python demo/server.py --port 9000

Endpoints
  GET  /                    the demo page
  GET  /api/status          which models are loaded
  GET  /api/search?q=...    Book title search
  GET  /api/personas        one-click example users
  GET  /api/replay          real held-out test users with the original saved predictions
  POST /api/recommend       {"picks": [{"item": id, "stars": 1-5}, ...], "rank": "rating"|"boost"}
  POST /api/llm             {"picks": [...]} -> live LLM preference picks (needs LM Studio running)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import Engine  # noqa: E402

import llm  # noqa: E402

ENGINE: Engine | None = None
STATIC = HERE / "static"          # plain fallback page (/classic)
WEB = HERE / "web/dist"            # built React app (cd demo/web && npm run build)
TYPES = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".woff": "font/woff",
         ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json"}
BUNDLE = HERE / "bundle"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send(self, payload, status=200, kind="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/classic":
                return self.send((STATIC / "index.html").read_bytes(), kind="text/html; charset=utf-8")
            if url.path in ("/", "/index.html"):
                page = WEB / "index.html" if (WEB / "index.html").exists() else STATIC / "index.html"
                return self.send(page.read_bytes(), kind="text/html; charset=utf-8")
            if url.path.startswith("/assets/"):
                file = (WEB / url.path.lstrip("/")).resolve()
                if WEB.resolve() in file.parents and file.is_file():
                    kind = TYPES.get(file.suffix, "application/octet-stream")
                    return self.send(file.read_bytes(), kind=kind)
            if url.path == "/api/status":
                return self.send({"overlaps": ENGINE.available(), "missing": ENGINE.missing, "llm": llm.status()})
            if url.path == "/api/search":
                return self.send(ENGINE.search(parse_qs(url.query).get("q", [""])[0]))
            if url.path == "/api/movies":
                return self.send(ENGINE.search_movies(parse_qs(url.query).get("q", [""])[0]))
            if url.path == "/api/suggest":
                return self.send(ENGINE.suggest(int(parse_qs(url.query).get("n", ["8"])[0])))
            if url.path == "/api/personas":
                return self.send(json.loads((BUNDLE / "personas.json").read_text()))
            if url.path == "/api/replay":
                return self.send((BUNDLE / "replay.json").read_bytes())
            return self.send({"error": "not found"}, 404)
        except Exception as exc:  # keep the demo alive
            traceback.print_exc()
            return self.send({"error": str(exc)}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            picks = body.get("picks", [])
            if url.path == "/api/recommend":
                return self.send(ENGINE.recommend(picks, top_k=int(body.get("top_k", 5)), rank=body.get("rank", "rating"), overlap=body.get("overlap", "10%")))
            if url.path == "/api/topical":
                return self.send(ENGINE.topical(picks, n=int(body.get("n", 8)), exclude=body.get("exclude", [])))
            if url.path == "/api/rate":
                return self.send(ENGINE.rate(picks, body.get("items", []), overlap=body.get("overlap", "10%")))
            if url.path == "/api/llm":
                titles = {int(p["item"]): ENGINE.book_title.get(int(p["item"]), "") for p in picks}
                return self.send(llm.ask(ENGINE, picks, titles, overlap=body.get("overlap", "10%")))
            return self.send({"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            return self.send({"error": str(exc)}, 500)


def main() -> None:
    global ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    started = time.time()
    ENGINE = Engine()
    print(f"models loaded in {time.time() - started:.1f}s; overlaps: {ENGINE.available()}; missing: {ENGINE.missing}", flush=True)
    print(f"open http://127.0.0.1:{args.port}", flush=True)

    def watch():  # pick up models that finish training later, without blocking any request
        import threading as _t
        while ENGINE.missing:
            time.sleep(30)
            try:
                ENGINE.refresh()
                print(f"loaded models; still missing: {ENGINE.missing}", flush=True)
            except Exception:
                traceback.print_exc()
    import threading
    threading.Thread(target=watch, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
