from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

from eval_core.local.protocol import LocalRequest, PROTOCOL_VERSION, profiles


@contextmanager
def gpu_lock(root: Path):
    """An advisory Linux lock shared by both profiles and all ports."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / "gpu.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another recurrent worker owns the GPU lock") from exc
        import os
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def make_server(backend, port: int) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, body: dict) -> None:
            payload = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            self.send_json(200, dict(backend.capabilities, ready=True))

        def do_POST(self) -> None:
            if self.path not in ("/tokenize", "/generate"):
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 262144:
                    raise ValueError("Request body must be 1..262144 bytes")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                self.connection.settimeout(30)
                value = json.loads(self.rfile.read(length))
                request = LocalRequest.parse(value, model=backend.model_id,
                    depth=backend.depth, max_total=backend.max_total)
            except (ValueError, TypeError, OSError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            try:
                # Tokenization and GPU execution share one queue, preventing tokenizer/RNG races.
                with lock:
                    if self.path == "/tokenize":
                        ids = backend.tokenize(request)
                        result = {"protocol_version": PROTOCOL_VERSION, "model": backend.model_id,
                            "depth": backend.depth, "input_tokens": len(ids), "input_token_ids": ids,
                            "prompt_format": request.prompt_format}
                    else:
                        result = backend.generate(request)
                self.send_json(200, result)
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                # No successful response is returned after OOM, loop mismatch, or other execution error.
                print(f"Generation failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fixed-depth recurrent inference worker")
    parser.add_argument("--profile", choices=profiles(), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    args = parser.parse_args()
    if args.depth not in profiles()[args.profile]["depths"]:
        parser.error("Unsupported depth for the selected profile")
    with gpu_lock(args.root):
        if args.profile == "dory":
            from eval_core.local.dory_backend import DoryBackend
            backend = DoryBackend(args.depth, args.root)
        else:
            from eval_core.local.backend import RecurrentBackend
            backend = RecurrentBackend(args.profile, args.depth, args.root)
        server = make_server(backend, args.port)
        print(json.dumps(dict(backend.capabilities, listening=f"http://127.0.0.1:{server.server_port}")), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
