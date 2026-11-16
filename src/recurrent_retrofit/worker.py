"""Local worker for retrofit parents. It serves the same HTTP contract as the
recurrent-model worker and shares its GPU lock, so only one of them runs at a time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_core.local.protocol import profiles
from eval_core.local.worker import gpu_lock, make_server
from recurrent_retrofit.loading import FAMILIES


def main() -> None:
    parents = [name for name, settings in profiles().items() if settings.get("family") in FAMILIES]
    parser = argparse.ArgumentParser(description="Local worker for a pretrained parent with a design installed")
    parser.add_argument("--profile", choices=parents, required=True)
    parser.add_argument("--depth", type=int, required=True,
        help="Passes over the repeated unit, counting the parent's own; 1 is the unmodified parent. A recirculation takes 2")
    parser.add_argument("--method", help="Design to install, e.g. block[9:18], block[15:19]/euler,anchor=0.5 or "
        "recirculate[18to9]/convex,alpha=0.07; omit to serve the plain parent. A checkpoint trained to loop "
        "takes its own design, e.g. gated[2:27]")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    args = parser.parse_args()
    if args.depth not in profiles()[args.profile]["depths"]:
        parser.error("Unsupported depth for the selected profile")
    with gpu_lock(args.root):
        from recurrent_retrofit.backend import RetrofitBackend
        try:
            backend = RetrofitBackend(args.profile, args.depth, args.root, spec=args.method)
        except ValueError as exc:
            parser.error(str(exc))
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
