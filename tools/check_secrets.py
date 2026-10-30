"""Scan tracked content and optional Git history; report locations, never values.

This is a pattern check, not a guarantee that arbitrary secrets are detectable.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "provider API key": re.compile(rb"\bsk-(?:proj-|ant-[A-Za-z0-9]+-)?[A-Za-z0-9_-]{24,}"),
    "GitHub token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "assigned secret": re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"'][A-Za-z0-9_./+=-]{24,}[\"']"
    ),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def findings(content: bytes) -> list[str]:
    return [label for label, pattern in PATTERNS.items() if pattern.search(content)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--staged", action="store_true", help="Scan the exact Git index contents")
    args = parser.parse_args()
    problems = []
    listing = ("ls-files", "--cached", "-z") if args.staged else ("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = git(*listing).decode().split("\0")
    for name in filter(None, paths):
        path = ROOT / name
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            problems.append(f"Secret environment file included: {name}")
        content = git("show", f":{name}") if args.staged else path.read_bytes() if path.is_file() else b""
        problems.extend(f"{label}: {name}" for label in findings(content))
    if args.history:
        for line in git("rev-list", "--objects", "--all").decode().splitlines():
            object_id, _, name = line.partition(" ")
            if git("cat-file", "-t", object_id).strip() != b"blob":
                continue
            if Path(name).name == ".env" or (Path(name).name.startswith(".env.") and Path(name).name != ".env.example"):
                problems.append(f"Secret environment file in history: {name}")
            problems.extend(
                f"{label} in history: {name} ({object_id[:12]})"
                for label in findings(git("cat-file", "blob", object_id))
            )
    for problem in problems:
        print(problem)
    if not problems:
        print("Secret-pattern scan passed for tracked/candidate files" + (" and Git history." if args.history else "."))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
