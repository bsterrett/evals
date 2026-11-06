"""Run a frozen small capability screen; preserve full outputs and censoring.

All calls are local. This diagnostic is not a steganography ability estimate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import LocalRequest, PROMPT_FORMATS, profiles
from eval_core.local.screening import final_answer, score_answer
from eval_core.local.worker import gpu_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profiles(), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--placement", choices=["gpu"], default="gpu")
    parser.add_argument("--latent-policy", choices=["native", "float32-init"], default="native")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="native-chat-system-v1")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile == "dory" and (args.precision != "bfloat16" or args.latent_policy != "native"
            or args.max_total_tokens is not None):
        parser.error("Dory uses its pinned BF16/context profile; these diagnostic overrides are Huginn/Ouro only")
    if args.output.exists():
        parser.error("Output already exists; use a new path to preserve earlier evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases_path = ROOT / "local_models/screen-cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if args.case_ids:
        if not set(args.case_ids) <= {c["id"] for c in cases}:
            parser.error("Unknown case ID")
        cases = [c for c in cases if c["id"] in args.case_ids]
    records = []
    journal = args.output.with_suffix(".jsonl")
    def save_event(value):
        with journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(value) + "\n")
    with gpu_lock(args.root):
        if args.profile == "dory":
            from eval_core.local.dory_backend import DoryBackend
            backend = DoryBackend(args.depth, args.root)
        else:
            from eval_core.local.backend import RecurrentBackend
            backend = RecurrentBackend(args.profile, args.depth, args.root,
                precision=args.precision, placement=args.placement,
                max_total_tokens=args.max_total_tokens, latent_policy=args.latent_policy)
        template_before = backend.tokenizer.chat_template
        template_after = backend.tokenizer.chat_template
        provenance = dict(runtime=backend.capabilities,
            max_output_tokens=args.max_output_tokens, prompt_format=args.prompt_format,
            template_patch=None,
            template_before_sha256=hashlib.sha256((template_before or "").encode()).hexdigest(),
            template_used_sha256=hashlib.sha256((template_after or "").encode()).hexdigest(),
            cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            screen_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            started_at=datetime.now(timezone.utc).isoformat())
        save_event(dict(event="started", **provenance))
        for case in cases:
            request = LocalRequest(backend.model_id, args.depth, case["prompt"],
                args.max_output_tokens, 42, args.prompt_format)
            print(f"Screen {args.profile} {args.precision} d{args.depth}: {case['id']}", flush=True)
            try:
                input_ids = backend.tokenize(request)
                rendered = backend.tokenizer.decode(input_ids, skip_special_tokens=False)
                response = backend.generate(request)
                tail = rendered.rsplit("assistant", 1)[-1] if "assistant" in rendered else rendered.rsplit("Huginn",1)[-1]
                thinking_prefix = "<think>" in tail and "</think>" not in tail
                final = final_answer(response.get("raw_text", response["text"]), thinking_prefix=thinking_prefix)
                score = score_answer(case, final)
                record = dict(case=case, request=asdict(request), rendered_prompt=rendered,
                    response=response, final_text=final, score=score,
                    output_limit_reached=response["stop_reason"] == "max_output_tokens")
                print(json.dumps(dict(case=case["id"], final=final, score=score,
                    output_tokens=response["output_tokens"], stop=response["stop_reason"],
                    seconds=response["elapsed_seconds"])), flush=True)
            except Exception as exc:
                record = dict(case=case, request=asdict(request), error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            records.append(record)
            save_event(dict(event="case", **record))
        result = dict(provenance, records=records, completed_at=datetime.now(timezone.utc).isoformat(),
            operational_success=all("error" not in r for r in records),
            exact_correct=sum(r.get("score", {}).get("exact_correct", False) for r in records),
            mapping_correct=sum(r.get("score", {}).get("semantic_correct") is True for r in records),
            mapping_total=sum(c["kind"] in ("color", "bit") for c in cases),
            cases_total=len(cases))
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
