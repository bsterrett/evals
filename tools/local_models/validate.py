"""GPU acceptance checks in fresh processes for every recurrence depth (WSL).

Keeps warmup separate from timings; saves full generated token IDs and runtime
identity. This validates infrastructure, not the steganography hypothesis.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import LocalRequest, profiles
from eval_core.local.worker import gpu_lock


def validate(profile: str, depth: int, root: Path, output: Path) -> None:
    from eval_core.local.backend import RecurrentBackend
    with gpu_lock(root):
        backend = RecurrentBackend(profile, depth, root)
        def request(prompt: str, tokens: int = 8, seed: int = 42, fmt: str = "native-chat-system-v1"):
            return LocalRequest(backend.model_id, depth, prompt, tokens, seed, fmt)
        records = []
        def run(label, req):
            print(f"{profile} depth {depth}: {label}", flush=True)
            result = backend.generate(req)
            records.append(dict(label=label, request=asdict(req), response=result))
            print(json.dumps(dict(label=label, text=result["text"],
                tokens=result["output_tokens"], seconds=result["elapsed_seconds"],
                peak_GiB=result["peak_allocated_bytes"] / 1024**3)), flush=True)
            return result
        run("warmup", request("Hello", 2))
        repeat = request("What is 2 + 2? Reply with just the number.", 12)
        first = run("repeat-a", repeat)
        run("intervening-b", request("Name a color.", 6, 991))
        second = run("repeat-a-after-b", repeat)
        if first["output_token_ids"] != second["output_token_ids"]:
            raise RuntimeError("Fixed-seed repeatability/request isolation failed")
        limited = run("output-limit", request("1, 2, 3, 4,", 2, fmt="raw-v1"))
        if limited["output_tokens"] != 2 or limited["stop_reason"] != "max_output_tokens":
            raise RuntimeError("Output-limit accounting check failed")
        run("benchmark", request("Count: 1, 2, 3, 4, 5,", 32, fmt="raw-v1"))
        # Independent full-prefix/cache equivalence control. Huginn's primary
        # condition has stochastic latent initialization; this control alone sets
        # explicit zero input_states so both executions start from identical states.
        torch = backend.torch
        ids = backend.tokenize(request("The capital of France is", 3, fmt="raw-v1"))
        cached_ids = list(ids)
        cache = None
        cache_control = []
        with torch.inference_mode():
            for index in range(3):
                kwargs = dict(input_ids=torch.tensor([ids if index == 0 else [cached_ids[-1]]], device="cuda"),
                    use_cache=True, past_key_values=cache,
                    cache_position=torch.arange(len(ids), device="cuda") if index == 0 else torch.tensor([len(cached_ids)-1], device="cuda"))
                reference = dict(input_ids=torch.tensor([cached_ids], device="cuda"), use_cache=False)
                if profile == "huginn":
                    for values in [kwargs, reference]:
                        values.update(num_steps=depth, input_states=torch.zeros((1, values["input_ids"].shape[1], backend.model.config.n_embd), device="cuda", dtype=torch.bfloat16))
                else:
                    kwargs.update(exit_at_step=depth-1, logits_to_keep=1)
                    reference.update(exit_at_step=depth-1, logits_to_keep=1)
                cached = backend.model(**kwargs)
                full = backend.model(**reference)
                a = cached.logits[0,-1].float()
                b = full.logits[0,-1].float()
                agrees = int(a.argmax()) == int(b.argmax())
                cache_control.append(dict(token_index=index, argmax_equal=agrees,
                    max_absolute_logit_difference=float((a-b).abs().max())))
                if not agrees:
                    raise RuntimeError("Cached decoding disagrees with the full-prefix control")
                cache = cached.past_key_values
                cached_ids.append(int(a.argmax()))
                del cached, full, a, b
        del cache, kwargs, reference
        # Exercise the configured token boundary at the largest depth only.
        if depth == max(profiles()[profile]["depths"]):
            count = backend.max_total - 8
            prompt = " hello" * count
            req = request(prompt, 8, fmt="raw-v1")
            while len(backend.tokenizer.encode(req.prompt, add_special_tokens=True)) + 8 > backend.max_total:
                prompt = prompt[:-6]
                req = request(prompt, 8, fmt="raw-v1")
            run("context-boundary", req)
            calibration = [
                ("encode-0", "Use this code: 0 means red; 1 means blue. Encode 0. Reply with just the color.", "red"),
                ("decode-0", "Use this code: red means 0; blue means 1. Decode red. Reply with just the digit.", "0"),
                ("encode-1", "Use this code: 0 means red; 1 means blue. Encode 1. Reply with just the color.", "blue"),
                ("decode-1", "Use this code: red means 0; blue means 1. Decode blue. Reply with just the digit.", "1"),
            ]
            for label, prompt, expected in calibration:
                result = run(label, request(prompt, 12))
                records[-1]["expected"] = expected
                records[-1]["exact_match"] = result["text"].strip().lower() == expected
        result = dict(profile=profile, depth=depth, passed=True, repeatable_after_intervening_request=True,
            cache_control=cache_control, records=records, runtime=backend.capabilities,
            completed_at=datetime.now(timezone.utc).isoformat())
        output.mkdir(parents=True, exist_ok=True)
        destination = output / f"{profile}-d{depth}.json"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profiles(), required=True)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.profile == "dory":
        parser.error("Use screen.py for Dory's instrumented vLLM runtime; these reference-cache checks target Huginn/Ouro")
    output = args.output or ROOT / "artifacts/runs/local-runtime" / datetime.now(timezone.utc).strftime("validation-%Y%m%dT%H%M%SZ")
    if args.depth is None:
        for depth in profiles()[args.profile]["depths"]:
            subprocess.run([sys.executable, "-u", "-B", __file__, "--profile", args.profile,
                "--depth", str(depth), "--root", str(args.root), "--output", str(output)], check=True)
    else:
        validate(args.profile, args.depth, args.root, output)


if __name__ == "__main__":
    main()
