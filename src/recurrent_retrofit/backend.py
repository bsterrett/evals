"""Worker backend: a parent checkpoint with a design installed.

It follows the local worker's contract. `depth` is the number of passes over the
repeated unit, counting the parent's own, so depth 1 is the unmodified parent.
For a recirculation, depth 2 is the design and depth 1 is the parent processed
one token at a time. A checkpoint trained to loop is served only with its gated
design, at a fixed number of passes: depth 1 is then one traversal and the gate,
not an unmodified parent.
The worker's model identity is the checkpoint's `repo_id@revision`, followed by
`+` and the program spec when one is given, so two designs cannot be mistaken
for each other.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

from eval_core.local.protocol import LocalRequest, PROMPT_FORMATS, encode_prompt, identity, profiles, verify_observations
from recurrent_retrofit.program import from_spec


def retrofit_identity(profile: str, spec: str | None) -> str:
    return identity(profile) + (f"+{spec}" if spec else "")


class RetrofitBackend:
    """Batch-one greedy decoding; the cache is request-local and every forward is checked."""

    def __init__(self, profile: str, depth: int, root: Path, *, spec: str | None = None,
            max_total_tokens: int | None = None) -> None:
        self.profile, self.depth = profile, depth
        self.settings = profiles()[profile]
        if depth not in self.settings["depths"]:
            raise ValueError("Depth is not supported for this profile")
        if spec in (None, "parent") and depth != 1:
            raise ValueError("More than one pass needs a loop spec such as block[9:18]")
        if spec is not None and spec.startswith("route"):
            raise ValueError("The worker serves the parent and loops; run routes through the library")
        # Reject a malformed or non-canonical design before loading any weights.
        # The layer range is checked against the real layer count after loading.
        design = from_spec(spec or "parent", 10**6, depth)
        from recurrent_retrofit.loading import load_parent, needs_gate
        if needs_gate(profile) != (design.kind == "gated"):
            raise ValueError("A checkpoint trained to loop is served with its gated design, and only it is")
        if getattr(design, "threshold", None) is not None:
            raise ValueError("The worker decodes with a cache, so it serves a gated loop at a fixed number of passes")
        self.model_id = retrofit_identity(profile, spec)
        self.max_total = max_total_tokens or self.settings["max_total_tokens"]
        from recurrent_retrofit.runner import install
        self.model, self.tokenizer, info = load_parent(profile, root)
        import torch
        self.torch = torch
        self.program = from_spec(spec or "parent", info["parent_layers"], depth)
        self.retrofit = install(self.model, self.program)
        self.eos_ids = set(info["eos_token_ids"])
        # The pass count is read from the hook on the first layer of the repeated unit,
        # or for a recirculation on the last layer, which every second pass reaches.
        if self.program.kind == "recirculate":
            self.witness = self.program.parent_layers - 1
        else:
            self.witness = self.program.unit[0] if self.program.unit else 0
        here = Path(__file__).parent
        self.capabilities = dict(info, protocol_version=1, model=self.model_id, depth=depth,
            max_total_tokens=self.max_total, decoding="greedy", adaptive_exit=False, cross_request_state=False,
            loop_policy="prefill-and-decode", prompt_formats=list(PROMPT_FORMATS),
            tokenizer_revision=self.settings["revision"], retrofit=self.program.describe(),
            adapter_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                for name in ("backend.py", "gated.py", "loading.py", "loopus.py", "program.py", "recirculation.py",
                    "runner.py", "worker.py")},
            dependency_versions=dict(sorted((dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions() if dist.metadata["Name"])),
            deterministic_algorithms=True, allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32)

    def tokenize(self, request: LocalRequest) -> list[int]:
        ids = encode_prompt(self.tokenizer, request)
        if not ids or len(ids) + request.max_output_tokens > self.max_total:
            raise ValueError(f"Prompt plus output limit exceeds {self.max_total} total tokens (no truncation)")
        return ids

    def generate(self, request: LocalRequest) -> dict:
        torch = self.torch
        ids = self.tokenize(request)
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        torch.use_deterministic_algorithms(True)
        output_ids: list[int] = []
        observed: list[int] = []
        times: list[float] = []
        stop_reason = "max_output_tokens"
        cache = self.retrofit.new_cache()
        self.retrofit.take_calls()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            current = torch.tensor([ids], device="cuda", dtype=torch.long)
            for _ in range(request.max_output_tokens):
                step_start = time.perf_counter()
                result = self.model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                calls = self.retrofit.take_calls()
                # A layer program runs once per forward pass; a recirculation once per token.
                runs = self.retrofit.runs_for(current.shape[1])
                self.retrofit.check_calls(calls, runs)
                self.retrofit.check_cache(cache, len(ids) + len(output_ids))
                observed.append(calls[self.witness] // runs)
                next_id = int(result.logits[0, -1].argmax().item())
                output_ids.append(next_id)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - step_start)
                del result
                if next_id in self.eos_ids:
                    stop_reason = "eos"
                    break
                current = torch.tensor([[next_id]], device="cuda", dtype=torch.long)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        verify_observations(observed, self.depth, len(output_ids))
        response = {
            "protocol_version": 1, "response_id": uuid.uuid4().hex, "model": self.model_id,
            "profile": self.profile, "depth": self.depth, "seed": request.seed,
            "prompt_format": request.prompt_format, "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
            "raw_text": self.tokenizer.decode(output_ids, skip_special_tokens=False),
            "input_tokens": len(ids), "output_tokens": len(output_ids), "output_token_ids": output_ids,
            "input_token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "stop_reason": stop_reason, "observed_iterations": observed,
            "forward_passes": len(observed), "prefill_iterations": observed[0], "decode_iterations": observed[1:],
            "layer_calls_per_forward": self.program.layer_calls,
            "elapsed_seconds": elapsed, "prefill_seconds": times[0], "decode_seconds": sum(times[1:]),
            "decode_tokens_per_second": (len(times) - 1) / sum(times[1:]) if len(times) > 1 else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "runtime": self.capabilities,
        }
        del cache, current
        return response
