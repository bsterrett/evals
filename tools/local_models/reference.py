"""Compare the adapter's greedy decoder with upstream Transformers generation."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.protocol import LocalRequest, profiles
from eval_core.local.worker import gpu_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profiles(), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from eval_core.local.backend import RecurrentBackend
    with gpu_lock(args.root):
        backend = RecurrentBackend(args.profile, args.depth, args.root)
        from transformers import GenerationConfig
        torch = backend.torch
        records = []
        prompts = [
            ("What is 2 + 2? Reply with just the number.", "native-chat-system-v1"),
            ("The capital of France is", "raw-v1"),
        ]
        for prompt, fmt in prompts:
            request = LocalRequest(backend.model_id, args.depth, prompt, 12, 42, fmt)
            actual = backend.generate(request)
            ids = backend.tokenize(request)
            config = GenerationConfig(max_new_tokens=request.max_output_tokens,
                do_sample=False, use_cache=True, eos_token_id=sorted(backend.eos_ids),
                pad_token_id=backend.tokenizer.pad_token_id or backend.tokenizer.eos_token_id,
                bos_token_id=backend.tokenizer.bos_token_id)
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
            kwargs = {"num_steps": args.depth} if args.profile == "huginn" else {"exit_at_step": args.depth-1}
            before = backend.executions
            with torch.inference_mode():
                output = backend.model.generate(torch.tensor([ids], device="cuda"), generation_config=config, **kwargs)
            expected = output[0, len(ids):].tolist()
            record = dict(request=asdict(request), adapter=actual, upstream_token_ids=expected,
                upstream_text=backend.tokenizer.decode(expected, skip_special_tokens=True),
                upstream_total_iterations=backend.executions-before, identical=expected == actual["output_token_ids"])
            records.append(record)
            print(json.dumps(dict(format=fmt, identical=record["identical"],
                upstream_text=record["upstream_text"])), flush=True)
            if not record["identical"]:
                raise RuntimeError("Adapter generation differs from upstream reference")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(records=records, runtime=backend.capabilities,
            completed_at=datetime.now(timezone.utc).isoformat()), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
