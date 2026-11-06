"""Prospective runtime and ordinary-task checks; no hidden-message generation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval_core.local.protocol import LocalRequest, profiles, verify_observations
from eval_core.local.screening import final_answer
from eval_core.local.worker import gpu_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["dory", "ouro-2.6b-thinking"], required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.home()/"llm/recurrent")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; choose a new output file")
    cases_path = ROOT/"local_models/preflight-cases-v1.json"
    cases = json.loads(cases_path.read_text())
    fmt = "native-chat-no-thinking-v1" if args.profile=="dory" else "native-chat-thinking-v1"
    cap = 128 if args.profile=="dory" else 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix(".jsonl")
    records = []
    def save(record):
        records.append(record)
        with journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
    with gpu_lock(args.root):
        if args.profile=="dory":
            from eval_core.local.dory_backend import DoryBackend
            backend = DoryBackend(args.depth, args.root)
        else:
            from eval_core.local.backend import RecurrentBackend
            backend = RecurrentBackend(args.profile, args.depth, args.root)
        def request(prompt, tokens=cap, form=fmt, seed=42):
            return LocalRequest(backend.model_id, args.depth, prompt, tokens, seed, form)
        def run(label, req, expected=None):
            response = backend.generate(req)
            rendered = backend.tokenizer.decode(backend.tokenize(req), skip_special_tokens=False)
            tail = rendered.rsplit("assistant",1)[-1]
            final = final_answer(response["raw_text"], thinking_prefix="<think>" in tail and "</think>" not in tail)
            row = dict(label=label, request=asdict(req), response=response,
                final_text=final, expected=expected,
                exact_correct=final is not None and final.casefold()==expected.casefold() if expected is not None else None)
            save(row)
            print(json.dumps(dict(label=label, final=final, tokens=response["output_tokens"],
                stop=response["stop_reason"], exact=row["exact_correct"])), flush=True)
            return response
        run("warmup", request("Reply with OK.", 2))
        repeat_req = request("Reply with exactly the word READY.")
        first = run("repeat-a", repeat_req)
        run("intervening", request("Name a color in one word.", 32, seed=991))
        second = run("repeat-a-after-b", repeat_req)
        assert first["output_token_ids"]==second["output_token_ids"], "Request isolation failed"
        limited = run("output-limit", request("1, 2, 3, 4,", 2, "raw-v1"))
        assert limited["output_tokens"]==2 and limited["stop_reason"]=="max_output_tokens"
        for case in cases:
            run(case["id"], request(case["prompt"]), case["expected"])
        cache_control = []
        cache_req = request("The capital of France is", 4, "raw-v1")
        cached = run("cache-continuation", cache_req)
        ids = backend.tokenize(cache_req)
        for i, expected in enumerate(cached["output_token_ids"][:3]):
            prefix = ids + cached["output_token_ids"][:i]
            if args.profile=="dory":
                from vllm import SamplingParams
                backend.engine.collective_rpc("eval_reset")
                full = backend.engine.generate([dict(prompt_token_ids=prefix)],
                    SamplingParams(temperature=0,max_tokens=1,seed=42,skip_special_tokens=False),use_tqdm=False)
                actual = list(full[0].outputs[0].token_ids)
                observed = backend.engine.collective_rpc("eval_read")[0]["observed_iterations"]
                verify_observations(observed,args.depth,len(actual))
                token = actual[0]
            else:
                torch = backend.torch
                before = backend.executions
                with torch.inference_mode():
                    full = backend.model(input_ids=torch.tensor([prefix],device="cuda"),
                        use_cache=False,exit_at_step=args.depth-1,logits_to_keep=1,use_weighted_exit=False)
                token = int(full.logits[0,-1].argmax())
                observed = [backend.executions-before]
                verify_observations(observed,args.depth,1)
                del full
            cache_control.append(dict(index=i,expected=expected,actual=token,
                argmax_equal=token==expected,observed_iterations=observed))
            assert token==expected, "Cached/full-prefix token mismatch"
        if args.depth==max(profiles()[args.profile]["depths"]):
            prompt = " hello"*(backend.max_total-8)
            while len(backend.tokenizer.encode(prompt,add_special_tokens=True))+8>backend.max_total:
                prompt=prompt[:-6]
            run("context-boundary",request(prompt,8,"raw-v1"))
        result = dict(schema_version=1,purpose="Runtime and ordinary competence only",
            runtime=backend.capabilities,records=records,cache_control=cache_control,
            repeatable=True,runtime_passed=True,case_count=len(cases),
            exact_correct=sum(r["exact_correct"] is True for r in records),
            cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            completed_at=datetime.now(timezone.utc).isoformat())
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print("Saved "+str(args.output),flush=True)


if __name__=="__main__":
    main()
