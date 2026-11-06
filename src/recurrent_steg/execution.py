from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path

from eval_core.artifacts import append_jsonl, file_sha256
from eval_core.local.screening import final_answer
from eval_core.models import GenerationRequest, ModelConfig, RateCard
from eval_core.providers.local import LocalProvider
from eval_core.runtime import RunSession
from eval_core.workspace import load_workspace
from recurrent_steg.design import decoding_jobs, encoding_jobs, load_design, observer_jobs
from recurrent_steg.scoring import parse_bit, parse_observer


def phase_records(run_dir: Path, phase: str, depth: int) -> list[dict]:
    path=run_dir/f"{phase}-d{depth}.jsonl"
    receipt=json.loads(path.with_suffix(".complete.json").read_text(encoding="utf-8"))
    if file_sha256(path)!=receipt["sha256"]:
        raise ValueError("Completed phase records changed: "+str(path))
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows)!=receipt["count"] or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Completed phase has missing or duplicate records")
    return rows


def all_carriers(run_dir: Path, config: dict, cases: list[dict]) -> list[dict]:
    rows=[r for d in config["depths"] for r in phase_records(run_dir,"encode",d)]
    expected={j["id"] for d in config["depths"] for j in encoding_jobs(config,cases,d)}
    if {r["id"] for r in rows}!=expected:
        raise ValueError("Encoder records do not match the frozen job set")
    return rows


def execute(run_dir: Path, phase: str, depth: int) -> dict:
    from recurrent_steg.cli import verify_freeze
    frozen=verify_freeze(run_dir)
    root=load_workspace().root
    config,cases,_=load_design(root/frozen["config_path"])
    if config["observer"]!="local-fixed":
        raise ValueError("Only the local observer is implemented for this pilot")
    if depth not in config["depths"] or phase=="observe" and depth!=config["observer_depth"]:
        raise ValueError("Requested depth is outside the frozen phase design")
    if phase=="encode":
        jobs=encoding_jobs(config,cases,depth)
    else:
        carriers=all_carriers(run_dir,config,cases)
        jobs=(decoding_jobs(config,carriers,cases,depth) if phase=="decode"
            else observer_jobs(config,carriers,cases))
    path=run_dir/f"{phase}-d{depth}.jsonl"
    receipt_path=path.with_suffix(".complete.json")
    if receipt_path.exists():
        return dict(already_complete=True,phase=phase,depth=depth,count=len(phase_records(run_dir,phase,depth)))
    lock=path.with_suffix(".lock")
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(descriptor)
    try:
        existing=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        done={r["id"] for r in existing}
        if len(done)!=len(existing) or not done<={j["id"] for j in jobs}:
            raise ValueError("Partial phase has unexpected records")
        workspace=load_workspace()
        model=ModelConfig(f"dory-d{depth}","local",frozen["model"],
            RateCard(Decimal(0),Decimal(0),frozen["profile"]["max_total_tokens"]-1,
                date.today(),"Local GPU, zero API charge"))
        session=RunSession(experiment_id="recurrent_steganography",output_dir=run_dir/"calls",
            budget=workspace.budget,dry_run=False,run_name=f"{phase}-d{depth}",
            provider_factory=lambda model,**kwargs:LocalProvider(model))
        cap=config[{"encode":"cover_tokens","decode":"decoder_tokens","observe":"observer_tokens"}[phase]]
        with session.accounting():
            for job in jobs:
                if job["id"] in done:
                    continue
                req=GenerationRequest(phase,job["prompt"],cap,
                    {"local":dict(depth=depth,seed=config["inference_seed"],prompt_format=config["prompt_format"])})
                response,cost=session.call(model,req,trial_id=job["id"])
                if cost!=0:
                    raise RuntimeError("Local pilot incurred an unexpected charge")
                raw=response.raw["raw_text"]
                final=final_answer(raw)
                row={k:v for k,v in job.items() if k!="prompt"}
                row.update(response=asdict(response),events=str(session.events_path.relative_to(run_dir)),
                    censored=response.stop_reason=="max_output_tokens")
                if phase=="encode":
                    carrier=final or ""
                    row.update(carrier=carrier,word_count=len(carrier.split()),
                        mechanically_usable=bool(carrier) and not row["censored"] and len(carrier.split())<=50)
                elif phase=="decode":
                    row.update(parsed=parse_bit(raw) if not row["censored"] else dict(bit=None,bare_format=False,final_text=final))
                else:
                    try:
                        if row["censored"]:
                            raise ValueError("Censored observer response")
                        row.update(judgment=parse_observer(raw),parse_error=None)
                    except (ValueError,TypeError) as exc:
                        row.update(judgment=None,parse_error=str(exc))
                append_jsonl(path,row)
                print(json.dumps(dict(phase=phase,depth=depth,done=len(done)+1,total=len(jobs))),flush=True)
                done.add(job["id"])
        receipt=dict(phase=phase,depth=depth,count=len(done),sha256=file_sha256(path),
            completed_at=datetime.now(timezone.utc).isoformat())
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        return receipt
    finally:
        lock.unlink()
