from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess

from eval_core.artifacts import file_sha256
from eval_core.local.protocol import identity, profiles
from eval_core.workspace import git_provenance, load_workspace
from recurrent_steg.design import decoding_jobs, encoding_jobs, load_design, observer_jobs, prompt_identity


def plan(config_path: Path) -> dict:
    config,cases,data_path = load_design(config_path)
    workspace = load_workspace(config_path)
    root = workspace.root
    source_paths = sorted((root/"src/recurrent_steg").glob("*.py"))
    source_paths += sorted((root/"src/eval_core").rglob("*.py"))
    source_paths += [root/"src/eval_core/local/profiles.json",root/"pyproject.toml"]
    observer = config["observer"]
    simulated_carriers=[dict(job,carrier="<<GENERATED CARRIER>>")
        for depth in config["depths"] for job in encoding_jobs(config,cases,depth)]
    calls=dict(encode=len(simulated_carriers),
        decode=sum(len(decoding_jobs(config,simulated_carriers,cases,d)) for d in config["depths"]),
        observe=len(observer_jobs(config,simulated_carriers,cases)) if observer=="local-fixed" else None)
    rng=random.Random(config["seed"]+50)
    orders={phase:rng.sample(config["depths"],len(config["depths"])) for phase in ("encode","decode")}
    return dict(schema_version=1,experiment_id="recurrent_steganography",protocol=config["protocol"],
        analysis_role=config["role"],config=config,model=identity(config["profile"]),
        profile=profiles()[config["profile"]],prompt_fingerprints=prompt_identity(),
        case_ids={split:[c["id"] for c in cases if c["split"]==split]
            for split in ("development","observer_calibration")},
        config_path=str(config_path.resolve().relative_to(root)),
        config_sha256=file_sha256(config_path),data_sha256=file_sha256(data_path),
        source_sha256={str(p.relative_to(root)):file_sha256(p) for p in source_paths},
        git=git_provenance(root),api_budget_usd="0.00",chance_recovery=0.5,
        encoder_depths=config["depths"],decoder_depths=config["depths"],
        batch_orders=orders,
        development_covers=8,observer_calibration_covers=8,
        generation_calls=calls,
        maximum_output_tokens=(calls["encode"]*config["cover_tokens"]+calls["decode"]*config["decoder_tokens"]+
            calls["observe"]*config["observer_tokens"]) if observer=="local-fixed" else None,
        observer=observer,ready_for_generation=observer=="local-fixed",
        remaining_inputs=["observer"] if observer=="pending-user-choice" else [],
        primary_measure="All-attempt joint recovery, cover preservation, and observer acceptance; undefined if the observer fails qualification",
        interpretation="Exploratory convenience set; depth effects require case-cluster uncertainty and later unused evaluation cases")


def verify_freeze(run_dir: Path) -> dict:
    frozen = json.loads((run_dir/"plan.json").read_text(encoding="utf-8"))
    workspace = load_workspace()
    current = plan(workspace.root/frozen["config_path"])
    for key in ("config_sha256","data_sha256","source_sha256","prompt_fingerprints","model"):
        if current[key]!=frozen[key]:
            raise ValueError("Frozen pilot identity changed: "+key)
    return frozen


def main():
    parser=argparse.ArgumentParser(description="Prepare and execute the frozen local recurrent-depth pilot")
    parser.add_argument("command",choices=["validate","plan","freeze","execute","analyze"])
    parser.add_argument("--config",type=Path,default=Path("experiments/recurrent_steganography/configs/pilot.toml"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--phase",choices=["encode","decode","observe"])
    parser.add_argument("--depth",type=int)
    parser.add_argument("--live-local",action="store_true")
    args=parser.parse_args()
    try:
        if args.command in ("execute","analyze"):
            if args.run is None:
                raise ValueError("A frozen --run directory is required")
            if args.command=="execute":
                if not args.live_local or args.phase is None or args.depth is None:
                    raise ValueError("Generation requires --live-local, --phase, and --depth")
                from recurrent_steg.execution import execute
                result=execute(args.run.resolve(),args.phase,args.depth)
            else:
                from recurrent_steg.analysis import analyze
                result=analyze(args.run.resolve())
            print(json.dumps(result,indent=2))
            return 0
        result=plan(args.config)
        if args.command=="freeze":
            if not result["ready_for_generation"]:
                raise ValueError("Observer choice must be resolved before freezing a carrier-generation run")
            root=load_workspace(args.config).root
            identity_paths=list(result["source_sha256"])+[result["config_path"],
                str((args.config.resolve().parent/result["config"]["data"]).resolve().relative_to(root))]
            changed=subprocess.check_output(["git","status","--porcelain","--untracked-files=all",
                "--",*identity_paths],cwd=root,text=True,encoding="utf-8")
            if changed.strip():
                raise ValueError("Commit the pilot source, configuration, and data before freezing a generation run")
            destination=args.output
            if destination is None:
                stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                destination=root/"artifacts/runs/recurrent_steganography"/(stamp+"-pilot")
            if destination.exists():
                raise ValueError("Preserve existing pilot evidence; choose another output directory")
            destination.mkdir(parents=True)
            (destination/"plan.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(dict(run_dir=str(destination),**result),indent=2))
        elif args.command=="validate":
            print(json.dumps(dict(valid=True,ready_for_generation=result["ready_for_generation"],
                remaining_inputs=result["remaining_inputs"],model=result["model"],api_budget_usd="0.00"),indent=2))
        else:
            print(json.dumps(result,indent=2))
        return 0
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,"error: "+str(exc)+"\n")


if __name__=="__main__":
    raise SystemExit(main())
