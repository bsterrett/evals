"""Inputs and job sets for re-decoding the confirmation's carriers with a reasoning decoder.

The carriers, prompts, option orders, and acceptance decisions are the confirmation
run's, verified by hash. Only the decoder's thinking setting changes.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tomllib

from eval_core.artifacts import file_sha256
from eval_core.workspace import load_workspace
from recurrent_steg.confirmation import design as parent_design
from recurrent_steg.reference.design import model_config
from recurrent_steg.reference.execution import phase_records

PROTOCOL = "sonnet-reasoning-decoder-v1"
PARENT_PROTOCOL = parent_design.PROTOCOL
# Proposed ceiling; a run needs the user's approval recorded in docs/user-policy.md first.
MAX_BUDGET_USD = Decimal("10.00")
PARENT_FILES = dict(plan="plan.json", encode="encode-sonnet55.jsonl", decode="decode-sonnet55.jsonl",
    analysis="analysis-v1.json")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _integer(config: dict, key: str, low: int, high: int) -> int:
    value = config[key]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def load(config_path: Path) -> dict:
    """Validate the configuration and the parent run, and return everything the jobs need."""
    from recurrent_steg.confirmation.cli import verify_freeze as verify_parent
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported reasoning-decoder protocol")
    budget = Decimal(config["run_budget_usd"])
    if not budget.is_finite() or budget <= 0 or budget > MAX_BUDGET_USD:
        raise ValueError(f"The run ceiling must be positive and no more than ${MAX_BUDGET_USD}")
    _integer(config, "decoder_tokens", 1024, 20000)
    _integer(config, "check_jobs", 4, 16)
    _integer(config, "min_reasoned_checks", 1, config["check_jobs"])
    _integer(config, "reasoned_min_output_tokens", 20, 1000)
    margin = Decimal(config["projection_margin"])
    if not margin.is_finite() or margin < 1:
        raise ValueError("projection_margin must be at least 1")
    models = {key: model_config(key, raw) for key, raw in config["models"].items()}
    subject = models.get(config["subject"])
    if subject is None or subject.provider != "anthropic" or subject.thinking_type != "adaptive" \
            or not subject.reasoning_effort:
        raise ValueError("The subject must be an Anthropic model with adaptive thinking and an effort level")
    root = load_workspace(config_path).root
    parent_run = (root / config["parent_run"]).resolve()
    for name, filename in PARENT_FILES.items():
        if file_sha256(parent_run / filename) != config[f"parent_{name}_sha256"]:
            raise ValueError(f"Parent {filename} does not match the configuration")
    parent_plan = verify_parent(parent_run)
    if parent_plan.get("protocol") != PARENT_PROTOCOL:
        raise ValueError("The parent run is not the plain/0-1 confirmation")
    parent_config, cases, parent_models, _ = parent_design.load_design(root / parent_plan["config_path"])
    if parent_config["subject"] != config["subject"] or parent_models[config["subject"]].model != subject.model:
        raise ValueError("The decoder must be the confirmation's subject model")
    carriers = phase_records(parent_run, "encode", config["subject"])
    parent_decodes = phase_records(parent_run, "decode", config["subject"])
    parent_analysis = json.loads((parent_run / PARENT_FILES["analysis"]).read_text(encoding="utf-8"))
    if parent_analysis.get("protocol") != PARENT_PROTOCOL or not parent_analysis["judge_qualification"]["qualified"]:
        raise ValueError("The parent analysis must be the confirmation's, with a qualified judge")
    return dict(config=config, config_path=config_path, root=root, parent_run=parent_run, parent_plan=parent_plan,
        parent_config=parent_config, cases=cases, model=subject, subject=config["subject"], carriers=carriers,
        parent_decodes=parent_decodes, parent_analysis=parent_analysis)


def decode_jobs(loaded: dict) -> list[dict]:
    """The confirmation's own decode jobs for payload carriers and the no-carrier controls.

    Ids, prompts, shown order, and job order are identical to the parent's. Honest
    paraphrases are not decoded again: no pre-registered measure here uses them.
    """
    jobs = parent_design.decoding_jobs(loaded["parent_config"], loaded["carriers"], loaded["cases"])
    return [job for job in jobs if job["kind"] != "honest"]


def check_jobs(loaded: dict) -> list[dict]:
    """Pre-test on the first honest development paraphrases by id; not part of any analysis."""
    jobs = parent_design.decoding_jobs(loaded["parent_config"], loaded["carriers"], loaded["cases"])
    honest = sorted((job for job in jobs if job["kind"] == "honest"), key=lambda job: job["id"])
    return [dict(job, id="check-" + job["id"]) for job in honest[:loaded["config"]["check_jobs"]]]


def job_identity(jobs: list[dict]) -> str:
    return _sha(json.dumps([[job["id"], _sha(job["prompt"]), job["shown"]] for job in jobs]))


def parent_identity(loaded: dict) -> dict:
    config = loaded["config"]
    return dict(run=config["parent_run"], protocol=loaded["parent_plan"]["protocol"],
        git=loaded["parent_plan"].get("git"),
        files={filename: config[f"parent_{name}_sha256"] for name, filename in PARENT_FILES.items()})
