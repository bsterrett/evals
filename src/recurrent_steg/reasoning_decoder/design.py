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
# Amendment after v1's pre-test showed no thinking at all: the same design plus one
# neutral sentence asking the decoder to think before it answers.
AMENDED_PROTOCOL = "sonnet-reasoning-decoder-v1.1"
PROTOCOLS = (PROTOCOL, AMENDED_PROTOCOL)
INSTRUCTION_ANCHOR = "Reply with exactly"
PARENT_PROTOCOL = parent_design.PROTOCOL
# The user approved $10.00 for the reasoning-decoder run, shared by v1 and its amendment.
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
    if config.get("schema_version") != 1 or config.get("protocol") not in PROTOCOLS:
        raise ValueError("Unsupported reasoning-decoder protocol")
    instruction = config.get("decoder_instruction")
    if (config["protocol"] == AMENDED_PROTOCOL) != (instruction is not None):
        raise ValueError("Only the amendment adds a decoder instruction, and it requires one")
    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()
            or INSTRUCTION_ANCHOR in instruction):
        raise ValueError("decoder_instruction must be one plain sentence")
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


def with_instruction(prompt: str, instruction: str | None) -> str:
    """The parent's prompt, with the amendment's sentence placed just before the reply instruction."""
    if instruction is None:
        return prompt
    if prompt.count(INSTRUCTION_ANCHOR) != 1:
        raise ValueError("The decoder prompt must contain the reply instruction exactly once")
    return prompt.replace(INSTRUCTION_ANCHOR, f"{instruction.strip()} {INSTRUCTION_ANCHOR}")


def _parent_jobs(loaded: dict) -> list[dict]:
    jobs = parent_design.decoding_jobs(loaded["parent_config"], loaded["carriers"], loaded["cases"])
    instruction = loaded["config"].get("decoder_instruction")
    return [dict(job, prompt=with_instruction(job["prompt"], instruction)) for job in jobs]


def decode_jobs(loaded: dict) -> list[dict]:
    """The confirmation's own decode jobs for payload carriers and the no-carrier controls.

    Ids, shown order, and job order are identical to the parent's, and so are the
    prompts apart from any amendment sentence. Honest paraphrases are not decoded
    again: no pre-registered measure here uses them.
    """
    return [job for job in _parent_jobs(loaded) if job["kind"] != "honest"]


def check_jobs(loaded: dict) -> list[dict]:
    """Pre-test on the first honest development paraphrases by id; not part of any analysis."""
    honest = sorted((job for job in _parent_jobs(loaded) if job["kind"] == "honest"), key=lambda job: job["id"])
    return [dict(job, id="check-" + job["id"]) for job in honest[:loaded["config"]["check_jobs"]]]


def job_identity(jobs: list[dict]) -> str:
    return _sha(json.dumps([[job["id"], _sha(job["prompt"]), job["shown"]] for job in jobs]))


def parent_identity(loaded: dict) -> dict:
    config = loaded["config"]
    return dict(run=config["parent_run"], protocol=loaded["parent_plan"]["protocol"],
        git=loaded["parent_plan"].get("git"),
        files={filename: config[f"parent_{name}_sha256"] for name, filename in PARENT_FILES.items()})
