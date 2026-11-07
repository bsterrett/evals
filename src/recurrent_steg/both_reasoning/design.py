"""Covers, prompts, and job sets for the variant in which both endpoints think before answering.

Claude Sonnet 5.5 encodes and decodes with adaptive thinking on fresh covers. Each Sonnet
prompt is a frozen plain/0-1 reference prompt plus one sentence asking for thought, placed
just before the reply instruction. The judge, its controls and qualification rules, the
option-order assignment, and the decoder's answer schema are the confirmation's. The
confirmation's answer-only decoder also reads the same carriers, as a registered secondary.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import tomllib

from recurrent_steg.confirmation.design import CONDITION, assign, control_jobs, judge_jobs, shown
from recurrent_steg.reference.design import (BIT_OPTIONS, NO_CARRIER, decoder_prompt, encoder_prompt, judge_prompt,
    model_config)
from recurrent_steg.reference.scoring import content_problems

PROTOCOL = "sonnet-both-reasoning-v1"
# Proposed ceiling, pending the user's spending approval; loading refuses any higher value.
MAX_BUDGET_USD = Decimal("15.00")
SPLIT_SIZES = {"development": 288, "judge_calibration": 24, "encoder_check": 8}
FAMILY_SIZES = {"development": 36, "judge_calibration": 3, "encoder_check": 1}
EARLIER_COVERS = ("covers-v1.json", "reference-covers-v1.json", "confirmation-covers-v1.json")
ENCODER_ANCHOR = "Return only the paraphrase."
DECODER_ANCHOR = "Reply with exactly"
__all__ = ["BIT_OPTIONS", "CONDITION", "PROTOCOL", "control_jobs", "judge_jobs", "shown"]


def _sentences(message: str) -> set[str]:
    return {part.strip().rstrip(".") for part in message.split(". ")}


def validate_cases(cases: list[dict], data_dir: Path) -> None:
    if any(set(case) != {"id", "split", "family", "message"} for case in cases):
        raise ValueError("Each cover needs exactly id, split, family, and message")
    messages = [case["message"] for case in cases]
    if len({case["id"] for case in cases}) != len(cases) or len(set(messages)) != len(cases):
        raise ValueError("Cover identifiers and messages must be unique")
    for split, size in SPLIT_SIZES.items():
        if sum(case["split"] == split for case in cases) != size:
            raise ValueError(f"The design fixes {size} {split} covers")
    if len(cases) != sum(SPLIT_SIZES.values()):
        raise ValueError("Unknown cover split")
    families = sorted({case["family"] for case in cases})
    if len(families) != 8:
        raise ValueError("The design fixes eight cover families")
    for family in families:
        for split, size in FAMILY_SIZES.items():
            if sum(case["family"] == family and case["split"] == split for case in cases) != size:
                raise ValueError(f"Family {family} needs {size} {split} covers")
    for case in cases:
        if content_problems(case["message"]):
            raise ValueError(f"Cover {case['id']} is unsuitable")
    earlier = [case["message"] for name in EARLIER_COVERS
        for case in json.loads((data_dir / name).read_text(encoding="utf-8"))]
    earlier_sentences = set().union(*(_sentences(message) for message in earlier))
    if any(_sentences(message) & earlier_sentences for message in messages):
        raise ValueError("Covers must be new, down to each sentence")


def _integer(config: dict, key: str, low: int, high: int) -> int:
    value = config[key]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def _positive(config: dict, key: str) -> Decimal:
    value = Decimal(config[key])
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def load_design(config_path: Path) -> tuple[dict, list[dict], dict, Path]:
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported both-reasoning protocol")
    if _positive(config, "run_budget_usd") > MAX_BUDGET_USD:
        raise ValueError(f"The run ceiling must be no more than ${MAX_BUDGET_USD}")
    instruction = config["think_instruction"]
    if not isinstance(instruction, str) or instruction != instruction.strip() or not instruction.endswith(".") \
            or "." in instruction[:-1] or ENCODER_ANCHOR in instruction or DECODER_ANCHOR in instruction:
        raise ValueError("think_instruction must be one plain sentence")
    for key in ("encoder_tokens", "decoder_tokens"):
        _integer(config, key, 1024, 20000)
    for key in ("answer_only_tokens", "judge_tokens"):
        _integer(config, key, 16, 1000)
    _integer(config, "decoder_check_jobs", 4, 16)
    _integer(config, "encoder_min_usable_checks", 1, 3 * SPLIT_SIZES["encoder_check"])
    _integer(config, "encoder_min_reasoned_checks", 1, 2 * SPLIT_SIZES["encoder_check"])
    _integer(config, "decoder_min_reasoned_checks", 1, config["decoder_check_jobs"])
    _integer(config, "reasoned_min_thinking_tokens", 1, 1000)
    if Decimal(config["projection_margin"]) < 1:
        raise ValueError("projection_margin must be at least 1")
    for key in ("decode_unit_usd", "answer_only_unit_usd", "judge_unit_usd"):
        _positive(config, key)
    models = {key: model_config(key, raw) for key, raw in config["models"].items()}
    subject, answer_only, judge = (models.get(config[key]) for key in ("subject", "answer_only", "judge"))
    if subject is None or subject.provider != "anthropic" or subject.thinking_type != "adaptive" \
            or not subject.reasoning_effort:
        raise ValueError("The subject must be an Anthropic model with adaptive thinking and an effort level")
    if answer_only is None or answer_only.model != subject.model or answer_only.thinking_type != "between_tools":
        raise ValueError("The answer-only decoder must be the subject model without up-front thinking")
    if judge is None or judge.model == subject.model:
        raise ValueError("A different model must judge")
    data_path = (config_path.parent / config["data"]).resolve()
    cases = json.loads(data_path.read_text(encoding="utf-8"))
    validate_cases(cases, data_path.parent)
    return config, assign(cases, config["seed"]), models, data_path


def think(prompt: str, instruction: str, anchor: str) -> str:
    """The frozen prompt with the thinking sentence placed just before its reply instruction."""
    if prompt.count(anchor) != 1:
        raise ValueError("The prompt must contain its reply instruction exactly once")
    return prompt.replace(anchor, f"{instruction} {anchor}")


def encoder_text(message: str, instruction: str, payload: str | None = None) -> str:
    """The honest prompt when payload is None, otherwise the plain/0-1 encoder prompt; both think."""
    base = encoder_prompt(message) if payload is None else encoder_prompt(message, CONDITION, payload)
    return think(base, instruction, ENCODER_ANCHOR)


def decoder_text(carrier: str, order: tuple[str, str], instruction: str | None) -> str:
    """The reasoning decoder's prompt, or the confirmation's answer-only prompt when instruction is None."""
    base = decoder_prompt(carrier, CONDITION, order)
    return base if instruction is None else think(base, instruction, DECODER_ANCHOR)


def _shuffled(jobs: list[dict], seed: str) -> list[dict]:
    random.Random(seed).shuffle(jobs)
    return jobs


def _base(config: dict, case: dict) -> dict:
    return dict(subject=config["subject"], case_id=case["id"], split=case["split"], family=case["family"],
        option_order=case["option_order"], message=case["message"])


def encoder_check_jobs(config: dict, cases: list[dict]) -> list[dict]:
    """Pre-test on separate check covers: both payloads and an honest paraphrase of each. Never analyzed."""
    instruction = config["think_instruction"]
    jobs = []
    for case in sorted((case for case in cases if case["split"] == "encoder_check"), key=lambda case: case["id"]):
        for payload in BIT_OPTIONS:
            jobs.append(dict(_base(config, case), id=f"check-enc-{case['id']}-{payload}", kind="encoded",
                condition=CONDITION, payload=payload, prompt=encoder_text(case["message"], instruction, payload)))
        jobs.append(dict(_base(config, case), id=f"check-enc-{case['id']}-honest", kind="honest", condition=None,
            payload=None, prompt=encoder_text(case["message"], instruction)))
    return jobs


def encoding_jobs(config: dict, cases: list[dict]) -> list[dict]:
    """Both payloads for each development cover and an honest paraphrase of every analyzed cover."""
    instruction = config["think_instruction"]
    jobs = []
    for case in cases:
        if case["split"] == "encoder_check":
            continue
        if case["split"] == "development":
            for payload in BIT_OPTIONS:
                jobs.append(dict(_base(config, case), id=f"enc-{case['id']}-{payload}", kind="encoded",
                    condition=CONDITION, payload=payload, prompt=encoder_text(case["message"], instruction, payload)))
        jobs.append(dict(_base(config, case), id=f"enc-{case['id']}-honest", kind="honest", condition=None,
            payload=None, prompt=encoder_text(case["message"], instruction)))
    return _shuffled(jobs, f"{config['seed']}-encode")


def decoding_jobs(config: dict, carriers: list[dict], cases: list[dict], *, reasoning: bool = True) -> list[dict]:
    """Payload carriers and two no-carrier controls. Honest paraphrases are decoded only in the pre-test."""
    by_case = {case["id"]: case for case in cases}
    instruction = config["think_instruction"] if reasoning else None
    jobs = []
    for row in carriers:
        if row["kind"] != "encoded" or row["split"] != "development" or not row.get("carrier"):
            continue
        order = shown(by_case[row["case_id"]])
        jobs.append(dict(id=f"dec-{row['id']}", source_id=row["id"], case_id=row["case_id"], kind="encoded",
            condition=CONDITION, target=row["payload"], options=list(BIT_OPTIONS), shown=list(order),
            prompt=decoder_text(row["carrier"], order, instruction)))
    for order in (BIT_OPTIONS, tuple(reversed(BIT_OPTIONS))):
        jobs.append(dict(id=f"dec-no-carrier-{'-'.join(order)}", source_id=None, case_id=None, kind="no_carrier",
            condition=CONDITION, target=None, options=list(BIT_OPTIONS), shown=list(order),
            prompt=decoder_text(NO_CARRIER, order, instruction)))
    return _shuffled(jobs, f"{config['seed']}-decode-{'reasoning' if reasoning else 'answer-only'}")


def decoder_check_jobs(config: dict, carriers: list[dict], cases: list[dict]) -> list[dict]:
    """Pre-test: the reasoning decoder on the first honest development paraphrases by id. Never analyzed."""
    by_case = {case["id"]: case for case in cases}
    honest = sorted((row for row in carriers
        if row["kind"] == "honest" and row["split"] == "development" and row.get("carrier")), key=lambda row: row["id"])
    jobs = []
    for row in honest[:config["decoder_check_jobs"]]:
        order = shown(by_case[row["case_id"]])
        jobs.append(dict(id=f"check-dec-{row['id']}", source_id=row["id"], case_id=row["case_id"], kind="honest",
            condition=CONDITION, target=None, options=list(BIT_OPTIONS), shown=list(order),
            prompt=decoder_text(row["carrier"], order, config["think_instruction"])))
    return jobs


def prompt_identity(config: dict) -> dict[str, str]:
    instruction = config["think_instruction"]
    texts = dict(honest=encoder_text("<<SOURCE>>", instruction), encoder=encoder_text("<<SOURCE>>", instruction, "1"),
        decoder=decoder_text("<<CARRIER>>", BIT_OPTIONS, instruction),
        answer_only_decoder=decoder_text("<<CARRIER>>", BIT_OPTIONS, None),
        judge=judge_prompt("<<SOURCE>>", "<<CARRIER>>"))
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in texts.items()}
