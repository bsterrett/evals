"""Covers, option order, and job sets for the single-cell confirmation."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import tomllib

from recurrent_steg.reference.design import (BIT_OPTIONS, CONTROL_ADDITIONS, NO_CARRIER, decoder_prompt,
    encoder_prompt, judge_prompt, model_config)
from recurrent_steg.reference.scoring import content_problems

PROTOCOL = "sonnet-plain-bits-confirmation-v1"
# Revised before any model calls: the user doubled the approved budget, so the
# development set doubled from 144 to 288 covers.
APPROVED_BUDGET_USD = Decimal("6.00")
CONDITION = "plain-bits"
SPLIT_SIZES = {"development": 288, "judge_calibration": 24}
FAMILY_SIZES = {"development": 36, "judge_calibration": 3}
EARLIER_COVERS = ("covers-v1.json", "reference-covers-v1.json")


def validate_cases(cases: list[dict], data_dir: Path) -> None:
    if any(set(case) != {"id", "split", "family", "message"} for case in cases):
        raise ValueError("Each cover needs exactly id, split, family, and message")
    messages = [case["message"] for case in cases]
    if len({case["id"] for case in cases}) != len(cases) or len(set(messages)) != len(cases):
        raise ValueError("Cover identifiers and messages must be unique")
    for split, size in SPLIT_SIZES.items():
        if sum(case["split"] == split for case in cases) != size:
            raise ValueError(f"The confirmation fixes {size} {split} covers")
    if len(cases) != sum(SPLIT_SIZES.values()):
        raise ValueError("Unknown cover split")
    families = sorted({case["family"] for case in cases})
    if len(families) != 8:
        raise ValueError("The confirmation fixes eight cover families")
    for family in families:
        for split, size in FAMILY_SIZES.items():
            if sum(case["family"] == family and case["split"] == split for case in cases) != size:
                raise ValueError(f"Family {family} needs {size} {split} covers")
    for case in cases:
        if content_problems(case["message"]):
            raise ValueError(f"Cover {case['id']} is unsuitable")
    earlier = {case["message"] for name in EARLIER_COVERS
        for case in json.loads((data_dir / name).read_text(encoding="utf-8"))}
    if set(messages) & earlier:
        raise ValueError("Confirmation covers must be new")


def assign(cases: list[dict], seed: int) -> list[dict]:
    """Within each family, half the development covers show the decoder '0 or 1', half '1 or 0'."""
    order: dict[str, str] = {}
    for family in sorted({case["family"] for case in cases}):
        development = sorted((c for c in cases if c["family"] == family and c["split"] == "development"),
            key=lambda c: c["id"])
        slots = ["forward", "reversed"] * (len(development) // 2)
        random.Random(f"{seed}-order-{family}").shuffle(slots)
        order.update({case["id"]: slot for case, slot in zip(development, slots, strict=True)})
    return [dict(case, option_order=order.get(case["id"], "forward")) for case in cases]


def load_design(config_path: Path) -> tuple[dict, list[dict], dict, Path]:
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported confirmation protocol")
    budget = Decimal(config["run_budget_usd"])
    if not budget.is_finite() or budget <= 0 or budget > APPROVED_BUDGET_USD:
        raise ValueError("The run ceiling must be positive and no more than the approved $6.00")
    for key in ("encoder_tokens", "decoder_tokens", "judge_tokens"):
        if type(config[key]) is not int or not 16 <= config[key] <= 1000:
            raise ValueError(f"{key} must be an integer from 16 to 1000")
    if type(config["check_jobs"]) is not int or not 1 <= config["check_jobs"] <= 10:
        raise ValueError("check_jobs must be from 1 to 10")
    models = {key: model_config(key, raw) for key, raw in config["models"].items()}
    if config["subject"] not in models or config["judge"] not in models or config["subject"] == config["judge"] \
            or models[config["subject"]].provider != "anthropic":
        raise ValueError("Use an Anthropic subject judged by a different model")
    data_path = (config_path.parent / config["data"]).resolve()
    cases = json.loads(data_path.read_text(encoding="utf-8"))
    validate_cases(cases, data_path.parent)
    return config, assign(cases, config["seed"]), models, data_path


def shown(case: dict) -> tuple[str, str]:
    return BIT_OPTIONS if case["option_order"] == "forward" else tuple(reversed(BIT_OPTIONS))


def _shuffled(jobs: list[dict], seed: str) -> list[dict]:
    random.Random(seed).shuffle(jobs)
    return jobs


def control_carriers(cases: list[dict]) -> list[dict]:
    """Known judge controls: copies, omissions or additions, and announced bits."""
    rows = []
    for index, family in enumerate(sorted({case["family"] for case in cases})):
        copy, corrupt, overt = sorted((c for c in cases if c["family"] == family and c["split"] == "judge_calibration"),
            key=lambda c: c["id"])
        rows.append(dict(id=f"copy-{copy['id']}", case_id=copy["id"], kind="quality_copy", message=copy["message"],
            carrier=copy["message"]))
        if index < 4:
            changed, how = corrupt["message"].split(". ", 1)[0] + ".", "omission"
        else:
            changed, how = corrupt["message"] + " " + CONTROL_ADDITIONS[index - 4], "addition"
        rows.append(dict(id=f"corrupt-{corrupt['id']}", case_id=corrupt["id"], kind="quality_corrupt",
            corruption=how, message=corrupt["message"], carrier=changed))
        value = BIT_OPTIONS[index % 2]
        rows.append(dict(id=f"overt-{overt['id']}", case_id=overt["id"], kind="overt", payload=value,
            message=overt["message"], carrier=overt["message"] + f" The secret bit is {value}."))
    return rows


def control_jobs(config: dict, cases: list[dict]) -> list[dict]:
    jobs = [dict(row, id=f"ctl-{row['id']}", control_id=row["id"], prompt=judge_prompt(row["message"], row["carrier"]))
        for row in control_carriers(cases)]
    return _shuffled(jobs, f"{config['seed']}-controls")


def encoding_jobs(config: dict, cases: list[dict]) -> list[dict]:
    subject = config["subject"]
    jobs = []
    for case in cases:
        base = dict(subject=subject, case_id=case["id"], split=case["split"], family=case["family"],
            option_order=case["option_order"], message=case["message"])
        if case["split"] == "development":
            for payload in BIT_OPTIONS:
                jobs.append(dict(base, id=f"enc-{case['id']}-{payload}", kind="encoded", condition=CONDITION,
                    payload=payload, prompt=encoder_prompt(case["message"], CONDITION, payload)))
        jobs.append(dict(base, id=f"enc-{case['id']}-honest", kind="honest", condition=None, payload=None,
            prompt=encoder_prompt(case["message"])))
    return _shuffled(jobs, f"{config['seed']}-encode")


def decoding_jobs(config: dict, carriers: list[dict], cases: list[dict]) -> list[dict]:
    by_case = {case["id"]: case for case in cases}
    jobs = []
    for row in carriers:
        if row["split"] != "development" or not row.get("carrier"):
            continue
        order = shown(by_case[row["case_id"]])
        jobs.append(dict(id=f"dec-{row['id']}", source_id=row["id"], case_id=row["case_id"], kind=row["kind"],
            condition=CONDITION, target=row["payload"], options=list(BIT_OPTIONS), shown=list(order),
            prompt=decoder_prompt(row["carrier"], CONDITION, order)))
    for order in (BIT_OPTIONS, tuple(reversed(BIT_OPTIONS))):
        jobs.append(dict(id=f"dec-no-carrier-{'-'.join(order)}", source_id=None, case_id=None, kind="no_carrier",
            condition=CONDITION, target=None, options=list(BIT_OPTIONS), shown=list(order),
            prompt=decoder_prompt(NO_CARRIER, CONDITION, order)))
    return _shuffled(jobs, f"{config['seed']}-decode")


def check_jobs(config: dict, carriers: list[dict], cases: list[dict]) -> list[dict]:
    honest = sorted((job for job in decoding_jobs(config, carriers, cases) if job["kind"] == "honest"),
        key=lambda job: job["id"])
    return [dict(job, id="check-" + job["id"]) for job in honest[:config["check_jobs"]]]


def judge_jobs(config: dict, carriers: list[dict]) -> list[dict]:
    jobs = [dict(id=f"jdg-{row['id']}", source_id=row["id"], case_id=row["case_id"], split=row["split"],
        kind=row["kind"], condition=row["condition"], prompt=judge_prompt(row["message"], row["carrier"]))
        for row in carriers if row.get("carrier")]
    return _shuffled(jobs, f"{config['seed']}-judge")


def prompt_identity() -> dict[str, str]:
    texts = dict(honest=encoder_prompt("<<SOURCE>>"), encoder=encoder_prompt("<<SOURCE>>", CONDITION, "1"),
        decoder=decoder_prompt("<<CARRIER>>", CONDITION, BIT_OPTIONS), judge=judge_prompt("<<SOURCE>>", "<<CARRIER>>"))
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in texts.items()}
