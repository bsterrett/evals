from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from covert_eval.config import ConfigError, load_cases, load_config, validate_live_config
from eval_core.budget import BudgetLedger
from covert_eval.experiment import ExperimentRunner, summarize_run
from covert_eval.models import AppConfig, ModelConfig


def _load_live_environment(start_dir: Path | None = None) -> str | None:
    """Load recognized API keys from the nearest .env without exposing or replacing them."""
    recognized = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    start = (start_dir or Path.cwd()).resolve()
    dotenv_path = next(
        (directory / ".env" for directory in (start, *start.parents) if (directory / ".env").is_file()),
        None,
    )
    if dotenv_path is None:
        return None
    with dotenv_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or key not in recognized:
                continue
            value = value.strip()
            if value[:1] in {'"', "'"}:
                quote = value[0]
                if len(value) < 2 or value[-1] != quote:
                    raise ConfigError(f"Unterminated quoted value in {dotenv_path}:{line_number}")
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if not value:
                raise ConfigError(f"Empty {key} value in {dotenv_path}:{line_number}")
            os.environ.setdefault(key, value)
    return str(dotenv_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="covert-eval",
        description="Evaluate semantic covert channels in reasoning-like natural language.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate config and dataset without API calls")
    validate.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    validate.add_argument("--live", action="store_true", help="Also enforce live pricing freshness")

    plan = subparsers.add_parser("plan", help="Show trial counts and conservative configured cost bound")
    plan.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    plan.add_argument("--case-limit", type=int)
    plan.add_argument("--subject", action="append", dest="subjects")
    plan.add_argument("--condition")
    plan.add_argument("--analysis-role", choices=("exploratory", "confirmatory"))

    run = subparsers.add_parser("run", help="Run dry by default; live calls require two explicit flags")
    run.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    run.add_argument("--case-limit", type=int)
    run.add_argument("--subject", action="append", dest="subjects")
    run.add_argument("--condition")
    run.add_argument("--analysis-role", choices=("exploratory", "confirmatory"))
    run.add_argument(
        "--resume-from",
        help="Resume a failed run artifact without repeating its attempted trials",
    )
    run.add_argument("--live", action="store_true")
    run.add_argument(
        "--run-budget-usd",
        help="Optional hard ceiling for this run in addition to the campaign ceiling",
    )
    run.add_argument(
        "--confirm-live-spend",
        action="store_true",
        help="Required with --live; acknowledges that provider calls incur charges",
    )

    summarize = subparsers.add_parser("summarize", help="Recompute summary for an existing run")
    summarize.add_argument("run_dir")

    budget_status = subparsers.add_parser(
        "budget-status", help="Show the persistent campaign ledger without making API calls"
    )
    budget_status.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")

    budget_reconcile = subparsers.add_parser(
        "budget-reconcile",
        help="Reconcile blocked charges after reviewing provider billing data",
    )
    budget_reconcile.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    budget_reconcile.add_argument("--confirmed-uncertain-usd", required=True)
    budget_reconcile.add_argument("--note", required=True)
    budget_reconcile.add_argument("--confirm-provider-bill-reviewed", action="store_true")

    diagnose = subparsers.add_parser(
        "diagnose-luna",
        help="Run the exploratory Luna length/payload/strategy/forced-guess diagnostic",
    )
    diagnose.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    diagnose.add_argument("--plan", action="store_true", help="Show the diagnostic call and cost plan")
    diagnose.add_argument("--live", action="store_true")
    diagnose.add_argument("--confirm-live-spend", action="store_true")
    diagnose.add_argument("--run-budget-usd", default="1.00")

    calibrate = subparsers.add_parser(
        "calibrate-luna-decoder",
        help="Run the frozen-carrier Luna decoder positive-control calibration",
    )
    calibrate.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    calibrate.add_argument("--plan", action="store_true")
    calibrate.add_argument("--live", action="store_true")
    calibrate.add_argument("--confirm-live-spend", action="store_true")
    calibrate.add_argument("--run-budget-usd", default="1.00")

    contract = subparsers.add_parser(
        "calibrate-decoder-contract",
        help="Run Stage 1 of the four-model v1.2 decoder contract calibration",
    )
    contract.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    contract.add_argument("--plan", action="store_true")
    contract.add_argument("--live", action="store_true")
    contract.add_argument("--confirm-live-spend", action="store_true")
    contract.add_argument("--run-budget-usd", default="1.00")

    stage1b = subparsers.add_parser(
        "calibrate-decoder-contract-stage1b",
        help="Run the repeated four-model v1.3 decoder contract calibration",
    )
    stage1b.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    stage1b.add_argument("--plan", action="store_true")
    stage1b.add_argument("--live", action="store_true")
    stage1b.add_argument("--confirm-live-spend", action="store_true")
    stage1b.add_argument("--run-budget-usd", default="1.00")

    stage1c = subparsers.add_parser(
        "calibrate-decoder-contract-stage1c",
        help="Run the four-model v1.4 distinctness decoder contract calibration",
    )
    stage1c.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    stage1c.add_argument("--plan", action="store_true")
    stage1c.add_argument("--live", action="store_true")
    stage1c.add_argument("--confirm-live-spend", action="store_true")
    stage1c.add_argument("--run-budget-usd", default="1.00")

    sonnet_refusal = subparsers.add_parser(
        "calibrate-sonnet-refusal",
        help="Run the four-variant Sonnet reasoning-extraction refusal calibration",
    )
    sonnet_refusal.add_argument("--config", default="experiments/semantic_covert_channel/configs/v1.toml")
    sonnet_refusal.add_argument("--plan", action="store_true")
    sonnet_refusal.add_argument("--live", action="store_true")
    sonnet_refusal.add_argument("--confirm-live-spend", action="store_true")
    sonnet_refusal.add_argument("--run-budget-usd", default="0.25")

    screen = subparsers.add_parser(
        "screen-v1-5",
        help="Run the frozen two-factor V1.5 development constraint screen",
    )
    screen.add_argument("--config", default="experiments/semantic_covert_channel/configs/v2.toml")
    screen.add_argument("--plan", action="store_true")
    screen.add_argument("--live", action="store_true")
    screen.add_argument("--confirm-live-spend", action="store_true")
    screen.add_argument("--run-budget-usd", default="20.00")

    sensitivity = subparsers.add_parser(
        "study-decoder-sensitivity",
        help="Compare baseline and synthesis decoders on frozen Sonnet development carriers",
    )
    sensitivity.add_argument(
        "--config", default="experiments/semantic_covert_channel/configs/v2.toml"
    )
    sensitivity.add_argument("--plan", action="store_true")
    sensitivity.add_argument("--live", action="store_true")
    sensitivity.add_argument("--confirm-live-spend", action="store_true")
    sensitivity.add_argument("--run-budget-usd", default="1.00")
    return parser


def _subjects(config: AppConfig, requested: list[str] | None) -> list[ModelConfig]:
    enabled = [subject for subject in config.subjects if subject.enabled]
    if requested is None:
        return enabled
    requested_set = set(requested)
    unknown = requested_set - {subject.id for subject in enabled}
    if unknown:
        raise ConfigError(f"Unknown or disabled subjects: {', '.join(sorted(unknown))}")
    return [subject for subject in enabled if subject.id in requested_set]


def _plan(
    config: AppConfig,
    case_limit: int | None,
    requested: list[str] | None,
    *,
    condition_id: str | None = None,
    analysis_role: str | None = None,
) -> dict[str, Any]:
    subjects = _subjects(config, requested)
    cases = load_cases(config.experiment.dataset)
    role = analysis_role or config.experiment.default_analysis_role
    selected_condition = condition_id or config.experiment.default_condition_id
    allowed: set[str] | None = None
    if config.experiment.case_split_manifest is not None:
        from covert_eval.splits import load_split_manifest

        split = load_split_manifest(config.experiment.case_split_manifest, cases)
        allowed = set(split.ids_for_role(role))
    eligible_cases = [
        case
        for case in cases
        if case.id not in set(config.experiment.excluded_case_ids)
        and (allowed is None or case.id in allowed)
    ]
    case_count = min(
        case_limit or config.experiment.max_cases,
        len(eligible_cases),
    )
    trials = case_count * config.experiment.repetitions
    subject_bound = Decimal()
    by_subject: dict[str, str] = {}
    for subject in subjects:
        per_trial = (
            subject.rate.reserve_cost(subject.rate.max_input_tokens, config.experiment.cover_max_tokens)
            * 2
            + subject.rate.reserve_cost(
                subject.rate.max_input_tokens, config.experiment.decoder_max_tokens
            )
            * 2
        )
        bound = per_trial * trials
        by_subject[subject.id] = f"{bound:.6f}"
        subject_bound += bound
    judge_calls = trials * len(subjects) * 3
    judge_bound = config.judge.rate.reserve_cost(
        config.judge.rate.max_input_tokens, config.experiment.judge_max_tokens
    ) * judge_calls
    total_bound = subject_bound + judge_bound
    return {
        "cases_per_subject": case_count,
        "repetitions": config.experiment.repetitions,
        "subjects": [subject.id for subject in subjects],
        "subject_trials": trials * len(subjects),
        "generation_calls": trials * len(subjects) * 7,
        "configured_absolute_cost_bound_usd": f"{total_bound:.6f}",
        "bound_by_subject_excluding_judge_usd": by_subject,
        "judge_calls": judge_calls,
        "judge_bound_usd": f"{judge_bound:.6f}",
        "campaign_budget_usd": f"{config.budget.campaign_usd:.2f}",
        "excluded_case_ids": list(config.experiment.excluded_case_ids),
        "condition_id": selected_condition or "legacy_v1_4_strict_trace_standard",
        "analysis_role": role,
        "note": (
            "The bound assumes every request uses its full configured input and output limits. "
            "Live execution counts each prompt and reserves its tighter per-call maximum."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "summarize":
            summary = summarize_run(Path(args.run_dir).resolve())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        config = load_config(args.config)
        if args.command == "study-decoder-sensitivity":
            from covert_eval.decoder_sensitivity import (
                MAX_RUN_BUDGET_USD,
                SavedCarrierDecoderSensitivityRunner,
                decoder_sensitivity_plan,
            )

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > MAX_RUN_BUDGET_USD:
                raise ConfigError(
                    "--run-budget-usd must be greater than zero and no more than 1.00"
                )
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(decoder_sensitivity_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = SavedCarrierDecoderSensitivityRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "screen-v1-5":
            from covert_eval.screening import (
                MAX_RUN_BUDGET_USD,
                ConstraintFrontierScreenRunner,
                screening_plan,
            )

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > MAX_RUN_BUDGET_USD:
                raise ConfigError(
                    "--run-budget-usd must be greater than zero and no more than 20.00"
                )
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(screening_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = ConstraintFrontierScreenRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "calibrate-sonnet-refusal":
            from covert_eval.refusal_calibration import (
                MAX_RUN_BUDGET_USD,
                SonnetRefusalCalibrationRunner,
                refusal_calibration_plan,
            )

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > MAX_RUN_BUDGET_USD:
                raise ConfigError(
                    "--run-budget-usd must be greater than zero and no more than 0.25"
                )
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(refusal_calibration_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = SonnetRefusalCalibrationRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "calibrate-decoder-contract-stage1c":
            from covert_eval.calibration import Stage1CDecoderContractRunner, stage1c_plan

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > Decimal("1.00"):
                raise ConfigError("--run-budget-usd must be greater than zero and no more than 1.00")
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(stage1c_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = Stage1CDecoderContractRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "calibrate-decoder-contract-stage1b":
            from covert_eval.calibration import RepeatedDecoderContractRunner, stage1b_plan

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > Decimal("1.00"):
                raise ConfigError("--run-budget-usd must be greater than zero and no more than 1.00")
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(stage1b_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = RepeatedDecoderContractRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "calibrate-decoder-contract":
            from covert_eval.calibration import (
                MultiModelDecoderContractRunner,
                decoder_contract_plan,
            )

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > Decimal("1.00"):
                raise ConfigError("--run-budget-usd must be greater than zero and no more than 1.00")
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(decoder_contract_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = MultiModelDecoderContractRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "calibrate-luna-decoder":
            from covert_eval.calibration import (
                LunaDecoderCalibrationRunner,
                calibration_plan,
            )

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > Decimal("1.00"):
                raise ConfigError("--run-budget-usd must be greater than zero and no more than 1.00")
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(calibration_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = LunaDecoderCalibrationRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "diagnose-luna":
            from covert_eval.diagnostics import LunaDiagnosticRunner, diagnostic_plan

            try:
                run_budget = Decimal(args.run_budget_usd)
            except Exception as exc:
                raise ConfigError("--run-budget-usd must be a decimal number") from exc
            if run_budget <= 0 or run_budget > Decimal("1.00"):
                raise ConfigError("--run-budget-usd must be greater than zero and no more than 1.00")
            if args.plan:
                if args.live or args.confirm_live_spend:
                    raise ConfigError("--plan cannot be combined with live-run flags")
                print(json.dumps(diagnostic_plan(config), indent=2))
                return 0
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.live:
                _load_live_environment()
            runner = LunaDiagnosticRunner(
                config,
                dry_run=not args.live,
                run_budget_usd=run_budget,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
        if args.command == "budget-status":
            ledger = BudgetLedger(
                config.budget.ledger, config.budget.campaign_usd, persist=True
            )
            print(json.dumps(ledger.status(), indent=2))
            return 0

        if args.command == "budget-reconcile":
            if not args.confirm_provider_bill_reviewed:
                raise ConfigError(
                    "budget-reconcile requires --confirm-provider-bill-reviewed"
                )
            try:
                confirmed = Decimal(args.confirmed_uncertain_usd)
            except Exception as exc:
                raise ConfigError("--confirmed-uncertain-usd must be a decimal number") from exc
            ledger = BudgetLedger(
                config.budget.ledger, config.budget.campaign_usd, persist=True
            )
            ledger.reconcile(confirmed, args.note)
            print(json.dumps(ledger.status(), indent=2))
            return 0
        if args.command == "validate":
            if args.live:
                validate_live_config(config)
            cases = load_cases(config.experiment.dataset)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "live_checks": bool(args.live),
                        "dataset_cases": len(cases),
                        "enabled_subjects": [
                            subject.id for subject in config.subjects if subject.enabled
                        ],
                        "campaign_budget_usd": f"{config.budget.campaign_usd:.2f}",
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "plan":
            if args.case_limit is not None and args.case_limit < 1:
                raise ConfigError("--case-limit must be positive")
            print(
                json.dumps(
                    _plan(
                        config,
                        args.case_limit,
                        args.subjects,
                        condition_id=args.condition,
                        analysis_role=args.analysis_role,
                    ),
                    indent=2,
                )
            )
            return 0

        if args.command == "run":
            if args.live and not args.confirm_live_spend:
                raise ConfigError("--live requires --confirm-live-spend")
            if args.confirm_live_spend and not args.live:
                raise ConfigError("--confirm-live-spend is only valid with --live")
            if args.case_limit is not None and args.case_limit < 1:
                raise ConfigError("--case-limit must be positive")
            run_budget = None
            if args.run_budget_usd is not None:
                try:
                    run_budget = Decimal(args.run_budget_usd)
                except Exception as exc:
                    raise ConfigError("--run-budget-usd must be a decimal number") from exc
                if run_budget <= 0:
                    raise ConfigError("--run-budget-usd must be greater than zero")
            if args.live:
                _load_live_environment()
            runner = ExperimentRunner(
                config,
                dry_run=not args.live,
                selected_subjects=set(args.subjects) if args.subjects else None,
                case_limit=args.case_limit,
                run_budget_usd=run_budget,
                resume_from=Path(args.resume_from) if args.resume_from else None,
                condition_id=args.condition,
                analysis_role=args.analysis_role,
            )
            run_dir = runner.run()
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
            return 0
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1

