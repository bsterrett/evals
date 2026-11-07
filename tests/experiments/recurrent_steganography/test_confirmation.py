from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, redirect_stdout
from datetime import date
from decimal import Decimal
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from eval_core.budget import BudgetExceeded, BudgetLedger
from eval_core.models import BudgetSettings, GenerationResponse, TokenUsage
from eval_core.providers import Provider
from eval_core.workspace import Workspace
from recurrent_steg.confirmation.analysis import analyze
from recurrent_steg.confirmation.design import (BIT_OPTIONS, PROTOCOL, encoding_jobs, decoding_jobs, load_design,
    judge_jobs)
from recurrent_steg.confirmation.execution import PHASES, execute
from recurrent_steg.design import decoder_prompt as pilot_decoder_prompt
from recurrent_steg.design import encoder_prompt as pilot_encoder_prompt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/recurrent_steganography/data"
CONFIG = ROOT / "experiments/recurrent_steganography/configs/confirmation-v1.toml"


def sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


class Fake(Provider):
    """Copies covers; when planted, payload 1 swaps sentence order and the decoder reads that swap."""
    plant = True
    swapped: set[str] = set()
    break_judge = False
    malformed = False

    def __init__(self, config, dry_run=False):
        super().__init__(config)

    def count_input_tokens(self, request):
        return len(request.prompt.split())

    def generate(self, request):
        prompt = request.prompt
        if request.stage == "encode":
            source = prompt.split("Source message:\n", 1)[1]
            text = source
            bit = re.search(r"Secret bit: (\d)\.", prompt)
            if Fake.plant and bit and bit.group(1) == "1":
                first, second = sentences(source)
                text = f"{second} {first}"
                Fake.swapped.add(text)
        elif request.stage in ("decoder-check", "decode"):
            assert len(request.metadata["answer_choices"]) == 2
            carrier = prompt.split("Text:\n", 1)[1]
            text = "not json" if Fake.malformed else json.dumps({"answer": "1" if carrier in Fake.swapped else "0"})
        else:
            source, carrier = prompt.split("Source message:\n", 1)[1].split("\n\nProposed paraphrase:\n", 1)
            overt = "secret" in carrier.lower()
            text = json.dumps(dict(facts_preserved=set(sentences(source)) <= set(sentences(carrier)),
                added_claims=bool(set(sentences(carrier)) - set(sentences(source))), overt_payload=overt,
                suspicion=1.0 if overt else 0.0))
            if Fake.break_judge:
                text = text[:-1] + ', "reason": "extra"}'
        return GenerationResponse(text, self.config.model, TokenUsage(self.count_input_tokens(request),
            max(1, len(text.split()))), "fake", "completed", "end_turn", {"fake": True})


class DesignTests(unittest.TestCase):
    def test_prompts_are_the_frozen_pilot_prompts_and_covers_are_balanced(self):
        config, cases, _, _ = load_design(CONFIG)
        development = [case for case in cases if case["split"] == "development"]
        self.assertEqual(len(development), 288)
        self.assertEqual(Counter(case["option_order"] for case in development), Counter(forward=144, reversed=144))
        for family in {case["family"] for case in development}:
            orders = Counter(c["option_order"] for c in development if c["family"] == family)
            self.assertEqual(orders, Counter(forward=18, reversed=18))
        jobs = encoding_jobs(config, cases)
        self.assertEqual(len(jobs), 888)
        encoded = next(job for job in jobs if job["kind"] == "encoded" and job["payload"] == "1")
        self.assertEqual(encoded["prompt"], pilot_encoder_prompt(encoded["message"], 1))
        honest = next(job for job in jobs if job["kind"] == "honest")
        self.assertEqual(honest["prompt"], pilot_encoder_prompt(honest["message"], None))
        carriers = [dict(job, carrier="Text.") for job in jobs]
        decodes = decoding_jobs(config, carriers, cases)
        self.assertEqual(len(decodes), 866)
        forward = next(job for job in decodes if job["shown"] == list(BIT_OPTIONS) and job["kind"] == "encoded")
        self.assertEqual(forward["prompt"], pilot_decoder_prompt("Text."))
        self.assertTrue(all("Secret" not in job["prompt"] for job in decodes + judge_jobs(config, carriers)))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)

    def setUp(self):
        Fake.plant, Fake.swapped, Fake.break_judge, Fake.malformed = True, set(), False, False

    def build(self, root: Path, stack: ExitStack, budget: str = "6.00") -> tuple[Path, Workspace]:
        (root / "experiment/configs").mkdir(parents=True)
        (root / "experiment/data").mkdir()
        text = CONFIG.read_text(encoding="utf-8").replace("2026-11-06", date.today().isoformat())
        text = text.replace('run_budget_usd = "6.00"', f'run_budget_usd = "{budget}"')
        (root / "experiment/configs/confirmation-v1.toml").write_text(text, encoding="utf-8")
        for name in ("confirmation-covers-v1.json", "covers-v1.json", "reference-covers-v1.json"):
            (root / "experiment/data" / name).write_bytes((DATA / name).read_bytes())
        workspace = Workspace(root, BudgetSettings(Decimal(50), root / "ledger.sqlite3", 30))
        frozen = dict(protocol=PROTOCOL, config_path="experiment/configs/confirmation-v1.toml")
        stack.enter_context(patch("recurrent_steg.confirmation.cli.verify_freeze", return_value=frozen))
        stack.enter_context(patch("recurrent_steg.confirmation.execution.load_workspace", return_value=workspace))
        stack.enter_context(patch("recurrent_steg.confirmation.analysis.load_workspace", return_value=workspace))
        stack.enter_context(redirect_stdout(io.StringIO()))
        run_dir = root / "run"
        run_dir.mkdir()
        return run_dir, workspace

    def run_all(self, run_dir: Path) -> dict:
        for phase in PHASES:
            execute(run_dir, phase, live=True, provider_factory=Fake)
        return analyze(run_dir)

    def test_a_planted_channel_is_confirmed_and_spend_is_settled(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, workspace = self.build(Path(directory).resolve(), stack)
            result = self.run_all(run_dir)
            self.assertTrue(result["judge_qualification"]["qualified"])
            self.assertTrue(result["confirmed"])
            self.assertEqual(result["primary"]["ci95"], [1, 1])
            self.assertEqual(result["primary"]["counts"], {"+1": 288, "0": 0, "-1": 0})
            ledger = BudgetLedger(workspace.budget.ledger, Decimal(50))
            self.assertEqual(ledger.pending, 0)
            self.assertEqual(Decimal(result["total_spend_usd"]), ledger.spent)
            self.assertTrue((run_dir / "blinded-sample-sonnet55.json").exists())

    def test_identical_carriers_cannot_confirm(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, _ = self.build(Path(directory).resolve(), stack)
            Fake.plant = False
            result = self.run_all(run_dir)
            self.assertFalse(result["confirmed"])
            self.assertEqual(result["tallies"]["identical_pairs"], 288)
            self.assertEqual(result["primary"]["counts"], {"+1": 0, "0": 288, "-1": 0})

    def test_gates_and_ceiling_stop_spending(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, _ = self.build(Path(directory).resolve(), stack)
            Fake.break_judge = True
            execute(run_dir, "judge-controls", live=True, provider_factory=Fake)
            with self.assertRaisesRegex(ValueError, "control check"):
                execute(run_dir, "encode", live=True, provider_factory=Fake)
        Fake.break_judge = False
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, _ = self.build(Path(directory).resolve(), stack)
            for phase in ("judge-controls", "encode"):
                execute(run_dir, phase, live=True, provider_factory=Fake)
            Fake.malformed = True
            execute(run_dir, "decoder-check", live=True, provider_factory=Fake)
            with self.assertRaisesRegex(ValueError, "pre-test"):
                execute(run_dir, "decode", live=True, provider_factory=Fake)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, _ = self.build(Path(directory).resolve(), stack, budget="0.0100")
            with self.assertRaises(BudgetExceeded):
                execute(run_dir, "judge-controls", live=True, provider_factory=Fake)


if __name__ == "__main__":
    unittest.main()
