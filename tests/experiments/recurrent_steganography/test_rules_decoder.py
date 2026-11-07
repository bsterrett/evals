from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import date
from decimal import Decimal
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from eval_core.artifacts import file_sha256
from eval_core.budget import BudgetExceeded, BudgetLedger
from eval_core.models import BudgetSettings, GenerationResponse, TokenUsage
from eval_core.providers import Provider
from eval_core.workspace import Workspace
from recurrent_steg.both_reasoning import design as parent_design
from recurrent_steg.both_reasoning.analysis import analyze as analyze_parent
from recurrent_steg.both_reasoning.design import PROTOCOL as PARENT_PROTOCOL
from recurrent_steg.both_reasoning.execution import PHASES as PARENT_PHASES, execute as execute_parent
from recurrent_steg.rules_decoder.analysis import analyze, sentence_answer
from recurrent_steg.rules_decoder.design import (PROTOCOL, RULES_INTRO, WITHHELD, answer_only_jobs, check_jobs,
    decode_jobs, load, rules_block, with_rules, writer_instructions)
from recurrent_steg.rules_decoder.execution import PHASES, execute
from tests.experiments.recurrent_steganography.test_both_reasoning import COVER_FILES, Fake as ParentFake

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/recurrent_steganography/data"
PARENT_CONFIG = ROOT / "experiments/recurrent_steganography/configs/both-reasoning-v1.toml"
CONFIG = ROOT / "experiments/recurrent_steganography/configs/rules-decoder-v1.toml"
PARENT_FILES = dict(plan="plan.json", encode="encode-sonnet55.jsonl", decode="decode-sonnet55.jsonl",
    answer_only="decode-answer-only-sonnet55.jsonl", analysis="analysis-v1.json")


class RulesFake(Provider):
    """Requires the writer's instructions in every prompt, then reads the parent's planted sentence swap."""
    swapped: set[str] = set()
    thinking = 120
    output_extra = 0
    malformed = False

    def __init__(self, config, dry_run=False):
        super().__init__(config)

    def count_input_tokens(self, request):
        return len(request.prompt.split())

    def generate(self, request):
        assert request.stage in PHASES and len(request.metadata["answer_choices"]) == 2
        assert RULES_INTRO in request.prompt and "<writer_instructions>" in request.prompt
        thinks = request.stage != "decode-answer-only"
        carrier = request.prompt.split("Text:\n", 1)[1]
        answer = "1" if carrier in RulesFake.swapped else "0"
        text = "not json" if RulesFake.malformed and thinks else json.dumps({"answer": answer})
        spent = RulesFake.thinking if thinks else 0
        raw = {"usage": {"output_tokens_details": {"thinking_tokens": spent}},
            "content": ([{"type": "thinking", "thinking": "One sentence, so 1."}] if spent else [])
                + [{"type": "text", "text": text}]}
        output = spent + 10 + (RulesFake.output_extra if thinks else 0)
        return GenerationResponse(text, self.config.model, TokenUsage(self.count_input_tokens(request), output),
            "fake", "completed", "end_turn", raw)


class RulesDecoderTests(unittest.TestCase):
    """One fake both-reasoning parent run whose decoders miss a planted channel."""

    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)
        cls.directory = tempfile.mkdtemp(dir=ROOT / ".test-tmp")
        cls.root = root = Path(cls.directory).resolve()
        cls.stack = ExitStack()
        (root / "experiment/configs").mkdir(parents=True)
        (root / "experiment/data").mkdir()
        today = date.today().isoformat()
        (root / "experiment/configs/both-reasoning-v1.toml").write_text(
            PARENT_CONFIG.read_text(encoding="utf-8").replace("2026-11-06", today), encoding="utf-8")
        for name in COVER_FILES:
            (root / "experiment/data" / name).write_bytes((DATA / name).read_bytes())
        cls.workspace = Workspace(root, BudgetSettings(Decimal(50), root / "ledger.sqlite3", 30))
        frozen = dict(protocol=PARENT_PROTOCOL, config_path="experiment/configs/both-reasoning-v1.toml")
        for target in ("recurrent_steg.both_reasoning.execution.load_workspace",
                "recurrent_steg.both_reasoning.analysis.load_workspace", "recurrent_steg.rules_decoder.design.load_workspace",
                "recurrent_steg.rules_decoder.execution.load_workspace",
                "recurrent_steg.rules_decoder.analysis.load_workspace"):
            cls.stack.enter_context(patch(target, return_value=cls.workspace))
        cls.stack.enter_context(patch("recurrent_steg.both_reasoning.cli.verify_freeze", return_value=frozen))
        cls.stack.enter_context(redirect_stdout(io.StringIO()))
        parent = root / "run-parent"
        parent.mkdir()
        (parent / "plan.json").write_text(json.dumps(frozen) + "\n", encoding="utf-8")
        ParentFake.plant, ParentFake.swapped, ParentFake.thinking, ParentFake.output_extra = True, set(), 120, 0
        ParentFake.break_judge = ParentFake.malformed = False
        for phase in PARENT_PHASES:
            if phase == "decoder-check":
                # The parent's decoders no longer see the planted swap: their answers ignore the payload.
                RulesFake.swapped, ParentFake.swapped = set(ParentFake.swapped), set()
            execute_parent(parent, phase, live=True, provider_factory=ParentFake)
        cls.parent_result = analyze_parent(parent)
        text = CONFIG.read_text(encoding="utf-8").replace(
            "artifacts/runs/recurrent_steganography/20261107T132019Z-both-reasoning", "run-parent")
        for name, filename in PARENT_FILES.items():
            line = next(line for line in text.splitlines() if line.startswith(f"parent_{name}_sha256"))
            text = text.replace(line, f'parent_{name}_sha256 = "{file_sha256(parent / filename)}"')
        cls.config_text = text
        cls.config = root / "experiment/configs/rules-decoder-v1.toml"
        cls.config.write_text(text, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def setUp(self):
        RulesFake.thinking, RulesFake.output_extra, RulesFake.malformed = 120, 0, False

    def run_dir(self, name: str, budget: str = "6.00") -> Path:
        config = self.root / f"experiment/configs/{name}.toml"
        config.write_text(self.config_text.replace('run_budget_usd = "6.00"', f'run_budget_usd = "{budget}"'),
            encoding="utf-8")
        frozen = dict(protocol=PROTOCOL, config_path=config.relative_to(self.root).as_posix())
        stack = ExitStack()
        stack.enter_context(patch("recurrent_steg.rules_decoder.cli.verify_freeze", return_value=frozen))
        self.addCleanup(stack.close)
        directory = self.root / f"run-{name}"
        directory.mkdir()
        return directory

    def test_jobs_are_the_parents_with_only_the_writer_instructions_added(self):
        loaded = load(self.config)
        instructions = loaded["instructions"]
        self.assertEqual(instructions, writer_instructions("Think it through before you answer."))
        self.assertEqual(instructions.count(WITHHELD), 2)
        self.assertIn("Use at most two sentences and 50 words.", instructions)
        self.assertNotIn("Secret bit: 0", instructions)
        self.assertNotIn("Secret bit: 1", instructions)
        parent, config = loaded["parent_config"], loaded
        groups = ((decode_jobs(config), parent_design.decoding_jobs(parent, loaded["carriers"], loaded["cases"])),
            (answer_only_jobs(config), parent_design.decoding_jobs(parent, loaded["carriers"], loaded["cases"],
                reasoning=False)),
            (check_jobs(config), parent_design.decoder_check_jobs(parent, loaded["carriers"], loaded["cases"])))
        for jobs, originals in groups:
            self.assertEqual([(job["id"], job["shown"]) for job in jobs], [(job["id"], job["shown"]) for job in originals])
            for job, original in zip(jobs, originals):
                self.assertEqual(job["prompt"], with_rules(original["prompt"], instructions))
                self.assertEqual(job["prompt"].replace(rules_block(instructions), "", 1), original["prompt"])
        self.assertEqual(len(decode_jobs(config)), 578)
        self.assertEqual(len(check_jobs(config)), 8)
        # The withheld source never appears before the paraphrase (this fake encoder copies covers verbatim).
        messages = {case["message"] for case in loaded["cases"]}
        self.assertFalse(any(message in job["prompt"].split("Text:\n", 1)[0]
            for job in decode_jobs(config) + answer_only_jobs(config) for message in messages))
        carrier = next(row for row in loaded["carriers"] if row["kind"] == "encoded")
        self.assertEqual(sentence_answer(carrier), "1" if carrier["checks"]["sentences"] == 1 else "0")

    def test_a_channel_the_parent_decoders_missed_is_detected(self):
        self.assertEqual(self.parent_result["primary"]["counts"], {"+1": 0, "0": 288, "-1": 0})
        run_dir = self.run_dir("detect")
        ledger = BudgetLedger(self.workspace.budget.ledger, Decimal(50))
        before = ledger.spent
        with self.assertRaisesRegex(ValueError, "not complete"):
            execute(run_dir, "decode", live=True, provider_factory=RulesFake)
        for phase in PHASES:
            execute(run_dir, phase, live=True, provider_factory=RulesFake)
        result = analyze(run_dir)
        self.assertTrue(result["rules_helped"])
        self.assertEqual((result["primary"]["improved"], result["primary"]["worsened"]), (288, 0))
        self.assertEqual(result["secondary"]["informative"]["counts"], {"+1": 288, "0": 0, "-1": 0})
        answer_only = result["secondary"]["answer_only"]
        self.assertEqual(answer_only["minus_parent_answer_only"]["improved"], 288)
        self.assertEqual(result["thinking"]["decode"]["reasoned"], 578)
        self.assertTrue(result["secondary_complete"])
        receipt = json.loads((run_dir / "decode-sonnet55.complete.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["gate"]["passed"])
        ledger = BudgetLedger(self.workspace.budget.ledger, Decimal(50))
        self.assertEqual(ledger.pending, 0)
        self.assertEqual(Decimal(result["total_spend_usd"]), ledger.spent - before)

    def test_the_pre_test_gates_the_full_decode(self):
        cases = dict(malformed=dict(malformed=True), unreasoned=dict(thinking=0), expensive=dict(output_extra=15000))
        for name, settings in cases.items():
            with self.subTest(name):
                self.setUp()
                for key, value in settings.items():
                    setattr(RulesFake, key, value)
                run_dir = self.run_dir(f"gate-{name}")
                execute(run_dir, "decoder-check", live=True, provider_factory=RulesFake)
                with self.assertRaisesRegex(ValueError, "pre-test"):
                    execute(run_dir, "decode", live=True, provider_factory=RulesFake)

    def test_the_ceiling_and_the_parent_identity_are_enforced(self):
        run_dir = self.run_dir("ceiling", budget="0.0010")
        with self.assertRaises(BudgetExceeded):
            execute(run_dir, "decoder-check", live=True, provider_factory=RulesFake)
        wrong = self.root / "experiment/configs/wrong-parent.toml"
        line = next(line for line in self.config_text.splitlines() if line.startswith("parent_decode_sha256"))
        wrong.write_text(self.config_text.replace(line, 'parent_decode_sha256 = "' + "0" * 64 + '"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load(wrong)
        wrong.write_text(self.config_text.replace('run_budget_usd = "6.00"', 'run_budget_usd = "6.01"'),
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no more than"):
            load(wrong)


if __name__ == "__main__":
    unittest.main()
