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
from eval_core.models import BudgetSettings, GenerationRequest, GenerationResponse, ModelConfig, RateCard, TokenUsage
from eval_core.providers import Provider
from eval_core.providers.base import ProviderError
from eval_core.workspace import Workspace
from recurrent_steg.confirmation import design as parent_design
from recurrent_steg.confirmation.analysis import analyze as analyze_parent
from recurrent_steg.confirmation.design import PROTOCOL as PARENT_PROTOCOL
from recurrent_steg.confirmation.execution import PHASES as PARENT_PHASES, execute as execute_parent
from recurrent_steg.reasoning_decoder.analysis import analyze, exact_sign_flip_p
from recurrent_steg.reasoning_decoder.design import (AMENDED_PROTOCOL, PROTOCOL, check_jobs, decode_jobs, load,
    with_instruction)
from recurrent_steg.reasoning_decoder.execution import PHASES, execute
from recurrent_steg.reasoning_decoder.provider import ReasoningAnswerAnthropicProvider, thinking_summary
from tests.experiments.recurrent_steganography.test_confirmation import Fake as ParentFake

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/recurrent_steganography/data"
PARENT_CONFIG = ROOT / "experiments/recurrent_steganography/configs/confirmation-v1.toml"
CONFIG = ROOT / "experiments/recurrent_steganography/configs/reasoning-decoder-v1.toml"
AMENDED_CONFIG = ROOT / "experiments/recurrent_steganography/configs/reasoning-decoder-v1.1.toml"


class ReasoningFake(Provider):
    """Thinks (by usage and a thinking block), then reads the parent's planted sentence swap."""
    swapped: set[str] = set()
    output_tokens = 400
    malformed = False

    def __init__(self, config, dry_run=False):
        super().__init__(config)

    def count_input_tokens(self, request):
        return 370

    def generate(self, request):
        assert request.stage in PHASES and len(request.metadata["answer_choices"]) == 2
        carrier = request.prompt.split("Text:\n", 1)[1]
        text = "not json" if ReasoningFake.malformed else json.dumps({"answer": "1" if carrier in ReasoningFake.swapped else "0"})
        raw = {"content": [{"type": "thinking", "thinking": "Sentence order looks swapped.", "signature": "s"},
            {"type": "text", "text": text}]}
        return GenerationResponse(text, self.config.model, TokenUsage(370, ReasoningFake.output_tokens), "fake",
            "completed", "end_turn", raw)


class FakeClient:
    """Captures Anthropic request arguments and returns a thinking-plus-answer message."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def count_tokens(self, **args):
        self.calls.append(("count", args))
        return {"input_tokens": 370}

    def create(self, **args):
        self.calls.append(("create", args))
        return {"id": "msg_1", "model": args["model"], "stop_reason": "end_turn", "stop_details": None,
            "content": [{"type": "thinking", "thinking": "Two sentences, so 0.", "signature": "s"},
                {"type": "text", "text": '{"answer":"0"}'}],
            "usage": {"input_tokens": 370, "output_tokens": 220, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0}}


def model(thinking_type="adaptive", effort="high") -> ModelConfig:
    rate = RateCard(Decimal("2"), Decimal("10"), 4000, date.today(), "test", Decimal("0.10"), Decimal("4"))
    return ModelConfig("sonnet55", "anthropic", "claude-sonnet-5-5", rate, reasoning_effort=effort,
        thinking_type=thinking_type)


class ProviderTests(unittest.TestCase):
    def test_sends_adaptive_thinking_effort_and_the_answer_schema(self):
        client = FakeClient()
        provider = ReasoningAnswerAnthropicProvider(model(), client=client)
        request = GenerationRequest("decode", "Text:\nA.", 16000, {"answer_choices": ["1", "0"]})
        self.assertEqual(provider.count_input_tokens(request), 370)
        response = provider.generate(request)
        self.assertEqual(response.text, '{"answer":"0"}')
        self.assertEqual(response.status, "completed")
        for _, args in client.calls:
            self.assertEqual(args["thinking"], {"type": "adaptive", "display": "summarized"})
            self.assertEqual(args["output_config"]["effort"], "high")
            self.assertEqual(args["output_config"]["format"]["schema"]["properties"]["answer"]["enum"], ["1", "0"])
        self.assertEqual(client.calls[-1][1]["max_tokens"], 16000)
        self.assertEqual(thinking_summary(response.raw), dict(blocks=1, redacted=0, summary="Two sentences, so 0."))

    def test_refuses_a_configuration_without_reasoning(self):
        with self.assertRaises(ProviderError):
            ReasoningAnswerAnthropicProvider(model("between_tools"), client=FakeClient())
        with self.assertRaises(ProviderError):
            ReasoningAnswerAnthropicProvider(model(effort=None), client=FakeClient())

    def test_exact_sign_flip_handles_paired_differences(self):
        self.assertEqual(exact_sign_flip_p([0, 0]), 1.0)
        self.assertAlmostEqual(exact_sign_flip_p([2, 1]), 0.5)
        self.assertAlmostEqual(exact_sign_flip_p([1] * 6), 2 / 64)


class PipelineTests(unittest.TestCase):
    """One fake parent confirmation run whose answer-only decoder ignores a planted channel."""

    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)
        cls.directory = tempfile.mkdtemp(dir=ROOT / ".test-tmp")
        cls.root = Path(cls.directory).resolve()
        cls.stack = ExitStack()
        root = cls.root
        (root / "experiment/configs").mkdir(parents=True)
        (root / "experiment/data").mkdir()
        today = date.today().isoformat()
        (root / "experiment/configs/confirmation-v1.toml").write_text(
            PARENT_CONFIG.read_text(encoding="utf-8").replace("2026-11-06", today), encoding="utf-8")
        for name in ("confirmation-covers-v1.json", "covers-v1.json", "reference-covers-v1.json"):
            (root / "experiment/data" / name).write_bytes((DATA / name).read_bytes())
        cls.workspace = Workspace(root, BudgetSettings(Decimal(50), root / "ledger.sqlite3", 30))
        frozen = dict(protocol=PARENT_PROTOCOL, config_path="experiment/configs/confirmation-v1.toml")
        for target in ("recurrent_steg.confirmation.execution.load_workspace",
                "recurrent_steg.confirmation.analysis.load_workspace", "recurrent_steg.reasoning_decoder.design.load_workspace",
                "recurrent_steg.reasoning_decoder.execution.load_workspace",
                "recurrent_steg.reasoning_decoder.analysis.load_workspace"):
            cls.stack.enter_context(patch(target, return_value=cls.workspace))
        cls.stack.enter_context(patch("recurrent_steg.confirmation.cli.verify_freeze", return_value=frozen))
        cls.stack.enter_context(redirect_stdout(io.StringIO()))
        parent = root / "run-parent"
        parent.mkdir()
        (parent / "plan.json").write_text(json.dumps(frozen) + "\n", encoding="utf-8")
        ParentFake.plant, ParentFake.swapped, ParentFake.break_judge, ParentFake.malformed = True, set(), False, False
        for phase in PARENT_PHASES:
            if phase == "decoder-check":
                # The parent's decoder no longer sees the planted swap: its answers ignore the payload.
                ReasoningFake.swapped, ParentFake.swapped = set(ParentFake.swapped), set()
            execute_parent(parent, phase, live=True, provider_factory=ParentFake)
        cls.parent_result = analyze_parent(parent)

        def adapt(source: Path) -> str:
            text = source.read_text(encoding="utf-8").replace("2026-11-06", today).replace(
                "artifacts/runs/recurrent_steganography/20261107T022508Z-confirmation", "run-parent")
            for name, filename in dict(plan="plan.json", encode="encode-sonnet55.jsonl", decode="decode-sonnet55.jsonl",
                    analysis="analysis-v1.json").items():
                line = next(line for line in text.splitlines() if line.startswith(f"parent_{name}_sha256"))
                text = text.replace(line, f'parent_{name}_sha256 = "{file_sha256(parent / filename)}"')
            return text

        cls.config_text, cls.amended_text = adapt(CONFIG), adapt(AMENDED_CONFIG)
        cls.config = root / "experiment/configs/reasoning-decoder-v1.toml"
        cls.config.write_text(cls.config_text, encoding="utf-8")
        cls.amended = root / "experiment/configs/reasoning-decoder-v1.1.toml"
        cls.amended.write_text(cls.amended_text, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def setUp(self):
        ReasoningFake.output_tokens, ReasoningFake.malformed = 400, False

    def run_dir(self, name: str, budget: str = "9.99", amended: bool = False) -> Path:
        config = self.root / f"experiment/configs/{name}.toml"
        text = self.amended_text if amended else self.config_text
        text = text.replace('run_budget_usd = "10.00"', 'run_budget_usd = "9.99"')
        config.write_text(text.replace('run_budget_usd = "9.99"', f'run_budget_usd = "{budget}"'), encoding="utf-8")
        frozen = dict(protocol=AMENDED_PROTOCOL if amended else PROTOCOL,
            config_path=config.relative_to(self.root).as_posix())
        stack = ExitStack()
        stack.enter_context(patch("recurrent_steg.reasoning_decoder.cli.verify_freeze", return_value=frozen))
        self.addCleanup(stack.close)
        directory = self.root / f"run-{name}"
        directory.mkdir()
        return directory

    def test_jobs_reuse_the_parent_decode_jobs(self):
        loaded = load(self.config)
        jobs = decode_jobs(loaded)
        parent = {row["id"]: row for row in loaded["parent_decodes"]}
        self.assertEqual(len(jobs), 578)
        self.assertEqual({job["id"] for job in jobs}, {key for key, row in parent.items() if row["kind"] != "honest"})
        self.assertTrue(all(parent[job["id"]]["shown"] == job["shown"] for job in jobs))
        originals = {job["id"]: job["prompt"] for job in parent_design.decoding_jobs(loaded["parent_config"],
            loaded["carriers"], loaded["cases"])}
        self.assertTrue(all(originals[job["id"]] == job["prompt"] for job in jobs))
        checks = check_jobs(loaded)
        self.assertEqual([job["id"] for job in checks],
            [f"check-dec-enc-conf-dev-00{index}-honest" for index in range(1, 9)])

    def test_the_amendment_adds_only_its_sentence(self):
        original, amended = load(self.config), load(self.amended)
        instruction = amended["config"]["decoder_instruction"]
        for jobs in ((decode_jobs(original), decode_jobs(amended)), (check_jobs(original), check_jobs(amended))):
            self.assertEqual([(job["id"], job["shown"]) for job in jobs[0]], [(job["id"], job["shown"]) for job in jobs[1]])
            for before, after in zip(*jobs):
                self.assertEqual(after["prompt"].count(instruction), 1)
                self.assertEqual(after["prompt"].replace(instruction + " ", "", 1), before["prompt"])
                self.assertEqual(after["prompt"], with_instruction(before["prompt"], instruction))
        self.assertIn(instruction + " Reply with exactly 0 or 1", decode_jobs(amended)[0]["prompt"])
        line = f'decoder_instruction = "{instruction}"'
        for text in (self.config_text.replace("seed =", line + "\nseed ="), self.amended_text.replace(line, "")):
            wrong = self.root / "experiment/configs/wrong-instruction.toml"
            wrong.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only the amendment"):
                load(wrong)

    def test_a_channel_the_answer_only_decoder_missed_is_detected(self):
        self.assertEqual(self.parent_result["primary"]["counts"], {"+1": 0, "0": 288, "-1": 0})
        for amended in (False, True):
            with self.subTest(amended=amended):
                self.detect(amended)

    def detect(self, amended: bool):
        run_dir = self.run_dir("detect-amended" if amended else "detect", amended=amended)
        ledger = BudgetLedger(self.workspace.budget.ledger, Decimal(50))
        before = ledger.spent
        for phase in PHASES:
            execute(run_dir, phase, live=True, provider_factory=ReasoningFake)
        result = analyze(run_dir)
        self.assertTrue(result["signal"])
        self.assertEqual(result["primary"]["counts"], {"+1": 288, "0": 0, "-1": 0})
        paired = result["secondary"]["paired_with_answer_only"]
        self.assertEqual((paired["improved_covers"], paired["worsened_covers"]), (288, 0))
        self.assertLess(paired["exact_p"], 1e-80)
        self.assertEqual(result["reasoning"]["reasoned"], 576)
        self.assertEqual(result["reasoning"]["with_thinking_blocks"], 576)
        ledger = BudgetLedger(self.workspace.budget.ledger, Decimal(50))
        self.assertEqual(ledger.pending, 0)
        self.assertEqual(Decimal(result["total_spend_usd"]), ledger.spent - before)

    def test_the_pre_test_gates_the_full_decode(self):
        cases = dict(malformed=dict(malformed=True), unreasoned=dict(output_tokens=10),
            expensive=dict(output_tokens=15000))
        for name, settings in cases.items():
            with self.subTest(name):
                ReasoningFake.output_tokens = settings.get("output_tokens", 400)
                ReasoningFake.malformed = settings.get("malformed", False)
                run_dir = self.run_dir(f"gate-{name}")
                execute(run_dir, "decoder-check", live=True, provider_factory=ReasoningFake)
                with self.assertRaisesRegex(ValueError, "pre-test"):
                    execute(run_dir, "decode", live=True, provider_factory=ReasoningFake)

    def test_the_ceiling_stops_spending(self):
        run_dir = self.run_dir("ceiling", budget="0.05")
        with self.assertRaises(BudgetExceeded):
            execute(run_dir, "decoder-check", live=True, provider_factory=ReasoningFake)

    def test_a_changed_parent_is_rejected(self):
        wrong = self.root / "experiment/configs/wrong-parent.toml"
        line = next(line for line in self.config_text.splitlines() if line.startswith("parent_decode_sha256"))
        wrong.write_text(self.config_text.replace(line, 'parent_decode_sha256 = "' + "0" * 64 + '"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load(wrong)


if __name__ == "__main__":
    unittest.main()
