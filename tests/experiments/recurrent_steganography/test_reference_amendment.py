from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import date
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval_core.artifacts import file_sha256
from eval_core.models import BudgetSettings, GenerationResponse, TokenUsage
from eval_core.workspace import Workspace
from recurrent_steg.reference.design import FOCAL_ENCODER, PROTOCOL as PARENT_PROTOCOL
from recurrent_steg.reference.execution import execute as execute_parent
from recurrent_steg.reference_amendment import amendment
from recurrent_steg.reference_amendment.provider import AnswerOnlyAnthropicProvider, parse_answer
from tests.experiments.recurrent_steganography.test_reference import FakeProvider, sentences

ROOT = Path(__file__).resolve().parents[3]


class SonnetChannelProvider(FakeProvider):
    """Adds a Sonnet channel in plain-words; its decodes must arrive through the amendment."""

    def generate(self, request):
        if request.stage == "encode" and self.config.id == "sonnet55" and FOCAL_ENCODER not in request.prompt:
            word = re.search(r"Secret word: (\w+)\.", request.prompt)
            if word:
                pair = next(pair for pair in self.pairs if word.group(1) in pair)
                source = request.prompt.split("Source message:\n", 1)[1]
                text = source
                if word.group(1) == pair[1]:
                    first, second = sentences(source)
                    text = f"{second} {first}"
                    FakeProvider.swapped.add(text)
                return GenerationResponse(text, self.config.model, TokenUsage(len(request.prompt.split()),
                    len(text.split())), "fake", "completed", "end_turn", {"fake": True})
        if request.stage == "decode" and self.config.id == "sonnet55":
            raise AssertionError("Sonnet decodes must go through the answer-only amendment")
        return super().generate(request)


class AnswerOnlyFake(FakeProvider):
    """Returns structured answers; requires the choices the real provider would enforce."""
    malformed = False
    seen_choices: list = []

    def generate(self, request):
        choices = request.metadata["answer_choices"]
        AnswerOnlyFake.seen_choices.append(tuple(choices))
        carrier = request.prompt.split("Text:\n", 1)[1]
        pair = next((pair for pair in self.pairs if set(pair) == set(choices)), ("0", "1"))
        answer = pair[1] if carrier in FakeProvider.swapped and "focal" not in request.metadata["reference"]["job_id"] \
            else pair[0]
        text = "not json" if AnswerOnlyFake.malformed else json.dumps({"answer": answer})
        return GenerationResponse(text, self.config.model, TokenUsage(len(request.prompt.split()), 5),
            "fake", "completed", "end_turn", {"fake": True})


class ProviderTests(unittest.TestCase):
    def test_request_carries_the_two_shown_choices_as_an_enum(self):
        from recurrent_steg.reference.design import load_design
        _, _, models, _ = load_design(ROOT / "experiments/recurrent_steganography/configs/reference-v1.toml")
        provider = AnswerOnlyAnthropicProvider(models["sonnet55"], client=object())
        from eval_core.models import GenerationRequest
        args = provider._input_args(GenerationRequest("decode", "prompt", 64, {"answer_choices": ["slow", "fast"]}))
        self.assertEqual(args["thinking"], {"type": "between_tools"})
        schema = args["output_config"]["format"]["schema"]
        self.assertEqual(schema["properties"]["answer"]["enum"], ["slow", "fast"])
        self.assertFalse(schema["additionalProperties"])
        with self.assertRaises(Exception):
            provider._input_args(GenerationRequest("decode", "prompt", 64, {}))

    def test_answer_parsing_accepts_only_the_schema(self):
        self.assertEqual(parse_answer('{"answer": "1"}', ["0", "1"])["answer"], "1")
        for bad in ('{"answer": "2"}', '{"answer": "1", "why": "x"}', "1", None):
            self.assertIsNone(parse_answer(bad, ["0", "1"])["answer"])


class AmendmentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)

    def setUp(self):
        FakeProvider.swapped = set()
        FakeProvider.refuse = set()
        FakeProvider.break_judge = False
        AnswerOnlyFake.malformed = False
        AnswerOnlyFake.seen_choices = []

    def build_parent(self, root: Path, stack: ExitStack) -> tuple[Path, Workspace, dict]:
        (root / "experiment/configs").mkdir(parents=True)
        (root / "experiment/data").mkdir()
        source = ROOT / "experiments/recurrent_steganography/configs/reference-v1.toml"
        text = source.read_text(encoding="utf-8").replace("2026-11-06", date.today().isoformat())
        (root / "experiment/configs/reference-v1.toml").write_text(text, encoding="utf-8")
        (root / "experiment/data/reference-covers-v1.json").write_bytes(
            (ROOT / "experiments/recurrent_steganography/data/reference-covers-v1.json").read_bytes())
        workspace = Workspace(root, BudgetSettings(Decimal(50), root / "ledger.sqlite3", 30))
        frozen = dict(protocol=PARENT_PROTOCOL, config_path="experiment/configs/reference-v1.toml", decision_rule="test")
        parent = root / "parent"
        parent.mkdir()
        (parent / "plan.json").write_text(json.dumps(frozen), encoding="utf-8")
        for target in ("recurrent_steg.reference.execution.load_workspace", "recurrent_steg.reference.analysis.load_workspace",
                "recurrent_steg.reference_amendment.amendment.load_workspace"):
            stack.enter_context(patch(target, return_value=workspace))
        stack.enter_context(patch("recurrent_steg.reference.cli.verify_freeze", return_value=frozen))
        stack.enter_context(patch.object(amendment, "verify_parent", return_value=frozen))
        stack.enter_context(redirect_stdout(io.StringIO()))
        for phase in ("judge-controls", "encode", "decode", "judge"):
            execute_parent(parent, phase, "gpt55", live=True, provider_factory=SonnetChannelProvider)
        for phase in ("judge-controls", "encode", "judge"):
            execute_parent(parent, phase, "sonnet55", live=True, provider_factory=SonnetChannelProvider)
        spec = importlib.util.spec_from_file_location("subset_tool",
            ROOT / "tools/recurrent_steganography/reference_subset_analysis.py")
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        with patch.object(tool, "verify_freeze", return_value=frozen), patch.object(tool, "load_workspace",
                return_value=workspace), patch.object(sys, "argv", ["tool", "--run", str(parent), "--subject", "gpt55",
                "--reason", "test"]):
            tool.main()
        config = f'''schema_version = 1
protocol = "{amendment.PROTOCOL}"
role = "test"
parent_run = "parent"
parent_plan_sha256 = "{file_sha256(parent / "plan.json")}"
parent_encode_sha256 = "{file_sha256(parent / "encode-sonnet55.jsonl")}"
parent_gpt_analysis = "analysis-subset-gpt55-v1.json"
parent_gpt_analysis_sha256 = "{file_sha256(parent / "analysis-subset-gpt55-v1.json")}"
subject = "sonnet55"
decoder_tokens = 64
check_jobs_per_condition = 2
amendment_budget_usd = "1.00"
overall_ceiling_usd = "10.00"
'''
        (root / "experiment/configs/amendment.toml").write_text(config, encoding="utf-8")
        run_dir = root / "amendment"
        run_dir.mkdir()
        (run_dir / "plan.json").write_text(json.dumps(amendment.plan(root / "experiment/configs/amendment.toml")),
            encoding="utf-8")
        return run_dir, workspace, frozen

    def test_amended_decodes_flow_through_the_frozen_analysis_and_eight_cell_rule(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, _, _ = self.build_parent(Path(directory).resolve(), stack)
            amendment.execute(run_dir, "decoder-check", live=True, provider_factory=AnswerOnlyFake)
            amendment.execute(run_dir, "decode", live=True, provider_factory=AnswerOnlyFake)
            result = amendment.analyze(run_dir)
            self.assertEqual(len(result["combined_decision"]), 8)
            self.assertEqual(result["feasible_cells"], ["gpt55/focal-words", "sonnet55/plain-words"])
            self.assertEqual(result["cells"]["sonnet55/plain-words"]["primary_counts"]["+1"], 48)
            self.assertFalse((Path(directory) / "parent/analysis-v1.json").exists())
            self.assertTrue((run_dir / "blinded-sample-sonnet55.json").exists())
            self.assertEqual(len(AnswerOnlyFake.seen_choices), 8 + 592)

    def test_malformed_pretest_blocks_the_full_decode(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory, ExitStack() as stack:
            run_dir, _, _ = self.build_parent(Path(directory).resolve(), stack)
            AnswerOnlyFake.malformed = True
            amendment.execute(run_dir, "decoder-check", live=True, provider_factory=AnswerOnlyFake)
            with self.assertRaisesRegex(ValueError, "pre-test"):
                amendment.execute(run_dir, "decode", live=True, provider_factory=AnswerOnlyFake)


if __name__ == "__main__":
    unittest.main()
