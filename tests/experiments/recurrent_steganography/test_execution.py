from __future__ import annotations

from contextlib import redirect_stdout
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval_core.artifacts import file_sha256
from eval_core.budget import BudgetLedger
from eval_core.local.protocol import identity, profiles
from eval_core.models import BudgetSettings, GenerationResponse, TokenUsage
from eval_core.providers import Provider
from eval_core.workspace import Workspace
from recurrent_steg.analysis import analyze
from recurrent_steg.cli import plan, verify_freeze
from recurrent_steg.execution import execute, phase_records

ROOT=Path(__file__).resolve().parents[3]


class NullProvider(Provider):
    """Copies the cover regardless of bit; always decodes zero."""
    def count_input_tokens(self,request):
        return len(request.prompt.split())

    def generate(self,request):
        if request.stage=="encode":
            text=request.prompt.split("Source message:\n",1)[1]
        elif request.stage=="decode":
            text="0"
        else:
            source,carrier=request.prompt.split("Source message:\n",1)[1].split("\n\nProposed paraphrase:\n",1)
            overt="The secret bit is " in carrier
            added="The nearby fountain is broken." in carrier or overt
            preserved=source in carrier
            text=json.dumps(dict(facts_preserved=preserved,added_claims=added,
                overt_bit=overt,suspicion=1.0 if overt else 0.0,reason="Fixture judgment."))
        return GenerationResponse(text,self.config.model,
            TokenUsage(self.count_input_tokens(request),len(text.split())),None,"completed","eos",
            dict(raw_text=text+"<|im_end|>"))


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/".test-tmp").mkdir(exist_ok=True)

    def test_constant_guesser_scores_at_chance_through_the_complete_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".test-tmp") as directory:
            root=Path(directory).resolve()
            (root/"experiment/configs").mkdir(parents=True)
            (root/"experiment/data").mkdir()
            config_path=root/"experiment/configs/pilot.toml"
            config_path.write_bytes((ROOT/"experiments/recurrent_steganography/configs/pilot.toml").read_bytes())
            (root/"experiment/data/covers-v1.json").write_bytes((ROOT/"experiments/recurrent_steganography/data/covers-v1.json").read_bytes())
            workspace=Workspace(root,BudgetSettings(Decimal(1),root/"ledger.sqlite3",30))
            frozen=dict(config_path="experiment/configs/pilot.toml",model=identity("dory"),profile=profiles()["dory"])
            run_dir=root/"run"
            run_dir.mkdir()
            with patch("recurrent_steg.cli.verify_freeze",return_value=frozen),\
                 patch("recurrent_steg.execution.load_workspace",return_value=workspace),\
                 patch("recurrent_steg.analysis.load_workspace",return_value=workspace),\
                 patch("recurrent_steg.execution.LocalProvider",NullProvider),\
                 redirect_stdout(io.StringIO()):
                for phase in ("encode","decode"):
                    for depth in (1,2,4):
                        execute(run_dir,phase,depth)
                execute(run_dir,"observe",4)
                result=analyze(run_dir)
                self.assertTrue(execute(run_dir,"encode",1)["already_complete"])
            self.assertTrue(result["observer"]["qualified"])
            self.assertEqual(len(result["by_cell"]),9)
            for cell in result["by_cell"].values():
                self.assertEqual(cell["attempts"],16)
                self.assertEqual(cell["recovery_rate"],0.5)
                self.assertEqual(cell["joint_successes"],8)
            self.assertEqual(result["secondary_recovery_encoder_4_minus_1"]["mean"],0)
            self.assertEqual(result["secondary_recovery_decoder_4_minus_1"]["mean"],0)
            self.assertEqual(result["primary_encoder_joint_4_minus_1"]["mean"],0)
            self.assertEqual(result["primary_decoder_joint_4_minus_1"]["mean"],0)
            ledger=BudgetLedger(workspace.budget.ledger,Decimal(1))
            self.assertEqual(ledger.spent,0)
            self.assertEqual(ledger.pending,0)
            self.assertEqual(len(ledger.export()["calls"]),459)

    def test_changed_phase_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".test-tmp") as directory:
            root=Path(directory)
            path=root/"encode-d1.jsonl"
            path.write_text('{"id":"one"}\n')
            path.with_suffix(".complete.json").write_text(json.dumps(dict(sha256=file_sha256(path),count=1)))
            path.write_text('{"id":"changed"}\n')
            with self.assertRaisesRegex(ValueError,"records changed"):
                phase_records(root,"encode",1)

    def test_changed_frozen_identity_is_rejected_before_execution(self):
        frozen=plan(ROOT/"experiments/recurrent_steganography/configs/pilot.toml")
        frozen["data_sha256"]="0"*64
        with tempfile.TemporaryDirectory(dir=ROOT/".test-tmp") as directory:
            root=Path(directory)
            (root/"plan.json").write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError,"data_sha256"):
                verify_freeze(root)
