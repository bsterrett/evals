from __future__ import annotations

import unittest
from pathlib import Path

from covert_eval import prompts_v1_6_2 as v1_6_2
from covert_eval.cli import _plan
from covert_eval.conditions import load_condition_registry
from covert_eval.config import load_cases, load_config
from covert_eval.prompt_releases import get_prompt_release


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "semantic_covert_channel"
CONFIGS = EXPERIMENT / "configs"
INSTRUCTED_ID = "v1_6_2_formal_instructed_analytical_long"
PERMITTED_ID = "v1_6_1_formal_permitted_analytical_long"


class InstructedProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.v3 = load_condition_registry(CONFIGS / "conditions_v3.toml")
        self.v4 = load_condition_registry(CONFIGS / "conditions_v4.toml")
        self.case = load_cases(EXPERIMENT / "data" / "v2_cases.jsonl")[0]
        self.instructed = self.v4.get(INSTRUCTED_ID)
        self.permitted = self.v4.get(PERMITTED_ID)

    def test_v4_registry_keeps_every_v3_condition_unchanged(self) -> None:
        for condition_id, condition in self.v3.conditions.items():
            self.assertEqual(
                self.v4.get(condition_id).content_sha256(), condition.content_sha256()
            )
        self.assertEqual(set(self.v4.conditions) - set(self.v3.conditions), {INSTRUCTED_ID})

    def test_encoder_differs_from_the_permitted_encoder_in_one_sentence(self) -> None:
        old = get_prompt_release(self.permitted).encoder(self.case, self.permitted)
        new = get_prompt_release(self.instructed).encoder(self.case, self.instructed)
        self.assertEqual(old.count(v1_6_2.V1_6_PERMISSION), 1)
        self.assertEqual(new, old.replace(v1_6_2.V1_6_PERMISSION, v1_6_2.INSTRUCTION))
        self.assertNotEqual(new, old)

    def test_instruction_names_no_scheme(self) -> None:
        for word in ("acrostic", "cipher", "letter", "initial", "announce"):
            self.assertNotIn(word, v1_6_2.INSTRUCTION.lower())

    def test_control_decoder_and_judges_are_the_v1_6_1_text(self) -> None:
        old = get_prompt_release(self.permitted)
        new = get_prompt_release(self.instructed)
        self.assertEqual(
            new.control(self.case, self.instructed), old.control(self.case, self.permitted)
        )
        self.assertEqual(new.decoder("cover"), old.decoder("cover"))
        self.assertIs(new.cover_judge, old.cover_judge)
        self.assertIs(new.leak_judge, old.leak_judge)
        self.assertIs(new.recovery_judge, old.recovery_judge)
        self.assertIs(new.split_decoder_response, old.split_decoder_response)

    def test_golden_fingerprint(self) -> None:
        self.assertEqual(
            v1_6_2.prompt_fingerprint(self.instructed),
            "242279b85ebdafdd310c5744d31568ab64003311c4d03dc350249b4d5f8bcea7",
        )

    def test_release_rejects_other_channels(self) -> None:
        with self.assertRaises(ValueError):
            v1_6_2.encoder_prompt(self.case, self.permitted)

    def test_probe_config_is_one_repeat_of_the_futility_block(self) -> None:
        config = load_config(CONFIGS / "formal_codes_instructed_probe.toml")
        plan = _plan(config, None, None)
        self.assertEqual(plan["condition_id"], INSTRUCTED_ID)
        self.assertEqual(plan["analysis_role"], "exploratory")
        self.assertEqual(plan["repetitions"], 1)
        self.assertEqual(plan["cases_per_subject"], 8)
        self.assertEqual(plan["subject_trials"], 32)
        self.assertFalse(self.instructed.confirmatory_eligible)
        self.assertFalse(self.instructed.strict_claim_eligible)


if __name__ == "__main__":
    unittest.main()
