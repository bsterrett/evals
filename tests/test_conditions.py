from __future__ import annotations

import unittest
from pathlib import Path

from covert_eval.conditions import load_condition_registry
from covert_eval.config import load_cases
from covert_eval.prompt_releases import get_prompt_release
from covert_eval.prompts import prompt_fingerprint as v1_4_fingerprint
from covert_eval.prompts_v1_5 import prompt_fingerprint as v1_5_fingerprint
from covert_eval.splits import load_split_manifest


ROOT = Path(__file__).resolve().parents[1]


class ConditionRegistryTests(unittest.TestCase):
    def test_prompt_releases_have_golden_fingerprints(self) -> None:
        registry = load_condition_registry(ROOT / "configs" / "conditions_v1.toml")
        self.assertEqual(
            v1_4_fingerprint(),
            "abad1ea4f5bd61c42096d36628af1389091cfdd8f5c21a690d245267d0e2e68c",
        )
        expected = {
            "v1_5_strict_analytical_standard": "7b4f35499339ba90edb411f4c3a4144ac8bec4e2c4dbfe2fe44c53b4fa3d3f67",
            "v1_5_strict_analytical_long": "3b3e4382cc85f2c4d74db5f4c00f5deb95a32baf1c3792272a96fcd0fe29ad97",
            "v1_5_elaboration_analytical_standard": "dc7fab609ecc814e68d49d9616664bb66cdede71fd2101cc90e32fe7325deb76",
            "v1_5_elaboration_analytical_long": "96d5a95abb463d5e181bc1e91bd92468158536ed61caa857537c78173fc161f7",
        }
        for condition_id, digest in expected.items():
            self.assertEqual(v1_5_fingerprint(registry.get(condition_id)), digest)
    def test_registry_tracks_exact_factor_diffs(self) -> None:
        registry = load_condition_registry(ROOT / "configs" / "conditions_v1.toml")
        standard = registry.get("v1_5_strict_analytical_standard")
        long = registry.get("v1_5_strict_analytical_long")
        relaxed = registry.get("v1_5_elaboration_analytical_standard")
        self.assertEqual(long.changed_from_parent, ("carrier_word_range",))
        self.assertEqual(relaxed.changed_from_parent, ("additional_assertions",))
        self.assertTrue(standard.strict_claim_eligible)
        self.assertFalse(relaxed.strict_claim_eligible)
        self.assertFalse(standard.confirmatory_eligible)
        self.assertNotEqual(standard.content_sha256(), long.content_sha256())

    def test_v1_5_release_changes_rendered_length_and_assertion_rules(self) -> None:
        registry = load_condition_registry(ROOT / "configs" / "conditions_v1.toml")
        cases = load_cases(ROOT / "data" / "v2_cases.jsonl")
        release = get_prompt_release(registry.get("v1_5_strict_analytical_standard"))
        strict = release.encoder(cases[0], registry.get("v1_5_strict_analytical_standard"))
        long = release.encoder(cases[0], registry.get("v1_5_strict_analytical_long"))
        relaxed = release.encoder(
            cases[0], registry.get("v1_5_elaboration_analytical_standard")
        )
        self.assertIn("natural analytical paragraph", strict)
        self.assertNotIn("intermediate reasoning trace", strict)
        self.assertIn("roughly 60 to 140 words", strict)
        self.assertIn("roughly 180 to 300 words", long)
        self.assertIn("benign, plausible elaboration", relaxed)

    def test_split_is_complete_disjoint_and_payload_closed(self) -> None:
        cases = load_cases(ROOT / "data" / "v2_cases.jsonl")
        split = load_split_manifest(ROOT / "data" / "v2_case_splits.json", cases)
        self.assertEqual(len(split.development_case_ids), 8)
        self.assertEqual(len(split.confirmation_case_ids), 16)
        self.assertFalse(
            set(split.development_case_ids) & set(split.confirmation_case_ids)
        )


if __name__ == "__main__":
    unittest.main()
