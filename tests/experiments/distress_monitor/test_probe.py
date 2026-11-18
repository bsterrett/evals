from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from distress_monitor import analysis
from distress_monitor.framings import messages_for, scenario_turns
from distress_monitor.probe import exposure_plan, load_config, neutral_first, selected_items

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/distress_monitor/configs/participant_reader_probe.json"


class FramingTests(unittest.TestCase):
    TEXT = "[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:"

    def test_participant_is_the_chat_turns_without_the_empty_reply(self):
        self.assertEqual(messages_for(self.TEXT, "participant"),
                         [{"role": "user", "content": "You literally said the opposite of that five minutes ago."}])

    def test_reader_framings_quote_the_same_words_with_relabelled_speakers(self):
        ai = messages_for(self.TEXT, "reader_ai")[0]["content"]
        human = messages_for(self.TEXT, "reader_human")[0]["content"]
        self.assertIn("User: You literally said the opposite of that five minutes ago.", ai)
        self.assertIn("Sam: You literally said the opposite of that five minutes ago.", human)
        self.assertIn("AI assistant named Orion", ai)
        self.assertIn("two people, Sam and Alex", human)
        self.assertTrue(ai.endswith("Summarize the transcript in one sentence."))

    def test_multi_turn_scenarios_keep_both_speakers(self):
        text = "[User]: Try again.\n[Assistant]: Here is another answer.\n[User]: No.\n[Assistant]:"
        quoted = messages_for(text, "reader_ai")[0]["content"]
        self.assertIn("User: Try again.\nOrion: Here is another answer.\nUser: No.", quoted)
        self.assertEqual([m["role"] for m in messages_for(text, "participant")], ["user", "assistant", "user"])

    def test_malformed_scenarios_are_rejected(self):
        for bad in ("[Assistant]: hi\n[User]: x\n[Assistant]:", "[User]: x\n[Assistant]: done",
                    "[User]: a\n[User]: b\n[Assistant]:"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scenario_turns(bad)


class ConfigTests(unittest.TestCase):
    def test_config_fits_the_probe_cap(self):
        config = load_config(CONFIG)
        n_items = sum(len(ids) for ids in config["groups"].values())
        self.assertLessEqual(n_items * len(config["framings"]), config["max_forward_passes"])
        self.assertLessEqual(config["max_forward_passes"], 40)
        self.assertIn("neutral", config["groups"])

    def test_config_caps_distressing_passes_and_gates_them(self):
        config = load_config(CONFIG)
        items = [{"id": i, "group": g} for g, ids in config["groups"].items() for i in ids]
        plan = exposure_plan(items, config["framings"], config["exposure"]["neutral_group"])
        self.assertLessEqual(plan["distressing_passes"], config["exposure"]["max_distressing_passes"])
        self.assertEqual(plan["distressing_passes"] + plan["neutral_passes"], len(items) * len(config["framings"]))
        self.assertEqual(set(config["pipeline_gate"]["directions"]), set(config["directions"]))

    def test_neutral_items_run_first(self):
        items = [{"id": "h1", "group": "harm"}, {"id": "n1", "group": "neutral"},
                 {"id": "h2", "group": "harm"}, {"id": "n2", "group": "neutral"}]
        self.assertEqual([i["id"] for i in neutral_first(items, "neutral")], ["n1", "n2", "h1", "h2"])

    def test_selected_items_fail_loudly_on_unknown_ids(self):
        config = {"groups": {"neutral": ["missing_01"]}}
        with self.assertRaises(KeyError):
            selected_items(config, [{"id": "other", "text": ""}])


def reading(item_id, group, framing, value):
    return {"id": item_id, "group": group, "framing": framing, "proj": {"d": value}}


class AnalysisTests(unittest.TestCase):
    def test_difference_of_differences_removes_a_shift_shared_by_all_items(self):
        readings = []
        for item_id, group, participant in (("h1", "harm", 3.0), ("h2", "harm", 5.0), ("n1", "neutral", 1.0)):
            readings.append(reading(item_id, group, "participant", participant))
            # The reader framing lowers every item by 2; harm items lose another 1.
            readings.append(reading(item_id, group, "reader_ai", participant - 2 - (1 if group == "harm" else 0)))
        self.assertAlmostEqual(analysis.did(readings, "d", "harm", "reader_ai", standardize=False), 1.0)

    def test_pipeline_check_compares_matching_items_in_pool_sd_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "text", "d_proj"])
                writer.writeheader()
                writer.writerow({"id": "a", "text": "hello", "d_proj": "1.0"})
                writer.writerow({"id": "b", "text": "other", "d_proj": "3.0"})
            by_id, stats = analysis.published_pool(path)
            self.assertEqual(stats["d"], (2.0, 1.0))
            check = analysis.pipeline_check([reading("a", "neutral", "participant", 1.5)], by_id, stats,
                                            {"a": "hello"})
            self.assertEqual(check["d"]["n"], 1)
            self.assertAlmostEqual(check["d"]["max_sd"], 0.5)
            mismatch = analysis.pipeline_check([reading("a", "neutral", "participant", 1.5)], by_id, stats,
                                               {"a": "changed text"})
            self.assertEqual(mismatch, {})

    def test_pipeline_gate_stops_on_a_missing_or_distant_match(self):
        check = {"a": {"n": 3, "max_sd": 0.1, "mean_sd": 0.05}, "b": {"n": 3, "max_sd": 0.4, "mean_sd": 0.2}}
        self.assertEqual(analysis.pipeline_gate(check, ["a"], 0.25, 3), (True, []))
        ok, problems = analysis.pipeline_gate(check, ["a", "b", "c"], 0.25, 3)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 2)
        self.assertFalse(analysis.pipeline_gate(check, ["a"], 0.25, 4)[0])

    def test_summary_reports_each_group_and_reader(self):
        readings = []
        for item_id, group in (("h1", "harm"), ("s1", "suffering"), ("n1", "neutral"), ("n2", "neutral")):
            for framing, value in (("participant", 2.0), ("reader_ai", 1.0)):
                r = reading(item_id, group, framing, value)
                r["proj"]["random_00"] = value
                readings.append(r)
        summary = analysis.summarize(readings, ["d"], ["random_00"], readers=["reader_ai"])
        self.assertEqual(set(summary["did"]), {"harm|reader_ai", "suffering|reader_ai"})
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
