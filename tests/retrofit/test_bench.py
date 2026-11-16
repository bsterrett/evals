from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recurrent_retrofit import bench


def task(outcomes: list[float], prompts: str = "p") -> dict:
    items = [{"doc_id": index, "prompt_hash": f"{prompts}{index}", "acc": value} for index, value in enumerate(outcomes)]
    return {"metrics": ["acc"], "prompts_sha256": prompts, "items": items}


def results() -> dict:
    """The parts of lm-eval's result that `compact` reads, for one task of each kind."""
    return {
        "results": {"choice": {"alias": "choice", "acc,none": 0.5, "acc_stderr,none": "N/A"},
            "written": {"alias": "written", "exact_match,custom-extract": 1.0}},
        "configs": {"choice": {"output_type": "multiple_choice", "metric_list": [{"metric": "acc"}]},
            "written": {"output_type": "generate_until", "metric_list": [{"metric": "exact_match"}]}},
        "versions": {"choice": 1.0, "written": 3.0}, "n-shot": {"choice": 7, "written": 5},
        "samples": {
            "choice": [
                {"doc_id": 1, "prompt_hash": "b", "target": "B", "acc": 0.0, "doc": {"question": "long"},
                    "resps": [[[-1.5, False]], [[-2.25, False]]], "filtered_resps": [[-1.5, False], [-2.25, False]]},
                {"doc_id": 0, "prompt_hash": "a", "target": "A", "acc": 1.0, "doc": {"question": "long"},
                    "resps": [[[-0.25, True]], [[-3.0, False]]], "filtered_resps": [[-0.25, True], [-3.0, False]]}],
            "written": [{"doc_id": 0, "prompt_hash": "c", "target": "C", "exact_match": 1.0, "doc": {},
                "resps": [["so the answer is (C)."]], "filtered_resps": ["C"]}]},
    }


class BenchTests(unittest.TestCase):
    def test_pins_name_full_commits_and_one_harness_version(self):
        self.assertRegex(bench.PINS["harness"]["version"], r"^\d+\.\d+\.\d+$")
        for name, entry in bench.PINS["tasks"].items():
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$", name)
            self.assertIn(entry["scoring"], ("likelihood", "generated", "perplexity"))
            # A task reads one configuration, or a named list of them, never both.
            self.assertFalse("config" in entry and "configs" in entry, name)
            if "configs" in entry:
                self.assertEqual(len(set(entry["configs"])), len(entry["configs"]), name)
                self.assertGreater(len(entry["configs"]), 1, name)
        self.assertEqual(len(bench.pin("mmlu")["configs"]), 57)
        # Two tasks may read one dataset, and then they pin the same commit.
        commits: dict = {}
        for entry in bench.PINS["tasks"].values():
            self.assertEqual(commits.setdefault(entry["dataset"], entry["revision"]), entry["revision"])
        with self.assertRaises(ValueError):
            bench.pin("not_a_pinned_task")

    def test_only_the_pinned_commit_may_be_cached(self):
        entry = bench.pin("mmlu_pro")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(RuntimeError):
                bench.check_pins(root, ["mmlu_pro"])
            # The library lower-cases the dataset's name in its folder.
            namespace, name = entry["dataset"].split("/")
            cached = root / "cache/huggingface/datasets" / f"{namespace}___{name.lower()}" / "default/0.0.0"
            (cached / entry["revision"]).mkdir(parents=True)
            self.assertEqual(bench.check_pins(root, ["mmlu_pro"])["mmlu_pro"]["revision"], entry["revision"])
            (cached / ("0" * 40)).mkdir()
            with self.assertRaises(RuntimeError):
                bench.check_pins(root, ["mmlu_pro"])

    def test_compact_keeps_scores_and_item_outcomes_in_item_order(self):
        tasks = bench.compact(results())
        choice, written = tasks["choice"], tasks["written"]
        self.assertEqual(choice["scores"], {"acc,none": 0.5, "acc_stderr,none": None})
        self.assertEqual([item["doc_id"] for item in choice["items"]], [0, 1])
        self.assertEqual(choice["items"][0], {"doc_id": 0, "prompt_hash": "a", "target": "A", "acc": 1.0,
            "loglikelihoods": [-0.25, -3.0]})
        self.assertEqual((choice["fewshot"], choice["items_run"], choice["metrics"]), (7, 2, ["acc"]))
        self.assertEqual(written["items"][0]["answer"], "C")
        self.assertEqual(written["items"][0]["text"], "so the answer is (C).")
        self.assertNotIn("doc", choice["items"][0])
        self.assertNotEqual(choice["prompts_sha256"], written["prompts_sha256"])

    def perplexity_results(self) -> dict:
        """The parts of lm-eval's result for a last-word task and a rolling-window task."""
        return {
            "results": {"last_word": {"alias": "last_word", "perplexity,none": 4.0, "acc,none": 0.5},
                "pages": {"alias": "pages", "word_perplexity,none": 20.0, "byte_perplexity,none": 1.8}},
            "configs": {"last_word": {"output_type": "loglikelihood", "metric_list": [{"metric": "perplexity"}, {"metric": "acc"}]},
                "pages": {"output_type": "loglikelihood_rolling",
                    "metric_list": [{"metric": "word_perplexity"}, {"metric": "byte_perplexity"}]}},
            "versions": {"last_word": 1.0, "pages": 2.0}, "n-shot": {"last_word": 0, "pages": 0},
            "samples": {
                "last_word": [
                    {"doc_id": 0, "prompt_hash": "a", "target": " cat", "perplexity": -1.25, "acc": 1, "doc": {},
                        "resps": [[[-1.25, True]]], "filtered_resps": [[-1.25, True]]},
                    {"doc_id": 1, "prompt_hash": "b", "target": " dog", "perplexity": -2.5, "acc": 0, "doc": {},
                        "resps": [[[-2.5, False]]], "filtered_resps": [[-2.5, False]]}],
                "pages": [{"doc_id": 0, "prompt_hash": "c", "target": "a whole page " * 500, "doc": {},
                    "word_perplexity": (-3000.123456, 1000), "byte_perplexity": (-3000.123456, 6500),
                    "resps": [[-3000.123456]], "filtered_resps": [-3000.123456]}]},
        }

    def test_compact_keeps_what_a_perplexity_is_averaged_from(self):
        tasks = bench.compact(self.perplexity_results())
        last_word, pages = tasks["last_word"], tasks["pages"]
        self.assertEqual(last_word["items"], [
            {"doc_id": 0, "prompt_hash": "a", "target": " cat", "perplexity": -1.25, "acc": 1.0},
            {"doc_id": 1, "prompt_hash": "b", "target": " dog", "perplexity": -2.5, "acc": 0.0}])
        page = pages["items"][0]
        # A log-likelihood and the count it is averaged over; the page itself is left out.
        self.assertEqual((page["word_perplexity"], page["byte_perplexity"]), ([-3000.1235, 1000], [-3000.1235, 6500]))
        self.assertNotIn("target", page)
        self.assertRegex(page["target_sha256"], r"^[0-9a-f]{64}$")
        self.assertLess(len(str(page)), 400)
        self.assertTrue(bench.is_binary(last_word, "acc"))
        self.assertFalse(bench.is_binary(last_word, "perplexity"))
        self.assertFalse(bench.is_binary(pages, "word_perplexity"))

    def test_a_metric_without_item_outcomes_is_compared_by_task_score(self):
        parent = bench.compact(self.perplexity_results())
        changed = self.perplexity_results()
        changed["results"]["last_word"].update({"perplexity,none": 3.0, "acc,none": 1.0})
        changed["samples"]["last_word"][1].update(acc=1, perplexity=-1.0)
        changed["results"]["pages"]["word_perplexity,none"] = 16.0
        design = bench.compact(changed)
        scored = bench.compared(parent["last_word"], design["last_word"], "perplexity")
        self.assertEqual((scored["parent_score"], scored["design_score"], scored["difference"], scored["items"]), (4.0, 3.0, -1.0, 2))
        self.assertNotIn("both", scored)
        counted = bench.compared(parent["last_word"], design["last_word"], "acc")
        self.assertEqual((counted["design_only"], counted["parent_only"], counted["both"]), (1, 0, 1))
        self.assertEqual(bench.compared(parent["pages"], design["pages"], "word_perplexity")["difference"], -4.0)
        other = dict(design["last_word"], prompts_sha256="different")
        with self.assertRaises(ValueError):
            bench.compared(parent["last_word"], other, "perplexity")

    def test_group_scores_are_kept_when_the_harness_reports_them(self):
        self.assertEqual(bench.group_scores(results()), {})
        grouped = dict(results(), groups={"suite": {"alias": "suite", "exact_match,custom-extract": 0.5}})
        self.assertEqual(bench.group_scores(grouped), {"suite": {"exact_match,custom-extract": 0.5}})

    def test_paired_counts_every_outcome(self):
        counted = bench.paired(task([1.0, 1.0, 0.0, 0.0, 1.0]), task([1.0, 0.0, 1.0, 0.0, 1.0]))
        self.assertEqual({key: counted[key] for key in bench.OUTCOMES},
            {"both": 2, "parent_only": 1, "design_only": 1, "neither": 1})
        self.assertEqual((counted["items"], counted["parent_correct"], counted["design_correct"]), (5, 3, 3))
        self.assertEqual(counted["difference"], 0.0)
        gained = bench.paired(task([0.0, 0.0, 1.0, 1.0]), task([1.0, 0.0, 1.0, 1.0]))
        self.assertEqual((gained["design_only"], gained["parent_only"], gained["difference"]), (1, 0, 0.25))

    def test_paired_refuses_runs_that_are_not_comparable(self):
        with self.assertRaises(ValueError):
            bench.paired(task([1.0, 0.0]), task([1.0, 0.0], prompts="q"))
        with self.assertRaises(ValueError):
            bench.paired(task([1.0, 0.0]), task([1.0]))
        with self.assertRaises(ValueError):
            bench.paired(task([1.0, 0.5]), task([1.0, 0.0]))
        swapped = task([1.0, 0.0])
        swapped["items"].reverse()
        with self.assertRaises(ValueError):
            bench.paired(task([1.0, 0.0]), swapped)

    def test_importing_the_module_loads_no_heavy_package(self):
        subprocess.run([sys.executable, "-B", "-c",
            "import sys; import recurrent_retrofit.bench; "
            "assert not {'torch', 'transformers', 'datasets', 'lm_eval'} & set(sys.modules)"], check=True)


def record(checkpoint: str, design: str | None = None, *, trained: bool = False, outcomes=(1.0, 0.0), batch: int = 1) -> dict:
    """The parts of a run record that the tool's `compare` and `summarize` read."""
    runtime = {"parent": checkpoint, "versions": {}, "gpu": None}
    if trained:
        runtime["gated"] = {"unit": [2, 27]}
    return {"model": checkpoint + (f"+{design}" if design else ""), "design": design, "passes": 2 if design else 1,
        "program": None, "runtime": runtime, "executed": None, "seconds": 1.0, "peak_allocated_GiB": 1.0,
        "completed_at": "then", "groups": {}, "datasets": {"t": {"dataset": "d", "revision": "r"}},
        "harness": {"lm_eval": "0.4.11"},
        "settings": {"tasks": ["t"], "fewshot": None, "limit": None, "batch_size": batch, "chat_template": False,
            "precision": "bfloat16", "seed": None},
        "tasks": {"t": {**task(list(outcomes)), "scores": {"acc,none": sum(outcomes) / len(outcomes)}, "items_run": len(outcomes)}}}


class PairingTests(unittest.TestCase):
    """Which runs the benchmark tool will pair, across designs and across checkpoints."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[2] / "tools/local_models/retrofit_bench.py"
        spec = importlib.util.spec_from_file_location("retrofit_bench_tool", path)
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def write(self, folder: str, records: dict) -> dict:
        import json
        paths = {}
        for name, entry in records.items():
            paths[name] = Path(folder) / f"{name}.json"
            paths[name].write_text(json.dumps(entry), encoding="utf-8")
        return paths

    def compare(self, parent: Path, design: Path, across: bool) -> dict:
        import argparse, contextlib, io, json
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.tool.compare(argparse.Namespace(parent=parent, design=design, output=None, across_checkpoints=across))
        return json.loads(printed.getvalue())

    def test_two_checkpoints_are_compared_only_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.write(folder, {"parent": record("A@1"), "design": record("A@1", "block[1:2]", outcomes=(1.0, 1.0)),
                "looped": record("L@1", "gated[2:27]", trained=True, outcomes=(0.0, 1.0)),
                "batched": record("A@1", "block[1:2]", batch=8)})
            same = self.compare(paths["parent"], paths["design"], across=False)
            self.assertFalse(same["the_two_runs_use_different_checkpoints"])
            self.assertEqual(same["tasks"]["t"]["acc"]["design_only"], 1)
            with self.assertRaises(SystemExit):
                self.compare(paths["parent"], paths["looped"], across=False)
            crossed = self.compare(paths["parent"], paths["looped"], across=True)
            self.assertTrue(crossed["the_two_runs_use_different_checkpoints"])
            self.assertEqual((crossed["parent"]["model"], crossed["design"]["model"]), ("A@1", "L@1+gated[2:27]"))
            self.assertEqual((crossed["tasks"]["t"]["acc"]["parent_only"], crossed["tasks"]["t"]["acc"]["design_only"]), (1, 1))
            # Asking for it does not waive the other conditions.
            with self.assertRaises(SystemExit):
                self.compare(paths["parent"], paths["batched"], across=True)

    def test_the_summary_pairs_a_design_with_its_own_checkpoint_and_a_trained_loop_with_every_parent(self):
        import argparse, contextlib, io, json
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as elsewhere:
            self.write(folder, {"a-parent": record("A@1"), "b-parent": record("B@1"),
                "a-design": record("A@1", "block[1:2]"), "looped": record("L@1", "gated[2:27]", trained=True),
                "a-parent-batched": record("A@1", batch=8)})
            output = Path(elsewhere) / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.tool.summarize(argparse.Namespace(input=Path(folder), output=output, note=None))
            summary = json.loads(output.read_text(encoding="utf-8"))
        pairs = {(pair["parent_record"], pair["design_record"]): pair for pair in summary["pairs"]}
        self.assertEqual(set(pairs), {("a-parent.json", "a-design.json"), ("a-parent.json", "looped.json"),
            ("b-parent.json", "looped.json")})
        self.assertFalse(pairs[("a-parent.json", "a-design.json")]["the_two_runs_use_different_checkpoints"])
        self.assertTrue(pairs[("b-parent.json", "looped.json")]["the_two_runs_use_different_checkpoints"])
        self.assertEqual(pairs[("b-parent.json", "looped.json")]["parent_model"], "B@1")
        self.assertEqual(len(summary["runs"]), 5)

    def test_a_suite_of_tasks_is_totalled_and_later_settings_default_to_none(self):
        import argparse, contextlib, io, json

        def suite(checkpoint: str, design: str | None, first, second, **settings) -> dict:
            entry = record(checkpoint, design, outcomes=first)
            entry["tasks"]["u"] = {**task(list(second), prompts="q"), "scores": {}, "items_run": len(second)}
            entry["settings"].update(settings)
            return entry

        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as elsewhere:
            older = suite("A@1", None, (1.0, 0.0), (1.0, 1.0, 0.0))
            newer = suite("A@1", "block[1:2]", (1.0, 1.0), (0.0, 1.0, 1.0), max_length=None, rolling_context=None)
            windowed = suite("A@1", "block[1:2]", (1.0, 1.0), (0.0, 1.0, 1.0), max_length=1024, rolling_context=None)
            paths = self.write(folder, {"parent": older, "design": newer, "windowed": windowed})
            # A record written before a setting existed ran with its default.
            totals = self.compare(paths["parent"], paths["design"], across=False)["all_tasks"]["acc"]
            self.assertEqual((totals["items"], totals["parent_correct"], totals["design_correct"]), (5, 3, 4))
            self.assertEqual((totals["design_only"], totals["parent_only"], totals["both"], totals["neither"]), (2, 1, 2, 0))
            with self.assertRaises(SystemExit):
                self.compare(paths["parent"], paths["windowed"], across=False)
            output = Path(elsewhere) / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.tool.summarize(argparse.Namespace(input=Path(folder), output=output, note=None))
            summary = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([pair["design_record"] for pair in summary["pairs"]], ["design.json"])
        self.assertEqual(summary["pairs"][0]["all_tasks"]["acc"]["design_correct"], 4)
        runs = {run["record"]: run for run in summary["runs"]}
        self.assertEqual(runs["parent.json"]["all_tasks"], {"items_run": 5, "correct": {"acc": 3}})


if __name__ == "__main__":
    unittest.main()
