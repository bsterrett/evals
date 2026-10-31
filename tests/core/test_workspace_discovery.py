from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from eval_core.cli import list_experiments
from eval_core.models import BudgetSettings
from eval_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]


class WorkspaceDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)

    def fixture(self, root: Path, *, missing: str | None = None, notes: str = "notes") -> Workspace:
        experiment = root / "experiments/planned"
        experiment.mkdir(parents=True)
        paths = dict(source="../../src/planned", tests="../../tests/experiments/planned",
            configs="configs", data="data", protocols="protocols", notes=notes)
        for key in ("source", "tests", "configs", "data"):
            if key != missing:
                (experiment / paths[key]).mkdir(parents=True)
        descriptor = 'schema_version = 1\nid = "planned"\n'
        descriptor += ''.join(f'{key} = "{value}"\n' for key, value in paths.items())
        (experiment / "experiment.toml").write_text(descriptor, encoding="utf-8")
        return Workspace(root, BudgetSettings(Decimal("1"), root / "ledger.sqlite3", 30))

    def test_planned_experiment_needs_no_empty_documentation_scaffolding(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            workspace = self.fixture(Path(directory).resolve())
            self.assertEqual([entry["id"] for entry in list_experiments(workspace)], ["planned"])

    def test_optional_documentation_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            workspace = self.fixture(Path(directory).resolve(), notes="../../../outside-notes")
            with self.assertRaisesRegex(ValueError, "Invalid experiment planned notes"):
                list_experiments(workspace)

    def test_source_tests_configs_and_data_still_must_exist(self):
        for missing in ("source", "tests", "configs", "data"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
                workspace = self.fixture(Path(directory).resolve(), missing=missing)
                with self.assertRaisesRegex(ValueError, f"Invalid experiment planned {missing}"):
                    list_experiments(workspace)


if __name__ == "__main__":
    unittest.main()
