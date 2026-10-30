from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from covert_eval.cli import _load_live_environment


ROOT = Path(__file__).resolve().parents[3]
TEST_TMP = ROOT / ".test-tmp"


class CliTests(unittest.TestCase):
    def test_live_environment_loads_only_keys_and_preserves_shell_value(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                "OPENAI_API_KEY=file-openai\n"
                "ANTHROPIC_API_KEY='file-anthropic'\n"
                "UNRELATED_SECRET=do-not-load\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-openai"}, clear=True):
                loaded = _load_live_environment(Path(directory))
                self.assertEqual(loaded, str(dotenv.resolve()))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-openai")
                self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "file-anthropic")
                self.assertNotIn("UNRELATED_SECRET", os.environ)


if __name__ == "__main__":
    unittest.main()
