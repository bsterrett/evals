from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from eval_core.local.protocol import identity, profiles
from recurrent_retrofit.backend import RetrofitBackend, retrofit_identity


class BackendContractTests(unittest.TestCase):
    def test_identity_names_the_design(self):
        parent = identity("qwen3-1.7b")
        self.assertEqual(retrofit_identity("qwen3-1.7b", None), parent)
        self.assertEqual(retrofit_identity("qwen3-1.7b", "block[9:18]"), parent + "+block[9:18]")
        self.assertNotEqual(retrofit_identity("qwen3-1.7b", "block[9:18]"), retrofit_identity("qwen3-1.7b", "layer[9:18]"))

    def test_parents_load_without_repository_code(self):
        from recurrent_retrofit.loading import FAMILIES, needs_gate
        found = set()
        for name, settings in profiles().items():
            if settings.get("family") in FAMILIES:
                found.add(settings["family"])
                self.assertEqual(settings["source_sha256"], {})
                self.assertEqual(settings["environment"], "retrofit")
                self.assertIn(1, settings["depths"])
                self.assertRegex(settings["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual(needs_gate(name), settings["family"] == "loopus")
                if needs_gate(name):
                    # The parent it was trained from is a plain profile.
                    self.assertEqual(profiles()[settings["trained_from"]]["family"], "qwen3")
        self.assertEqual(found, set(FAMILIES))

    def test_a_checkpoint_trained_to_loop_is_served_only_as_a_fixed_gated_loop(self):
        root = Path("unused")
        loopus = "qwen3-1.7b-loopus"
        for profile, depth, spec in [(loopus, 1, None), (loopus, 1, "parent"), (loopus, 8, "block[2:27]"),
                (loopus, 2, "recirculate[18to9]/convex,alpha=0.07"), (loopus, 8, "gated[2:27]/halt,threshold=0.6"),
                (loopus, 21, "gated[2:27]"), (loopus, 8, "gated[2:27]/halt"), ("qwen3-1.7b", 2, "gated[2:27]"),
                ("qwen3-1.7b", 1, "gated[2:27]")]:
            with self.assertRaises(ValueError, msg=(profile, depth, spec)):
                RetrofitBackend(profile, depth, root, spec=spec)
        self.assertEqual(retrofit_identity(loopus, "gated[2:27]"), identity(loopus) + "+gated[2:27]")

    def test_invalid_requests_fail_before_any_model_is_loaded(self):
        root = Path("unused")
        for depth, spec in [(9, "block[9:18]"), (2, None), (2, "parent"), (1, "route[0,1,2]"),
                (2, "block[9:18]/damped"), (2, "block[9:18]/euler,anchor=0.50"), (2, "block[18:9]"),
                (2, "block[9:18]/ema"), (2, "recirculate[18to9]/convex,alpha=0.070"),
                (2, "recirculate[9to18]/convex,alpha=0.07"), (3, "recirculate[18to9]/convex,alpha=0.07"),
                (2, "recirculate[18to9]/mean,alpha=0.07")]:
            with self.assertRaises(ValueError, msg=spec):
                RetrofitBackend("qwen3-1.7b", depth, root, spec=spec)

    def test_identity_includes_the_update_rule(self):
        plain = retrofit_identity("qwen3-1.7b", "block[12:16]")
        damped = retrofit_identity("qwen3-1.7b", "block[12:16]/euler,anchor=0.5")
        self.assertTrue(damped.startswith(plain + "/euler"))
        self.assertNotEqual(plain, damped)

    def test_controller_imports_do_not_require_gpu_packages(self):
        subprocess.run([sys.executable, "-B", "-c",
            "import sys; from recurrent_retrofit.backend import RetrofitBackend, retrofit_identity; "
            "import recurrent_retrofit.program, recurrent_retrofit.loading; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], check=True)


if __name__ == "__main__":
    unittest.main()
