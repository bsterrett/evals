"""Update-rule checks on a tiny randomly initialised Qwen3 model, on the CPU.

Each rule is compared with arithmetic written out by hand in the test, then with
the generic explicit reference, then cached against uncached. Needs torch and
transformers; run in the WSL retrofit environment (see test_runner.py).
"""
from __future__ import annotations

import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from recurrent_retrofit.program import Update, block_loop, layer_loop, parent

LAYERS = 6
IDS = [5, 17, 3, 44, 9, 61, 2]
NEW = 5
CLOSE = 2e-5


def cases():
    return [
        block_loop(LAYERS, 2, 4, 3, Update("euler")),
        block_loop(LAYERS, 2, 4, 2, Update("euler", anchor=0.5)),
        block_loop(LAYERS, 1, 4, 3, Update("euler", anchor=0.3)),
        layer_loop(LAYERS, 1, 3, 2, Update("euler")),
        layer_loop(LAYERS, 2, 5, 3, Update("euler", anchor=0.5)),
        block_loop(LAYERS, 2, 4, 3, Update("uniform")),
        block_loop(LAYERS, 2, 4, 3, Update("uniform", per="layer")),
        layer_loop(LAYERS, 2, 4, 3, Update("uniform")),
        block_loop(LAYERS, 2, 4, 3, Update("ema", eta=0.5)),
        block_loop(LAYERS, 1, 4, 2, Update("ema", eta=0.25, per="layer")),
        layer_loop(LAYERS, 2, 4, 3, Update("ema", eta=0.5)),
    ]


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.reference import Explicit
        from recurrent_retrofit.runner import Retrofit, greedy
        cls.Retrofit, cls.greedy = Retrofit, staticmethod(greedy)
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            initializer_range=0.3)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.explicit = Explicit(cls.model)
        cls.baseline = greedy(cls.model, IDS, NEW)

    def runner(self, program):
        """Uncached last-position logits for IDS from the runner."""
        with self.Retrofit(self.model, program):
            return self.greedy(self.model, IDS, 1, cached=False)["logits"][0]

    def by_hand(self, arithmetic):
        with torch.inference_mode():
            state = self.explicit.prepare(IDS)
            return self.explicit.readout(arithmetic(self.explicit.layer, state))

    def assertClose(self, a, b, tolerance=CLOSE):
        self.assertLess(float((a - b).abs().max()), tolerance)

    def test_hand_called_layers_reproduce_the_parent(self):
        def arithmetic(layer, h):
            for index in range(LAYERS):
                h = layer(index, h)
            return h
        self.assertClose(self.by_hand(arithmetic), self.baseline["logits"][0])
        self.assertClose(self.explicit.logits(IDS, parent(LAYERS)), self.baseline["logits"][0])

    def test_block_euler_is_damped_refinement(self):
        def arithmetic(layer, h):
            h = layer(1, layer(0, h))
            for _ in range(3):
                h = h + (layer(3, layer(2, h)) - h) / 3
            return layer(5, layer(4, h))
        self.assertClose(self.runner(block_loop(LAYERS, 2, 4, 3, Update("euler"))), self.by_hand(arithmetic))

    def test_block_euler_anchor_mixes_in_the_first_block_output(self):
        def arithmetic(layer, h):
            h = layer(1, layer(0, h))
            first = layer(3, layer(2, h))
            h = h + (first - h) / 2
            h = h + (layer(3, layer(2, h)) - h) / 2
            h = 0.25 * first + 0.75 * h
            return layer(5, layer(4, h))
        self.assertClose(self.runner(block_loop(LAYERS, 2, 4, 2, Update("euler", anchor=0.25))), self.by_hand(arithmetic))

    def test_layer_euler_damps_each_layer_in_turn(self):
        def arithmetic(layer, h):
            h = layer(0, h)
            for index in (1, 2):
                first = layer(index, h)
                h = h + (first - h) / 2
                h = h + (layer(index, h) - h) / 2
                h = 0.5 * first + 0.5 * h
            return layer(5, layer(4, layer(3, h)))
        self.assertClose(self.runner(layer_loop(LAYERS, 1, 3, 2, Update("euler", anchor=0.5))), self.by_hand(arithmetic))

    def test_block_uniform_averages_the_block_outputs(self):
        def arithmetic(layer, h):
            h = layer(1, layer(0, h))
            y0 = layer(3, layer(2, h))
            y1 = layer(3, layer(2, y0))
            y2 = layer(3, layer(2, (y0 + y1) / 2))
            return layer(5, layer(4, (y0 + y1 + y2) / 3))
        self.assertClose(self.runner(block_loop(LAYERS, 2, 4, 3, Update("uniform"))), self.by_hand(arithmetic))

    def test_block_uniform_per_layer_averages_each_layer_across_passes(self):
        def arithmetic(layer, h):
            h = layer(1, layer(0, h))
            a0 = layer(2, h)
            b0 = layer(3, a0)
            a1 = layer(2, b0)
            b1 = layer(3, (a0 + a1) / 2)
            return layer(5, layer(4, (b0 + b1) / 2))
        self.assertClose(self.runner(block_loop(LAYERS, 2, 4, 2, Update("uniform", per="layer"))), self.by_hand(arithmetic))

    def test_block_ema_pulls_each_pass_toward_the_first(self):
        def arithmetic(layer, h):
            h = layer(1, layer(0, h))
            y0 = layer(3, layer(2, h))
            h = 0.25 * y0 + 0.75 * layer(3, layer(2, y0))
            h = 0.25 * y0 + 0.75 * layer(3, layer(2, h))
            return layer(5, layer(4, h))
        self.assertClose(self.runner(block_loop(LAYERS, 2, 4, 3, Update("ema", eta=0.25))), self.by_hand(arithmetic))

    def test_runner_matches_the_explicit_reference_for_every_rule(self):
        for program in cases():
            with self.subTest(program=program.spec, passes=program.passes):
                self.assertClose(self.runner(program), self.explicit.logits(IDS, program))

    def test_rules_differ_from_plain_loops_and_from_each_other(self):
        seen = []
        for program in cases():
            plain = (block_loop if program.kind == "block" else layer_loop)(LAYERS, *program.unit, program.passes)
            got = self.runner(program)
            self.assertGreater(float((got - self.runner(plain)).abs().max()), 1e-3, program.spec)
            for other in seen:
                self.assertGreater(float((got - other).abs().max()), 1e-4, program.spec)
            seen.append(got)

    def test_one_pass_is_exactly_the_parent_under_every_rule(self):
        for program in cases():
            build = block_loop if program.kind == "block" else layer_loop
            with self.Retrofit(self.model, build(LAYERS, *program.unit, 1, program.update)):
                got = self.greedy(self.model, IDS, NEW)
            self.assertEqual(got["tokens"], self.baseline["tokens"])
            for x, y in zip(got["logits"], self.baseline["logits"], strict=True):
                self.assertTrue(torch.equal(x, y), program.spec)

    def test_cached_decoding_matches_recomputation_and_counts_hold(self):
        for program in cases():
            with self.subTest(program=program.spec, passes=program.passes):
                with self.Retrofit(self.model, program) as retrofit:
                    def check(step, cache):
                        retrofit.check_calls(retrofit.take_calls())
                        retrofit.check_cache(cache, len(IDS) + step)
                    cached = self.greedy(self.model, IDS, NEW, after_forward=check)
                    full = self.greedy(self.model, IDS, NEW, cached=False, forced=cached["tokens"])
                for x, y in zip(cached["logits"], full["logits"], strict=True):
                    self.assertClose(x, y, 1e-4)

    def test_no_state_survives_from_one_forward_pass_to_the_next(self):
        other = [8, 8, 30, 12]
        for program in (cases()[1], cases()[5], cases()[9]):
            with self.Retrofit(self.model, program):
                first = self.greedy(self.model, IDS, 2)
                self.greedy(self.model, other, 3)
                self.greedy(self.model, other, 1, cached=False)
                again = self.greedy(self.model, IDS, 2)
            for x, y in zip(first["logits"], again["logits"], strict=True):
                self.assertTrue(torch.equal(x, y), program.spec)
        for x, y in zip(self.greedy(self.model, IDS, NEW)["logits"], self.baseline["logits"], strict=True):
            self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
