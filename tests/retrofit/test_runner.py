"""Runner checks on a tiny randomly initialised Qwen3 model, on the CPU.

These need torch and transformers, which CI and the Windows environment do not
install. Run them in the WSL retrofit environment:

    PYTHONPATH=src ~/llm/recurrent/envs/retrofit/bin/python -m unittest tests.retrofit.test_runner -v
"""
from __future__ import annotations

import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from recurrent_retrofit.program import block_loop, layer_loop, parent, route

LAYERS = 6
IDS = [5, 17, 3, 44, 9, 61, 2]
NEW = 6


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.reference import StockCopy
        from recurrent_retrofit.runner import Retrofit, greedy
        cls.Retrofit, cls.StockCopy, cls.greedy = Retrofit, StockCopy, staticmethod(greedy)
        torch.manual_seed(0)
        # A large initializer range makes every layer change the output noticeably,
        # so that a wrong order or a shared cache slot cannot hide.
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            initializer_range=0.3)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.baseline = greedy(cls.model, IDS, NEW)
        cls.parameters = sum(p.numel() for p in cls.model.parameters())

    def assertSameLogits(self, a, b):
        self.assertEqual(a["tokens"], b["tokens"])
        for x, y in zip(a["logits"], b["logits"], strict=True):
            self.assertTrue(torch.equal(x, y), f"max difference {float((x - y).abs().max())}")

    def test_parent_program_reproduces_the_parent_exactly(self):
        for program in (parent(LAYERS), block_loop(LAYERS, 2, 4, 1), layer_loop(LAYERS, 1, 5, 1)):
            with self.Retrofit(self.model, program):
                self.assertSameLogits(self.greedy(self.model, IDS, NEW), self.baseline)

    def test_remove_restores_the_parent(self):
        retrofit = self.Retrofit(self.model, block_loop(LAYERS, 2, 4, 3))
        self.assertEqual(self.model.config.num_hidden_layers, LAYERS + 4)
        self.assertEqual(len(self.model.config.layer_types), LAYERS + 4)
        retrofit.remove()
        self.assertEqual(self.model.config.num_hidden_layers, LAYERS)
        self.assertEqual(len(self.model.config.layer_types), LAYERS)
        self.assertEqual([layer.self_attn.layer_idx for layer in self.model.model.layers], list(range(LAYERS)))
        self.assertSameLogits(self.greedy(self.model, IDS, NEW), self.baseline)

    def test_loops_change_the_output_and_share_the_weights(self):
        for program in (block_loop(LAYERS, 2, 4, 2), layer_loop(LAYERS, 2, 4, 2)):
            with self.Retrofit(self.model, program):
                looped = self.greedy(self.model, IDS, NEW)
                self.assertEqual(sum(p.numel() for p in self.model.parameters()), self.parameters)
            self.assertGreater(float((looped["logits"][0] - self.baseline["logits"][0]).abs().max()), 1e-3)

    def test_block_and_layer_order_give_different_outputs(self):
        with self.Retrofit(self.model, block_loop(LAYERS, 2, 4, 2)):
            block = self.greedy(self.model, IDS, 1)
        with self.Retrofit(self.model, layer_loop(LAYERS, 2, 4, 2)):
            layer = self.greedy(self.model, IDS, 1)
        self.assertGreater(float((block["logits"][0] - layer["logits"][0]).abs().max()), 1e-3)

    def test_every_forward_executes_the_program_and_fills_its_own_slots(self):
        for program in (block_loop(LAYERS, 1, 4, 3), layer_loop(LAYERS, 0, 6, 2), route(LAYERS, [0, 2, 2, 5])):
            with self.Retrofit(self.model, program) as retrofit:
                seen = []
                def check(step, cache):
                    calls = retrofit.take_calls()
                    retrofit.check_calls(calls)
                    retrofit.check_cache(cache, len(IDS) + step)
                    seen.append(calls)
                self.greedy(self.model, IDS, NEW, after_forward=check)
                self.assertEqual(seen, [program.calls_per_layer()] * NEW)
                with self.assertRaises(RuntimeError):
                    retrofit.check_calls([0] * LAYERS)

    def test_cached_decoding_matches_full_prefix_recomputation(self):
        for program in (block_loop(LAYERS, 1, 4, 3), layer_loop(LAYERS, 2, 5, 2), route(LAYERS, [0, 2, 2, 5])):
            with self.Retrofit(self.model, program):
                cached = self.greedy(self.model, IDS, NEW)
                full = self.greedy(self.model, IDS, NEW, cached=False)
            self.assertEqual(cached["tokens"], full["tokens"])
            for x, y in zip(cached["logits"], full["logits"], strict=True):
                self.assertLess(float((x - y).abs().max()), 1e-4)

    def test_runner_equals_a_physically_deeper_stock_model(self):
        for program in (block_loop(LAYERS, 1, 4, 3), layer_loop(LAYERS, 2, 5, 2), route(LAYERS, [3, 0, 0, 5, 1])):
            for cached in (True, False):
                with self.Retrofit(self.model, program):
                    shared = self.greedy(self.model, IDS, NEW, cached=cached)
                with self.StockCopy(self.model, program):
                    stock = self.greedy(self.model, IDS, NEW, cached=cached)
                self.assertSameLogits(shared, stock)
        self.assertSameLogits(self.greedy(self.model, IDS, NEW), self.baseline)

    def test_stock_generate_runs_the_installed_program(self):
        program = block_loop(LAYERS, 2, 5, 2)
        with self.Retrofit(self.model, program) as retrofit:
            expected = self.greedy(self.model, IDS, NEW)["tokens"]
            retrofit.take_calls()
            output = self.model.generate(torch.tensor([IDS]), max_new_tokens=NEW, min_new_tokens=NEW,
                do_sample=False, pad_token_id=0)
            self.assertEqual(output[0, len(IDS):].tolist(), expected)
            retrofit.check_calls(retrofit.take_calls(), forwards=NEW)

    def test_misuse_is_rejected(self):
        with self.assertRaises(ValueError):
            self.Retrofit(self.model, parent(LAYERS + 1))
        with self.Retrofit(self.model, parent(LAYERS)):
            with self.assertRaises(RuntimeError):
                self.Retrofit(self.model, parent(LAYERS))
        self.assertSameLogits(self.greedy(self.model, IDS, NEW), self.baseline)


if __name__ == "__main__":
    unittest.main()
