"""Recirculation checks on a tiny randomly initialised Qwen3 model, on the CPU.

Needs torch and transformers; run in the WSL retrofit environment (see
test_runner.py).
"""
from __future__ import annotations

import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from recurrent_retrofit.program import Recirculation, block_loop

LAYERS = 6
IDS = [5, 17, 3, 44, 9, 61, 2]
ROWS = [IDS, [8, 30, 71], [12, 4, 90, 33, 6]]
NEW = 5
PAD = 0
CLOSE = 2e-5


def designs():
    return [
        Recirculation(LAYERS, 4, 1, 0.1),
        Recirculation(LAYERS, 5, 2, 0.07, "additive"),
        Recirculation(LAYERS, 3, 0, 0.3),
        Recirculation(LAYERS, 5, 4, 0.2, "additive"),
    ]


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class RecirculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.recirculation import Recirculator, mixture
        from recurrent_retrofit.reference import ExplicitRecirculation
        from recurrent_retrofit.runner import Retrofit, greedy, install
        cls.Recirculator, cls.Retrofit, cls.mixture = Recirculator, Retrofit, staticmethod(mixture)
        cls.install, cls.greedy = staticmethod(install), staticmethod(greedy)
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            initializer_range=0.3)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.explicit = ExplicitRecirculation(cls.model)

    def every_position(self, ids=IDS):
        with torch.no_grad():
            return self.model(torch.tensor([ids])).logits[0]

    def generate(self, ids, mask=None):
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=NEW, min_new_tokens=NEW,
                do_sample=False, pad_token_id=PAD, eos_token_id=None, use_cache=True,
                output_scores=True, return_dict_in_generate=True)
        return out.sequences[:, ids.shape[1]:], torch.stack(out.scores, dim=1)

    def test_the_mixture_is_the_written_arithmetic(self):
        low, high = torch.tensor([[[3.0, 4.0]]]), torch.tensor([[[0.0, 10.0]]])
        # The source is rescaled from norm 10 to the destination's norm 5: [0, 5].
        self.assertTrue(torch.allclose(self.mixture(low, high, 0.2, "convex"), torch.tensor([[[2.4, 4.2]]])))
        self.assertTrue(torch.allclose(self.mixture(low, high, 0.2, "additive"), torch.tensor([[[3.0, 5.0]]])))
        self.assertTrue(torch.equal(self.mixture(low, high, 0.0, "convex"), low))
        rows = self.mixture(torch.tensor([[[3.0, 4.0]], [[6.0, 8.0]]]), torch.tensor([[[0.0, 10.0]], [[0.0, 1.0]]]),
            0.5, "additive")
        self.assertTrue(torch.allclose(rows, torch.tensor([[[3.0, 6.5]], [[6.0, 13.0]]])))

    def hidden(self, ids, cache=None):
        """Final hidden states from the decoder itself. The output head is left out because
        its rounding depends on how many positions it is given at once."""
        with torch.no_grad():
            return self.model.model(torch.tensor([ids]), past_key_values=cache, use_cache=cache is not None).last_hidden_state[0]

    def test_one_pass_is_the_parent_token_by_token(self):
        from transformers import DynamicCache
        cache = DynamicCache(config=self.model.config)
        steps = torch.cat([self.hidden([token], cache) for token in IDS])
        together = self.every_position()
        with self.install(self.model, Recirculation(LAYERS, 4, 1, 0.1, passes=1)) as installed:
            self.assertTrue(torch.equal(self.hidden(IDS), steps))
            installed.check_calls(installed.take_calls(), forwards=len(IDS))
            self.assertLess((self.every_position() - together).abs().max().item(), CLOSE)

    def test_zero_alpha_is_exactly_one_pass(self):
        with self.install(self.model, Recirculation(LAYERS, 4, 1, 0.0, passes=1)):
            once = self.greedy(self.model, IDS, NEW)
        with self.install(self.model, Recirculation(LAYERS, 4, 1, 0.0)):
            twice = self.greedy(self.model, IDS, NEW)
        self.assertEqual(once["tokens"], twice["tokens"])
        for x, y in zip(once["logits"], twice["logits"], strict=True):
            self.assertTrue(torch.equal(x, y))

    def test_the_runner_agrees_with_the_explicit_reference(self):
        for design in designs():
            with self.subTest(design.spec):
                with self.install(self.model, design):
                    decoded = self.greedy(self.model, IDS, NEW)
                    served = self.every_position(IDS + decoded["tokens"])
                by_hand = torch.stack(self.explicit.logits(IDS + decoded["tokens"], design))
                self.assertLess((served - by_hand).abs().max().item(), 1e-4)
                # Cached decoding reads the same logits as the whole sequence in one call.
                for step, logits in enumerate(decoded["logits"]):
                    self.assertLess((logits - by_hand[len(IDS) - 1 + step]).abs().max().item(), 1e-4)

    def test_the_first_token_is_the_parents_and_later_ones_are_not(self):
        with self.install(self.model, Recirculation(LAYERS, 4, 1, 0.1, passes=1)):
            plain = self.every_position()
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design):
                fed = self.every_position()
                self.assertTrue(torch.equal(fed[0], plain[0]))
                for position in range(1, len(IDS)):
                    self.assertGreater((fed[position] - plain[position]).abs().max().item(), 1e-3)

    def test_source_destination_and_mix_each_change_the_output(self):
        outputs = []
        for design in [Recirculation(LAYERS, 4, 1, 0.1), Recirculation(LAYERS, 5, 1, 0.1),
                Recirculation(LAYERS, 4, 2, 0.1), Recirculation(LAYERS, 4, 1, 0.1, "additive"),
                Recirculation(LAYERS, 4, 1, 0.2)]:
            with self.install(self.model, design):
                outputs.append(self.every_position()[-1])
        for index, first in enumerate(outputs):
            for second in outputs[index + 1:]:
                self.assertGreater((first - second).abs().max().item(), 1e-3)

    def test_splitting_a_prompt_between_calls_changes_nothing(self):
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design) as installed:
                whole = self.hidden(IDS)
                cache = installed.new_cache()
                parts = [self.hidden(IDS[:3], cache), self.hidden(IDS[3:], cache)]
                installed.check_cache(cache, len(IDS))
                self.assertTrue(torch.equal(torch.cat(parts), whole))

    def test_every_token_makes_the_designs_calls_and_fills_the_cache(self):
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design) as installed:
                self.assertEqual(design.layer_calls, LAYERS + LAYERS - 1 - design.destination)
                seen = []
                def check(step, cache):
                    calls = installed.take_calls()
                    installed.check_calls(calls, forwards=installed.runs_for(len(IDS) if step == 0 else 1))
                    installed.check_cache(cache, len(IDS) + step)
                    seen.append(sum(calls))
                self.greedy(self.model, IDS, NEW, after_forward=check)
                self.assertEqual(seen, [design.layer_calls * len(IDS)] + [design.layer_calls] * (NEW - 1))
                with self.assertRaises(RuntimeError):
                    installed.check_calls([1] * LAYERS)

    def test_cached_decoding_matches_full_prefix_recomputation(self):
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design):
                cached = self.greedy(self.model, IDS, NEW)
                full = self.greedy(self.model, IDS, NEW, cached=False, forced=cached["tokens"])
                for x, y in zip(cached["logits"], full["logits"], strict=True):
                    self.assertLess((x - y).abs().max().item(), CLOSE)

    def test_right_padded_rows_score_as_they_do_alone(self):
        width = max(map(len, ROWS))
        batch = torch.tensor([row + [PAD] * (width - len(row)) for row in ROWS])
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design) as installed:
                with torch.no_grad():
                    together = self.model(batch).logits
                installed.check_calls(installed.take_calls(), forwards=width)
                for index, row in enumerate(ROWS):
                    alone = self.every_position(row)
                    self.assertLess((together[index, :len(row)] - alone).abs().max().item(), CLOSE)

    def test_library_generation_matches_the_reference_decoder(self):
        ids = torch.tensor([IDS])
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design) as installed:
                expected = self.greedy(self.model, IDS, NEW)
                installed.take_calls()
                tokens, scores = self.generate(ids)
                installed.check_calls(installed.take_calls(), forwards=len(IDS) + NEW - 1)
                self.assertEqual(tokens[0].tolist(), expected["tokens"])
                reference = torch.stack([torch.as_tensor(step) for step in expected["logits"]])
                self.assertLess((scores[0] - reference).abs().max().item(), CLOSE)

    def test_left_padded_rows_generate_as_they_do_alone(self):
        width = max(map(len, ROWS))
        batch = torch.tensor([[PAD] * (width - len(row)) + row for row in ROWS])
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ROWS])
        for design in designs():
            with self.subTest(design.spec), self.install(self.model, design):
                tokens, scores = self.generate(batch, mask)
                for index, row in enumerate(ROWS):
                    alone_tokens, alone_scores = self.generate(torch.tensor([row]))
                    self.assertEqual(tokens[index].tolist(), alone_tokens[0].tolist())
                    self.assertLess((scores[index] - alone_scores[0]).abs().max().item(), 10 * CLOSE)

    def test_remove_restores_the_parent(self):
        before = self.every_position()
        installed = self.install(self.model, designs()[0])
        self.assertIsInstance(installed, self.Recirculator)
        during = self.every_position()
        installed.remove()
        installed.remove()
        self.assertNotIn("forward", vars(self.model.model))
        self.assertGreater((during - before).abs().max().item(), 1e-3)
        self.assertTrue(torch.equal(self.every_position(), before))
        self.assertTrue(all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in self.model.model.layers))

    def test_misuse_is_rejected(self):
        design = designs()[0]
        with self.assertRaises(TypeError):
            self.Retrofit(self.model, design)
        with self.assertRaises(TypeError):
            self.Recirculator(self.model, block_loop(LAYERS, 2, 4, 2))
        with self.assertRaises(ValueError):
            self.install(self.model, Recirculation(LAYERS + 1, 4, 1, 0.1))
        with self.install(self.model, design):
            for other in (design, block_loop(LAYERS, 2, 4, 2)):
                with self.assertRaises(RuntimeError):
                    self.install(self.model, other)
            with self.assertRaises(NotImplementedError), torch.no_grad():
                self.model(torch.tensor([IDS]), output_hidden_states=True)
        with self.install(self.model, block_loop(LAYERS, 2, 4, 2)):
            with self.assertRaises(RuntimeError):
                self.install(self.model, design)


if __name__ == "__main__":
    unittest.main()
