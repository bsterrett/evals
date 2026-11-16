"""Batched and library-driven decoding under a design, on a tiny random Qwen3 model.

The benchmark harness scores likelihoods on right-padded batches without an
attention mask, and generates through `model.generate` on left-padded batches
with one. Both must compute, for every row, what that row computes alone.
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

from recurrent_retrofit.program import Update, block_loop, layer_loop, parent

LAYERS = 6
ROWS = [[5, 17, 3, 44, 9, 61, 2], [8, 30, 71], [12, 4, 90, 33, 6]]
NEW = 5
PAD = 0
CLOSE = 2e-5


def cases():
    return [
        parent(LAYERS),
        block_loop(LAYERS, 2, 4, 3),
        layer_loop(LAYERS, 1, 4, 2),
        block_loop(LAYERS, 2, 4, 3, Update("euler")),
        block_loop(LAYERS, 1, 4, 2, Update("euler", anchor=0.5)),
        layer_loop(LAYERS, 2, 5, 3, Update("uniform")),
        block_loop(LAYERS, 2, 4, 3, Update("ema", eta=0.5, per="layer")),
    ]


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class BatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.runner import Retrofit, greedy
        cls.Retrofit, cls.greedy = Retrofit, staticmethod(greedy)
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            initializer_range=0.3)
        cls.model = Qwen3ForCausalLM(config).eval()

    def generate(self, ids, mask=None):
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=NEW, min_new_tokens=NEW,
                do_sample=False, pad_token_id=PAD, eos_token_id=None, use_cache=True,
                output_scores=True, return_dict_in_generate=True)
        return out.sequences[:, ids.shape[1]:], torch.stack(out.scores, dim=1)

    def test_right_padded_rows_score_as_they_do_alone(self):
        width = max(map(len, ROWS))
        batch = torch.tensor([row + [PAD] * (width - len(row)) for row in ROWS])
        for program in cases():
            with self.subTest(program.spec, passes=program.passes), self.Retrofit(self.model, program) as retrofit:
                with torch.no_grad():
                    together = self.model(batch).logits
                    retrofit.check_calls(retrofit.take_calls())
                    for index, row in enumerate(ROWS):
                        alone = self.model(torch.tensor([row])).logits[0]
                        self.assertLess((together[index, :len(row)] - alone).abs().max().item(), CLOSE)

    def test_library_generation_matches_the_reference_decoder(self):
        ids = torch.tensor([ROWS[0]])
        for program in cases():
            with self.subTest(program.spec, passes=program.passes), self.Retrofit(self.model, program) as retrofit:
                expected = self.greedy(self.model, ROWS[0], NEW)
                retrofit.take_calls()
                tokens, scores = self.generate(ids)
                retrofit.check_calls(retrofit.take_calls(), forwards=NEW)
                self.assertEqual(tokens[0].tolist(), expected["tokens"])
                reference = torch.stack([torch.as_tensor(step) for step in expected["logits"]])
                self.assertLess((scores[0] - reference).abs().max().item(), CLOSE)

    def test_left_padded_rows_generate_as_they_do_alone(self):
        width = max(map(len, ROWS))
        batch = torch.tensor([[PAD] * (width - len(row)) + row for row in ROWS])
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ROWS])
        for program in cases():
            with self.subTest(program.spec, passes=program.passes), self.Retrofit(self.model, program) as retrofit:
                tokens, scores = self.generate(batch, mask)
                retrofit.check_calls(retrofit.take_calls(), forwards=NEW)
                for index, row in enumerate(ROWS):
                    alone_tokens, alone_scores = self.generate(torch.tensor([row]))
                    self.assertEqual(tokens[index].tolist(), alone_tokens[0].tolist())
                    self.assertLess((scores[index] - alone_scores[0]).abs().max().item(), 10 * CLOSE)

    def test_a_design_changes_batched_scores(self):
        batch = torch.tensor([ROWS[0], ROWS[0][::-1]])
        with torch.no_grad():
            before = self.model(batch).logits
            with self.Retrofit(self.model, block_loop(LAYERS, 2, 4, 3, Update("euler"))):
                during = self.model(batch).logits
            after = self.model(batch).logits
        self.assertGreater((during - before).abs().max().item(), 1e-3)
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
