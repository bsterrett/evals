"""Gated-loop checks on a tiny randomly initialised Qwen3 model with a random gate, on the CPU.

Needs torch and transformers; run in the WSL retrofit environment (see
test_runner.py).
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from recurrent_retrofit.program import GatedLoop, Recirculation, block_loop, route

LAYERS = 6
UNIT = (2, 5)
IDS = [5, 17, 3, 44, 9, 61, 2]
ROWS = [IDS, [8, 30, 71], [12, 4, 90, 33, 6], [40, 41, 42, 43]]
NEW = 5
PAD = 0
CLOSE = 2e-5
BUDGET = 4


def designs():
    return [GatedLoop(LAYERS, *UNIT, 1), GatedLoop(LAYERS, *UNIT, 2), GatedLoop(LAYERS, *UNIT, 4)]


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class GatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.gated import Gate, Gated, Loop
        from recurrent_retrofit.reference import ExplicitGated
        from recurrent_retrofit.runner import Retrofit, greedy, install
        cls.Gate, cls.Gated, cls.Loop, cls.Retrofit = Gate, Gated, Loop, Retrofit
        cls.install, cls.greedy = staticmethod(install), staticmethod(greedy)
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            initializer_range=0.3)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.loop = cls.random_loop(UNIT)
        cls.model.gated_loop = cls.loop
        cls.explicit = ExplicitGated(cls.model)
        # Each row's confidence after each pass, from the reference run with no threshold.
        cls.confidences = [cls.trace(row) for row in ROWS]
        cls.threshold = cls.spread_threshold()
        cls.stops = [cls.stop_pass(trace, cls.threshold) for trace in cls.confidences]

    @classmethod
    def random_loop(cls, unit, head: bool = True):
        loop = cls.Loop(32, 2, unit, head=head, trained_passes=BUDGET, threshold=0.5)
        with torch.no_grad():
            loop.gate.log_rate.normal_(0.0, 0.5)
            if head:
                # Wide enough that sequences differ in when they pass a threshold.
                loop.head[1].weight.mul_(10)
        return loop.eval()

    @classmethod
    def trace(cls, ids, explicit=None) -> list[float]:
        """By hand: the head's probability after each of BUDGET passes of an unstopped loop."""
        explicit, values = explicit or cls.explicit, []
        with torch.inference_mode():
            state = explicit.explicit.prepare(ids)
            for index in range(UNIT[0]):
                state = explicit.explicit.layer(index, state)
            for _ in range(BUDGET):
                output = state
                for index in range(*UNIT):
                    output = explicit.explicit.layer(index, output)
                values.append(explicit.confidence(output - state))
                state = explicit.gate(output, state)
        return values

    @staticmethod
    def stop_pass(trace: list[float], threshold: float) -> int:
        return next((index + 1 for index, value in enumerate(trace) if value > threshold), len(trace))

    @classmethod
    def spread_threshold(cls) -> float:
        """A threshold at which the rows stop after as many different numbers of passes as possible."""
        values = sorted(value for trace in cls.confidences for value in trace)
        candidates = [round((low + high) / 2, 6) for low, high in zip(values, values[1:]) if high - low > 1e-3]
        return max(candidates, key=lambda t: len({cls.stop_pass(trace, t) for trace in cls.confidences}))

    @contextmanager
    def loop_as(self, loop):
        """Swap the model's trained loop, or take it away with None."""
        del self.model.gated_loop
        if loop is not None:
            self.model.gated_loop = loop
        try:
            yield loop
        finally:
            if loop is not None:
                del self.model.gated_loop
            self.model.gated_loop = self.loop

    def every_position(self, ids=IDS, **kwargs):
        with torch.no_grad():
            return self.model(torch.tensor([ids]), **kwargs).logits[0]

    def hidden(self, ids=IDS):
        """Final hidden states from the decoder itself, whose rounding does not depend on the output head."""
        with torch.no_grad():
            return self.model.model(torch.tensor([ids]), use_cache=False).last_hidden_state[0]

    def generate(self, ids, mask=None, **kwargs):
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=NEW, min_new_tokens=NEW,
                do_sample=False, pad_token_id=PAD, eos_token_id=None, output_scores=True,
                return_dict_in_generate=True, **{"use_cache": True, **kwargs})
        return out.sequences[:, ids.shape[1]:], torch.stack(out.scores, dim=1)

    # The gate.

    def test_the_gate_is_the_written_arithmetic(self):
        gate = self.Gate(2, 1)
        with torch.no_grad():
            gate.narrow.weight.copy_(torch.tensor([[1.0, -1.0]]))
            gate.widen.weight.copy_(torch.tensor([[2.0], [0.5]]))
            gate.widen.bias.copy_(torch.tensor([0.0, 1.0]))
            gate.log_rate.copy_(torch.tensor([0.0, math.log(2.0)]))
            new, old = torch.tensor([[[3.0, 1.0]]]), torch.tensor([[[1.0, 2.0]]])
            # new - old = [2, -1], narrowed to 3, widened to [6, 2.5].
            keep = [math.exp(-math.log1p(math.exp(6.0)) * 1.0), math.exp(-math.log1p(math.exp(2.5)) * 2.0)]
            expected = [keep[0] * 3.0 + (1 - keep[0]) * 1.0, keep[1] * 1.0 + (1 - keep[1]) * 2.0]
            self.assertTrue(torch.allclose(gate(new, old), torch.tensor([[expected]]), atol=1e-6))
            # The result lies between the two states, nearer the old one here.
            self.assertLess(abs(expected[0] - 1.0), abs(expected[0] - 3.0))
            gate.log_rate.fill_(-math.inf)
            self.assertTrue(torch.equal(gate(new, old), new))
            gate.log_rate.fill_(50.0)
            self.assertTrue(torch.equal(gate(new, old), old))

    def test_the_gate_runs_in_its_own_precision_and_returns_the_states(self):
        gate = self.loop.gate
        new, old = torch.randn(1, 3, 32, dtype=torch.float64), torch.randn(1, 3, 32, dtype=torch.float64)
        with torch.no_grad():
            self.assertEqual(gate(new, old).dtype, torch.float64)
            self.assertTrue(torch.allclose(gate(new, old).float(), gate(new.float(), old.float()), atol=1e-6))

    def test_confidence_reads_the_last_valid_token(self):
        change = torch.randn(4, 5, 32)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]])
        with torch.no_grad():
            head = lambda vector: torch.sigmoid(self.loop.head(vector)).squeeze(-1)
            masked = self.loop.confidence(change, mask)
            self.assertTrue(torch.allclose(masked, head(torch.stack([change[0, 4], change[1, 1], change[2, 4], change[3, 0]]))))
            self.assertTrue(torch.allclose(self.loop.confidence(change, None), head(change[:, -1])))
            self.assertTrue(torch.allclose(self.loop.confidence(change, mask.bool()), masked))

    # Fixed number of passes.

    def test_an_open_gate_is_a_plain_block_loop(self):
        with self.loop_as(self.random_loop(UNIT)) as loop:
            with torch.no_grad():
                loop.gate.log_rate.fill_(-math.inf)
            gated = {}
            for passes in (1, 2, 3):
                with self.install(self.model, GatedLoop(LAYERS, *UNIT, passes)):
                    gated[passes] = self.hidden()
        with self.loop_as(None):
            for passes, state in gated.items():
                with self.install(self.model, block_loop(LAYERS, *UNIT, passes)):
                    self.assertTrue(torch.equal(self.hidden(), state), passes)

    def test_a_closed_gate_leaves_the_unit_out(self):
        with self.loop_as(self.random_loop(UNIT)) as loop:
            with torch.no_grad():
                loop.gate.log_rate.fill_(50.0)
            gated = []
            for passes in (1, 3):
                with self.install(self.model, GatedLoop(LAYERS, *UNIT, passes)):
                    gated.append(self.hidden())
        with self.loop_as(None), self.install(self.model, route(LAYERS, [0, 1, 5])):
            skipped = self.hidden()
        for state in gated:
            self.assertTrue(torch.equal(state, skipped))

    def test_the_runner_agrees_with_the_explicit_reference(self):
        cases = [(UNIT, 1), (UNIT, 2), (UNIT, 4), ((0, 3), 3), ((3, 6), 3), ((0, 6), 2), ((4, 5), 5)]
        for unit, passes in cases:
            design = GatedLoop(LAYERS, *unit, passes)
            with self.subTest(design.spec, passes=passes), self.loop_as(self.random_loop(unit)):
                with self.install(self.model, design):
                    decoded = self.greedy(self.model, IDS, NEW)
                    served = self.every_position(IDS + decoded["tokens"])
                by_hand = self.explicit_for().logits(IDS + decoded["tokens"], design)
                self.assertLess((served - by_hand).abs().max().item(), 1e-4)
                for step, logits in enumerate(decoded["logits"]):
                    self.assertLess((logits - by_hand[len(IDS) - 1 + step]).abs().max().item(), 1e-4)

    def explicit_for(self):
        from recurrent_retrofit.reference import ExplicitGated
        return ExplicitGated(self.model)

    def test_one_pass_is_gated_and_every_pass_changes_the_output(self):
        plain = self.every_position()
        outputs = [plain]
        for design in designs():
            with self.install(self.model, design):
                outputs.append(self.every_position())
        for index, first in enumerate(outputs):
            for second in outputs[index + 1:]:
                self.assertGreater((first - second).abs().max().item(), 1e-3)

    def test_every_forward_makes_the_designs_calls_and_fills_the_cache(self):
        for design in designs():
            with self.subTest(passes=design.passes), self.install(self.model, design) as installed:
                seen = []
                def check(step, cache):
                    calls = installed.take_calls()
                    installed.check_calls(calls)
                    installed.check_cache(cache, len(IDS) + step)
                    self.assertEqual(installed.taken["gated"], design.passes)
                    seen.append(calls)
                self.greedy(self.model, IDS, NEW, after_forward=check)
                self.assertEqual(seen, [design.calls_per_layer()] * NEW)
                self.assertEqual(sum(seen[0]), LAYERS + (design.passes - 1) * 3)
                installed.take_calls()
                with self.assertRaises(RuntimeError):
                    installed.check_calls([1] * LAYERS)
                self.every_position()
                wrong = installed.take_calls()
                wrong[UNIT[0]] -= 1
                with self.assertRaises(RuntimeError):
                    installed.check_calls(wrong)

    def test_cached_decoding_matches_full_prefix_recomputation(self):
        for design in designs():
            with self.subTest(passes=design.passes), self.install(self.model, design):
                cached = self.greedy(self.model, IDS, NEW)
                full = self.greedy(self.model, IDS, NEW, cached=False, forced=cached["tokens"])
                for x, y in zip(cached["logits"], full["logits"], strict=True):
                    self.assertLess((x - y).abs().max().item(), CLOSE)

    def test_right_padded_rows_score_as_they_do_alone(self):
        width = max(map(len, ROWS))
        batch = torch.tensor([row + [PAD] * (width - len(row)) for row in ROWS])
        for design in designs():
            with self.subTest(passes=design.passes), self.install(self.model, design):
                with torch.no_grad():
                    together = self.model(batch).logits
                for index, row in enumerate(ROWS):
                    self.assertLess((together[index, :len(row)] - self.every_position(row)).abs().max().item(), CLOSE)

    def test_library_generation_matches_the_reference_decoder(self):
        ids = torch.tensor([IDS])
        for design in designs():
            with self.subTest(passes=design.passes), self.install(self.model, design) as installed:
                expected = self.greedy(self.model, IDS, NEW)
                installed.take_calls()
                tokens, scores = self.generate(ids)
                installed.check_calls(installed.take_calls(), forwards=NEW)
                self.assertEqual(tokens[0].tolist(), expected["tokens"])
                reference = torch.stack([torch.as_tensor(step) for step in expected["logits"]])
                self.assertLess((scores[0] - reference).abs().max().item(), CLOSE)

    def test_left_padded_rows_generate_as_they_do_alone(self):
        width = max(map(len, ROWS))
        batch = torch.tensor([[PAD] * (width - len(row)) + row for row in ROWS])
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ROWS])
        for design in designs():
            with self.subTest(passes=design.passes), self.install(self.model, design):
                tokens, scores = self.generate(batch, mask)
                for index, row in enumerate(ROWS):
                    alone_tokens, alone_scores = self.generate(torch.tensor([row]))
                    self.assertEqual(tokens[index].tolist(), alone_tokens[0].tolist())
                    self.assertLess((scores[index] - alone_scores[0]).abs().max().item(), 10 * CLOSE)

    # Stopping.

    def test_the_rows_stop_after_different_numbers_of_passes(self):
        # A property of the fixture that the stopping tests below rely on.
        self.assertGreaterEqual(len(set(self.stops)), 3)
        self.assertLess(min(self.stops), BUDGET)

    def test_a_threshold_of_one_never_stops(self):
        with self.install(self.model, GatedLoop(LAYERS, *UNIT, 3)):
            fixed = self.every_position(use_cache=False)
        with self.install(self.model, GatedLoop(LAYERS, *UNIT, 3, 1.0)) as installed:
            self.assertTrue(torch.equal(self.every_position(), fixed))
            calls = installed.take_calls()
            installed.check_calls(calls)
            self.assertEqual(calls, [1, 1, 3, 3, 3, 1])
            self.assertEqual(installed.taken, {"forwards": 1, "gated": 3, "passes_per_sequence": {3: 1}})

    def test_a_threshold_of_zero_stops_after_one_pass(self):
        with self.install(self.model, GatedLoop(LAYERS, *UNIT, 1)):
            once = self.every_position(use_cache=False)
        with self.install(self.model, GatedLoop(LAYERS, *UNIT, 3, 0.0)) as installed:
            self.assertTrue(torch.equal(self.every_position(), once))
            calls = installed.take_calls()
            installed.check_calls(calls)
            # The layers of the two passes nobody needed were not called.
            self.assertEqual(calls, [1] * LAYERS)
            self.assertEqual(installed.taken, {"forwards": 1, "gated": 1, "passes_per_sequence": {1: 1}})

    def test_stopping_agrees_with_the_explicit_reference(self):
        design = GatedLoop(LAYERS, *UNIT, BUDGET, self.threshold)
        explicit = self.explicit_for()
        for row, stop in zip(ROWS, self.stops):
            with self.subTest(row=row):
                by_hand = explicit.logits(row, design)
                self.assertEqual(explicit.passes_used, stop)
                with self.install(self.model, design) as installed:
                    served = self.every_position(row)
                    calls = installed.take_calls()
                    installed.check_calls(calls)
                self.assertEqual(installed.taken["passes_per_sequence"], {stop: 1})
                self.assertEqual(calls, [1, 1, stop, stop, stop, 1])
                self.assertLess((served - by_hand).abs().max().item(), 1e-4)
                # Stopping early is the fixed loop of that many passes.
                with self.install(self.model, GatedLoop(LAYERS, *UNIT, stop)):
                    self.assertTrue(torch.equal(self.every_position(row, use_cache=False), served))

    def test_confidence_reads_the_change_before_the_gate(self):
        # With a closed gate the state never changes, so the change after the gate is zero for every
        # sequence and every pass. The head must still tell the sequences apart, from what the layers
        # proposed: a sequence stops after the first pass or never.
        with self.loop_as(self.random_loop(UNIT)) as loop:
            with torch.no_grad():
                loop.gate.log_rate.fill_(50.0)
            first = [self.trace(row, self.explicit_for())[0] for row in ROWS]
            ordered = sorted(first)
            gap, low = max((high - low, low) for low, high in zip(ordered, ordered[1:]))
            self.assertGreater(gap, 1e-3)
            threshold = low + gap / 2
            expected = [1 if value > threshold else 3 for value in first]
            self.assertEqual(set(expected), {1, 3})
            with self.install(self.model, GatedLoop(LAYERS, *UNIT, 3, threshold)) as installed:
                for row, passes in zip(ROWS, expected):
                    self.every_position(row)
                    self.assertEqual(installed.mixer.last_depths, [passes])

    def padded(self, side: str):
        width = max(map(len, ROWS))
        pad = lambda row: [PAD] * (width - len(row))
        if side == "left":
            return (torch.tensor([pad(row) + row for row in ROWS]),
                torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ROWS]))
        return (torch.tensor([row + pad(row) for row in ROWS]),
            torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ROWS]))

    def test_each_row_of_a_batch_stops_on_its_own_last_token(self):
        design = GatedLoop(LAYERS, *UNIT, BUDGET, self.threshold)
        with self.install(self.model, design) as installed:
            alone = [self.every_position(row) for row in ROWS]
            for side in ("left", "right"):
                with self.subTest(side):
                    batch, mask = self.padded(side)
                    installed.take_calls()
                    with torch.no_grad():
                        together = self.model(batch, attention_mask=mask).logits
                    calls = installed.take_calls()
                    installed.check_calls(calls)
                    self.assertEqual(installed.mixer.last_depths, self.stops)
                    self.assertEqual(calls[UNIT[0]], max(self.stops))
                    for index, row in enumerate(ROWS):
                        mine = together[index, -len(row):] if side == "left" else together[index, :len(row)]
                        self.assertLess((mine - alone[index]).abs().max().item(), 10 * CLOSE)

    def test_padding_tokens_do_not_decide_a_masked_row(self):
        design = GatedLoop(LAYERS, *UNIT, BUDGET, self.threshold)
        batch, mask = self.padded("right")
        other = batch.clone()
        other[mask == 0] = 7
        with self.install(self.model, design) as installed, torch.no_grad():
            first = self.model(batch, attention_mask=mask).logits
            depths = installed.mixer.last_depths
            second = self.model(other, attention_mask=mask).logits
            self.assertEqual(installed.mixer.last_depths, depths)
            for index, row in enumerate(ROWS):
                self.assertTrue(torch.equal(first[index, :len(row)], second[index, :len(row)]))

    def test_stopping_takes_whole_sequences_and_keeps_no_cache(self):
        from transformers import DynamicCache
        design = GatedLoop(LAYERS, *UNIT, BUDGET, self.threshold)
        with self.install(self.model, GatedLoop(LAYERS, *UNIT, 2)):
            cache = DynamicCache(config=self.model.config)
            self.every_position(IDS[:3], past_key_values=cache, use_cache=True)
        with self.install(self.model, design) as installed:
            with torch.no_grad():
                out = self.model(torch.tensor([IDS]), use_cache=True)
            self.assertIsNone(out.past_key_values)
            with self.assertRaises(NotImplementedError):
                self.every_position(IDS[3:], past_key_values=cache, use_cache=True)
            with self.assertRaises(NotImplementedError):
                self.every_position(IDS[-1:], cache_position=torch.tensor([len(IDS) - 1]))
            with self.assertRaises(NotImplementedError):
                self.every_position(IDS[-1:], attention_mask=torch.ones(1, len(IDS), dtype=torch.long))
            with self.assertRaises(NotImplementedError):
                self.generate(torch.tensor([IDS]))
            with self.assertRaises(RuntimeError):
                installed.check_cache(cache, 3)
            # Without a cache the library recomputes the prefix each step, which is supported.
            tokens, scores = self.generate(torch.tensor([IDS]), use_cache=False)
            prefix = list(IDS)
            for step in range(NEW):
                logits = self.every_position(prefix)[-1]
                self.assertLess((scores[0, step] - logits).abs().max().item(), CLOSE)
                self.assertEqual(int(tokens[0, step]), int(logits.argmax()))
                prefix.append(int(tokens[0, step]))

    # Installing and removing.

    def test_remove_restores_the_model(self):
        before = self.every_position()
        for design in (GatedLoop(LAYERS, *UNIT, 3), GatedLoop(LAYERS, *UNIT, 3, self.threshold)):
            installed = self.install(self.model, design)
            self.assertIsInstance(installed, self.Gated)
            during = self.every_position()
            installed.remove()
            installed.remove()
            self.assertNotIn("forward", vars(self.model.model))
            self.assertGreater((during - before).abs().max().item(), 1e-3)
            self.assertTrue(torch.equal(self.every_position(), before))
            self.assertEqual(self.model.config.num_hidden_layers, LAYERS)
            self.assertTrue(all(not layer._forward_pre_hooks for layer in self.model.model.layers))

    def test_misuse_is_rejected(self):
        design = GatedLoop(LAYERS, *UNIT, 2)
        with self.assertRaises(TypeError):
            self.Gated(self.model, block_loop(LAYERS, *UNIT, 2))
        # A checkpoint trained with a gate runs as a gated loop only.
        for other in (block_loop(LAYERS, *UNIT, 2), Recirculation(LAYERS, 4, 1, 0.1)):
            with self.assertRaises(ValueError):
                self.install(self.model, other)
        with self.assertRaises(ValueError):
            self.Retrofit(self.model, block_loop(LAYERS, *UNIT, 2))
        # The gate belongs to the unit it was trained for.
        for wrong in (GatedLoop(LAYERS, 2, 4, 2), GatedLoop(LAYERS, 1, 5, 2), GatedLoop(LAYERS + 1, *UNIT, 2)):
            with self.assertRaises(ValueError):
                self.install(self.model, wrong)
        with self.loop_as(None):
            with self.assertRaises(ValueError):
                self.install(self.model, design)
        with self.loop_as(self.random_loop(UNIT, head=False)):
            with self.assertRaises(ValueError):
                self.install(self.model, GatedLoop(LAYERS, *UNIT, 2, 0.5))
            with self.install(self.model, design):
                self.every_position()
        with self.install(self.model, design):
            with self.assertRaises(RuntimeError):
                self.install(self.model, design)
        with self.install(self.model, GatedLoop(LAYERS, *UNIT, 2, 0.5)):
            with self.assertRaises(RuntimeError):
                self.install(self.model, design)
            with self.assertRaises(NotImplementedError):
                self.every_position(attention_mask=torch.ones(1, 1, len(IDS), len(IDS)))


if __name__ == "__main__":
    unittest.main()
