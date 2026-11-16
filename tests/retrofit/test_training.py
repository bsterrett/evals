"""Training a gate over a frozen parent: the pinned text and its packing, which need no model, and
the trainer on a tiny Qwen3 model that was first taught a simple rule, on the CPU.

The trainer tests need torch and transformers; run them in the WSL retrofit
environment (see test_runner.py).
"""
from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import numpy
    import pyarrow
    import pyarrow.parquet
except ImportError:
    numpy = None
try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from recurrent_retrofit import training
from recurrent_retrofit.program import GatedLoop

LAYERS, UNIT, VOCAB, WIDTH = 6, (2, 5), 97, 12
PARENT = "example/parent@0123"
REVISION = "0123456789abcdef0123456789abcdef01234567"


class Letters:
    """A tokenizer for these tests: one token per character."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


class PinTests(unittest.TestCase):
    def test_every_pinned_text_names_a_commit_a_size_and_a_hash(self):
        self.assertIn("fineweb-edu-2025-26-000", training.TEXTS)
        for name, entry in training.TEXTS.items():
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$", name)
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", name)
            self.assertGreater(entry["bytes"], 0, name)
            self.assertTrue(entry["dataset"] and entry["file"] and entry["field"], name)

    def test_an_unpinned_text_is_refused(self):
        with self.assertRaises(ValueError):
            training.text_pin("some other text")

    def test_a_cached_file_is_accepted_only_if_it_is_the_pinned_one(self):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest("needs huggingface_hub")
        import hashlib
        content = b"the pinned bytes"
        pin = {"dataset": "example/text", "revision": REVISION, "file": "data/part.parquet", "field": "text",
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(training.TEXTS, {"tiny": pin}):
            root = Path(folder)
            path = root / "cache/huggingface/hub/datasets--example--text/snapshots" / REVISION / "data/part.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            self.assertEqual(training.text_file(root, "tiny").read_bytes(), content)
            path.write_bytes(b"the pinned bytez")  # the same size
            with self.assertRaises(RuntimeError):
                training.text_file(root, "tiny")
            path.write_bytes(content + b"!")
            with self.assertRaises(RuntimeError):
                training.text_file(root, "tiny")


@unittest.skipIf(numpy is None, "needs numpy and pyarrow; run in the WSL retrofit environment")
class PackingTests(unittest.TestCase):
    def test_documents_are_joined_with_the_separator_and_cut_into_full_rows(self):
        rows, opened = training.pack(iter([[1, 2, 3], [4], [5, 6, 7, 8, 9], [10, 11]]), 4, 0, 3)
        self.assertEqual(rows.tolist(), [[1, 2, 3, 0], [4, 0, 5, 6], [7, 8, 9, 0]])
        self.assertEqual((opened, str(rows.dtype)), (3, "uint32"))

    def test_the_last_document_is_cut_off_where_the_rows_end(self):
        rows, opened = training.pack(iter([[1, 2, 3, 4, 5, 6]]), 2, 0, 2)
        self.assertEqual((rows.tolist(), opened), ([[1, 2], [3, 4]], 1))

    def test_text_that_runs_out_is_refused(self):
        with self.assertRaises(RuntimeError):
            training.pack(iter([[1, 2, 3]]), 4, 0, 2)

    def write(self, folder: str, documents: list, group: int = 10) -> Path:
        path = Path(folder) / "text.parquet"
        table = pyarrow.table({"text": documents, "score": [1.0] * len(documents)})
        pyarrow.parquet.write_table(table, str(path), row_group_size=group)
        return path

    def documents(self, count: int = 30) -> list:
        return [f"<{index:02d}>" + "x" * (index % 5) for index in range(count)]

    def prepared(self, folder: str, **sizes):
        return training.prepare_text(self.write(folder, self.documents()), "text", Letters(), Path(folder) / "rows",
            context=8, separator=0, facts={"text": "tiny"}, **sizes)

    def test_training_rows_come_from_the_start_and_held_out_rows_from_the_end(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = self.prepared(folder, train_rows=6, heldout_rows=3)
            train, heldout, loaded = training.load_tokens(Path(folder) / "rows")
            train, heldout = train.tolist(), heldout.tolist()
        self.assertEqual(loaded, manifest)
        documents = self.documents()

        def stream(texts: list) -> list[int]:
            return [token for text in texts for token in [*map(ord, text), 0]]

        self.assertEqual([token for row in train for token in row], stream(documents)[:48])
        self.assertEqual([token for row in heldout for token in row], stream(documents[::-1])[:24])
        first, last = manifest["train"]["documents"], manifest["heldout"]["documents"]
        self.assertEqual((first[0], last[1], manifest["documents_in_file"]), (0, 30, 30))
        self.assertLessEqual(first[1], last[0])
        # The counts are the documents the rows really hold.
        self.assertEqual(len(stream(documents[:first[1] - 1])) < 48 <= len(stream(documents[:first[1]])), True)
        self.assertEqual(len(stream(documents[last[0] + 1:])) < 24 <= len(stream(documents[last[0]:])), True)
        self.assertEqual((manifest["text"], manifest["context"], manifest["separator"]), ("tiny", 8, 0))

    def test_rows_that_would_share_a_document_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepared(folder, train_rows=23, heldout_rows=3)  # documents 0 to 26, and 27 to 29
            with self.assertRaises(RuntimeError):
                self.prepared(folder, train_rows=24, heldout_rows=3)  # both would need document 27

    def test_held_out_rows_stay_inside_the_last_row_group(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                self.prepared(folder, train_rows=2, heldout_rows=9)

    def test_an_empty_document_still_gets_its_separator(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, ["ab", None, "", "cd"] + ["zz"] * 6, group=5)
            training.prepare_text(path, "text", Letters(), Path(folder) / "rows", context=4, separator=0,
                train_rows=2, heldout_rows=1, facts={})
            train, _, _ = training.load_tokens(Path(folder) / "rows")
            self.assertEqual(train.tolist(), [[97, 98, 0, 0], [0, 99, 100, 0]])

    def test_changed_rows_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepared(folder, train_rows=6, heldout_rows=3)
            path = Path(folder) / "rows/train.npy"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(bytes(data))
            with self.assertRaises(RuntimeError):
                training.load_tokens(Path(folder) / "rows")

    def test_the_row_order_is_a_fixed_shuffle_of_every_row(self):
        order = training.row_order(3, 50)
        self.assertEqual(sorted(order.tolist()), list(range(50)))
        self.assertEqual(order.tolist(), training.row_order(3, 50).tolist())
        self.assertNotEqual(order.tolist(), training.row_order(4, 50).tolist())
        self.assertNotEqual(order.tolist(), list(range(50)))


class ScoredPassTests(unittest.TestCase):
    def test_scored_passes_depend_only_on_the_seed_and_the_batch(self):
        draws = [training.scored_passes(0, batch, 8, 2) for batch in range(40)]
        self.assertEqual(draws, [training.scored_passes(0, batch, 8, 2) for batch in range(40)])
        for draw in draws:
            self.assertEqual(len(draw), 2)
            self.assertTrue(draw <= set(range(8)))
        self.assertGreater(len(set(draws)), 10)
        self.assertEqual(set().union(*draws), set(range(8)))
        self.assertNotEqual(draws, [training.scored_passes(1, batch, 8, 2) for batch in range(40)])
        self.assertEqual(training.scored_passes(0, 0, 3, 3), frozenset(range(3)))


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.gated import Loop
        from recurrent_retrofit.runner import install
        cls.Loop, cls.install = Loop, staticmethod(install)
        torch.manual_seed(0)
        cls.rule = torch.randperm(VOCAB)
        # A large initializer range makes every layer matter, so looping the unit changes the output.
        config = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            initializer_range=0.3)
        model = Qwen3ForCausalLM(config)
        # Teach the parent the rule, so that it has something for a loop to spoil.
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for index in range(300):
            ids = cls.rows(16, index)
            model(input_ids=ids, labels=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        cls.model = model.eval()
        cls.heldout = cls.rows(24, 10_000)

    @classmethod
    def rows(cls, count: int, seed: int):
        """Rows in which every token is its predecessor's image under a fixed permutation."""
        generator = torch.Generator().manual_seed(seed)
        columns = [torch.randint(0, VOCAB, (count,), generator=generator)]
        for _ in range(WIDTH - 1):
            columns.append(cls.rule[columns[-1]])
        return torch.stack(columns, dim=1)

    def trainer(self, **settings):
        defaults = {"passes": 4, "supervised": 2, "learning_rate": 1e-2, "gate_rank": 4}
        return training.Trainer(self.model, UNIT, **{**defaults, **settings})

    def random_loop(self):
        loop = self.Loop(32, 4, UNIT, trained_passes=4)
        with torch.no_grad():
            loop.gate.log_rate.normal_(0.0, 0.5)
            loop.gate.widen.bias.normal_(0.0, 1.0)
        return loop.eval()

    def through_the_runner(self, loop, rows, passes: int):
        """Logits of every row from the installed gated loop, one row per forward pass."""
        self.model.gated_loop = loop
        try:
            with self.install(self.model, GatedLoop(LAYERS, *UNIT, passes)), torch.no_grad():
                return torch.cat([self.model(input_ids=row[None]).logits for row in rows])
        finally:
            del self.model.gated_loop

    def mean_loss(self, logits, rows) -> float:
        return float(torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), rows[:, 1:].flatten()))

    def by_hand(self, loop, rows, passes: int):
        parts = training.Parts(self.model, UNIT)
        with torch.no_grad():
            state = parts.below(rows)
            for _ in range(passes):
                state = loop.gate(parts.unit(state), state)
            return parts.logits(state)

    def test_the_parent_learned_the_rule_and_looping_spoils_it(self):
        with torch.no_grad():
            parent = self.mean_loss(self.model(input_ids=self.heldout).logits, self.heldout)
        self.assertLess(parent, 0.05)
        loop = self.Loop(32, 4, UNIT)
        training.open_gate(loop)
        for passes in (2, 3, 4):
            self.assertGreater(self.mean_loss(self.through_the_runner(loop, self.heldout, passes), self.heldout), 10 * parent)

    def test_a_fully_open_gate_reproduces_the_parent_on_the_first_pass(self):
        loop = self.Loop(32, 4, UNIT)
        training.open_gate(loop, -40.0)
        with torch.no_grad():
            parent = self.model(input_ids=self.heldout).logits
        self.assertTrue(torch.equal(self.through_the_runner(loop, self.heldout, 1), parent))
        # A loop as the class makes it lets about half through, which is not the parent.
        self.assertGreater(float((self.through_the_runner(self.Loop(32, 4, UNIT), self.heldout, 1) - parent).abs().max()), 0.1)

    def test_the_starting_gate_lets_the_same_share_through_for_every_input(self):
        loop = self.trainer().loop
        share = training.share_let_through(loop.gate, torch.randn(3, 5, 32) * 7, torch.randn(3, 5, 32))
        expected = math.exp(-math.log1p(math.exp(training.OPEN_BIAS)))
        self.assertTrue(torch.allclose(share, torch.full_like(share, expected), atol=1e-6))
        self.assertTrue(0.997 < expected < 0.998)
        self.assertEqual(float(loop.gate.widen.weight.detach().abs().max()), 0.0)
        self.assertGreater(float(loop.gate.narrow.weight.detach().abs().max()), 0.0)

    def test_the_share_let_through_is_the_gates_own_arithmetic(self):
        loop = self.random_loop()
        new, old = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
        with torch.no_grad():
            share = training.share_let_through(loop.gate, new, old)
            self.assertTrue(torch.allclose(share * new + (1 - share) * old, loop.gate(new, old), atol=1e-6))
        self.assertGreater(float(share.max() - share.min()), 0.1)

    def test_the_by_hand_forward_pass_agrees_with_the_runner(self):
        loop = self.random_loop()
        for passes in (1, 2, 4):
            gap = (self.by_hand(loop, self.heldout, passes) - self.through_the_runner(loop, self.heldout, passes)).abs().max()
            self.assertLess(float(gap), 1e-4, passes)
        # The gate matters here: without it the logits differ.
        self.assertGreater(float((self.by_hand(loop, self.heldout, 2) - self.through_the_runner(self.Loop(32, 4, UNIT),
            self.heldout, 2)).abs().max()), 0.01)

    def test_loss_by_passes_is_the_runners_loss_at_each_number_of_passes(self):
        loop = self.random_loop()
        measured = training.loss_by_passes(training.Parts(self.model, UNIT), loop, self.heldout.numpy().astype("uint32"), 4, 5)
        self.assertEqual(measured["rows"], 24)
        for passes in (1, 2, 3, 4):
            expected = self.mean_loss(self.through_the_runner(loop, self.heldout, passes), self.heldout)
            self.assertAlmostEqual(measured["loss_after_pass"][passes - 1], expected, places=4)
        with torch.no_grad():
            parent = self.mean_loss(self.model(input_ids=self.heldout).logits, self.heldout)
        self.assertAlmostEqual(measured["parent_loss"], parent, places=4)
        self.assertEqual(len(measured["share_let_through"]), 4)
        self.assertTrue(all(0 < share < 1 for share in measured["share_let_through"]))

    def test_only_the_loop_is_trained_and_the_parent_is_untouched(self):
        before = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        mark = training.fingerprint(self.model)
        trainer = self.trainer()
        start = copy.deepcopy(trainer.loop.state_dict())
        for index in range(3):
            measured = trainer.step(self.rows(4, index))
            self.assertEqual([row["pass"] - 1 for row in measured], sorted(training.scored_passes(0, index, 4, 2)))
        self.assertEqual((trainer.batches, trainer.updates), (3, 3))
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))
        self.assertEqual({id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]},
            {id(parameter) for parameter in trainer.loop.parameters()})
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertEqual(training.fingerprint(self.model), mark)
        changed = [name for name, tensor in trainer.loop.state_dict().items() if not torch.equal(tensor, start[name])]
        self.assertEqual(sorted(changed), sorted(start))

    def test_a_bf16_parent_trains_an_fp32_loop(self):
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        trainer = training.Trainer(model, UNIT, passes=3, supervised=2, learning_rate=1e-2, gate_rank=4)
        start = copy.deepcopy(trainer.loop.state_dict())
        mark = training.fingerprint(model)
        for index in range(2):
            self.assertTrue(all(math.isfinite(row["loss"]) for row in trainer.step(self.rows(4, index))))
        self.assertEqual({parameter.dtype for parameter in trainer.loop.parameters()}, {torch.float32})
        self.assertFalse(torch.equal(trainer.loop.gate.widen.weight, start["gate.widen.weight"]))
        for name, tensor in model.state_dict().items():
            self.assertEqual(tensor.dtype, torch.bfloat16, name)
            self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertEqual(training.fingerprint(model), mark)
        self.assertNotEqual(mark, training.fingerprint(self.model))
        measured = training.loss_by_passes(trainer.parts, trainer.loop, self.heldout.numpy().astype("uint32"), 3, 8)
        self.assertTrue(all(math.isfinite(value) for value in measured["loss_after_pass"]))
        # The runner serves the FP32 loop on the BF16 parent, and agrees with the by-hand pass as far as BF16 allows.
        ids = self.heldout[:4]
        model.gated_loop = trainer.loop
        with self.install(model, GatedLoop(LAYERS, *UNIT, 2)), torch.no_grad():
            served = model(input_ids=ids).logits
        with torch.no_grad():
            state = trainer.parts.below(ids)
            for _ in range(2):
                state = trainer.loop.gate(trainer.parts.unit(state), state)
            by_hand = trainer.parts.logits(state)
        self.assertEqual((served.dtype, by_hand.dtype), (torch.bfloat16, torch.bfloat16))
        self.assertLess(float((served.float() - by_hand.float()).abs().mean()), 0.2)

    def test_the_fingerprint_sees_one_changed_weight_and_ignores_an_attached_loop(self):
        model = copy.deepcopy(self.model)
        mark = training.fingerprint(model)
        model.gated_loop = self.random_loop()
        self.assertEqual(training.fingerprint(model), mark)
        weight = model.model.layers[3].mlp.up_proj.weight
        with torch.no_grad():
            weight[1, 2] = torch.nextafter(weight[1, 2], torch.tensor(10.0))
        self.assertNotEqual(training.fingerprint(model), mark)

    def test_a_batch_accumulates_the_gradient_of_each_scored_pass_alone(self):
        functional = torch.nn.functional
        # Two scored passes in a row, where a gradient could cross between them, and a batch with an unscored pass.
        for passes, supervised in ((2, 2), (3, 2)):
            trainer = self.trainer(passes=passes, supervised=supervised, clip=0.0)
            loop = trainer.loop
            with torch.no_grad():  # away from the starting gate, whose narrowing map has no gradient yet
                loop.gate.widen.weight.normal_(0.0, 0.05)
                loop.gate.widen.bias.normal_(0.0, 1.0)
                loop.gate.log_rate.normal_(0.0, 0.5)
            taken = []
            trainer._update = lambda: taken.append({name: parameter.grad.clone() for name, parameter in loop.named_parameters()})
            ids = self.rows(4, 77)
            trainer.step(ids)
            # By hand: every scored pass is its own graph, starting from a state that carries no gradient.
            parts, chosen = training.Parts(self.model, UNIT), training.scored_passes(0, 0, passes, supervised)
            expected = {name: torch.zeros_like(parameter) for name, parameter in loop.named_parameters()}
            with torch.no_grad():
                state = parts.below(ids)
            for index in range(passes):
                with torch.no_grad():
                    new = parts.unit(state)
                    before = parts.loss(state, ids)[0]
                mixed = loop.gate(new, state)
                if index in chosen:
                    loss, correct = parts.loss(mixed, ids)
                    score = loop.head((new - state)[:, -1]).squeeze(-1)
                    total = loss + functional.silu(loss - before) + functional.binary_cross_entropy_with_logits(score, correct)
                    gradients = torch.autograd.grad(total / supervised, list(loop.parameters()))
                    for (name, _), gradient in zip(loop.named_parameters(), gradients, strict=True):
                        expected[name] += gradient
                state = mixed.detach()
            self.assertEqual(len(taken), 1)
            for name, gradient in expected.items():
                self.assertTrue(torch.allclose(taken[0][name], gradient, atol=1e-6), (passes, name))
                self.assertGreater(float(gradient.abs().max()), 0.0, (passes, name))

    def test_updates_wait_for_the_accumulated_batches(self):
        trainer = self.trainer(accumulate=3)
        start = copy.deepcopy(trainer.loop.state_dict())
        for index in range(2):
            trainer.step(self.rows(4, index))
        self.assertEqual(trainer.updates, 0)
        self.assertTrue(all(torch.equal(tensor, start[name]) for name, tensor in trainer.loop.state_dict().items()))
        with self.assertRaises(RuntimeError):
            trainer.state()
        trainer.step(self.rows(4, 2))
        self.assertEqual(trainer.updates, 1)
        self.assertFalse(torch.equal(trainer.loop.gate.widen.bias, start["gate.widen.bias"]))
        trainer.state()

    def test_the_learning_rate_rises_over_the_warm_up_and_then_stays(self):
        trainer = self.trainer(warmup_updates=4)
        rates = []
        for index in range(6):
            trainer.step(self.rows(4, index))
            rates.append(trainer.optimizer.param_groups[0]["lr"])
        for rate, expected in zip(rates, [0.0025, 0.005, 0.0075, 0.01, 0.01, 0.01], strict=True):
            self.assertAlmostEqual(rate, expected, places=9)

    def test_a_resumed_run_is_the_unbroken_run(self):
        def run(trainer, first: int, last: int):
            for index in range(first, last):
                trainer.step(self.rows(4, index))
            return trainer

        straight = run(self.trainer(accumulate=2, warmup_updates=3), 0, 8)
        broken = run(self.trainer(accumulate=2, warmup_updates=3), 0, 4)
        stored = io.BytesIO()
        torch.save(broken.state(), stored)
        stored.seek(0)
        resumed = self.trainer(accumulate=2, warmup_updates=3)
        resumed.restore(torch.load(stored, weights_only=True))
        self.assertEqual((resumed.batches, resumed.updates), (4, 2))
        run(resumed, 4, 8)
        self.assertEqual((resumed.batches, resumed.updates), (straight.batches, straight.updates))
        for name, tensor in straight.loop.state_dict().items():
            self.assertTrue(torch.equal(resumed.loop.state_dict()[name], tensor), name)
        # The comparison can fail: a run resumed without its optimizer's state differs.
        forgetful = self.trainer(accumulate=2, warmup_updates=3)
        forgetful.loop.load_state_dict(broken.loop.state_dict())
        forgetful.batches, forgetful.updates = 4, 2
        run(forgetful, 4, 8)
        self.assertFalse(all(torch.equal(forgetful.loop.state_dict()[name], tensor)
            for name, tensor in straight.loop.state_dict().items()))

    def test_unusable_settings_are_refused(self):
        for settings in ({"supervised": 5}, {"supervised": 0}, {"accumulate": 0}):
            with self.assertRaises(ValueError, msg=settings):
                self.trainer(**settings)
        with self.assertRaises(ValueError):
            training.Trainer(self.model, (4, 7), passes=2, supervised=1, learning_rate=1e-3)
        self.model.train()
        try:
            with self.assertRaises(ValueError):
                self.trainer()
        finally:
            self.model.eval()

    def test_training_lowers_the_loss_it_is_given(self):
        trainer = self.trainer(accumulate=2, warmup_updates=5)
        parts, rows = trainer.parts, self.heldout.numpy().astype("uint32")
        before = training.loss_by_passes(parts, trainer.loop, rows, 4, 8)
        for index in range(240):
            trainer.step(self.rows(8, 500 + index))
        after = training.loss_by_passes(parts, trainer.loop, rows, 4, 8)
        # Scored passes are drawn evenly, so what training lowers is the loss averaged over the passes.
        self.assertLess(sum(after["loss_after_pass"]), 0.75 * sum(before["loss_after_pass"]))
        for index in (2, 3):
            self.assertLess(after["loss_after_pass"][index], 0.8 * before["loss_after_pass"][index], index + 1)
        self.assertGreater(after["share_let_through"][0], after["share_let_through"][3] + 0.1)
        # In this tiny model a linear map cannot tell the first pass's change from a later one's, so
        # the gate settles on a compromise: it closes part of the way on the first pass too, and
        # the first pass ends worse than the parent. The parent's loss on these rows is unchanged.
        self.assertGreater(after["loss_after_pass"][0], before["loss_after_pass"][0])
        self.assertEqual(after["parent_loss"], before["parent_loss"])
        # The runner computes the same losses from the trained loop.
        for passes in (1, 4):
            expected = self.mean_loss(self.through_the_runner(trainer.loop, self.heldout, passes), self.heldout)
            self.assertAlmostEqual(after["loss_after_pass"][passes - 1], expected, places=4)

    def saved(self, folder: str, **facts) -> tuple:
        loop = self.random_loop()
        manifest = training.save_loop(Path(folder), loop, {"parent": PARENT, "tokens_seen": 4096, **facts})
        return loop, manifest

    def test_a_saved_loop_loads_as_it_was_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            loop, manifest = self.saved(folder, note="kept")
            loaded, read = training.load_loop(Path(folder))
        self.assertEqual(read, manifest)
        self.assertEqual({name: manifest[name] for name in ("format", "parent", "unit", "hidden", "gate_rank", "head",
            "trained_passes", "tokens_seen", "note")}, {"format": training.FORMAT, "parent": PARENT, "unit": [2, 5],
            "hidden": 32, "gate_rank": 4, "head": True, "trained_passes": 4, "tokens_seen": 4096, "note": "kept"})
        self.assertEqual(loaded.unit, UNIT)
        self.assertEqual(set(loaded.state_dict()), set(loop.state_dict()))
        for name, tensor in loop.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], tensor), name)
        model = copy.deepcopy(self.model)
        training.attach(model, loaded, read, PARENT)
        with self.install(model, GatedLoop(LAYERS, *UNIT, 3)), torch.no_grad():
            served = model(input_ids=self.heldout[:1]).logits
        self.assertLess(float((served - self.by_hand(loop, self.heldout[:1], 3)).abs().max()), 1e-4)

    def test_a_loop_is_attached_only_to_the_parent_it_was_trained_over(self):
        with tempfile.TemporaryDirectory() as folder:
            self.saved(folder)
            loaded, manifest = training.load_loop(Path(folder))
        model = copy.deepcopy(self.model)
        for label, changed in {"another parent": {"parent": "example/other@0123"}, "another width": {"hidden": 64},
                "a unit beyond the model": {"unit": [2, 9]}, "an empty unit": {"unit": [3, 3]}}.items():
            with self.assertRaises(ValueError, msg=label):
                training.attach(model, loaded, {**manifest, **changed}, PARENT)
            self.assertFalse(hasattr(model, "gated_loop"), label)

    def test_a_changed_or_incomplete_file_is_refused(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.saved(folder)
            original = (path / "loop.safetensors").read_bytes()
            manifest = json.loads((path / "loop.json").read_text(encoding="utf-8"))
            changed = bytearray(original)
            changed[-1] ^= 1
            (path / "loop.safetensors").write_bytes(bytes(changed))
            with self.assertRaises(ValueError):
                training.load_loop(path)
            # A file the manifest does name, but which lacks a tensor or carries one too many.
            (path / "loop.safetensors").write_bytes(original)
            tensors = load_file(str(path / "loop.safetensors"))
            for label, altered in {"missing": {name: tensor for name, tensor in tensors.items() if name != "gate.log_rate"},
                    "extra": {**tensors, "gate.other": torch.zeros(2)}}.items():
                save_file(altered, str(path / "loop.safetensors"))
                (path / "loop.json").write_text(json.dumps({**manifest,
                    "tensors_sha256": training.file_sha256(path / "loop.safetensors")}), encoding="utf-8")
                with self.assertRaises(RuntimeError, msg=label):
                    training.load_loop(path)
            (path / "loop.safetensors").write_bytes(original)
            (path / "loop.json").write_text(json.dumps({**manifest, "format": "something else"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                training.load_loop(path)
            (path / "loop.json").write_text(json.dumps(manifest), encoding="utf-8")
            training.load_loop(path)


if __name__ == "__main__":
    unittest.main()
