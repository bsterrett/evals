"""Rebuilding a LoopUS checkpoint: the naming and configuration rules, which need no model, and
loading a synthetic checkpoint written in the released layout.

The loading tests need torch and transformers; run them in the WSL retrofit
environment (see test_runner.py).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from recurrent_retrofit import loopus
from recurrent_retrofit.program import GatedLoop

LAYERS = 6
BLOCKS = {"encoder": [0], "reasoning": [1, 2, 3], "decoder": [4, 5]}
IDS = [5, 17, 3, 44, 9, 61, 2]


def configuration() -> dict:
    """A checkpoint configuration in the released layout, with the fields a later library writes."""
    return {
        "N": 4, "model_type": "lds", "q_threshold": 0.9, "q_eval_interval": 1, "halting_strategy": "threshold",
        "convergence_epsilon": 0.01, "base_model_name_or_path": "example/parent", "transformers_version": "5.3.0",
        "encoder_layer_indices": BLOCKS["encoder"], "reasoning_layer_indices": BLOCKS["reasoning"],
        "decoder_layer_indices": BLOCKS["decoder"], "hidden_size": 32, "vocab_size": 97,
        "base_config_dict": {
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "vocab_size": 97, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": LAYERS, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "hidden_act": "silu", "max_position_embeddings": 64, "initializer_range": 0.3,
            "rms_norm_eps": 1e-06, "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 1,
            "eos_token_id": 2, "pad_token_id": None, "dtype": "float32", "tie_word_embeddings": False,
            "rope_parameters": {"rope_theta": 5000, "rope_type": "default"}, "sliding_window": None,
            "use_sliding_window": False, "layer_types": ["full_attention"] * LAYERS, "id2label": {"0": "LABEL_0"},
            "transformers_version": "5.3.0", "use_cache": True},
    }


class NamingTests(unittest.TestCase):
    def test_block_positions_map_to_parent_layers(self):
        config = configuration()
        cases = {
            "encoder.layers.0.mlp.up_proj.weight": ("model", "model.layers.0.mlp.up_proj.weight"),
            "reasoning.layers.0.self_attn.q_proj.weight": ("model", "model.layers.1.self_attn.q_proj.weight"),
            "reasoning.layers.2.input_layernorm.weight": ("model", "model.layers.3.input_layernorm.weight"),
            "decoder.layers.1.self_attn.k_norm.weight": ("model", "model.layers.5.self_attn.k_norm.weight"),
            "encoder.embed_tokens.weight": ("model", "model.embed_tokens.weight"),
            "decoder.norm.weight": ("model", "model.norm.weight"),
            "decoder.lm_head.weight": ("model", "lm_head.weight"),
            "reasoning.gate.dt_input_proj.weight": ("loop", "gate.narrow.weight"),
            "reasoning.gate.delta_proj.weight": ("loop", "gate.widen.weight"),
            "reasoning.gate.delta_proj.bias": ("loop", "gate.widen.bias"),
            "reasoning.gate.A_log": ("loop", "gate.log_rate"),
            "q_head.0.weight": ("loop", "head.0.weight"), "q_head.1.bias": ("loop", "head.1.bias"),
        }
        for key, expected in cases.items():
            self.assertEqual(loopus.rename(key, config), expected, key)

    def test_the_released_layout_maps_as_the_paper_describes(self):
        # Qwen3-1.7B: encoder layers 0-1, decoder layer 27, the rest looped.
        config = dict(configuration(), encoder_layer_indices=[0, 1], reasoning_layer_indices=list(range(2, 27)),
            decoder_layer_indices=[27])
        self.assertEqual(loopus.rename("reasoning.layers.0.mlp.up_proj.weight", config)[1], "model.layers.2.mlp.up_proj.weight")
        self.assertEqual(loopus.rename("reasoning.layers.24.mlp.up_proj.weight", config)[1], "model.layers.26.mlp.up_proj.weight")
        self.assertEqual(loopus.rename("decoder.layers.0.mlp.up_proj.weight", config)[1], "model.layers.27.mlp.up_proj.weight")

    def test_unrecognised_tensors_are_refused(self):
        config = configuration()
        for key in ["reasoning.layers.3.mlp.up_proj.weight", "encoder.layers.1.mlp.up_proj.weight", "plateau.layers.0.x",
                "reasoning.layers.x.weight", "reasoning.layers.0", "reasoning.gate.B", "q_head.2.weight", "lm_head.weight",
                "decoder.embed_tokens.weight", "encoder.rotary_emb.inv_freq"]:
            with self.assertRaises(ValueError, msg=key):
                loopus.rename(key, config)

    def read(self, config: dict) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            return loopus.read_config(Path(folder))

    def test_a_supported_configuration_is_read(self):
        self.assertEqual(self.read(configuration())["N"], 4)

    def test_unsupported_configurations_are_refused(self):
        def changed(**fields) -> dict:
            return dict(configuration(), **fields)

        def base(**fields) -> dict:
            config = configuration()
            config["base_config_dict"].update(fields)
            return config

        for label, config in {
            "another format": changed(model_type="qwen3"),
            "another parent family": base(model_type="llama"),
            "blocks out of order": changed(encoder_layer_indices=[4, 5], decoder_layer_indices=[0]),
            "a layer in two blocks": changed(encoder_layer_indices=[0, 1]),
            "a layer in no block": changed(decoder_layer_indices=[5]),
            "nothing looped": changed(encoder_layer_indices=[0, 1, 2, 3], reasoning_layer_indices=[]),
            "another stopping rule": changed(halting_strategy="cdf"),
            "head read every other pass": changed(q_eval_interval=2),
            "sliding window": base(sliding_window=16),
            "scaled rotary positions": base(rope_parameters={"rope_theta": 5000, "rope_type": "yarn"}),
            "no rotary base": base(rope_parameters={"rope_type": "default"}),
        }.items():
            with self.assertRaises(ValueError, msg=label):
                self.read(config)


@unittest.skipIf(torch is None, "needs torch and transformers; run in the WSL retrofit environment")
class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recurrent_retrofit.gated import Loop
        torch.manual_seed(3)
        base = configuration()["base_config_dict"]
        stock = Qwen3Config(**{name: base[name] for name in loopus.ARCHITECTURE if name in base},
            rope_theta=5000.0, sliding_window=None, tie_word_embeddings=False)
        cls.reference = Qwen3ForCausalLM(stock).eval()
        cls.loop = Loop(32, 2, (1, 4), trained_passes=4, threshold=0.9)
        with torch.no_grad():
            cls.loop.gate.log_rate.normal_(0.0, 0.5)
            cls.loop.head[1].weight.mul_(10)
        # The checkpoint file, in the released names.
        cls.tensors = {}
        state = cls.reference.state_dict()
        for block, indices in BLOCKS.items():
            for position, index in enumerate(indices):
                prefix = f"model.layers.{index}."
                for name, tensor in state.items():
                    if name.startswith(prefix):
                        cls.tensors[f"{block}.layers.{position}.{name[len(prefix):]}"] = tensor.clone()
        cls.tensors["encoder.embed_tokens.weight"] = state["model.embed_tokens.weight"].clone()
        cls.tensors["decoder.norm.weight"] = state["model.norm.weight"].clone()
        cls.tensors["decoder.lm_head.weight"] = state["lm_head.weight"].clone()
        gate, head = cls.loop.gate, cls.loop.head
        cls.tensors.update({"reasoning.gate.dt_input_proj.weight": gate.narrow.weight, "reasoning.gate.delta_proj.weight": gate.widen.weight,
            "reasoning.gate.delta_proj.bias": gate.widen.bias, "reasoning.gate.A_log": gate.log_rate,
            "q_head.0.weight": head[0].weight, "q_head.0.bias": head[0].bias,
            "q_head.1.weight": head[1].weight, "q_head.1.bias": head[1].bias})
        cls.tensors = {name: tensor.detach().clone() for name, tensor in cls.tensors.items()}

    def write(self, folder: str, tensors: dict | None = None, config: dict | None = None) -> Path:
        from safetensors.torch import save_file
        path = Path(folder)
        (path / "config.json").write_text(json.dumps(config or configuration()), encoding="utf-8")
        save_file(dict(self.tensors if tensors is None else tensors), str(path / "model.safetensors"))
        return path

    def test_the_file_has_one_tensor_per_parameter(self):
        self.assertEqual(len(self.tensors), LAYERS * 11 + 3 + 8)
        self.assertFalse(torch.equal(self.tensors["decoder.lm_head.weight"], self.tensors["encoder.embed_tokens.weight"]))

    def test_every_tensor_lands_where_the_parent_has_it(self):
        with tempfile.TemporaryDirectory() as folder:
            model, facts = loopus.assemble(self.write(folder), torch.float32, "cpu")
        self.assertEqual(type(model).__module__, "transformers.models.qwen3.modeling_qwen3")
        self.assertFalse(model.training)
        expected = self.reference.state_dict()
        loaded = {name: tensor for name, tensor in model.state_dict().items() if not name.startswith("gated_loop.")}
        self.assertEqual(set(loaded), set(expected))
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(loaded[name], tensor), name)
        # The layers differ from each other, so a shuffled mapping could not pass the check above.
        self.assertFalse(torch.equal(expected["model.layers.1.mlp.up_proj.weight"], expected["model.layers.2.mlp.up_proj.weight"]))
        for name, tensor in self.loop.state_dict().items():
            self.assertTrue(torch.equal(model.gated_loop.state_dict()[name], tensor), name)
        self.assertIsNot(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(model.gated_loop.unit, (1, 4))
        self.assertEqual(facts, {"format": "lds", "unit": [1, 4], "trained_passes": 4, "checkpoint_threshold": 0.9,
            "tensors": len(self.tensors), "parent_named_in_config": "example/parent", "written_by_transformers": "5.3.0"})

    def test_settings_written_by_a_later_library_are_translated(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = loopus.assemble(self.write(folder), torch.float32, "cpu")
        config = model.config
        self.assertEqual(config.rope_theta, 5000.0)
        self.assertTrue(torch.equal(model.model.rotary_emb.inv_freq, self.reference.model.rotary_emb.inv_freq))
        self.assertEqual(config.layer_types, ["full_attention"] * LAYERS)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual((config.num_hidden_layers, config.eos_token_id, config.model_type), (LAYERS, 2, "qwen3"))
        self.assertEqual(config._attn_implementation, "sdpa")

    def test_the_rebuilt_model_computes_what_the_parts_do_by_hand(self):
        from recurrent_retrofit.reference import ExplicitGated
        from recurrent_retrofit.runner import install
        with tempfile.TemporaryDirectory() as folder:
            model, _ = loopus.assemble(self.write(folder), torch.float32, "cpu")
        self.reference.gated_loop = self.loop
        try:
            by_hand = ExplicitGated(self.reference)
            for design in (GatedLoop(LAYERS, 1, 4, 1), GatedLoop(LAYERS, 1, 4, 3)):
                with install(model, design), torch.no_grad():
                    rebuilt = model(torch.tensor([IDS])).logits[0]
                self.assertLess((rebuilt - by_hand.logits(IDS, design)).abs().max().item(), 1e-4)
        finally:
            del self.reference.gated_loop

    def test_precision_is_applied_to_the_gate_too(self):
        with tempfile.TemporaryDirectory() as folder:
            model, _ = loopus.assemble(self.write(folder), torch.bfloat16, "cpu")
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.bfloat16})

    def test_an_incomplete_or_padded_file_is_refused(self):
        missing = {name: tensor for name, tensor in self.tensors.items() if name != "reasoning.layers.1.mlp.up_proj.weight"}
        no_gate = {name: tensor for name, tensor in self.tensors.items() if name != "reasoning.gate.A_log"}
        no_head = {name: tensor for name, tensor in self.tensors.items() if name != "q_head.1.bias"}
        extra = dict(self.tensors, **{"reasoning.layers.3.mlp.up_proj.weight": self.tensors["reasoning.layers.2.mlp.up_proj.weight"]})
        wrong_shape = dict(self.tensors, **{"decoder.norm.weight": torch.ones(31)})
        for label, tensors in {"missing layer tensor": missing, "missing gate tensor": no_gate, "missing head tensor": no_head,
                "tensor beyond the block": extra, "wrong shape": wrong_shape}.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    loopus.assemble(self.write(folder, tensors), torch.float32, "cpu")

    def test_a_block_split_that_disagrees_with_the_file_is_refused(self):
        config = copy.deepcopy(configuration())
        config["encoder_layer_indices"], config["reasoning_layer_indices"] = [0, 1], [2, 3]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises((ValueError, RuntimeError)):
                loopus.assemble(self.write(folder, config=config), torch.float32, "cpu")

    def test_the_tokenizer_is_built_from_the_checkpoints_own_file(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        vocabulary = {"<|endoftext|>": 0, "<|im_end|>": 1, "<|im_start|>": 2, "hello": 3, "world": 4, "[UNK]": 5}
        raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        raw.add_special_tokens(["<|endoftext|>", "<|im_end|>", "<|im_start|>"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            raw.save(str(path / "tokenizer.json"))
            # `extra_special_tokens` as a list is what the pinned library cannot parse.
            (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
                "extra_special_tokens": ["<|im_start|>", "<|im_end|>"], "model_max_length": 128,
                "tokenizer_class": "Qwen2Tokenizer", "backend": "tokenizers"}), encoding="utf-8")
            (path / "chat_template.jinja").write_text("{{ messages[0]['content'] }}", encoding="utf-8")
            tokenizer = loopus.tokenizer_of(path)
        self.assertEqual(tokenizer.encode("hello world"), [3, 4])
        self.assertEqual((tokenizer.eos_token_id, tokenizer.pad_token_id), (1, 0))
        self.assertEqual(tokenizer.decode([3, 1, 4], skip_special_tokens=True), "hello world")
        self.assertEqual(tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True), [3])


if __name__ == "__main__":
    unittest.main()
