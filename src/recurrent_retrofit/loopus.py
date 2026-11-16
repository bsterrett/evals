"""Rebuild a released LoopUS checkpoint as a stock decoder with a gated loop.

LoopUS (Park et al., arXiv 2605.11011) trains a pretrained decoder to loop its
middle layers. A released checkpoint stores the decoder's layers under three
block names, in the order they have in the parent:

    encoder.embed_tokens, encoder.layers.*     the first layers, run once
    reasoning.layers.*, reasoning.gate.*       the looped layers and their gate
    decoder.layers.*, decoder.norm, decoder.lm_head
    q_head.*                                   the confidence head

The authors' model class is not in the checkpoint repository and is not used.
This module puts the layers back into the installed library's own model class,
with the output head separate from the embedding as the checkpoint has it, and
attaches the gate and head as `model.gated_loop` for `gated.py` to run. Every
tensor in the file must be used and every parameter of the model filled.

The configuration was written by a later library release than the pinned one, so
the fields the pinned release reads are picked out by name. The same goes for the
tokenizer, which is built from the checkpoint's own `tokenizer.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT = "lds"
# Fields of the checkpoint's base configuration that the pinned library takes under the same name.
ARCHITECTURE = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "initializer_range", "rms_norm_eps",
    "attention_bias", "attention_dropout", "bos_token_id", "eos_token_id", "pad_token_id")
GATE = {"reasoning.gate.dt_input_proj.weight": "gate.narrow.weight", "reasoning.gate.delta_proj.weight": "gate.widen.weight",
    "reasoning.gate.delta_proj.bias": "gate.widen.bias", "reasoning.gate.A_log": "gate.log_rate"}
HEAD = {"q_head.0.weight": "head.0.weight", "q_head.0.bias": "head.0.bias",
    "q_head.1.weight": "head.1.weight", "q_head.1.bias": "head.1.bias"}


def read_config(snapshot: Path) -> dict:
    """The checkpoint's configuration, checked to be one this module can run."""
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    base = config.get("base_config_dict") or {}
    if config.get("model_type") != FORMAT or base.get("model_type") != "qwen3":
        raise ValueError("Not a LoopUS checkpoint of a Qwen3 model")
    layers = base["num_hidden_layers"]
    first, middle, last = (config[f"{name}_layer_indices"] for name in ("encoder", "reasoning", "decoder"))
    if first + middle + last != list(range(layers)) or not middle:
        raise ValueError("The three blocks are not the parent's layers in order")
    if config.get("halting_strategy") != "threshold" or config.get("q_eval_interval") != 1:
        raise ValueError("Only the threshold stopping rule, read on every pass, is supported")
    if base.get("use_sliding_window") or base.get("sliding_window"):
        raise ValueError("Sliding-window checkpoints are not supported")
    rope = base.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or "rope_theta" not in rope:
        raise ValueError("Unrecognised rotary-position settings")
    return config


def rename(key: str, config: dict) -> tuple[str, str]:
    """Where a checkpoint tensor belongs: ("model", name in the stock model) or ("loop", name in the loop)."""
    if key in GATE:
        return "loop", GATE[key]
    if key in HEAD:
        return "loop", HEAD[key]
    fixed = {"encoder.embed_tokens.weight": "model.embed_tokens.weight", "decoder.norm.weight": "model.norm.weight",
        "decoder.lm_head.weight": "lm_head.weight"}
    if key in fixed:
        return "model", fixed[key]
    block, kind, position, rest = (key.split(".", 3) + [""] * 4)[:4]
    indices = config.get(f"{block}_layer_indices") if block in ("encoder", "reasoning", "decoder") else None
    if indices is None or kind != "layers" or not position.isdigit() or int(position) >= len(indices) or not rest:
        raise ValueError(f"Unrecognised tensor in the checkpoint: {key}")
    return "model", f"model.layers.{indices[int(position)]}.{rest}"


def assemble(snapshot: Path, dtype, device: str):
    """The stock model carrying the checkpoint's weights, with the trained loop attached."""
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from recurrent_retrofit.gated import Loop
    config = read_config(snapshot)
    base = config["base_config_dict"]
    settings = {name: base[name] for name in ARCHITECTURE if name in base}
    stock = Qwen3Config(**settings, rope_theta=float(base["rope_parameters"]["rope_theta"]), rope_scaling=None,
        use_sliding_window=False, sliding_window=None, max_window_layers=base["num_hidden_layers"],
        tie_word_embeddings=False, use_cache=True, attn_implementation="sdpa")
    tensors = load_file(str(snapshot / "model.safetensors"), device="cpu")
    parts: dict[str, dict] = {"model": {}, "loop": {}}
    for key, tensor in tensors.items():
        part, name = rename(key, config)
        if name in parts[part]:
            raise ValueError(f"Two checkpoint tensors map to {name}")
        parts[part][name] = tensor
    with init_empty_weights():
        model = Qwen3ForCausalLM(stock)
    # Strict: a missing or unexpected name, or a wrong shape, stops the load.
    model.load_state_dict(parts["model"], strict=True, assign=True)
    hidden = stock.hidden_size
    unit = (config["reasoning_layer_indices"][0], config["reasoning_layer_indices"][-1] + 1)
    loop = Loop(hidden, parts["loop"]["gate.narrow.weight"].shape[0], unit, head=True,
        trained_passes=config["N"], threshold=config["q_threshold"])
    loop.load_state_dict(parts["loop"], strict=True)
    model.gated_loop = loop
    model = model.to(device=device, dtype=dtype).eval()
    facts = {"format": FORMAT, "unit": list(unit), "trained_passes": config["N"],
        "checkpoint_threshold": config["q_threshold"], "tensors": len(tensors),
        "parent_named_in_config": config.get("base_model_name_or_path"),
        "written_by_transformers": config.get("transformers_version")}
    return model, facts


def tokenizer_of(snapshot: Path):
    """The checkpoint's own tokenizer file and chat template, without the settings file the pinned
    library cannot parse."""
    from transformers import PreTrainedTokenizerFast
    settings = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snapshot / "tokenizer.json"),
        eos_token=settings["eos_token"], pad_token=settings["pad_token"],
        additional_special_tokens=list(settings.get("extra_special_tokens") or []),
        clean_up_tokenization_spaces=False, model_max_length=settings.get("model_max_length"))
    template = snapshot / "chat_template.jinja"
    if template.exists():
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    return tokenizer
