"""Load a pinned checkpoint for retrofitting, offline and without repository code.

Two families load here. `qwen3` is a plain parent, loaded by the library. `loopus`
is a released LoopUS checkpoint: a Qwen3 decoder trained to loop, which
`loopus.py` rebuilds as the library's own model class with its trained gate
attached. It runs only with a `gated[...]` design installed.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import time

from eval_core.local.protocol import identity, profiles

FAMILIES = ("qwen3", "loopus")
HEADROOM_BYTES = 2 * 1024**3


def needs_gate(profile: str) -> bool:
    """True for a checkpoint that was trained to loop and runs only as a gated loop."""
    return profiles()[profile].get("family") == "loopus"


def load_parent(profile: str, root: Path, *, precision: str = "bfloat16", placement: str = "gpu") -> tuple:
    """Return `(model, tokenizer, info)` for a parent profile, on one device in one precision.

    The default, BF16 on the GPU, is the serving condition. FP32 and the CPU exist
    for numerical controls: BF16 rounding is amplified by some loops, so a tight
    comparison needs FP32, and a parent whose FP32 weights exceed the GPU can run
    that control on the CPU.
    """
    settings = profiles()[profile]
    if settings.get("family") not in FAMILIES:
        raise ValueError(f"Profile {profile!r} is not a retrofit parent")
    if precision not in ("bfloat16", "float32") or placement not in ("gpu", "cpu"):
        raise ValueError("Unsupported precision or placement")
    root = root.resolve()
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, precision)
    device = "cuda:0" if placement == "gpu" else "cpu"
    if importlib.metadata.version("transformers") != settings["transformers"]:
        raise RuntimeError("Transformers version does not match the profile")
    snapshot = Path(snapshot_download(settings["repo_id"], revision=settings["revision"],
        cache_dir=str(root / "cache/huggingface/hub"), local_files_only=True))
    if settings["source_sha256"] or list(snapshot.glob("*.py")):
        raise RuntimeError("Retrofit parents must load without repository code")
    total = None
    if placement == "gpu":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        # The released weights are BF16, so FP32 needs twice the file size.
        weight_bytes = sum(path.stat().st_size for path in snapshot.glob("*.safetensors")) * (2 if precision == "float32" else 1)
        free, total = torch.cuda.mem_get_info()
        if free < weight_bytes + HEADROOM_BYTES:
            raise RuntimeError(f"Insufficient free GPU memory: {free / 1024**3:.2f} GiB; "
                f"need {(weight_bytes + HEADROOM_BYTES) / 1024**3:.2f} GiB")
    start = time.perf_counter()
    gated = None
    if settings["family"] == "loopus":
        from recurrent_retrofit import loopus
        tokenizer = loopus.tokenizer_of(snapshot)
        model, gated = loopus.assemble(snapshot, dtype, device)
        if model.config._attn_implementation != "sdpa":
            raise RuntimeError("The rebuilt model does not use the expected attention implementation")
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=dtype,
            device_map={"": device}, attn_implementation="sdpa", local_files_only=True).eval()
    if not type(model).__module__.startswith("transformers.models."):
        raise RuntimeError("The model class did not come from the installed Transformers package")
    if any(str(p.device) != device for p in model.parameters()):
        raise RuntimeError(f"Model was not placed entirely on {device}")
    if any(p.is_floating_point() and p.dtype != dtype for p in model.parameters()):
        raise RuntimeError("Model parameter precision does not match the request")
    if placement == "gpu":
        torch.cuda.synchronize()
    eos = set()
    for value in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        eos.update([value] if isinstance(value, int) else value or [])
    end_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_turn is not None and end_turn != tokenizer.unk_token_id:
        eos.add(end_turn)
    if gated is not None and tokenizer.pad_token_id is not None:
        # The checkpoint was trained on plain text; its end-of-text token is its padding token.
        eos.add(tokenizer.pad_token_id)
    info = {
        "profile": profile, "parent": identity(profile), "model_class": type(model).__name__,
        "parent_layers": model.config.num_hidden_layers, "precision": precision, "placement": placement,
        "attention": "sdpa", "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "eos_token_ids": sorted(eos), "load_seconds": time.perf_counter() - start,
        "gpu": torch.cuda.get_device_name() if placement == "gpu" else None, "gpu_total_bytes": total,
        "torch_cuda": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "accelerate", "safetensors", "huggingface-hub"]},
    }
    if gated is not None:
        info["gated"] = gated
    return model, tokenizer, info
