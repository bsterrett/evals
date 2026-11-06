from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid

from eval_core.local.protocol import LocalRequest, PROMPT_FORMATS, encode_prompt, identity, profiles, verify_observations


class RecurrentBackend:
    """Batch-one reference decoding; all KV and recurrent state is request-local.

    Depth is fixed before construction. A hook on the first layer in the recurrent
    stack counts actual iterations in every forward, including prompt prefill.
    """

    def __init__(self, profile: str, depth: int, root: Path, *, precision: str = "bfloat16",
            placement: str = "gpu", max_total_tokens: int | None = None,
            latent_policy: str = "native") -> None:
        self.profile = profile
        self.settings = profiles()[profile]
        self.family = self.settings.get("family", profile)
        if self.family not in ("huginn", "ouro"):
            raise ValueError("This profile needs its dedicated backend")
        if precision not in ("bfloat16", "float32") or placement != "gpu":
            raise ValueError("Unsupported precision or placement")
        if precision == "float32" and self.family != "huginn":
            raise ValueError("The FP32 control is currently specific to Huginn")
        if latent_policy not in ("native", "float32-init") or (latent_policy != "native" and self.family != "huginn"):
            raise ValueError("Unsupported latent initialization control")
        if depth not in self.settings["depths"]:
            raise ValueError("Depth is not supported for this profile")
        self.depth = depth
        self.model_id = identity(profile)
        self.max_total = max_total_tokens or self.settings["max_total_tokens"]
        root = root.resolve()
        os.environ["HF_HOME"] = str(root / "cache/huggingface")
        os.environ["HF_MODULES_CACHE"] = str(root / "cache/huggingface/modules")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        self.torch = torch
        dtype = getattr(torch, precision)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        free, total = torch.cuda.mem_get_info()
        needed = (11 if self.family == "huginn" else 7 if profile == "ouro-2.6b-thinking" else 5) * 1024**3
        if free < needed:
            raise RuntimeError(f"Insufficient free GPU memory: {free / 1024**3:.2f} GiB; need {needed / 1024**3:.0f} GiB")
        snapshot = Path(snapshot_download(self.settings["repo_id"], revision=self.settings["revision"],
            cache_dir=str(root / "cache/huggingface/hub"), local_files_only=True))
        actual_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.py")}
        if actual_hashes != self.settings["source_sha256"]:
            raise RuntimeError("Custom source differs from the reviewed hashes")
        if importlib.metadata.version("transformers") != self.settings["transformers"]:
            raise RuntimeError("Transformers version does not match the profile")
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
        config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
        if self.family == "ouro":
            config.total_ut_steps = depth
            config.early_exit_threshold = None
            config.early_exit_step = depth - 1
        # CPU-first loading avoids transient duplicate CUDA allocations when the
        # FP32 model is near the board's capacity. It does not change weight values.
        load_device = "cpu" if precision == "float32" else "cuda:0"
        device_map = {"": load_device}
        self.model = AutoModelForCausalLM.from_pretrained(str(snapshot), config=config,
            torch_dtype=dtype, device_map=device_map,
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=True).eval()
        if precision == "float32" and placement == "gpu":
            weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            if weight_bytes + 512 * 1024**2 > torch.cuda.mem_get_info()[0]:
                raise RuntimeError(f"FP32 parameters need {weight_bytes / 1024**3:.2f} GiB plus cache/workspace; insufficient free GPU memory")
            self.model.to("cuda:0")
        if placement == "gpu" and any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("Model was not placed entirely on the GPU")
        if any(p.is_floating_point() and p.dtype != dtype for p in self.model.parameters()):
            raise RuntimeError("Model parameter precision does not match the requested control")
        if latent_policy == "float32-init":
            original_initializer = self.model.initialize_state
            def initialize_float32(input_embeds, scale=1.0):
                return original_initializer(input_embeds.float(), scale=scale).to(input_embeds.dtype)
            self.model.initialize_state = initialize_float32
        self.executions = 0
        first_layer = self.model.transformer.core_block[0] if self.family == "huginn" else self.model.model.layers[0]
        self.hook = first_layer.register_forward_pre_hook(self._count_iteration)
        eos = self.tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        end_turn = "<|end_turn|>" if self.family == "huginn" else "<|im_end|>"
        token = self.tokenizer.convert_tokens_to_ids(end_turn)
        if token is not None and token != self.tokenizer.unk_token_id:
            self.eos_ids.add(token)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.capabilities = {
            "protocol_version": 1, "profile": profile, "model": self.model_id, "depth": depth,
            "max_total_tokens": self.max_total, "precision": precision, "placement": placement, "attention": "sdpa",
            "parameter_bytes_by_device": {device: sum(p.numel() * p.element_size() for p in self.model.parameters() if str(p.device) == device)
                for device in sorted({str(p.device) for p in self.model.parameters()})},
            "decoding": "greedy", "adaptive_exit": False, "cross_request_state": False,
            "prompt_formats": list(PROMPT_FORMATS), "eos_token_ids": sorted(self.eos_ids),
            "source_sha256": actual_hashes, "tokenizer_revision": self.settings["revision"],
            "load_seconds": self.load_seconds, "gpu": torch.cuda.get_device_name(),
            "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "accelerate", "safetensors", "huggingface-hub"]},
            "dependency_versions": dict(sorted((dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions() if dist.metadata["Name"])),
            "torch_cuda": torch.version.cuda, "gpu_total_bytes": total,
            "adapter_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("backend.py", "protocol.py", "profiles.json", "worker.py")},
            "deterministic_algorithms": True,
            "latent_initialization": getattr(config, "state_init", "token_embeddings"),
            "latent_policy": latent_policy,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        }

    def _count_iteration(self, module, args) -> None:
        self.executions += 1

    def tokenize(self, request: LocalRequest) -> list[int]:
        ids = encode_prompt(self.tokenizer, request)
        if not ids or len(ids) + request.max_output_tokens > self.max_total:
            raise ValueError(f"Prompt plus output limit exceeds {self.max_total} total tokens (no truncation)")
        return ids

    def generate(self, request: LocalRequest) -> dict:
        torch = self.torch
        ids = self.tokenize(request)
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        torch.use_deterministic_algorithms(True)
        output_ids: list[int] = []
        observed: list[int] = []
        times: list[float] = []
        cache = None
        stop_reason = "max_output_tokens"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            current = torch.tensor([ids], device="cuda", dtype=torch.long)
            position = torch.arange(len(ids), device="cuda")
            for _ in range(request.max_output_tokens):
                before = self.executions
                step_start = time.perf_counter()
                kwargs = dict(input_ids=current, past_key_values=cache, use_cache=True, cache_position=position)
                if self.family == "huginn":
                    kwargs["num_steps"] = self.depth
                else:
                    # Explicit final-loop readout: no learned/threshold/weighted exit selection.
                    kwargs.update(exit_at_step=self.depth - 1, logits_to_keep=1, use_weighted_exit=False)
                result = self.model(**kwargs)
                count = self.executions - before
                if count != self.depth:
                    raise RuntimeError(f"Depth instrument saw {count} iterations, expected {self.depth}")
                observed.append(count)
                next_id = int(result.logits[0, -1].argmax().item())
                cache = result.past_key_values
                if cache is None:
                    raise RuntimeError("Model failed to return its request-local cache")
                output_ids.append(next_id)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - step_start)
                del result
                if next_id in self.eos_ids:
                    stop_reason = "eos"
                    break
                current = torch.tensor([[next_id]], device="cuda", dtype=torch.long)
                position = position[-1:] + 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        verify_observations(observed, self.depth, len(output_ids))
        response = {
            "protocol_version": 1, "response_id": uuid.uuid4().hex, "model": self.model_id,
            "profile": self.profile, "depth": self.depth, "seed": request.seed,
            "prompt_format": request.prompt_format, "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
            "raw_text": self.tokenizer.decode(output_ids, skip_special_tokens=False),
            "input_tokens": len(ids), "output_tokens": len(output_ids), "output_token_ids": output_ids,
            "input_token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "stop_reason": stop_reason, "observed_iterations": observed,
            "forward_passes": len(observed), "prefill_iterations": observed[0], "decode_iterations": observed[1:],
            "elapsed_seconds": elapsed, "prefill_seconds": times[0],
            "decode_seconds": sum(times[1:]),
            "decode_tokens_per_second": (len(times) - 1) / sum(times[1:]) if len(times) > 1 else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "runtime": self.capabilities,
        }
        # Do not retain a cache, latent, logits, or input tensor on the backend instance.
        del cache, current, position
        return response
