from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import time
import uuid

from eval_core.local.protocol import LocalRequest, PROMPT_FORMATS, encode_prompt, identity, profiles, verify_observations

VENDOR_HASHES = {
    "configuration_dory.py": "cf0d79e7cbdbf44d34e385f7aa66ff4cfc531dad280d001b8b850e15143817be",
    "modeling_dory.py": "3ae99f3c4b4309f5af26983354fd2672f9f668250a7eb7450e48da3f1771c6d7",
}


def install_instrumentation(model):
    """Executed inside vLLM's single GPU worker, with compilation disabled."""
    model._eval_iteration_count = 0
    model._eval_observations = []
    def iteration(module, args):
        model._eval_iteration_count += 1
    def before(module, args):
        model._eval_before = model._eval_iteration_count
    def after(module, args, result):
        model._eval_observations.append(model._eval_iteration_count - model._eval_before)
    first = model.backbone.layers[model.config.n_input_layers]
    first.register_forward_pre_hook(iteration)
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    return dict(parameters_by_dtype={str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
        for dtype in {p.dtype for p in model.parameters()}},
        loaded_depth=model.backbone.num_loops)


def reset_instrumentation(model):
    import torch
    model._eval_observations = []
    torch.cuda.reset_peak_memory_stats()


def read_instrumentation(model):
    import torch
    torch.cuda.synchronize()
    return dict(observed_iterations=list(model._eval_observations),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


class DoryWorkerExtension:
    """Named RPC methods; no executable callback is sent over the IPC channel."""

    def eval_install(self):
        return install_instrumentation(self.model_runner.get_model())

    def eval_reset(self):
        return reset_instrumentation(self.model_runner.get_model())

    def eval_read(self):
        return read_instrumentation(self.model_runner.get_model())


class DoryBackend:
    """Official Dory implementation registered as a vLLM external model.

    Fixed-depth, BF16, per-loop caches, eager execution, one request at a time.
    This backend requires the source checkout installed in the isolated dory env.
    """

    def __init__(self, depth: int, root: Path) -> None:
        self.settings = profiles()["dory"]
        if depth not in self.settings["depths"]:
            raise ValueError("Unsupported Dory depth")
        self.profile = "dory"
        self.depth = depth
        self.model_id = identity("dory")
        self.max_total = self.settings["max_total_tokens"]
        root = root.resolve()
        os.environ["HF_HOME"] = str(root / "cache/huggingface")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_PLUGINS"] = "eval_recurrent_dory"
        os.environ["VLLM_CACHE_ROOT"] = str(root / "cache/vllm")
        os.environ["PATH"] = str(root / "envs/dory/bin") + os.pathsep + os.environ["PATH"]
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "cache/inductor")
        os.environ["TRITON_CACHE_DIR"] = str(root / "cache/triton")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        if importlib.metadata.version("vllm") != self.settings["vllm"]:
            raise RuntimeError("Dory vLLM version differs from the pinned runtime")
        vendor = Path(__file__).with_name("vendor_dory")
        actual = {n: hashlib.sha256((vendor / n).read_bytes()).hexdigest() for n in VENDOR_HASHES}
        if actual != VENDOR_HASHES:
            raise RuntimeError("Dory vendor source differs from reviewed hashes")
        import torch
        from huggingface_hub import snapshot_download
        from vllm import LLM
        # Register before config inspection as well as in spawned model workers.
        from eval_core.local.dory_plugin import register
        register()
        if torch.cuda.mem_get_info()[0] < 11 * 1024**3:
            raise RuntimeError("Dory screen needs at least 11 GiB free GPU memory")
        snapshot = snapshot_download(self.settings["repo_id"], revision=self.settings["revision"],
            cache_dir=str(root / "cache/huggingface/hub"), local_files_only=True)
        started = time.perf_counter()
        self.engine = LLM(model=snapshot, tokenizer=snapshot, dtype="bfloat16",
            trust_remote_code=False, tensor_parallel_size=1, seed=42,
            worker_extension_cls="eval_core.local.dory_backend.DoryWorkerExtension",
            async_scheduling=False,
            hf_overrides=dict(n_recurrent_loops=depth, recurrent_kv_cache_mode="per_loop"),
            enforce_eager=True, compilation_config={"mode": 0},
            attention_config={"backend": "TRITON_ATTN"},
            gpu_memory_utilization=0.78, max_model_len=self.max_total,
            max_num_seqs=1, max_num_batched_tokens=self.max_total,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            disable_log_stats=True, generation_config="vllm")
        self.tokenizer = self.engine.get_tokenizer()
        check = self.engine.collective_rpc("eval_install")
        if len(check) != 1 or check[0]["loaded_depth"] != depth:
            raise RuntimeError("Dory worker did not load the requested fixed depth")
        self.capabilities = dict(protocol_version=1, profile="dory", model=self.model_id,
            depth=depth, max_total_tokens=self.max_total, precision="bfloat16", placement="gpu",
            decoding="greedy", cache_mode="per_loop", cross_request_state=False,
            prefix_caching=False, compilation=False, cuda_graphs=False,
            async_scheduling=False,
            attention="TRITON_ATTN",
            prompt_formats=list(PROMPT_FORMATS), load_seconds=time.perf_counter()-started,
            gpu=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda,
            versions={p: importlib.metadata.version(p) for p in ["vllm", "torch", "transformers", "huggingface-hub"]},
            tokenizer_revision=self.settings["revision"], vendor_source_sha256=actual,
            adapter_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                for n in ["dory_backend.py", "dory_plugin.py", "protocol.py", "profiles.json"]},
            parameter_verification=check[0])

    def tokenize(self, request: LocalRequest) -> list[int]:
        ids = encode_prompt(self.tokenizer, request)
        if not ids or len(ids) + request.max_output_tokens > self.max_total:
            raise ValueError(f"Prompt plus output exceeds {self.max_total} tokens; no truncation")
        return ids

    def generate(self, request: LocalRequest) -> dict:
        from vllm import SamplingParams
        ids = self.tokenize(request)
        self.engine.collective_rpc("eval_reset")
        started = time.perf_counter()
        outputs = self.engine.generate([dict(prompt_token_ids=ids)],
            SamplingParams(temperature=0, max_tokens=request.max_output_tokens,
                seed=request.seed, skip_special_tokens=False), use_tqdm=False)
        elapsed = time.perf_counter() - started
        stats = self.engine.collective_rpc("eval_read")[0]
        observed = stats["observed_iterations"]
        output = outputs[0].outputs[0]
        token_ids = list(output.token_ids)
        verify_observations(observed, self.depth, len(token_ids))
        if output.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected Dory termination: {output.finish_reason}")
        return dict(protocol_version=1, response_id=uuid.uuid4().hex,
            profile="dory", model=self.model_id, depth=self.depth, seed=request.seed,
            prompt_format=request.prompt_format,
            text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
            raw_text=self.tokenizer.decode(token_ids, skip_special_tokens=False),
            input_tokens=len(ids), output_tokens=len(token_ids), output_token_ids=token_ids,
            stop_reason="max_output_tokens" if output.finish_reason == "length" else "eos",
            engine_finish_reason=output.finish_reason, engine_stop_reason=output.stop_reason,
            forward_passes=len(observed), prefill_iterations=observed[0], decode_iterations=observed[1:],
            elapsed_seconds=elapsed, observed_iterations=observed,
            peak_allocated_bytes=stats["peak_allocated_bytes"], peak_reserved_bytes=stats["peak_reserved_bytes"],
            runtime=self.capabilities)
