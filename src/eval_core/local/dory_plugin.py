"""Register the reviewed NVIDIA Dory implementation in the isolated vLLM runtime."""


def register() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from eval_core.local.vendor_dory.configuration_dory import DoryConfig
    AutoConfig.register("dory", DoryConfig, exist_ok=True)
    ModelRegistry.register_model("DoryForCausalLM",
        "eval_core.local.vendor_dory.modeling_dory:DoryForCausalLM")
