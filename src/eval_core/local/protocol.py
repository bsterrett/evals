from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from importlib.resources import files
import json
from typing import Any

PROTOCOL_VERSION = 1
PROMPT_FORMATS = ("raw-v1", "native-chat-v1", "native-chat-system-v1",
    "native-chat-thinking-v1", "native-chat-no-thinking-v1")


def encode_prompt(tokenizer, request):
    if request.prompt_format == "raw-v1":
        return tokenizer.encode(request.prompt, add_special_tokens=True)
    messages = [{"role": "user", "content": request.prompt}]
    if request.prompt_format != "native-chat-v1":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    kwargs = {}
    if request.prompt_format in ("native-chat-thinking-v1", "native-chat-no-thinking-v1"):
        kwargs["enable_thinking"] = request.prompt_format == "native-chat-thinking-v1"
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
    # Transformers 4 returns a list here; newer releases can return BatchEncoding.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Local inference supports batch size one")
        encoded = encoded[0]
    return list(encoded)


def profiles() -> dict[str, dict[str, Any]]:
    return json.loads(files("eval_core.local").joinpath("profiles.json").read_text(encoding="utf-8"))


def identity(profile: str) -> str:
    data = profiles()[profile]
    return f"{data['repo_id']}@{data['revision']}"


def integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


@dataclass(frozen=True)
class LocalRequest:
    model: str
    depth: int
    prompt: str
    max_output_tokens: int
    seed: int
    prompt_format: str

    @classmethod
    def parse(cls, value: dict[str, Any], *, model: str, depth: int, max_total: int) -> LocalRequest:
        expected = {"model", "depth", "prompt", "max_output_tokens", "seed", "prompt_format"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(f"Request must contain exactly: {', '.join(sorted(expected))}")
        if value["model"] != model or type(value["depth"]) is not int or value["depth"] != depth:
            raise ValueError("Requested model/revision/depth does not match the loaded worker")
        if not isinstance(value["prompt"], str) or not value["prompt"].strip():
            raise ValueError("prompt must be a nonempty string")
        if value["prompt_format"] not in PROMPT_FORMATS:
            raise ValueError("Unsupported prompt_format")
        return cls(model, depth, value["prompt"],
            integer(value["max_output_tokens"], "max_output_tokens", 1, max_total - 1),
            integer(value["seed"], "seed", 0, 2**32 - 1), value["prompt_format"])


def verify_observations(observed: list[int], depth: int, output_tokens: int) -> None:
    # One full prefill predicts the first output token; subsequent forwards decode one token.
    if len(observed) != output_tokens or any(type(n) is not int or n != depth for n in observed):
        raise ValueError("Observed recurrent execution differs from the requested depth")
