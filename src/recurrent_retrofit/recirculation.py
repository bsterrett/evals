"""Install a recirculation on a loaded Hugging Face decoder.

A recirculation feeds a deep layer's state back into a shallow layer across token
steps, so a token cannot be processed in parallel with the tokens before it. The
decoder's own forward pass is kept and driven one token at a time, with the stock
attention cache, whether the caller hands over one token or a whole prompt. After
each token's ordinary pass:

1. the outputs of the destination and source layers at that token are mixed
   (`mixture`);
2. the cache entries that the layers above the destination wrote for the token
   are dropped;
3. those layers run again at that token from the mixed state, with the arguments
   the ordinary pass gave them, and write the entries later tokens will attend to.

The token's hidden state, and so its prediction, comes from the ordinary pass.
Nothing is carried between calls except the cache, and no weight changes.

With `passes=1` steps 1 to 3 are skipped, which leaves the parent processed one
token at a time. That is the baseline whose arithmetic matches a recirculation's.
"""
from __future__ import annotations

import torch

from recurrent_retrofit.program import Recirculation
from recurrent_retrofit.runner import Counted, owner_of


def mixture(low: torch.Tensor, high: torch.Tensor, alpha: float, mix: str) -> torch.Tensor:
    """The state written at the destination: its own output `low`, plus the source's
    output `high` rescaled to the norm of `low` at each position. Computed in FP32."""
    own, deep = low.float(), high.float()
    fed = deep * (own.norm(dim=-1, keepdim=True) / deep.norm(dim=-1, keepdim=True))
    keep = 1.0 if mix == "additive" else 1.0 - alpha
    return (keep * own + alpha * fed).to(low.dtype)


class Recirculator(Counted):
    """A recirculation installed on a model. Call `remove()` to restore the parent exactly."""

    def __init__(self, model, design: Recirculation) -> None:
        if not isinstance(design, Recirculation):
            raise TypeError("Recirculator installs recirculations; use install() for any design")
        if getattr(model, "gated_loop", None) is not None:
            raise ValueError("This checkpoint was trained with a gate; it runs as a gated loop only")
        owner = owner_of(model)
        if design.parent_layers != len(owner.layers):
            raise ValueError(f"The design is for {design.parent_layers} layers; the model has {len(owner.layers)}")
        if getattr(owner, "has_sliding_layers", False):
            raise ValueError("Models with sliding-window layers are not supported")
        self.program = design
        self._watch(model, owner)
        self._low = self._high = self._arguments = None
        self._second_pass = False
        if design.passes == 2:
            layers = self.parents
            self._hooks += [
                layers[design.destination].register_forward_hook(self._keep("_low")),
                layers[design.source].register_forward_hook(self._keep("_high")),
                layers[design.destination + 1].register_forward_pre_hook(self._keep_arguments, with_kwargs=True),
            ]
        self._stock = owner.forward
        owner.forward = self._forward
        self.installed = True

    def _keep(self, name: str):
        def keep(module, args, output) -> None:
            if not self._second_pass:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError("Recirculation needs a layer that returns its hidden state")
                setattr(self, name, output)
        return keep

    def _keep_arguments(self, module, args, kwargs) -> None:
        if not self._second_pass:
            self._arguments = dict(kwargs)

    def _forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
            inputs_embeds=None, use_cache=None, cache_position=None, **kwargs):
        from transformers import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if isinstance(attention_mask, dict):
            raise NotImplementedError("A prepared attention mask cannot be split into token steps")
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            raise NotImplementedError("Hidden states and attentions are not collected across token steps")
        cache = DynamicCache(config=self.model.config) if past_key_values is None else past_key_values
        if not isinstance(cache, DynamicCache):
            raise NotImplementedError("Recirculation needs the dynamic attention cache")
        width = (inputs_embeds if input_ids is None else input_ids).shape[1]
        states = []
        for offset in range(width):
            column = slice(offset, offset + 1)
            after = width - offset - 1
            output = self._stock(
                input_ids=None if input_ids is None else input_ids[:, column],
                attention_mask=None if attention_mask is None else attention_mask[:, :attention_mask.shape[1] - after],
                position_ids=None if position_ids is None else position_ids[:, column],
                past_key_values=cache,
                inputs_embeds=None if inputs_embeds is None else inputs_embeds[:, column],
                use_cache=True,
                cache_position=None if cache_position is None else cache_position[column],
                **kwargs)
            states.append(output.last_hidden_state)
            if self.program.passes == 2:
                self._recirculate(cache)
        return BaseModelOutputWithPast(last_hidden_state=torch.cat(states, dim=1),
            past_key_values=cache if use_cache else None)

    def _recirculate(self, cache) -> None:
        design = self.program
        low, high, arguments = self._low, self._high, self._arguments
        self._low = self._high = self._arguments = None
        if low is None or high is None or arguments is None:
            raise RuntimeError("The token step did not reach the source and destination layers")
        state = mixture(low, high, design.alpha, design.mix)
        self._second_pass = True
        try:
            for index in range(design.destination + 1, design.parent_layers):
                cache.layers[index].crop(-1)
                state = self.parents[index](state, **arguments)
        finally:
            self._second_pass = False

    def runs_for(self, tokens: int) -> int:
        return tokens

    def check_cache(self, cache, tokens: int) -> None:
        """Every parent layer keeps one slot, holding one entry per token."""
        if len(cache.layers) != self.program.parent_layers:
            raise RuntimeError(f"Cache has {len(cache.layers)} slots for {self.program.parent_layers} layers")
        lengths = [layer.get_seq_length() for layer in cache.layers]
        if any(length != tokens for length in lengths):
            raise RuntimeError(f"Cache slot lengths {lengths} differ from {tokens} tokens")

    def remove(self) -> None:
        if not self.installed:
            return
        for hook in self._hooks:
            hook.remove()
        del self._owner.forward
        self.installed = False
