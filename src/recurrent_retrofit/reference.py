"""References for checking the runner.

`StockCopy` is a physically deeper stock model. It gives every program position
its own ordinary layer module, built by the model's own layer class with a fixed
cache index, and loads the parent layer's tensors into it. That is the structure
of an exported deeper checkpoint (a passthrough merge), with none of the runner's
position wrappers. The tensors are shared with the parent, so the copy costs no
extra weight memory. The runner and this copy perform the same arithmetic in the
same order, so their outputs should be identical. It cannot express an update
rule, because a stock model has none.

`Explicit` covers the update rules. It calls the parent's layers by hand on a
whole sequence, with no cache, no installed program and none of the runner's
mixing code, and writes each rule's arithmetic out as plain loops.

`ExplicitRecirculation` does the same for recirculation, token by token.

`ExplicitGated` does the same for a gated loop. It reads the gate's and the
confidence head's tensors and writes their arithmetic out, without calling the
modules that the runner calls.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from recurrent_retrofit.program import Program, Update
from recurrent_retrofit.runner import owner_of


def _mixed(update: Update, passes: int, index: int, state, output, outputs: list):
    """The state after one cell. `outputs` holds this cell's outputs so far, ending with `output`."""
    if update.rule == "euler":
        mixed = state + (output - state) / passes
        if update.anchor is not None and index == passes - 1:
            mixed = update.anchor * outputs[0] + (1 - update.anchor) * mixed
        return mixed
    if index == 0:
        return output
    if update.rule == "uniform":
        return sum(outputs) / len(outputs)
    return update.eta * outputs[0] + (1 - update.eta) * output


class Explicit:
    """Hand-called layers of an unmodified Qwen3 parent, for one whole sequence at a time."""

    def __init__(self, model) -> None:
        self.model, self.core = model, owner_of(model)

    def prepare(self, ids: list[int]):
        device = next(self.model.parameters()).device
        state = self.core.embed_tokens(torch.tensor([ids], device=device, dtype=torch.long))
        self.positions = torch.arange(len(ids), device=device)[None]
        self.rotary = self.core.rotary_emb(state, self.positions)
        blocked = torch.full((len(ids), len(ids)), torch.finfo(state.dtype).min, dtype=state.dtype, device=device)
        self.mask = blocked.triu(1)[None, None]
        return state

    def layer(self, index: int, state):
        return self.core.layers[index](state, attention_mask=self.mask, position_ids=self.positions,
            position_embeddings=self.rotary)

    def readout(self, state):
        """Logits at the last position."""
        return self.model.lm_head(self.core.norm(state))[0, -1].float().cpu()

    def logits(self, ids: list[int], program: Program):
        update, passes = program.update, program.passes
        with torch.inference_mode():
            state = self.prepare(ids)
            if update.rule == "plain" or passes == 1:
                for index in program.steps:
                    state = self.layer(index, state)
                return self.readout(state)
            start, end = program.unit
            for index in range(start):
                state = self.layer(index, state)
            if program.kind == "layer":
                for index in range(start, end):
                    outputs = []
                    for turn in range(passes):
                        outputs.append(self.layer(index, state))
                        state = _mixed(update, passes, turn, state, outputs[-1], outputs)
            elif update.per == "layer":
                outputs = {index: [] for index in range(start, end)}
                for turn in range(passes):
                    for index in range(start, end):
                        outputs[index].append(self.layer(index, state))
                        state = _mixed(update, passes, turn, state, outputs[index][-1], outputs[index])
            else:
                outputs = []
                for turn in range(passes):
                    output = state
                    for index in range(start, end):
                        output = self.layer(index, output)
                    outputs.append(output)
                    state = _mixed(update, passes, turn, state, output, outputs)
            for index in range(end, program.parent_layers):
                state = self.layer(index, state)
            return self.readout(state)


class ExplicitRecirculation:
    """A recirculation computed with hand-called layers and no attention cache.

    It uses none of the runner's code. For every layer it keeps the inputs that
    layer was given at the tokens so far, in the version later tokens attend to:
    the ordinary pass's input up to the destination layer, and the second pass's
    above it. A new token is computed by calling each layer on those inputs with
    the new token's own appended, and keeping the last position. That equals
    attending to cached entries, because what a layer contributes to attention at
    a position depends only on its input there.
    """

    def __init__(self, model) -> None:
        self.model, self.core = model, owner_of(model)

    def logits(self, ids: list[int], design) -> list:
        """The logits read after each token of `ids`."""
        core, count = self.core, design.parent_layers
        device = next(self.model.parameters()).device
        seen: list = [None] * count
        results = []
        with torch.inference_mode():
            embedded = core.embed_tokens(torch.tensor([ids], device=device, dtype=torch.long))
            for token in range(len(ids)):
                positions = torch.arange(token + 1, device=device)[None]
                rotary = core.rotary_emb(embedded, positions)
                blocked = torch.full((token + 1, token + 1), torch.finfo(embedded.dtype).min,
                    dtype=embedded.dtype, device=device)
                mask = blocked.triu(1)[None, None]

                def call(index: int, state):
                    whole = state if seen[index] is None else torch.cat([seen[index], state], dim=1)
                    return core.layers[index](whole, attention_mask=mask, position_ids=positions,
                        position_embeddings=rotary)[:, -1:]

                state = embedded[:, token:token + 1]
                given = []
                for index in range(count):
                    given.append(state)
                    state = call(index, state)
                    if index == design.destination:
                        low = state
                    if index == design.source:
                        high = state
                results.append(self.model.lm_head(core.norm(state))[0, -1].float().cpu())
                if design.passes == 2:
                    scale = low.float().pow(2).sum(-1, keepdim=True).sqrt() / high.float().pow(2).sum(-1, keepdim=True).sqrt()
                    fed = design.alpha * scale * high.float()
                    kept = low.float() if design.mix == "additive" else (1 - design.alpha) * low.float()
                    state = (kept + fed).to(low.dtype)
                    for index in range(design.destination + 1, count):
                        given[index] = state
                        state = call(index, state)
                for index in range(count):
                    seen[index] = given[index] if seen[index] is None else torch.cat([seen[index], given[index]], dim=1)
        return results


class ExplicitGated:
    """A gated loop computed with hand-called layers on one whole sequence, with no cache.

    `logits` returns the logits at every position. `passes_used` then holds how
    many passes ran, which is fewer than the design's when it stopped early.
    """

    def __init__(self, model) -> None:
        self.model, self.explicit = model, Explicit(model)
        self.loop = model.gated_loop
        self.passes_used: int | None = None

    def keep(self, new, old):
        """The share of `new` the gate lets through at each position and channel, in the gate's precision."""
        gate = self.loop.gate
        kind = gate.log_rate.dtype
        narrowed = (new.to(kind) - old.to(kind)) @ gate.narrow.weight.T
        raised = narrowed @ gate.widen.weight.T + gate.widen.bias
        step = torch.logaddexp(raised, torch.zeros_like(raised))  # softplus
        return torch.exp(-(step * torch.exp(gate.log_rate)))

    def gate(self, new, old):
        keep = self.keep(new, old)
        return (keep * new.to(keep.dtype) + (1 - keep) * old.to(keep.dtype)).to(new.dtype)

    def confidence(self, change) -> float:
        """The head's probability for the change at the last token."""
        norm, linear = self.loop.head[0], self.loop.head[1]
        value = change[0, -1].to(linear.weight.dtype)
        centred = value - value.mean()
        scaled = centred / torch.sqrt(centred.pow(2).mean() + norm.eps) * norm.weight + norm.bias
        score = (scaled * linear.weight[0]).sum() + linear.bias[0]
        return float(1 / (1 + torch.exp(-score.float())))

    def logits(self, ids: list[int], design):
        explicit = self.explicit
        with torch.inference_mode():
            state = explicit.prepare(ids)
            for index in range(design.start):
                state = explicit.layer(index, state)
            self.passes_used = 0
            for _ in range(design.passes):
                output = state
                for index in range(design.start, design.end):
                    output = explicit.layer(index, output)
                self.passes_used += 1
                stop = design.threshold is not None and self.confidence(output - state) > design.threshold
                state = self.gate(output, state)
                if stop:
                    break
            for index in range(design.end, design.parent_layers):
                state = explicit.layer(index, state)
            return self.model.lm_head(explicit.core.norm(state))[0].float().cpu()


class StockCopy:
    def __init__(self, model, program: Program) -> None:
        owner = owner_of(model)
        if program.parent_layers != len(owner.layers):
            raise ValueError("The program does not match the model's layer count")
        self.model, self._owner, self.parents = model, owner, owner.layers
        config = model.config
        self._saved = {"num_hidden_layers": config.num_hidden_layers}
        deep = copy.deepcopy(config)
        deep.num_hidden_layers = program.layer_calls
        types = getattr(config, "layer_types", None)
        if types is not None:
            self._saved["layer_types"] = types
            deep.layer_types = [types[step] for step in program.steps]
        layers = []
        for slot, step in enumerate(program.steps):
            source = self.parents[step]
            layer = type(source)(deep, slot)
            layer.load_state_dict(source.state_dict(), assign=True)
            layers.append(layer.train(source.training))
        owner.layers = nn.ModuleList(layers)
        config.num_hidden_layers = deep.num_hidden_layers
        if types is not None:
            config.layer_types = deep.layer_types

    def remove(self) -> None:
        self._owner.layers = self.parents
        for name, value in self._saved.items():
            setattr(self.model.config, name, value)

    def __enter__(self) -> StockCopy:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
