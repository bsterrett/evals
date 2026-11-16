"""Install a layer program on a loaded Hugging Face decoder.

The parent's own forward pass is kept. Its layer list is replaced by one
`Position` per program step. Each position runs a parent layer's weights, which
stay shared, and points that layer's attention at the position's own cache slot
for the duration of the call. The model's `num_hidden_layers` and `layer_types`
are set to the program's length so that the stock cache allocates one slot per
position. Nothing is copied and no weight changes.

Supported families are listed in `FAMILIES`. A family is supported when its
forward pass iterates `layers`, reads nothing else from a layer except
`attention_type`, and its attention picks the cache slot from `layer_idx`.
"""
from __future__ import annotations

import torch
from torch import nn

from recurrent_retrofit.program import Cell, GatedLoop, Program, Recirculation, Update

# model_type -> the module that owns the `layers` list.
FAMILIES = {"qwen3": lambda model: model.model}


class Mixer:
    """Applies a program's update rule at the end of each cell.

    The states it keeps belong to one forward pass and are dropped when the next
    one starts. Mixing is position-wise along the sequence, so a cached decoding
    step, which sees only the new token, computes what a full recomputation would.
    """

    def __init__(self, update: Update, passes: int) -> None:
        self.update, self.passes = update, passes
        self.inputs: dict[int, torch.Tensor] = {}
        self.outputs: dict[int, list[torch.Tensor]] = {}

    def reset(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def skips(self, position) -> bool:
        """An update rule never skips a layer; a gated loop that has stopped does."""
        return False

    def enter(self, cell: Cell, state: torch.Tensor) -> None:
        if self.update.rule == "euler":
            self.inputs[cell.key] = state

    def leave(self, cell: Cell, output: torch.Tensor) -> torch.Tensor:
        rule, last = self.update.rule, cell.index == self.passes - 1
        history = self.outputs.setdefault(cell.key, [])
        if len(history) != cell.index:
            raise RuntimeError("A cell was reached out of order")
        if rule == "euler":
            state = self.inputs.pop(cell.key)
            mixed = state + (output - state) / self.passes
            anchored = self.update.anchor is not None
            history.append(output if anchored and cell.index == 0 else None)
            if anchored and last:
                mixed = self.update.anchor * history[0] + (1 - self.update.anchor) * mixed
        elif rule == "uniform":
            history.append(output)
            mixed = output if cell.index == 0 else torch.stack(history).mean(dim=0)
        else:  # ema
            history.append(output if cell.index == 0 else None)
            mixed = output if cell.index == 0 else self.update.eta * history[0] + (1 - self.update.eta) * output
        if last:
            self.outputs.pop(cell.key)
        return mixed


class Position(nn.Module):
    """One executed position: a parent layer's weights with this position's cache slot."""

    def __init__(self, layer: nn.Module, slot: int, parent_index: int) -> None:
        super().__init__()
        self.layer = layer
        self.slot = slot
        self.parent_index = parent_index
        self.attention_type = getattr(layer, "attention_type", "full_attention")
        # Set by Retrofit when the program has an update rule or a gate.
        self.mixer = None
        self.opens: Cell | None = None
        self.closes: Cell | None = None
        self.inside: Cell | None = None

    def forward(self, hidden_states, *args, **kwargs):
        if self.mixer is not None:
            if self.slot == 0:
                self.mixer.reset()
            if self.mixer.skips(self):
                return hidden_states
            if self.opens is not None:
                self.mixer.enter(self.opens, hidden_states)
        attention = self.layer.self_attn
        home = attention.layer_idx
        attention.layer_idx = self.slot
        try:
            output = self.layer(hidden_states, *args, **kwargs)
        finally:
            attention.layer_idx = home
        if self.closes is not None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Update rules need a layer that returns its hidden state")
            output = self.mixer.leave(self.closes, output)
        return output

    def extra_repr(self) -> str:
        return f"slot={self.slot}, parent_layer={self.parent_index}"


def owner_of(model) -> nn.Module:
    """The module holding `layers`, after checking the model is a supported, unmodified parent."""
    family = getattr(model.config, "model_type", None)
    if family not in FAMILIES:
        raise ValueError(f"Unsupported model family {family!r}; supported: {sorted(FAMILIES)}")
    owner = FAMILIES[family](model)
    layers = owner.layers
    if any(isinstance(layer, Position) for layer in layers) or "forward" in vars(owner):
        raise RuntimeError("A program is already installed on this model")
    if len(layers) != model.config.num_hidden_layers:
        raise RuntimeError("The layer list does not match the configuration")
    for index, layer in enumerate(layers):
        if getattr(layer.self_attn, "layer_idx", None) != index:
            raise RuntimeError(f"Layer {index} does not carry its own cache index")
    return owner


class Counted:
    """What every installed design shares: counted layer calls, a fresh cache, and use as a context.

    The count is an instrument independent of how the design is installed: a hook
    on each real parent layer counts the calls that reach its weights.
    """

    program: Program | Recirculation | GatedLoop

    def _watch(self, model, owner) -> None:
        self.model, self._owner, self.parents = model, owner, owner.layers
        self.calls = [0] * len(self.parents)
        self._hooks = [layer.register_forward_pre_hook(self._counter(index))
            for index, layer in enumerate(self.parents)]

    def _counter(self, index: int):
        def count(module, args) -> None:
            self.calls[index] += 1
        return count

    def take_calls(self) -> list[int]:
        """Calls per parent layer since the last take."""
        calls, self.calls = self.calls, [0] * len(self.parents)
        return calls

    def check_calls(self, calls: list[int], forwards: int = 1) -> None:
        """`forwards` is how many times the design ran: see `runs_for`."""
        expected = [count * forwards for count in self.program.calls_per_layer()]
        if calls != expected:
            raise RuntimeError(f"Executed layer calls {calls} differ from the program's {expected}")

    def runs_for(self, tokens: int) -> int:
        """How many times the design runs when the model is called on `tokens` new tokens."""
        return 1

    def forwards_in(self, calls: list[int]) -> int:
        """How many times the design ran, worked out from `calls`: a whole number of runs of its first layer."""
        forwards, remainder = divmod(calls[0], self.program.calls_per_layer()[0])
        if remainder or not forwards:
            raise RuntimeError("The first layer's calls are not a whole number of runs of the design")
        return forwards

    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.model.config)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


class Retrofit(Counted):
    """A program installed on a model. Call `remove()` to restore the parent exactly."""

    def __init__(self, model, program: Program) -> None:
        if not isinstance(program, Program):
            raise TypeError("Retrofit installs layer programs; use install() for any design")
        if getattr(model, "gated_loop", None) is not None:
            raise ValueError("This checkpoint was trained with a gate; it runs as a gated loop only")
        self._install(model, program, Mixer(program.update, program.passes) if program.cells() else None)

    def _install(self, model, program, mixer) -> None:
        """Put one position per step of `program` in place of the layer list. `mixer` acts at the
        ends of the program's cells."""
        owner = owner_of(model)
        if program.parent_layers != len(owner.layers):
            raise ValueError(f"The program is for {program.parent_layers} layers; the model has {len(owner.layers)}")
        self.program = program
        self._watch(model, owner)
        config = model.config
        self._saved = {"num_hidden_layers": config.num_hidden_layers}
        types = getattr(config, "layer_types", None)
        if types is not None:
            self._saved["layer_types"] = types
            config.layer_types = [types[step] for step in program.steps]
        config.num_hidden_layers = program.layer_calls
        positions = nn.ModuleList(Position(self.parents[step], slot, step)
            for slot, step in enumerate(program.steps))
        if mixer is not None:
            for position in positions:
                position.mixer = mixer
            for cell in program.cells():
                positions[cell.first].opens = cell
                positions[cell.last].closes = cell
                for slot in range(cell.first, cell.last + 1):
                    positions[slot].inside = cell
        owner.layers = positions
        self.installed = True

    def check_cache(self, cache, tokens: int) -> None:
        """Every position must have written its own slot, once per token."""
        if len(cache.layers) != self.program.layer_calls:
            raise RuntimeError(f"Cache has {len(cache.layers)} slots for {self.program.layer_calls} positions")
        lengths = [layer.get_seq_length() for layer in cache.layers]
        if any(length != tokens for length in lengths):
            raise RuntimeError(f"Cache slot lengths {lengths} differ from {tokens} tokens")

    def remove(self) -> None:
        if not self.installed:
            return
        for hook in self._hooks:
            hook.remove()
        self._owner.layers = self.parents
        for name, value in self._saved.items():
            setattr(self.model.config, name, value)
        self.installed = False


def install(model, design: Program | Recirculation | GatedLoop):
    """Install any design and return it installed: a layer program, a recirculation or a gated loop."""
    if isinstance(design, GatedLoop):
        from recurrent_retrofit.gated import Gated
        return Gated(model, design)
    if isinstance(design, Recirculation):
        from recurrent_retrofit.recirculation import Recirculator
        return Recirculator(model, design)
    return Retrofit(model, design)


def greedy(model, ids: list[int], new_tokens: int, *, cached: bool = True, forced: list[int] | None = None,
        after_forward=None) -> dict:
    """Batch-one greedy decoding that returns each step's last-position logits.

    `cached=False` recomputes the whole prefix at every step without a cache, as
    the control for cached decoding. `forced` supplies the tokens to append
    instead of each step's own argmax, so two runs can be compared on identical
    prefixes. `after_forward(step, cache)` runs after each forward pass, for
    instruments.
    """
    from transformers import DynamicCache
    device = next(model.parameters()).device
    tokens, logits = list(ids), []
    cache = DynamicCache(config=model.config) if cached else None
    current = torch.tensor([tokens], device=device, dtype=torch.long)
    with torch.inference_mode():
        for step in range(new_tokens):
            if cached:
                output = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
            else:
                output = model(input_ids=torch.tensor([tokens], device=device, dtype=torch.long),
                    use_cache=False, logits_to_keep=1)
            last = output.logits[0, -1].float().cpu()
            if after_forward is not None:
                after_forward(step, cache)
            logits.append(last)
            tokens.append(int(last.argmax()) if forced is None else forced[step])
            current = torch.tensor([[tokens[-1]]], device=device, dtype=torch.long)
            del output
    return {"tokens": tokens[len(ids):], "logits": logits, "cache": cache}
