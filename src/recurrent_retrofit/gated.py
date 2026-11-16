"""Install a gated loop on a model that carries a trained gate.

A gated loop is a block loop in which a trained gate follows every traversal of
the unit, the first included. `Loop` holds what was trained: the gate, and
optionally a confidence head for stopping early. A loader attaches it to the
model as `model.gated_loop`; `loopus.py` does that for released LoopUS
checkpoints.

The loop is installed the way a layer program is: the model's own forward pass
runs one position per layer call, and every pass has its own attention-cache
slots. The gate mixes a traversal's output with the state that entered it at the
same sequence position, so cached decoding computes what recomputing the whole
prefix would.

Stopping is different. With a threshold, each sequence stops looping after the
first pass on which the head's probability, read from the change at its last
token, exceeds the threshold. A stopped sequence keeps its state while the others
continue. The number of passes then depends on the last token, so a later call
cannot extend an earlier one: every call must hold the whole sequence, and no
cache is returned.

The arithmetic follows the forward pass of the LoopUS authors' model class
(github.com/Thrillcrazyer/LoopUS at commit 003ad7a5, `models/modeling_lds.py`),
which was read and rewritten here, not run.
"""
from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.nn import functional

from recurrent_retrofit.program import Cell, GatedLoop
from recurrent_retrofit.runner import Retrofit


class Gate(nn.Module):
    """How much of a traversal's output `new` replaces the state `old` that entered it.

        keep  = exp(-softplus(widen(narrow(new - old))) * exp(log_rate))
        state = keep * new + (1 - keep) * old

    `keep` is between 0 and 1 at every position and channel. The arithmetic runs
    in the gate's own precision.
    """

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.narrow = nn.Linear(hidden, rank, bias=False)
        self.widen = nn.Linear(rank, hidden)
        self.log_rate = nn.Parameter(torch.zeros(hidden))

    def forward(self, new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        kind = self.log_rate.dtype
        fresh, held = new.to(kind), old.to(kind)
        step = functional.softplus(self.widen(self.narrow(fresh - held)))
        keep = torch.exp(-step * torch.exp(self.log_rate))
        return (keep * fresh + (1 - keep) * held).to(new.dtype)


class Loop(nn.Module):
    """The trained parts of a gated loop over layers `unit`, and what they were trained with."""

    def __init__(self, hidden: int, rank: int, unit: tuple[int, int], *, head: bool = True,
            trained_passes: int | None = None, threshold: float | None = None) -> None:
        super().__init__()
        self.gate = Gate(hidden, rank)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1)) if head else None
        self.unit = (int(unit[0]), int(unit[1]))
        self.trained_passes, self.threshold = trained_passes, threshold

    def confidence(self, change: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """The head's probability for each row, read from `change` at the row's last valid token.

        `mask` is the two-dimensional attention mask, or None when every token is
        valid. A row with no valid token is read at its first position.
        """
        if self.head is None:
            raise RuntimeError("This loop has no confidence head")
        rows, width = change.shape[:2]
        if mask is None:
            last = change[:, -1]
        else:
            valid = (mask != 0).to(change.device)
            where = (valid * torch.arange(1, width + 1, device=change.device)).argmax(dim=1)
            last = change[torch.arange(rows, device=change.device), where]
        kind = self.head[1].weight.dtype
        return torch.sigmoid(self.head(last.to(kind)).squeeze(-1))


class GateMixer:
    """Applies the gate at the end of every pass, and the stopping rule when there is one.

    It is the gated loop's counterpart of the runner's `Mixer`, and keeps two
    tallies for the checks: forward passes begun, and passes of the unit that
    were gated.
    """

    def __init__(self, design: GatedLoop, loop: Loop) -> None:
        self.design, self.loop = design, loop
        self.mask: torch.Tensor | None = None
        self.forwards = self.gated = 0
        self.depths: Counter = Counter()
        self.last_depths: list[int] | None = None
        self.reset(count=False)

    def reset(self, count: bool = True) -> None:
        """Called when a forward pass starts."""
        self.state = None
        self.active = None
        self.used = None
        self.done = False
        if count:
            self.forwards += 1

    def skips(self, position) -> bool:
        """True for the layers of a pass that no sequence needs."""
        return self.done and position.inside is not None

    def enter(self, cell: Cell, state: torch.Tensor) -> None:
        self.state = state

    def leave(self, cell: Cell, output: torch.Tensor) -> torch.Tensor:
        state, self.state = self.state, None
        if state is None:
            raise RuntimeError("A pass ended that was never entered")
        self.gated += 1
        mixed = self.loop.gate(output, state)
        last = cell.index == self.design.passes - 1
        if self.design.threshold is None:
            return mixed
        if self.active is None:
            self.active = torch.ones(output.shape[0], dtype=torch.bool, device=output.device)
            self.used = torch.zeros(output.shape[0], dtype=torch.long, device=output.device)
        # A sequence that has stopped keeps its state; the pass still ran for the others.
        mixed = torch.where(self.active[:, None, None], mixed, state)
        self.used += self.active
        confident = self.loop.confidence(output - state, self.mask) > self.design.threshold
        self.active = self.active & ~confident.to(self.active.device)
        self.done = not bool(self.active.any())
        if self.done or last:
            self.last_depths = self.used.tolist()
            self.depths.update(self.last_depths)
        return mixed


class Gated(Retrofit):
    """A gated loop installed on a model. Call `remove()` to restore the model exactly."""

    def __init__(self, model, design: GatedLoop) -> None:
        if not isinstance(design, GatedLoop):
            raise TypeError("Gated installs gated loops; use install() for any design")
        loop = getattr(model, "gated_loop", None)
        if not isinstance(loop, Loop):
            raise ValueError("This model carries no trained gate; a gated loop needs a checkpoint trained with one")
        if design.unit != loop.unit:
            raise ValueError(f"The gate was trained for layers [{loop.unit[0]}:{loop.unit[1]}], not {design.spec}")
        if design.threshold is not None and loop.head is None:
            raise ValueError("This checkpoint has no confidence head, so it cannot stop early")
        self.mixer = GateMixer(design, loop)
        self.taken: dict | None = None
        self._install(model, design, self.mixer)
        self._stock = None
        if design.threshold is not None:
            self._stock = self._owner.forward
            self._owner.forward = self._forward

    def _forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
            inputs_embeds=None, use_cache=None, cache_position=None, **kwargs):
        """With stopping, every call holds a whole sequence and runs without a cache."""
        width = (inputs_embeds if input_ids is None else input_ids).shape[1]
        partial = (past_key_values is not None and past_key_values.get_seq_length() > 0) \
            or (cache_position is not None and int(cache_position[0]) != 0)
        if attention_mask is not None:
            if isinstance(attention_mask, dict) or attention_mask.dim() != 2:
                raise NotImplementedError("Stopping reads the last valid token from a two-dimensional attention mask")
            partial = partial or attention_mask.shape[1] != width
        if partial:
            raise NotImplementedError("With a stopping threshold every call must hold the whole sequence: "
                "the number of passes depends on the last token, so a cache cannot be extended. "
                "Pass use_cache=False")
        self.mixer.mask = attention_mask
        try:
            return self._stock(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=None, inputs_embeds=inputs_embeds, use_cache=False,
                cache_position=cache_position, **kwargs)
        finally:
            self.mixer.mask = None

    def take_calls(self) -> list[int]:
        """Calls per parent layer since the last take. The gate's own tallies are taken with them,
        into `taken`: forward passes begun, passes gated, and passes used per sequence."""
        mixer = self.mixer
        self.taken = {"forwards": mixer.forwards, "gated": mixer.gated,
            "passes_per_sequence": dict(sorted(mixer.depths.items()))}
        mixer.forwards = mixer.gated = 0
        mixer.depths = Counter()
        return super().take_calls()

    def forwards_in(self, calls: list[int]) -> int:
        return self.taken["forwards"]

    def check_calls(self, calls: list[int], forwards: int = 1) -> None:
        """Check `calls` against the tallies taken with them. Without a threshold the calls are the
        design's. With one, the layers of the unit must have run once for each pass the gate saw,
        and every other layer once per forward pass."""
        taken, design = self.taken, self.program
        if taken is None:
            raise RuntimeError("Take the calls before checking them")
        if taken["forwards"] != forwards:
            raise RuntimeError(f"{taken['forwards']} forward passes began; {forwards} were expected")
        if design.threshold is None:
            if taken["gated"] != forwards * design.passes:
                raise RuntimeError(f"The gate ran {taken['gated']} times in {forwards} forward passes of {design.passes}")
            return super().check_calls(calls, forwards)
        if not forwards <= taken["gated"] <= forwards * design.passes:
            raise RuntimeError(f"The gate ran {taken['gated']} times in {forwards} forward passes of at most {design.passes}")
        expected = [taken["gated"] if design.start <= index < design.end else forwards
            for index in range(design.parent_layers)]
        if calls != expected:
            raise RuntimeError(f"Executed layer calls {calls} differ from the gated passes' {expected}")

    def check_cache(self, cache, tokens: int) -> None:
        if self.program.threshold is not None:
            raise RuntimeError("A gated loop with a stopping threshold keeps no cache")
        super().check_cache(cache, tokens)

    def remove(self) -> None:
        if not self.installed:
            return
        if self._stock is not None:
            del self._owner.forward
        super().remove()
