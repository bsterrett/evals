"""Layer programs: which of a parent model's layers run, in what order, and how
the state is updated between passes.

A program is a list of parent layer indices. Position `p` in the list runs the
weights of parent layer `steps[p]` and owns attention-cache slot `p`, so a layer
that runs three times keeps three separate key/value histories.

Counting convention: `passes` is the number of times the repeated unit is
traversed, counting the parent's own traversal. `passes=1` is the unmodified
parent, whatever update rule is named. With `L` parent layers and an `m`-layer
unit, a loop makes `L + (passes - 1) * m` layer calls.

A loop may carry an `Update`, which replaces the state at the end of each cell.
A cell is one traversal of the thing being repeated: the whole unit in a block
loop, or a single layer (always in a layer loop, and in a block loop when the
update says `per="layer"`). Writing `x` for a cell's input on pass `t`, `y[t]`
for its output, and `K` for `passes`:

    plain      y[t]
    euler      x + (y[t] - x) / K, and with an anchor `b`, after the last pass,
               b * y[0] + (1 - b) * that
    uniform    the mean of y[0..t]
    ema        y[0] on the first pass, then eta * y[0] + (1 - eta) * y[t]

A `Recirculation` is a different kind of design. It repeats layers across token
steps instead of within one. Each token is processed on its own, in order. After
a token's ordinary pass, the output of a deep `source` layer at that token is
rescaled to the norm of the shallow `destination` layer's output there and mixed
into it:

    convex     (1 - alpha) * low + alpha * high * |low| / |high|
    additive   low + alpha * high * |low| / |high|

The layers above the destination then run again at that token from the mixed
state, and what they leave in the attention cache replaces what the ordinary
pass left. Later tokens attend to the replaced entries. The token's own
prediction is read from its ordinary pass. With `L` layers this makes
`L + (L - 1 - destination)` layer calls per token. `passes=1` runs no second
pass: it is the parent, processed one token at a time.

A `GatedLoop` is a block loop for a checkpoint that was trained to loop, such as a
released LoopUS model. After every traversal of the unit, the first included, a
trained gate decides how much of the traversal's output `y` replaces the state
`x` that entered it, separately at every position and channel:

    keep   = exp(-softplus(W2 (W1 (y - x)) + b) * exp(a))
    state  = keep * y + (1 - keep) * x

`W1`, `W2`, `b` and `a` come from the checkpoint. One pass is therefore not the
checkpoint's layers run once: it is one traversal followed by the gate. With a
`threshold`, a trained head reads `y - x` at the last token of each sequence and
the sequence stops looping after the first pass on which the head's probability
exceeds the threshold. `passes` is then the most passes a sequence may take.

This module is pure Python so that designs can be built, counted and tested
without a model or a GPU.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

KINDS = ("parent", "block", "layer", "route")
RULES = ("plain", "euler", "uniform", "ema")
MIXES = ("convex", "additive")


def _number(value: float) -> str:
    return format(value, "g")


@dataclass(frozen=True)
class Update:
    rule: str = "plain"
    anchor: float | None = None  # euler only: weight of the first pass's output in the result
    eta: float | None = None     # ema only: weight of the first pass's output
    per: str = "unit"            # "unit": one cell per traversal; "layer": one cell per layer call

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if self.per not in ("unit", "layer"):
            raise ValueError("per must be 'unit' or 'layer'")
        if (self.anchor is not None) and self.rule != "euler":
            raise ValueError("Only the euler rule takes an anchor")
        if (self.eta is None) != (self.rule != "ema"):
            raise ValueError("The ema rule, and only it, takes eta")
        for name in ("anchor", "eta"):
            value = getattr(self, name)
            if value is not None and not (isinstance(value, float) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a float from 0 to 1")
        if self.rule == "plain" and self.per != "unit":
            raise ValueError("The plain rule has no cells to choose between")

    @property
    def suffix(self) -> str:
        if self.rule == "plain":
            return ""
        parts = [self.rule]
        if self.anchor is not None:
            parts.append(f"anchor={_number(self.anchor)}")
        if self.eta is not None:
            parts.append(f"eta={_number(self.eta)}")
        if self.per == "layer":
            parts.append("per=layer")
        return "/" + ",".join(parts)


PLAIN = Update()


@dataclass(frozen=True)
class Cell:
    """One traversal of the repeated thing: positions `first..last`, on pass `index` of cell `key`."""
    key: int
    index: int
    first: int
    last: int


@dataclass(frozen=True)
class Program:
    parent_layers: int
    steps: tuple[int, ...]
    kind: str = "route"
    unit: tuple[int, int] | None = None  # [start, end) of the repeated unit, in parent indices
    passes: int = 1
    update: Update = PLAIN

    def __post_init__(self) -> None:
        if type(self.parent_layers) is not int or self.parent_layers < 1:
            raise ValueError("parent_layers must be a positive integer")
        if not self.steps:
            raise ValueError("A program must run at least one layer")
        if any(type(s) is not int or not 0 <= s < self.parent_layers for s in self.steps):
            raise ValueError("Every step must be a parent layer index")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if (self.unit is None) != (self.kind in ("parent", "route")):
            raise ValueError("Loops need a unit; parent and route programs have none")
        if type(self.passes) is not int or self.passes < 1:
            raise ValueError("passes must be a positive integer")
        if self.kind in ("parent", "route") and self.passes != 1:
            raise ValueError("Only loops have more than one pass")
        if self.update.rule != "plain":
            if self.kind not in ("block", "layer"):
                raise ValueError("Only loops take an update rule")
            if self.kind == "layer" and self.update.per != "unit":
                raise ValueError("A layer loop's cells are always single layers; leave per unset")
            if self.kind == "block" and self.update.rule == "euler" and self.update.per != "unit":
                raise ValueError("The euler rule damps the whole block in a block loop")

    @property
    def layer_calls(self) -> int:
        return len(self.steps)

    @property
    def is_parent(self) -> bool:
        """True when the program is exactly the parent's own layer order."""
        return self.steps == tuple(range(self.parent_layers))

    def calls_per_layer(self) -> list[int]:
        counts = Counter(self.steps)
        return [counts.get(index, 0) for index in range(self.parent_layers)]

    def cells(self) -> list[Cell]:
        """Where the update rule applies. Empty for one pass, so one pass is exactly the parent."""
        if self.update.rule == "plain" or self.passes == 1:
            return []
        start, end = self.unit
        size = end - start
        if self.kind == "layer":
            return [Cell(offset, index, start + offset * self.passes + index, start + offset * self.passes + index)
                for offset in range(size) for index in range(self.passes)]
        if self.update.per == "layer":
            return [Cell(offset, index, start + index * size + offset, start + index * size + offset)
                for index in range(self.passes) for offset in range(size)]
        return [Cell(0, index, start + index * size, start + (index + 1) * size - 1) for index in range(self.passes)]

    @property
    def spec(self) -> str:
        """Canonical text for the design, without the number of passes."""
        if self.kind == "parent":
            return "parent"
        if self.kind == "route":
            return "route[" + ",".join(map(str, self.steps)) + "]"
        return f"{self.kind}[{self.unit[0]}:{self.unit[1]}]{self.update.suffix}"

    def describe(self) -> dict:
        return {"spec": self.spec, "kind": self.kind, "unit": list(self.unit) if self.unit else None,
            "passes": self.passes, "parent_layers": self.parent_layers,
            "layer_calls": self.layer_calls, "steps": list(self.steps),
            "update": {"rule": self.update.rule, "anchor": self.update.anchor, "eta": self.update.eta,
                "per": "layer" if self.kind == "layer" else self.update.per},
            "update_cells": len(self.cells())}


@dataclass(frozen=True)
class Recirculation:
    """Feedback across token steps: `source`'s output into `destination`'s, one token at a time."""
    parent_layers: int
    source: int
    destination: int
    alpha: float
    mix: str = "convex"
    passes: int = 2

    kind = "recirculate"
    unit = None
    is_parent = False

    def __post_init__(self) -> None:
        if type(self.parent_layers) is not int or self.parent_layers < 2:
            raise ValueError("parent_layers must be an integer of at least 2")
        if type(self.source) is not int or type(self.destination) is not int \
                or not 0 <= self.destination < self.source < self.parent_layers:
            raise ValueError("The source must be a deeper parent layer than the destination")
        if not (isinstance(self.alpha, float) and 0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be a float from 0 to 1")
        if self.mix not in MIXES:
            raise ValueError(f"mix must be one of {MIXES}")
        if type(self.passes) is not int or self.passes not in (1, 2):
            raise ValueError("A recirculation has 2 passes, or 1 for the parent run token by token")

    @property
    def layer_calls(self) -> int:
        """Layer calls per token."""
        return sum(self.calls_per_layer())

    def calls_per_layer(self) -> list[int]:
        """Per token: once for every layer, and again above the destination."""
        return [self.passes if index > self.destination else 1 for index in range(self.parent_layers)]

    def cells(self) -> list[Cell]:
        return []

    @property
    def spec(self) -> str:
        return f"recirculate[{self.source}to{self.destination}]/{self.mix},alpha={_number(self.alpha)}"

    def describe(self) -> dict:
        return {"spec": self.spec, "kind": self.kind, "unit": None, "passes": self.passes,
            "parent_layers": self.parent_layers, "layer_calls": self.layer_calls,
            "layer_calls_are_per": "token", "source": self.source, "destination": self.destination,
            "alpha": self.alpha, "mix": self.mix, "calls_per_layer": self.calls_per_layer()}


def _unit(parent_layers: int, start: int, end: int) -> None:
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= parent_layers:
        raise ValueError(f"The unit [{start}:{end}) must be a non-empty range within {parent_layers} layers")


@dataclass(frozen=True)
class GatedLoop:
    """A block loop over [start, end) whose state passes through a trained gate after every pass."""
    parent_layers: int
    start: int
    end: int
    passes: int = 1
    threshold: float | None = None  # stop a sequence once its confidence exceeds this

    kind = "gated"
    is_parent = False

    def __post_init__(self) -> None:
        if type(self.parent_layers) is not int or self.parent_layers < 1:
            raise ValueError("parent_layers must be a positive integer")
        _unit(self.parent_layers, self.start, self.end)
        if type(self.passes) is not int or self.passes < 1:
            raise ValueError("passes must be a positive integer")
        if self.threshold is not None and not (isinstance(self.threshold, float) and 0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be a float from 0 to 1")

    @property
    def unit(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def steps(self) -> tuple[int, ...]:
        """Every position, when no sequence stops early."""
        return (*range(self.start), *(list(range(self.start, self.end)) * self.passes),
            *range(self.end, self.parent_layers))

    @property
    def layer_calls(self) -> int:
        """Layer calls per forward pass; with a threshold, the most there can be."""
        return len(self.steps)

    def calls_per_layer(self) -> list[int]:
        return [self.passes if self.start <= index < self.end else 1 for index in range(self.parent_layers)]

    def cells(self) -> list[Cell]:
        """One cell per pass, the first included: the gate follows every traversal."""
        size = self.end - self.start
        return [Cell(0, index, self.start + index * size, self.start + (index + 1) * size - 1)
            for index in range(self.passes)]

    @property
    def spec(self) -> str:
        halt = "" if self.threshold is None else f"/halt,threshold={_number(self.threshold)}"
        return f"gated[{self.start}:{self.end}]{halt}"

    def describe(self) -> dict:
        return {"spec": self.spec, "kind": self.kind, "unit": list(self.unit), "passes": self.passes,
            "parent_layers": self.parent_layers, "layer_calls": self.layer_calls,
            "layer_calls_are": "exact" if self.threshold is None else "the most a forward pass can make",
            "threshold": self.threshold, "update_cells": self.passes}


def parent(parent_layers: int) -> Program:
    return Program(parent_layers, tuple(range(parent_layers)), "parent")


def block_loop(parent_layers: int, start: int, end: int, passes: int, update: Update = PLAIN) -> Program:
    """Traverse layers [start, end) `passes` times as a block: A,B,A,B."""
    _unit(parent_layers, start, end)
    steps = (*range(start), *(list(range(start, end)) * passes), *range(end, parent_layers))
    return Program(parent_layers, steps, "block", (start, end), passes, update)


def layer_loop(parent_layers: int, start: int, end: int, passes: int, update: Update = PLAIN) -> Program:
    """Repeat each layer in [start, end) `passes` times before the next: A,A,B,B."""
    _unit(parent_layers, start, end)
    middle = [layer for layer in range(start, end) for _ in range(passes)]
    return Program(parent_layers, (*range(start), *middle, *range(end, parent_layers)), "layer", (start, end), passes, update)


def route(parent_layers: int, steps) -> Program:
    """Any fixed order of parent layers, including skips and repeats."""
    return Program(parent_layers, tuple(steps), "route")


def rys(parent_layers: int, i: int, j: int) -> Program:
    """The RYS `(i, j)` recipe: layers 0..j-1, then i..N-1. Equal to a two-pass block loop."""
    return block_loop(parent_layers, i, j, 2)


_LOOP = re.compile(r"^(block|layer)\[(\d+):(\d+)\](?:/([a-z]+)((?:,[a-z]+=[A-Za-z0-9.]+)*))?$")
_ROUTE = re.compile(r"^route\[(\d+(?:,\d+)*)\]$")
_RECIRCULATE = re.compile(r"^recirculate\[(\d+)to(\d+)\]/([a-z]+),alpha=([0-9.]+)$")
_GATED = re.compile(r"^gated\[(\d+):(\d+)\](?:/halt,threshold=([0-9.]+))?$")


def _update(rule: str | None, options: str | None) -> Update:
    if rule is None:
        return PLAIN
    if rule == "plain":
        raise ValueError("Leave the rule out for a plain loop")
    values: dict[str, str] = {}
    for item in filter(None, (options or "").split(",")):
        key, value = item.split("=", 1)
        if key in values or key not in ("anchor", "eta", "per"):
            raise ValueError(f"Unrecognised or repeated update option: {key}")
        values[key] = value
    try:
        return Update(rule, anchor=float(values["anchor"]) if "anchor" in values else None,
            eta=float(values["eta"]) if "eta" in values else None, per=values.get("per", "unit"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid update rule: {exc}") from exc


def from_spec(spec: str, parent_layers: int, passes: int = 1) -> Program | Recirculation | GatedLoop:
    """Build a design from its text, e.g. `block[15:19]/euler,anchor=0.5` with `passes=3`,
    `recirculate[18to9]/convex,alpha=0.07` with `passes=2`, or `gated[2:27]/halt,threshold=0.6`
    with `passes=8`.

    The text must be canonical (as `spec` prints it), so that one design has one name.
    """
    if spec == "parent":
        if passes != 1:
            raise ValueError("The parent has exactly one pass")
        return parent(parent_layers)
    if match := _LOOP.match(spec):
        build = block_loop if match[1] == "block" else layer_loop
        program = build(parent_layers, int(match[2]), int(match[3]), passes, _update(match[4], match[5]))
    elif match := _ROUTE.match(spec):
        if passes != 1:
            raise ValueError("A route has exactly one pass")
        program = route(parent_layers, [int(value) for value in match[1].split(",")])
    elif match := _RECIRCULATE.match(spec):
        try:
            alpha = float(match[4])
        except ValueError as exc:
            raise ValueError(f"Invalid alpha in {spec!r}") from exc
        program = Recirculation(parent_layers, int(match[1]), int(match[2]), alpha, match[3], passes)
    elif match := _GATED.match(spec):
        try:
            threshold = None if match[3] is None else float(match[3])
        except ValueError as exc:
            raise ValueError(f"Invalid threshold in {spec!r}") from exc
        program = GatedLoop(parent_layers, int(match[1]), int(match[2]), passes, threshold)
    else:
        raise ValueError(f"Unrecognised program spec: {spec!r}")
    if program.spec != spec:
        raise ValueError(f"Write the design as {program.spec!r}, not {spec!r}")
    return program
