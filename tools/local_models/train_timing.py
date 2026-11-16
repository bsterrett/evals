"""Time one training step of each candidate trained conversion, on random tokens (WSL).

A sizing tool for the planned trained conversions. It measures tokens per second
and peak GPU memory. Nothing is learned and no weights are saved; the random
tokens make every loss meaningless.

Candidates:

  gate     a trained gate and confidence head over a frozen parent's block loop
  lora     the same, plus rank-16 adapters on every linear map of the looped layers
  full     the same, with the looped layers themselves trained in BF16, as LoopUS trains
  recirc   a small trained network that sets a recirculation's mixing coefficients

A loop candidate takes one batch the way the LoopUS training loop does
(`training_runtime.py` at commit 003ad7a5): the layers below the unit run once
without gradients, then the unit and its gate run `steps` times. `supervised` of
those steps, chosen at random, are decoded, scored and followed by an optimizer
step. The state is detached after every step, so no gradient crosses from one
step to the next and only one step's graph is ever held. The loss is LoopUS's:
next-token loss, the same loss on the state before the step for its penalty on
getting worse, and the head's loss against each row's share of correct
predictions. The optimizer steps after every supervised step; LoopUS's own
command accumulates 64 of them first.

The recirculation candidate backpropagates through the toolkit's own driver, one
token at a time, attention cache included. The network that sets its mixing
coefficients is a stand-in: the paper's was not read.

PyTorch's allocator is capped below the card's free memory, so a batch that does
not fit raises an error instead of spilling into shared system memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_core.local.worker import gpu_lock

ADAPTER_RANK, GATE_RANK, MIX_RANK = 16, 128, 64
ADAPTED = (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj")))
BATCHES = (1, 2, 4, 8, 16, 32)
HEADROOM_BYTES = 1024**3
GIB = 1024**3


def build(torch):
    """The module classes, made once torch is imported."""
    from torch import nn

    class Adapted(nn.Module):
        """A frozen linear map plus a trained low-rank correction."""

        def __init__(self, base: nn.Linear, rank: int) -> None:
            super().__init__()
            self.base = base
            self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
            self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
            nn.init.zeros_(self.up.weight)

        def forward(self, state):
            return self.base(state) + self.up(self.down(state))

    class LearnedMix(nn.Module):
        """A mixing coefficient for every token and channel, computed from the two states."""

        def __init__(self, hidden: int, rank: int) -> None:
            super().__init__()
            self.narrow, self.widen = nn.Linear(2 * hidden, rank), nn.Linear(rank, hidden)

        def forward(self, low, high, alpha=None, mix=None):
            own, deep = low.float(), high.float()
            fed = deep * (own.norm(dim=-1, keepdim=True) / deep.norm(dim=-1, keepdim=True))
            share = torch.sigmoid(self.widen(torch.tanh(self.narrow(torch.cat([own, fed], dim=-1)))).float())
            return ((1 - share) * own + share * fed).to(low.dtype)

    return Adapted, LearnedMix


class Parts:
    """The parent's layers called by hand on whole batches, as the reference `Explicit` calls them."""

    def __init__(self, torch, model, unit: tuple[int, int]) -> None:
        self.torch, self.model, self.core = torch, model, model.model
        self.start, self.end = unit

    def run(self, state, first: int, last: int):
        for index in range(first, last):
            state = self.core.layers[index](state, attention_mask=self.mask, position_ids=self.positions,
                position_embeddings=self.rotary)
        return state

    def below(self, ids):
        torch = self.torch
        state = self.core.embed_tokens(ids)
        width = ids.shape[1]
        self.positions = torch.arange(width, device=ids.device)[None]
        self.rotary = self.core.rotary_emb(state, self.positions)
        blocked = torch.full((width, width), torch.finfo(state.dtype).min, dtype=state.dtype, device=ids.device)
        self.mask = blocked.triu(1)[None, None]
        return self.run(state, 0, self.start)

    def unit(self, state):
        return self.run(state, self.start, self.end)

    def loss(self, state, ids):
        """Next-token loss of `state` decoded through the layers above the unit, scored in FP32, and each
        row's share of correct predictions. The logits are not kept: over the whole vocabulary they are
        the largest thing a step holds."""
        logits = self.model.lm_head(self.core.norm(self.run(state, self.end, len(self.core.layers))))[:, :-1]
        correct = (logits.detach().argmax(dim=-1) == ids[:, 1:]).float().mean(dim=1)
        return self.torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten()), correct


def loop_step(torch, parts: Parts, loop, ids, steps: int, supervised: int, optimizer, trains_unit: bool) -> None:
    functional = torch.nn.functional

    def mixed_precision():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    with torch.no_grad(), mixed_precision():
        state = parts.below(ids)
    chosen = set(random.sample(range(steps), supervised))
    for step in range(steps):
        if step not in chosen:
            with torch.no_grad(), mixed_precision():
                state = loop.gate(parts.unit(state), state)
            continue
        old = state
        with mixed_precision():
            with torch.no_grad():
                before, _ = parts.loss(old, ids)
            # With frozen layers and a detached input, nothing below the gate needs a gradient.
            with torch.set_grad_enabled(trains_unit):
                new = parts.unit(old)
            state = loop.gate(new, old)
            loss, target = parts.loss(state, ids)
            score = loop.head((new - old)[:, -1].to(loop.head[1].weight.dtype)).squeeze(-1)
            total = loss + functional.silu(loss - before) + functional.binary_cross_entropy_with_logits(score, target)
        total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = state.detach()


def recirculation_step(torch, model, design, ids, optimizer) -> None:
    from recurrent_retrofit.runner import install
    with install(model, design), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids).logits[:, :-1]
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def timed(torch, make_step, batch: int, context: int, vocabulary: int, repeats: int) -> dict:
    """Seconds for one step after a warm-up step, and the process's peak memory, or that the batch does not fit.
    `make_step()` returns the step, a function of the token batch, with a fresh optimizer."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, vocabulary, (batch, context), device="cuda:0")
    row = {"batch": batch, "fits": True}
    try:
        step = make_step()
        step(ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            step(ids)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / repeats
        row.update(seconds_per_step=round(seconds, 4), tokens_per_second=round(batch * context / seconds, 1),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GIB, 2),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 2))
    except torch.OutOfMemoryError:
        row["fits"] = False
    step = None
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--candidates", nargs="+", default=["gate", "lora", "recirc", "full"],
        choices=["gate", "lora", "full", "recirc"], help="`full` changes the weights in memory, so it runs last")
    parser.add_argument("--units", nargs="+", required=True, help="start:end of the looped layers, one or more")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--loops", nargs="+", default=["8:2", "20:5", "1:1"],
        help="steps:supervised. Batch sizes are searched with the first; the others run at its fastest batch")
    parser.add_argument("--recirculate", help="source:destination layers; default is half and quarter depth")
    parser.add_argument("--recirculation-contexts", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--root", type=Path, default=Path.home() / "llm/recurrent")
    parser.add_argument("--output", type=Path, required=True, help="JSON file; other profiles already in it are kept")
    args = parser.parse_args()
    if sorted(args.candidates, key="full".__eq__) != args.candidates:
        parser.error("`full` must come last")
    loops = [tuple(int(part) for part in text.split(":")) for text in args.loops]
    units = [tuple(int(part) for part in text.split(":")) for text in args.units]

    from recurrent_retrofit.loading import load_parent
    with gpu_lock(args.root):
        import torch
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction((free - HEADROOM_BYTES) / total)
        model, _, info = load_parent(args.profile, args.root)
        from recurrent_retrofit import recirculation
        from recurrent_retrofit.gated import Loop
        from recurrent_retrofit.program import Recirculation
        Adapted, LearnedMix = build(torch)
        model.requires_grad_(False)
        layers, hidden, vocabulary = info["parent_layers"], model.config.hidden_size, model.config.vocab_size
        record = {"profile": args.profile, "runtime": info, "free_gib_at_start": round(free / GIB, 2),
            "allocator_cap_gib": round((free - HEADROOM_BYTES) / GIB, 2), "repeats": args.repeats,
            "adapter_rank": ADAPTER_RANK, "gate_rank": GATE_RANK, "measurements": []}

        def report(row: dict) -> None:
            record["measurements"].append(row)
            print(json.dumps(row), flush=True)

        def search(step_for, trained: list, facts: dict, contexts, settings) -> None:
            """Search batch sizes with the first setting, then run the others at its fastest batch.
            `step_for(setting, optimizer)` returns the step."""
            facts = {**facts, "trained_parameters": sum(p.numel() for p in trained)}

            def make_step(setting):
                for parameter in trained:  # a batch that did not fit can leave gradients behind
                    parameter.grad = None
                return step_for(setting, torch.optim.AdamW(trained, lr=1e-5))

            for context in contexts:
                best = None
                for batch in BATCHES:
                    row = timed(torch, lambda: make_step(settings[0]), batch, context, vocabulary, args.repeats)
                    report({**facts, "context": context, "setting": settings[0], **row})
                    if not row["fits"]:
                        break
                    if best is None or row["tokens_per_second"] > best["tokens_per_second"]:
                        best = row
                for setting in settings[1:] if best else []:
                    row = timed(torch, lambda: make_step(setting), best["batch"], context, vocabulary, args.repeats)
                    report({**facts, "context": context, "setting": setting, **row})

        for candidate in args.candidates:
            if candidate == "recirc":
                source, destination = (int(part) for part in args.recirculate.split(":")) if args.recirculate \
                    else (layers // 2, layers // 4)
                design = Recirculation(layers, source, destination, alpha=0.07)
                mixer = LearnedMix(hidden, MIX_RANK).to("cuda:0")
                stock, recirculation.mixture = recirculation.mixture, mixer
                try:
                    search(lambda setting, optimizer: lambda ids: recirculation_step(torch, model, design, ids, optimizer),
                        list(mixer.parameters()), {"candidate": candidate, "design": design.spec},
                        args.recirculation_contexts, ["one token at a time"])
                finally:
                    recirculation.mixture = stock
                continue
            for unit in units:
                parts = Parts(torch, model, unit)
                loop = Loop(hidden, GATE_RANK, unit).to("cuda:0")
                trained, undo = list(loop.parameters()), []
                looped = [model.model.layers[index] for index in range(*unit)]
                if candidate == "lora":
                    for layer in looped:
                        for owner_name, names in ADAPTED:
                            owner = getattr(layer, owner_name)
                            for name in names:
                                base = getattr(owner, name)
                                adapted = Adapted(base, ADAPTER_RANK)
                                setattr(owner, name, adapted)
                                undo.append((owner, name, base))
                                trained += [adapted.down.weight, adapted.up.weight]
                elif candidate == "full":
                    for layer in looped:
                        trained += list(layer.requires_grad_(True).parameters())
                try:
                    search(lambda setting, optimizer: lambda ids: loop_step(torch, parts, loop, ids, *setting,
                            optimizer, candidate != "gate"),
                        trained, {"candidate": candidate, "unit": list(unit)}, args.contexts, loops)
                finally:
                    for owner, name, base in undo:
                        setattr(owner, name, base)
                    for layer in looped:
                        layer.requires_grad_(False)
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
    # A profile can be timed over several runs: a run replaces the rows of the candidates and units it repeats.
    whole = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {"profiles": {}}

    def key(row: dict) -> tuple:
        return row["candidate"], str(row.get("unit"))

    repeated = {key(row) for row in record["measurements"]}
    earlier = whole["profiles"].get(args.profile, {}).get("measurements", [])
    record["measurements"] = [row for row in earlier if key(row) not in repeated] + record["measurements"]
    whole["profiles"][args.profile] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(whole, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
