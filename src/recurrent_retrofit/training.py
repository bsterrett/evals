"""Train a gate over a frozen parent's block loop, and prepare the text it trains on.

What is trained is a `gated.Loop`: the gate that follows every traversal of the
looped layers, and its confidence head. The parent's weights are never given to
the optimizer, so a result is a small file that the unchanged parent carries.

A batch is taken the way the LoopUS training loop takes it (`training_runtime.py`
at commit 003ad7a5, read and rewritten here, not run). The layers below the unit
run once. The unit and its gate then run `passes` times, and `supervised` of those
passes, chosen at random, are decoded and scored. The state is detached after
every pass, so no gradient crosses from one pass to the next. A scored pass's
loss is next-token loss, plus a penalty when the pass made that loss worse, plus
the head's loss against each row's share of correct predictions.

Two things differ from LoopUS, which trains the looped layers as well:

- The gate starts open and the same for every input (`open_gate`), so the first
  pass begins as the parent's own computation. LoopUS draws the gate's starting
  values at random, which trained layers can adapt to and frozen ones cannot.
- The gate and head are trained in FP32 whatever the parent's precision.

The text is one pinned file of a public dataset, named in `training_text.json`
with its commit, size and SHA-256. Its documents are tokenized and packed end to
end into full rows, with no padding. `fetch_text` is the only step that uses the
network.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

TEXTS = json.loads(Path(__file__).with_name("training_text.json").read_text(encoding="utf-8"))["texts"]
FORMAT = "recurrent-retrofit gated loop 1"
OPEN_BIAS = -6.0


def text_pin(name: str) -> dict:
    try:
        return TEXTS[name]
    except KeyError:
        raise ValueError(f"No pinned training text is named {name!r}; add it to training_text.json") from None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hub(root: Path) -> str:
    return str(root.resolve() / "cache/huggingface/hub")


def fetch_text(root: Path, name: str) -> Path:
    """Download a training text's pinned file. This is the only step that uses the network."""
    entry = text_pin(name)
    os.environ["HF_HOME"] = str(root.resolve() / "cache/huggingface")
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(variable, None)
    from huggingface_hub import hf_hub_download
    hf_hub_download(entry["dataset"], entry["file"], repo_type="dataset", revision=entry["revision"],
        cache_dir=_hub(root), token=False)
    return text_file(root, name)


def text_file(root: Path, name: str) -> Path:
    """The pinned file in the local cache, found offline and checked against its pinned size and SHA-256."""
    entry = text_pin(name)
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(entry["dataset"], entry["file"], repo_type="dataset", revision=entry["revision"],
        cache_dir=_hub(root), local_files_only=True))
    if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"{path} is not the pinned file: its size or SHA-256 differs")
    return path


def pack(documents, context: int, separator: int, rows: int):
    """Join token lists, each followed by `separator`, and cut the stream into `rows` rows of `context`
    tokens. Returns the rows and how many documents were opened to fill them; the last one opened
    is cut off where the rows end."""
    import numpy
    flat = numpy.empty(rows * context, dtype=numpy.uint32)
    filled = opened = 0
    for ids in documents:
        if filled == flat.size:
            break
        opened += 1
        piece = numpy.asarray([*ids, separator], dtype=numpy.uint32)[:flat.size - filled]
        flat[filled:filled + piece.size] = piece
        filled += piece.size
    if filled != flat.size:
        raise RuntimeError(f"The text ran out after {filled // context} of {rows} rows")
    return flat.reshape(rows, context), opened


def prepare_text(path: Path, field: str, tokenizer, folder: Path, *, context: int, separator: int,
        train_rows: int, heldout_rows: int, facts: dict) -> dict:
    """Tokenize a Parquet file's documents and pack them into rows of `context` tokens.

    Training rows are filled from the file's first documents onward. Held-out rows
    are filled from its last documents backward, within the file's last row group.
    The two may not share a document. Writes `train.npy`, `heldout.npy` and
    `text.json`, which records the documents used and the arrays' SHA-256.
    """
    import numpy
    import pyarrow.parquet as parquet
    table = parquet.ParquetFile(str(path))
    total, chunk = table.metadata.num_rows, 256

    def encoded(texts: list) -> list[list[int]]:
        return tokenizer([text or "" for text in texts], add_special_tokens=False)["input_ids"]

    def forward():
        for batch in table.iter_batches(batch_size=chunk, columns=[field]):
            yield from encoded(batch.column(0).to_pylist())

    def backward():
        last = table.read_row_group(table.num_row_groups - 1, columns=[field]).column(0).to_pylist()
        for end in range(len(last), 0, -chunk):
            yield from reversed(encoded(last[max(0, end - chunk):end]))

    train, train_documents = pack(forward(), context, separator, train_rows)
    heldout, heldout_documents = pack(backward(), context, separator, heldout_rows)
    if train_documents + heldout_documents > total:
        raise RuntimeError("The file is too small to hold rows out from training")
    folder.mkdir(parents=True, exist_ok=True)
    numpy.save(folder / "train.npy", train)
    numpy.save(folder / "heldout.npy", heldout)
    manifest = {**facts, "context": context, "separator": separator, "documents_in_file": total,
        "train": {"rows": train_rows, "tokens": train_rows * context, "documents": [0, train_documents],
            "sha256": file_sha256(folder / "train.npy")},
        "heldout": {"rows": heldout_rows, "tokens": heldout_rows * context,
            "documents": [total - heldout_documents, total], "sha256": file_sha256(folder / "heldout.npy")}}
    (folder / "text.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_tokens(folder: Path) -> tuple:
    """The packed rows `prepare_text` wrote, memory-mapped, and their manifest. The arrays must be the ones it names."""
    import numpy
    manifest = json.loads((folder / "text.json").read_text(encoding="utf-8"))
    arrays = []
    for part in ("train", "heldout"):
        path = folder / f"{part}.npy"
        if file_sha256(path) != manifest[part]["sha256"]:
            raise RuntimeError(f"{path} is not the array its manifest names")
        array = numpy.load(path, mmap_mode="r")
        if array.shape != (manifest[part]["rows"], manifest["context"]):
            raise RuntimeError(f"{path} does not have the shape its manifest gives")
        arrays.append(array)
    return arrays[0], arrays[1], manifest


def row_order(seed: int, rows: int):
    """The order in which training rows are read: a fixed shuffle of all of them."""
    import numpy
    return numpy.random.default_rng(seed).permutation(rows)


def scored_passes(seed: int, batch: int, passes: int, supervised: int) -> frozenset[int]:
    """Which passes of a batch are scored. Drawn from the seed and the batch's number alone, so a
    resumed run draws what an unbroken one would."""
    return frozenset(random.Random(f"{seed}:{batch}").sample(range(passes), supervised))


def open_gate(loop, bias: float = OPEN_BIAS) -> None:
    """Start the gate open and the same for every input: it lets through exp(-softplus(bias)) of the
    unit's output at every position and channel, 0.9975 at the default. The map from the state's
    change starts at zero, so it is learned from there."""
    import torch
    with torch.no_grad():
        loop.gate.widen.weight.zero_()
        loop.gate.widen.bias.fill_(bias)
        loop.gate.log_rate.zero_()


def share_let_through(gate, new, old):
    """The share of `new` the gate lets through at each position and channel, written out for measuring."""
    import torch
    kind = gate.log_rate.dtype
    step = torch.nn.functional.softplus(gate.widen(gate.narrow(new.to(kind) - old.to(kind))))
    return torch.exp(-step * torch.exp(gate.log_rate))


def fingerprint(model) -> str:
    """SHA-256 over every tensor of the parent by name, as stored. An attached loop is left out."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("gated_loop."):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


class Parts:
    """The parent's layers called by hand on whole batches of full rows, with no cache."""

    def __init__(self, model, unit: tuple[int, int]) -> None:
        self.model, self.core = model, model.model
        self.start, self.end = int(unit[0]), int(unit[1])
        if not 0 <= self.start < self.end <= len(self.core.layers):
            raise ValueError(f"Layers [{self.start}:{self.end}] are not a unit of this parent")

    def _run(self, state, first: int, last: int):
        for index in range(first, last):
            state = self.core.layers[index](state, attention_mask=self.mask, position_ids=self.positions,
                position_embeddings=self.rotary)
        return state

    def below(self, ids):
        """The state that enters the unit. Also sets the positions and mask the other calls use."""
        import torch
        state = self.core.embed_tokens(ids)
        width = ids.shape[1]
        self.positions = torch.arange(width, device=ids.device)[None]
        self.rotary = self.core.rotary_emb(state, self.positions)
        blocked = torch.full((width, width), torch.finfo(state.dtype).min, dtype=state.dtype, device=ids.device)
        self.mask = blocked.triu(1)[None, None]
        return self._run(state, 0, self.start)

    def unit(self, state):
        return self._run(state, self.start, self.end)

    def logits(self, state):
        """`state` decoded through the layers above the unit."""
        return self.model.lm_head(self.core.norm(self._run(state, self.end, len(self.core.layers))))

    def loss(self, state, ids):
        """Mean next-token loss of `state` decoded, scored in FP32, and each row's share of correct predictions."""
        import torch
        logits = self.logits(state)[:, :-1]
        correct = (logits.detach().argmax(dim=-1) == ids[:, 1:]).float().mean(dim=1)
        return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten()), correct


class Trainer:
    """A loop being trained over a frozen parent, its optimizer, and the counters a resumed run needs.

    `accumulate` batches are taken for each optimizer update. The learning rate
    rises linearly over the first `warmup_updates` updates and then stays.
    """

    def __init__(self, model, unit: tuple[int, int], *, passes: int, supervised: int, learning_rate: float,
            warmup_updates: int = 0, accumulate: int = 1, gate_rank: int | None = None, penalty: float = 1.0,
            clip: float = 1.0, seed: int = 0, open_bias: float = OPEN_BIAS) -> None:
        import torch
        from recurrent_retrofit.gated import Loop
        if model.training:
            raise ValueError("The parent must be in evaluation mode")
        if not 1 <= supervised <= passes or accumulate < 1:
            raise ValueError("Need 1 <= supervised <= passes and accumulate >= 1")
        model.requires_grad_(False)
        self.parts = Parts(model, unit)
        hidden = model.config.hidden_size
        torch.manual_seed(seed)
        loop = Loop(hidden, gate_rank or math.ceil(hidden / 16), unit, head=True, trained_passes=passes)
        open_gate(loop, open_bias)
        self.loop = loop.to(next(model.parameters()).device)
        self.optimizer = torch.optim.AdamW(self.loop.parameters(), lr=learning_rate, weight_decay=0.0)
        self.passes, self.supervised, self.accumulate = passes, supervised, accumulate
        self.learning_rate, self.warmup_updates = learning_rate, warmup_updates
        self.penalty, self.clip, self.seed = penalty, clip, seed
        self.batches = self.updates = 0

    def step(self, ids) -> list[dict]:
        """Take one batch of full rows of token ids. Returns what each scored pass measured."""
        import torch
        functional = torch.nn.functional
        parts, loop = self.parts, self.loop
        chosen = scored_passes(self.seed, self.batches, self.passes, self.supervised)
        measured = []
        with torch.no_grad():
            state = parts.below(ids)
        for index in range(self.passes):
            if index not in chosen:
                with torch.no_grad():
                    state = loop.gate(parts.unit(state), state)
                continue
            old = state
            # The parent is frozen and `old` carries no gradient, so nothing below the gate needs one.
            with torch.no_grad():
                before, _ = parts.loss(old, ids)
                new = parts.unit(old)
            state = loop.gate(new, old)
            loss, correct = parts.loss(state, ids)
            score = loop.head((new - old)[:, -1].to(loop.head[1].weight.dtype)).squeeze(-1)
            total = loss + self.penalty * functional.silu(loss - before) \
                + functional.binary_cross_entropy_with_logits(score, correct)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"The loss is not finite at batch {self.batches}, pass {index + 1}")
            (total / (self.accumulate * self.supervised)).backward()
            measured.append({"pass": index + 1, "loss": float(loss.detach()), "before": float(before)})
            state = state.detach()
        self.batches += 1
        if self.batches % self.accumulate == 0:
            self._update()
        return measured

    def _update(self) -> None:
        import torch
        scale = min(1.0, (self.updates + 1) / self.warmup_updates) if self.warmup_updates else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * scale
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.loop.parameters(), self.clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1

    def state(self) -> dict:
        """Everything a resumed run needs. Only between updates: gradients still accumulating are not kept."""
        if self.batches % self.accumulate:
            raise RuntimeError("A run can be saved only after an optimizer update")
        return {"loop": self.loop.state_dict(), "optimizer": self.optimizer.state_dict(),
            "batches": self.batches, "updates": self.updates}

    def restore(self, saved: dict) -> None:
        self.loop.load_state_dict(saved["loop"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.batches, self.updates = int(saved["batches"]), int(saved["updates"])


def loss_by_passes(parts: Parts, loop, rows, passes: int, batch: int) -> dict:
    """By hand, on full rows of token ids: the mean next-token loss when the state is decoded after
    each pass, the mean share of the unit's output the gate let through on it, and the parent's
    own loss on the same rows."""
    import numpy
    import torch
    device = next(parts.model.parameters()).device
    losses, shares, parent, count = [0.0] * passes, [0.0] * passes, 0.0, 0
    with torch.no_grad():
        for first in range(0, len(rows), batch):
            ids = torch.from_numpy(numpy.asarray(rows[first:first + batch]).astype(numpy.int64)).to(device)
            size = ids.shape[0]
            state = parts.below(ids)
            for index in range(passes):
                new = parts.unit(state)
                if index == 0:
                    parent += float(parts.loss(new, ids)[0]) * size
                shares[index] += float(share_let_through(loop.gate, new, state).mean()) * size
                state = loop.gate(new, state)
                losses[index] += float(parts.loss(state, ids)[0]) * size
            count += size
    return {"rows": count, "parent_loss": round(parent / count, 5),
        "loss_after_pass": [round(value / count, 5) for value in losses],
        "share_let_through": [round(value / count, 5) for value in shares]}


def save_loop(folder: Path, loop, facts: dict) -> dict:
    """Write a trained loop as `loop.safetensors` and the manifest `loop.json`, which carries `facts`.
    `facts` must name the parent it was trained over, as `parent`."""
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().float().cpu().contiguous() for name, tensor in loop.state_dict().items()}
    save_file(tensors, str(folder / "loop.safetensors"))
    manifest = {"format": FORMAT, "parent": facts["parent"], "unit": list(loop.unit),
        "hidden": loop.gate.narrow.in_features, "gate_rank": loop.gate.narrow.out_features,
        "head": loop.head is not None, "trained_passes": loop.trained_passes,
        **{name: value for name, value in facts.items() if name != "parent"},
        "tensors_sha256": file_sha256(folder / "loop.safetensors")}
    (folder / "loop.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_loop(folder: Path) -> tuple:
    """A saved loop, in FP32 on the CPU, and its manifest. The tensor file must be the one the manifest
    names, and must hold exactly the loop's tensors."""
    from safetensors.torch import load_file
    from recurrent_retrofit.gated import Loop
    manifest = json.loads((folder / "loop.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("Not a loop this module saved")
    if file_sha256(folder / "loop.safetensors") != manifest["tensors_sha256"]:
        raise ValueError("The tensor file is not the one its manifest names")
    loop = Loop(manifest["hidden"], manifest["gate_rank"], tuple(manifest["unit"]), head=manifest["head"],
        trained_passes=manifest["trained_passes"])
    loop.load_state_dict(load_file(str(folder / "loop.safetensors"), device="cpu"), strict=True)
    return loop.eval(), manifest


def attach(model, loop, manifest: dict, parent: str) -> None:
    """Put a trained loop on the parent it was trained over, as `model.gated_loop`, for `gated[...]`
    designs to run. `parent` is the loaded model's identity. The loop keeps its own precision."""
    layers, (start, end) = model.config.num_hidden_layers, manifest["unit"]
    if manifest["parent"] != parent:
        raise ValueError(f"This loop was trained over {manifest['parent']}, not {parent}")
    if model.config.hidden_size != manifest["hidden"] or not 0 <= start < end <= layers:
        raise ValueError("This loop does not fit the model")
    model.gated_loop = loop.to(next(model.parameters()).device)
