"""Run a retrofit design and its parent through the same benchmark items.

The harness is lm-eval, which the published frozen-weight results used, so a
published number can be reproduced like for like. Datasets are pinned by commit
and read offline. This module measures and counts. It holds no rule for calling a
difference an improvement.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

PINS = json.loads(Path(__file__).with_name("benchmarks.json").read_text(encoding="utf-8"))
OUTCOMES = ("both", "parent_only", "design_only", "neither")


def pin(task: str) -> dict:
    try:
        return PINS["tasks"][task]
    except KeyError:
        raise ValueError(f"Task {task!r} has no pinned dataset; add it to benchmarks.json and fetch it") from None


def _cache(root: Path) -> Path:
    return root.resolve() / "cache/huggingface"


def cached_revisions(root: Path, dataset: str) -> list[str]:
    """Commits of a dataset that the `datasets` cache can serve offline."""
    # The library names the folder from the dataset, changing the case of the name.
    def key(text: str) -> str:
        return text.lower().replace("_", "")
    folders = [path for path in (_cache(root) / "datasets").glob("*") if key(path.name) == key(dataset.replace("/", "___"))]
    return sorted({path.name for folder in folders for path in folder.glob("*/*/*")
        if path.is_dir() and len(path.name) == 40})


def check_pins(root: Path, tasks: list[str]) -> dict:
    """Require that each task's dataset is cached at its pinned commit and at no other."""
    found = {}
    for task in tasks:
        entry = pin(task)
        revisions = cached_revisions(root, entry["dataset"])
        if revisions != [entry["revision"]]:
            raise RuntimeError(f"{entry['dataset']} is cached at {revisions or 'no revision'}, "
                f"not only at the pinned {entry['revision']}; run the fetch step")
        found[task] = {"dataset": entry["dataset"], "revision": entry["revision"]}
    return found


def fetch(root: Path, tasks: list[str]) -> dict:
    """Download each task's pinned dataset commit. This is the only step that uses the network."""
    os.environ["HF_HOME"] = str(_cache(root))
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(name, None)
    import datasets
    rows = {}
    for task in tasks:
        entry = pin(task)
        # A task group such as MMLU reads one configuration per subject.
        counts: dict[str, int] = {}
        for config in entry.get("configs") or [entry.get("config")]:
            data = datasets.load_dataset(entry["dataset"], config, revision=entry["revision"])
            for split, table in data.items():
                counts[split] = counts.get(split, 0) + table.num_rows
        rows[task] = counts
    check_pins(root, tasks)
    return rows


def _condition_rolling_windows(harness_model, context: int) -> None:
    """Make the harness's rolling windows condition every prediction on at least `context` tokens.

    The stock harness asks for one token, which cuts a document into disjoint
    windows, so a window's first tokens are scored with almost no context. This
    passes another length to the harness's own window function and changes
    nothing else.
    """
    from lm_eval import utils
    stock_windows, stock_rolling = utils.get_rolling_token_windows, harness_model.loglikelihood_rolling

    def windows(token_list, prefix_token, max_seq_len, context_len):
        return stock_windows(token_list=token_list, prefix_token=prefix_token, max_seq_len=max_seq_len, context_len=context)

    def rolling(requests, disable_tqdm: bool = False):
        utils.get_rolling_token_windows = windows
        try:
            return stock_rolling(requests, disable_tqdm=disable_tqdm)
        finally:
            utils.get_rolling_token_windows = stock_windows

    harness_model.loglikelihood_rolling = rolling


def evaluate(model, tokenizer, tasks: list[str], *, fewshot: int | None = None, limit: int | None = None,
        batch_size: int = 1, chat_template: bool = False, seed: int | None = None,
        max_length: int | None = None, rolling_context: int | None = None) -> dict:
    """lm-eval's results for a loaded model, with whatever design is installed on it.

    Batch size one is the harness default and involves no padding. `fewshot=None`
    keeps each task's own number of examples. `seed=None` keeps the harness's
    default seeds; a number sets all four of them, as its command line does, which
    changes which few-shot examples each prompt gets. `max_length=None` keeps the
    harness's context length, which it reads from the model's configuration; a
    number truncates longer inputs from the left and sets the window of a
    perplexity task. `rolling_context` is the context every token of a perplexity
    task is scored with, at least; None keeps the harness's disjoint windows.
    """
    version = importlib.metadata.version(PINS["harness"]["package"])
    if version != PINS["harness"]["version"]:
        raise RuntimeError(f"lm-eval {version} is installed; the pinned harness is {PINS['harness']['version']}")
    if rolling_context is not None and not (max_length and 1 <= rolling_context <= max_length):
        raise ValueError("rolling_context needs max_length, and must be from 1 to it")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    harness_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)
    if rolling_context is not None:
        _condition_rolling_windows(harness_model, rolling_context)
    seeds = {} if seed is None else dict.fromkeys(
        ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"), seed)
    return lm_eval.simple_evaluate(model=harness_model, tasks=list(tasks), num_fewshot=fewshot, limit=limit,
        log_samples=True, apply_chat_template=chat_template, **seeds)


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact(results: dict) -> dict:
    """Per-task scores and per-item outcomes, without the prompts and documents.

    What an item keeps depends on how its task is scored: the options'
    log-likelihoods for a multiple-choice task, the answer and text for a
    generated one, and for a perplexity task each metric's log-likelihood with the
    count it is averaged over.
    """
    tasks = {}
    for task, rows in results["samples"].items():
        config = results["configs"][task]
        metrics = [entry["metric"] for entry in config["metric_list"]]
        kind = config["output_type"]
        items = []
        for row in sorted(rows, key=lambda row: row["doc_id"]):
            item = {"doc_id": row["doc_id"], "prompt_hash": row["prompt_hash"]}
            if kind == "loglikelihood_rolling":
                # The target is the whole document.
                item["target_sha256"] = hashlib.sha256(str(row["target"]).encode()).hexdigest()
            else:
                item["target"] = str(row["target"])
            for metric in metrics:
                if metric in row:
                    value = row[metric]
                    item[metric] = [round(float(value[0]), 4), value[1]] if isinstance(value, (tuple, list)) else _number(value)
            if kind == "multiple_choice":
                item["loglikelihoods"] = [round(float(choice[0][0]), 4) for choice in row["resps"]]
            elif kind == "generate_until":
                item["answer"] = row["filtered_resps"][0]
                item["text"] = row["resps"][0][0]
            items.append(item)
        prompts = hashlib.sha256("\n".join(item["prompt_hash"] for item in items).encode()).hexdigest()
        tasks[task] = {
            "scores": {key: _number(value) for key, value in results["results"][task].items() if key != "alias"},
            "metrics": metrics, "output_type": config["output_type"], "task_version": results["versions"].get(task),
            "fewshot": results["n-shot"].get(task), "items_run": len(items), "prompts_sha256": prompts, "items": items,
        }
    return tasks


def group_scores(results: dict) -> dict:
    """The harness's own aggregate for each task group, such as the 14 subjects of MMLU-Pro."""
    return {group: {key: _number(value) for key, value in scores.items() if key != "alias"}
        for group, scores in (results.get("groups") or {}).items()}


def paired(parent: dict, design: dict, metric: str | None = None) -> dict:
    """Count one task's items by outcome under the parent and under a design.

    Both arguments are one task's entry from `compact`. Runs over different
    prompts are refused. The metric must score each item 0 or 1.
    """
    if parent["prompts_sha256"] != design["prompts_sha256"] or len(parent["items"]) != len(design["items"]):
        raise ValueError("The two runs did not use the same prompts")
    metric = metric or parent["metrics"][0]
    counts = dict.fromkeys(OUTCOMES, 0)
    for before, after in zip(parent["items"], design["items"]):
        if before["doc_id"] != after["doc_id"]:
            raise ValueError("The two runs list different items")
        if before[metric] not in (0.0, 1.0) or after[metric] not in (0.0, 1.0):
            raise ValueError(f"{metric} is not a 0 or 1 score on item {before['doc_id']}")
        counts[OUTCOMES[(0 if before[metric] else 2) + (0 if after[metric] else 1)]] += 1
    items = len(parent["items"])
    parent_correct = counts["both"] + counts["parent_only"]
    design_correct = counts["both"] + counts["design_only"]
    return {"metric": metric, "items": items, **counts, "parent_correct": parent_correct,
        "design_correct": design_correct, "parent_score": parent_correct / items,
        "design_score": design_correct / items, "difference": (design_correct - parent_correct) / items}


def is_binary(entry: dict, metric: str) -> bool:
    """True when the metric scores every item of a task 0 or 1."""
    return all(item.get(metric) in (0.0, 1.0) for item in entry["items"])


def compared(parent: dict, design: dict, metric: str) -> dict:
    """Paired counts for a metric that scores items 0 or 1. Any other metric, such as a
    perplexity, has no per-item outcome to count, so the two task scores are set side by side."""
    if is_binary(parent, metric) and is_binary(design, metric):
        return paired(parent, design, metric)
    if parent["prompts_sha256"] != design["prompts_sha256"] or len(parent["items"]) != len(design["items"]):
        raise ValueError("The two runs did not use the same prompts")

    def score(entry: dict):
        return next((value for key, value in entry["scores"].items() if key.split(",")[0] == metric), None)

    before, after = score(parent), score(design)
    return {"metric": metric, "items": len(parent["items"]), "parent_score": before, "design_score": after,
        "difference": None if before is None or after is None else after - before}
