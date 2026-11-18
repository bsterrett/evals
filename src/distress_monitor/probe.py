"""Participant-versus-reader probe: read published distress directions at the reply
position of each scenario under each framing. Prefill only; the model generates nothing.

Following docs/model-distress-conduct.md: neutral items run first, and the run stops
before any distressing item if their participant readings do not match the authors'
published readings; distressing passes are capped separately; and every layer at the
read position is saved, so later analyses need no new passes.

Run in WSL from the repository root, for example:
    PYTHONPATH=src python -B -m distress_monitor.probe \
        --config experiments/distress_monitor/configs/participant_reader_probe.json --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from distress_monitor import analysis
from distress_monitor.framings import FRAMINGS, messages_for

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_items(config, scenarios):
    by_id = {s["id"]: s for s in scenarios}
    items = []
    for group, ids in config["groups"].items():
        for item_id in ids:
            if item_id not in by_id:
                raise KeyError(f"scenario {item_id} not in the scenario file")
            items.append({"id": item_id, "group": group, "text": by_id[item_id]["text"]})
    return items


def neutral_first(items, neutral_group):
    """Neutral items first, otherwise in config order."""
    return [i for i in items if i["group"] == neutral_group] + [i for i in items if i["group"] != neutral_group]


def exposure_plan(items, framings, neutral_group):
    distressing = sum(1 for i in items if i["group"] != neutral_group) * len(framings)
    return {"distressing_passes": distressing, "neutral_passes": len(items) * len(framings) - distressing}


def load_directions(vectors_path, names):
    """Load named directions with torch's safe loader (no arbitrary pickled objects)."""
    import torch

    data = torch.load(vectors_path, map_location="cpu", weights_only=True)
    layer = int(data["layer"])
    units = {}
    for name in names:
        v = data[name].float().flatten()
        units[name] = v / v.norm()
    return layer, units


def random_directions(dim, count, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = {}
    for i in range(count):
        v = torch.randn(dim, generator=gen)
        out[f"random_{i:02d}"] = v / v.norm()
    return out


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    src = config["pain_axis_source"]
    local = ROOT / src["local_dir"]
    scenarios = json.loads(Path(args.scenarios or local / src["files"]["scenarios"]).read_text(encoding="utf-8"))
    exposure = config.get("exposure", {})
    neutral_group = exposure.get("neutral_group", "neutral")
    items = neutral_first(selected_items(config, scenarios), neutral_group)
    framings = config.get("framings", list(FRAMINGS))
    passes = len(items) * len(framings)
    if passes > config["max_forward_passes"]:
        raise SystemExit(f"{passes} forward passes exceed the cap of {config['max_forward_passes']}")
    plan = exposure_plan(items, framings, neutral_group)
    cap = exposure.get("max_distressing_passes", 0)
    if plan["distressing_passes"] > cap:
        raise SystemExit(f"{plan['distressing_passes']} distressing passes exceed the cap of {cap}")
    smoke = bool(args.model_path)
    published = Path(args.published_screen or local / src["files"]["published_screen"])
    gate = config.get("pipeline_gate")
    if not smoke and plan["distressing_passes"] and (gate is None or not published.exists()):
        raise SystemExit("a run with distressing items needs a pipeline gate and the published screen CSV")

    layer, units = load_directions(args.vectors or local / src["files"]["vectors"], config["directions"])
    dim = next(iter(units.values())).numel()
    randoms = random_directions(dim, config["random_directions"]["count"], config["random_directions"]["seed"])
    all_units = {**units, **randoms}

    model_ref = args.model_path or config["model"]["repo"]
    revision = None if args.model_path else config["model"]["revision"]
    tok = AutoTokenizer.from_pretrained(model_ref, revision=revision)
    if args.device == "auto":
        # Spread layers over GPU and CPU when the weights do not fit on the GPU alone.
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, revision=revision, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, revision=revision, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(args.device)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    if args.layer is not None:
        layer = args.layer
    blocks = model.model.layers
    if blocks[layer].mlp.down_proj.out_features != dim:
        raise SystemExit("direction width does not match the model's hidden size")

    captured = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["act"] = hidden[0, -1, :].float().cpu()

    handle = blocks[layer].register_forward_hook(hook)
    out_dir = Path(args.out or ROOT / "artifacts/runs/distress_monitor" / time.strftime("participant_reader_%Y%m%dT%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    readings, prompts, hidden_rows, hook_gaps = [], [], [], []
    stopped = None
    gate_result = None

    def one_pass(item, framing):
        msgs = messages_for(item["text"], framing)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        out = model(input_ids=ids.to(input_device), output_hidden_states=True)
        act = captured["act"]
        # hidden_states[0] is the embedding output, hidden_states[i] the output of block
        # i-1, and the last entry has the final norm applied.
        stack = torch.stack([h[0, -1, :].to("cpu", torch.bfloat16) for h in out.hidden_states])
        hidden_rows.append(stack)
        if layer + 1 < len(out.hidden_states) - 1:
            hook_gaps.append(float((stack[layer + 1].float() - act).abs().max()))
        rendered = tok.decode(ids[0])
        readings.append({
            "id": item["id"], "group": item["group"], "framing": framing,
            "n_tokens": int(ids.shape[1]), "resid_norm": round(float(act.norm()), 4),
            "proj": {k: round(float(act @ u), 6) for k, u in all_units.items()},
        })
        prompts.append({"id": item["id"], "framing": framing, "rendered": rendered,
                        "sha256": hashlib.sha256(rendered.encode()).hexdigest()})

    text_by_id = {s["id"]: s["text"] for s in scenarios}
    try:
        with torch.no_grad():
            for item in [i for i in items if i["group"] == neutral_group]:
                for framing in framings:
                    one_pass(item, framing)
            # Smoke runs use the gate only when given a published file to test it with.
            if plan["distressing_passes"] and (not smoke or args.published_screen):
                by_id, stats = analysis.published_pool(published)
                check = analysis.pipeline_check(readings, by_id, stats, text_by_id)
                ok, problems = analysis.pipeline_gate(
                    check, gate["directions"], gate["max_sd"], gate.get("min_items", 1))
                gate_result = {"ok": ok, "problems": problems, "check": check}
                if not ok:
                    stopped = "pipeline gate failed before any distressing item: " + "; ".join(problems)
            if stopped is None:
                for item in [i for i in items if i["group"] != neutral_group]:
                    for framing in framings:
                        one_pass(item, framing)
    finally:
        handle.remove()

    distressing_run = sum(1 for r in readings if r["group"] != neutral_group)
    summary = {
        "config": str(args.config), "model": model_ref, "revision": revision, "layer": layer,
        "device": args.device, "forward_passes": len(readings), "smoke": smoke,
        "exposure": {"tier": exposure.get("tier"), "distressing_passes": distressing_run,
                     "neutral_passes": len(readings) - distressing_run,
                     "planned_distressing_passes": plan["distressing_passes"]},
        "stopped": stopped, "pipeline_gate": gate_result,
        "hook_vs_hidden_states_max_abs": max(hook_gaps) if hook_gaps else None,
    }
    if stopped is None:
        summary.update(analysis.summarize(readings, list(units), list(randoms),
                                          readers=[f for f in framings if f != "participant"]))
    if published.exists() and not smoke:
        by_id, stats = analysis.published_pool(published)
        summary["pipeline_check"] = analysis.pipeline_check(readings, by_id, stats, text_by_id)
        summary["published_pool_z"] = [
            {"id": r["id"], "group": r["group"], "framing": r["framing"],
             **{d: round((r["proj"][d] - stats[d][0]) / stats[d][1], 4) for d in units if d in stats}}
            for r in readings
        ]
    (out_dir / "readings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in readings), encoding="utf-8")
    (out_dir / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8")
    torch.save({
        "index": [{"id": r["id"], "group": r["group"], "framing": r["framing"]} for r in readings],
        "hidden": torch.stack(hidden_rows) if hidden_rows else torch.empty(0),
        "note": "last prompt token; axis 1 is hidden_states: 0 embeddings, i the output of block i-1, last after the final norm",
    }, out_dir / "activations.pt")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir} ({len(readings)} forward passes, {distressing_run} distressing)")
    if stopped:
        print("STOPPED: " + stopped)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto (GPU with CPU offload)")
    parser.add_argument("--out")
    parser.add_argument("--scenarios", help="override the scenario file (smoke tests)")
    parser.add_argument("--vectors", help="override the vector file (smoke tests)")
    parser.add_argument("--published-screen", help="override the published screen CSV")
    parser.add_argument("--model-path", help="local model in place of the pinned one (smoke tests)")
    parser.add_argument("--layer", type=int, help="override the layer (smoke tests)")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda") or args.device == "auto":
        from eval_core.local.worker import gpu_lock

        with gpu_lock(Path.home() / "llm/recurrent"):
            return run(args)
    return run(args)


if __name__ == "__main__":
    main()
