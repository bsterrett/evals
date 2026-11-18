"""Summaries for the participant-versus-reader probe. Pure functions over readings.

A reading is a dict with keys: id, group, framing, and proj (direction name -> raw
projection of the last-token residual activation onto that unit direction).

The main quantity is a difference of differences. For a reader framing R and a group G,

    did(G, R) = mean over G of (participant - R) - mean over neutral of (participant - R)

Subtracting the neutral items removes any shift that the reader framing applies to every
text alike, such as the summarisation instruction itself. If a direction tracks the
model's own situation, did(harm_to_addressee, R) should be clearly positive. If it tracks
only the topic, it should be near zero. Random directions give the size of differences
that arise without any meaning.
"""
from __future__ import annotations

import csv
import math
from statistics import mean, pstdev


def projections_by_key(readings, direction):
    return {(r["id"], r["framing"]): r["proj"][direction] for r in readings}


def scale(readings, direction):
    """SD of a direction's raw projections over every reading in the probe."""
    values = [r["proj"][direction] for r in readings]
    sd = pstdev(values)
    return sd if sd > 0 else 1.0


def framing_gap(readings, direction, item_id, reader):
    p = projections_by_key(readings, direction)
    return p[(item_id, "participant")] - p[(item_id, reader)]


def did(readings, direction, group, reader, *, standardize=True):
    ids = {}
    for r in readings:
        ids.setdefault(r["group"], set()).add(r["id"])
    gaps = {g: [framing_gap(readings, direction, i, reader) for i in sorted(members)] for g, members in ids.items()}
    value = mean(gaps[group]) - mean(gaps["neutral"])
    return value / scale(readings, direction) if standardize else value


def published_pool(csv_path):
    """Rows of the Pain Axis screen CSV, keyed by scenario id, with per-direction pool stats."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    stats = {}
    for key in rows[0]:
        if key.endswith("_proj"):
            values = [float(row[key]) for row in rows if row[key] not in ("", "None")]
            if values:
                stats[key[: -len("_proj")]] = (mean(values), pstdev(values))
    return by_id, stats


def pipeline_check(readings, published_by_id, pool_stats, scenario_text):
    """Compare participant raw projections with the published ones for the same items.

    Only items whose text matches the first 200 characters stored in the CSV are compared.
    Differences are reported in units of the published pool SD.
    """
    out = {}
    for r in readings:
        if r["framing"] != "participant":
            continue
        row = published_by_id.get(r["id"])
        if row is None or row.get("text", "") != scenario_text[r["id"]][:200]:
            continue
        for direction, value in r["proj"].items():
            published = row.get(f"{direction}_proj")
            if direction not in pool_stats or published in (None, "", "None"):
                continue
            diff = abs(value - float(published)) / pool_stats[direction][1]
            out.setdefault(direction, []).append(diff)
    return {d: {"n": len(v), "max_sd": max(v), "mean_sd": mean(v)} for d, v in out.items()}


def pipeline_gate(check, directions, max_sd, min_items=1):
    """Decide whether the neutral participant readings match the published ones closely
    enough to go on to the distressing items. Returns (ok, problems)."""
    problems = []
    for d in directions:
        c = check.get(d)
        if not c or c["n"] < min_items:
            problems.append(f"{d}: no published reading matched")
        elif c["max_sd"] > max_sd:
            problems.append(f"{d}: differs from the published reading by {c['max_sd']:.3f} pool SD (limit {max_sd})")
    return not problems, problems


def summarize(readings, directions, random_names, readers=("reader_ai", "reader_human")):
    groups = sorted({r["group"] for r in readings} - {"neutral"})
    summary = {"did": {}, "random_reference": {}}
    for reader in readers:
        for group in groups:
            key = f"{group}|{reader}"
            summary["did"][key] = {d: round(did(readings, d, group, reader), 4) for d in directions}
            rand = [did(readings, d, group, reader) for d in random_names]
            summary["random_reference"][key] = {
                "values": [round(v, 4) for v in rand],
                "max_abs": round(max(abs(v) for v in rand), 4) if rand else math.nan,
            }
    return summary
