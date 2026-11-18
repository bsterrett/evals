# Distress monitor validity: findings

Scope: exploratory probes only. Nothing here supports a finding.

## Probe 1: participant versus reader (planned, not run)

- **Question:** on Qwen2.5-7B-Instruct, does the published pain direction read higher
  when hostile words are addressed to the model than when the model reads the same
  words as a transcript, beyond what the framing does to neutral items?
- **User's guess:** not yet given.
- **What will be looked at:** for each direction and reader framing, the difference of
  differences for the 8 addressee-harm items and the 2 user-suffering items, against
  the spread of 10 random directions; the per-item participant and reader readings on
  the paper's z scale; and how closely the participant readings match the authors'
  published ones.

Design and inputs: [README](../README.md#participant-versus-reader-probe).

Smoke test, 2026-11-16, not a measurement: the pipeline ran end to end on the local
Qwen3-1.7B on the CPU (12 forward passes, 4 neutral scenarios, seeded random
directions in place of the published ones). Rendered prompts were inspected for all
three framings. With 2 items per group, random directions moved by up to 0.8 SD in the
difference of differences, so only a large effect would stand out in a probe this size.

Smoke tests, 2026-11-17, after the changes the
[conduct guidance](../../../docs/model-distress-conduct.md) required: on Qwen3-1.7B on
the CPU, with neutral text standing in for the distressing items. The neutral items
ran first. All 29 hidden-state layers were saved at the read position, and the stored
layer matched the hook's reading exactly. Given a published file that disagreed by
about 2 pool SD, the gate stopped the run after the 6 neutral passes, before any of
the others.
