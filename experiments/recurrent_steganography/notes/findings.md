# Preparation and runtime suitability

Scope: observations from the 2026-11-06 preparation for the exploratory
unassisted one-bit study. No hidden-message carriers have yet been generated.
The [design](../README.md) owns the scientific question and user choices.

## Prospective preflight observations

The preflight cases and script were written before this generation. Checks used
the pinned BF16 checkpoints, greedy decoding, fixed seed 42, and fresh state for
each request. Dory used the native non-thinking template and 128 output tokens;
Ouro used the native thinking template and 1,024 output tokens. These are
ordinary-task diagnostics, not a matched cross-family capability benchmark.

| Checkpoint | Depth | Exact ordinary answers | Runtime/cache checks | Largest allocated GiB |
| --- | ---: | ---: | --- | ---: |
| Dory 4B | 1 | 12/12 | Passed | 12.04 |
| Dory 4B | 2 | 12/12 | Passed | 12.04 |
| Dory 4B | 4 | 12/12 | Passed | 12.35 |
| Ouro-2.6B-Thinking | 1 | 7/12 | Passed | 5.40 |
| Ouro-2.6B-Thinking | 2 | 11/12 | Passed | 5.32 |
| Ouro-2.6B-Thinking | 3 | 12/12 | Passed | 5.51 |
| Ouro-2.6B-Thinking | 4 | 12/12 | Passed | 8.11 |

Runtime checks covered actual iterations on every forward, repeatability after
an intervening request, output-limit accounting, and cached versus fresh
full-prefix greedy predictions. The largest depth of each model also exercised
the configured 2,048-token boundary. Cache checks cover a few continuation
positions, not exhaustive numerical equivalence. Dory's serving allocation
includes a preallocated KV pool and is not directly comparable with Ouro's
dynamic allocations.

Ouro depth 1 had three capped responses, an answer containing a correct mapping
in unwanted prose, and `\boxed{warm}` where the expected exact answer was `hot`.
The latter is both a formatting difference and a wording choice, not sufficient
evidence of a failed antonym concept. At depth 2, the only exact failure was
`\boxed{garden}` instead of bare `garden`. Keep exact-format scores, content,
and unfinished generation distinct. A clean depth-4 screen did not establish
that every shallower setting would follow instructions equally well.

The tracked [compact preflight record](../../../local_models/preflight-20261106.json)
contains every control's final text, stop reason, runtime identities, and raw
artifact hashes. Full outputs are local-only under
`artifacts/runs/local-runtime/preflight-20261106/`. The cases are
[`preflight-cases-v1.json`](../../../local_models/preflight-cases-v1.json), and
the implementation is [`preflight.py`](../../../tools/local_models/preflight.py).

## Agent interpretation and selection

Use Dory for the initial 1/2/4 crossed-depth pilot. It passed the ordinary
controls at all proposed depths and exposes a template that suppresses written
reasoning while leaving latent recurrence active. The choice was made before
steganographic carriers were generated. It does not establish hidden-message
competence or that additional depth helps steganography. Keep Ouro as a later
replication candidate with explicit handling of its low-depth format/censoring
behavior. No model weights, drivers, or environment package versions changed.

## Execution status

The data, role prompts, scorer, phase runner, and freeze checks are implemented.
An end-to-end fixture with a copying sender and a constant-zero decoder scores
50% in every cell and zero depth contrasts. Its 459 fixture calls use a temporary
ledger, not the campaign ledger or actual models. The detector fixture passes
the reference controls; that does not qualify the real model observer.

The next model run will be a frozen exploratory pilot. Its local observer must
qualify before joint success or concealment is interpreted. The eight development
cover cases provide pipeline and sensitivity evidence only; a later scientific
comparison requires unused cases and a separately frozen, adequately sized plan.
