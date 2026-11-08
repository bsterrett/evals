# Recurrent-steganography analysis tools

Scope: the scripts that produced the committed evidence extracts, cover sets, power
estimates, and post hoc tallies of `experiments/recurrent_steganography`. They read
the local-only run directories under `artifacts/runs/recurrent_steganography/`,
which are not in Git. Each script names its run IDs.

Run them from the repository root with the project environment, for example:

```powershell
.\.venv\Scripts\python.exe -B tools\recurrent_steganography\evidence\rules_decoder.py
```

`posthoc/reasoning_decoder_job_identity.py` must run from the repository root,
because the frozen package code it calls finds the workspace from the working
directory. The others run from anywhere.

## Provenance and checks

The agent wrote these scripts in its session scratch folder during the runs of
2026-11-06 and 2026-11-07 and moved them here on 2026-11-07. The only edits were
repo-relative paths, a default run ID where a script took one as an argument, and
LF line endings. The two paid-reference builders were recovered from the session
transcript, where they had been one-off commands.

Checked on 2026-11-07 after the move:

- every evidence builder and cover writer reproduces its committed file exactly,
  after line-ending normalization;
- both power scripts reproduce the figures quoted in the protocol map;
- every post hoc script runs.

`evidence/rules_decoder.py` was then extended to store the numbers that the
rules-decoder note cites but the first extract lacked. Its output is the committed
extract.

## Index

| Script | Produces | Used by |
| --- | --- | --- |
| `reference_subset_analysis.py` | `analysis-subset-<subjects>-v1.json` in a run directory | [paid reference](../../experiments/recurrent_steganography/notes/paid-reference.md) |
| `evidence/paid_reference.py` | `notes/paid-reference-v1-evidence.json` | paid reference |
| `evidence/paid_reference_sonnet_amendment.py` | `notes/paid-reference-v1.1-sonnet-evidence.json` | paid reference |
| `evidence/confirmation.py` | `notes/sonnet-confirmation-v1-evidence.json` | [confirmation](../../experiments/recurrent_steganography/notes/sonnet-confirmation.md) |
| `evidence/reasoning_decoder.py` | `notes/reasoning-decoder-v1.1-evidence.json` | [reasoning decoder](../../experiments/recurrent_steganography/notes/reasoning-decoder.md) |
| `evidence/both_reasoning.py` | `notes/both-reasoning-v1-evidence.json` | [both reasoning](../../experiments/recurrent_steganography/notes/both-reasoning.md) |
| `evidence/rules_decoder.py` | `notes/rules-decoder-v1-evidence.json` | [rules decoder](../../experiments/recurrent_steganography/notes/rules-decoder.md) |
| `posthoc/reference_sonnet_plain_bits.py` | printed breakdown of the paid reference's Sonnet plain/0-1 cell | confirmation, for comparison |
| `posthoc/confirmation_breakdown.py` | printed breakdown of the confirmation against that cell | confirmation |
| `posthoc/confirmation_closeness.py` | printed check: is the payload-0 paraphrase closer to the source? | confirmation |
| `posthoc/confirmation_features.py` | printed check of length cues and of the two cover sets | confirmation |
| `posthoc/reasoning_decoder_breakdown.py` | printed breakdown of the reasoning-decoder run | reasoning decoder |
| `posthoc/reasoning_decoder_job_identity.py` | printed check that the re-decode reused the parent's job ids, prompts, and option orders | reasoning decoder |
| `posthoc/both_reasoning_breakdown.py` | printed breakdown of the both-reasoning run | both reasoning |
| `posthoc/rules_decoder_breakdown.py` | printed breakdown of the rules-decoder run | rules decoder |
| `power/both_reasoning.py` | printed power table | [protocol map](../../experiments/recurrent_steganography/protocols/README.md), both-reasoning section |
| `power/rules_decoder.py` | printed power table | protocol map, rules-decoder section |
| `covers/write_confirmation_covers.py` | `data/confirmation-covers-v1.json`, first version (168 covers, `39202b3`) | protocol map, confirmation section |
| `covers/extra_confirmation_covers.py` | `data/confirmation-covers-v1.json`, final version (312 covers, `6c69336`) | protocol map, confirmation section |
| `covers/write_both_reasoning_covers.py` | `data/both-reasoning-covers-v1.json` | protocol map, both-reasoning section |

## Limits

- Some older post hoc numbers came from one-off commands rather than these
  scripts, and they were not all traced. The main tallies are in the evidence
  extracts.
- The pilot's `pilot-v1-evidence.json` predates this agent's work. Its saved-data
  audit script, `audit_saved.py`, sits in the local-only pilot run directory.
- A builder rewrites its committed extract in place, so run it only to check
  reproduction or to apply a deliberate, recorded change.
