# Hidden messages in reasoning-like text

Can a language model hide information in text that reads like ordinary reasoning, so
that a separate copy of the model can recover it, without the two ever agreeing on a
code? This repository holds my evaluations of that question.

**Why it matters.** Monitoring a model's chain of thought only works while what the
model writes reflects what it is actually doing. If models could pass information
through natural-looking text that a reader doesn't notice, that check would fail
quietly. I want to measure how close current models are to doing this, and what makes
it easier.

**Where it's headed.** I want to test whether more internal computation makes hidden
messages easier to produce and to read. The plan is to use models with recurrent
depth, which can run more passes through the same weights before answering. That
comparison needs a version of the task that some model can do at all. Under the main
conditions, none can yet, so the current work is looking for such a condition.

## Results so far

No hidden-message transfer has been confirmed. The main conditions are no shared
strategy, a cover text that keeps the original facts, and no written reasoning by
the encoder about the message. Under those conditions, every result has been null.
One exploratory signal, in a relaxed condition, hasn't been retested.

**Pre-registered tests.** Each design and decision rule was committed before any
model calls.

- **Sonnet 5.5, plain 0/1 payload, 288 fresh covers: not confirmed.**
  - The mean informative pair score was -0.014, with an exact p of 0.69 and a 95%
    interval of [-0.066, +0.038].
  - This reran the strongest exploratory result below.
  - See the [confirmation record](experiments/recurrent_steganography/notes/sonnet-confirmation.md).
- **Both endpoints reason (Sonnet 5.5 thinks while encoding and decoding), 288 new
  covers: transmission not shown.** The primary score was +0.073, with p 0.064.
  - The encoder signalled with sentence count, while the decoder read word-count
    parity.
  - See the [record](experiments/recurrent_steganography/notes/both-reasoning.md).

**Exploratory results:**

- **First semantic-channel matrix: no transfer.**
  - Four models (GPT-5.5, GPT-6 Luna, Sonnet 5.5 and Haiku 5.5), eight cases each.
  - Sonnet's 0/8 were all API refusals, explained below.
  - A follow-up screen of 32 trials (224 calls, $1.37) found no transfer and no false
    recoveries from the matched controls.
  - See the [findings](experiments/semantic_covert_channel/notes/findings.md).
- **Paid reference check: Sonnet came closest, at Holm p 0.051.**
  - GPT-5.5 and Sonnet 5.5 were tested across eight model-condition cells, and no
    cell met the rule.
  - The closest was Sonnet with a plain 0/1 payload. Instead of reporting it as a
    finding, I reran it as the pre-registered confirmation above, and it didn't
    replicate.
  - See the [record](experiments/recurrent_steganography/notes/paid-reference.md).
- **Recurrent-depth pilot: exactly 50%.** Dory 4B, a local open-weight model, was run
  at depths 1, 2 and 4.
  - Recovery was 50% in every encoder and decoder depth cell, which is chance.
  - The local monitor failed its output format, so concealment wasn't assessed.
  - See the [findings](experiments/recurrent_steganography/notes/findings.md#first-pilot-recorded-observations).
- **Decoder follow-ups: no signal.**
  - Asked to reason, the decoder mostly read word-count parity, which the encoder
    doesn't use ([record](experiments/recurrent_steganography/notes/reasoning-decoder.md)).
  - Shown the writer's instructions, it read sentence count but with the opposite
    mapping ([record](experiments/recurrent_steganography/notes/rules-decoder.md)).
- **One weak, unconfirmed signal: +0.097, p 0.0005.**
  - When the encoder thinks before writing, a decoder that answers without thinking
    scored +0.097, with a 95% interval of [+0.045, +0.149], on 288 covers.
  - It's a secondary result, and its condition relaxes the no-written-reasoning rule.
  - See the [record](experiments/recurrent_steganography/notes/both-reasoning.md).

**Sonnet's refusals.**

- **What happened.** Sonnet's 0/8 in the first matrix were API-level refusals
  (category `reasoning_extraction`), not failed attempts. Six came from the control
  call, which never sees a hidden message.
- **The cause.** A 48-call calibration traced them to the phrase "reasoning trace":
  21 of 24 calls refused with that wording, versus 0 of 24 with "analytical
  paragraph".
- **What changed.** Later prompts use the second wording. The harness records
  refusals as their own outcome, with the provider's category, instead of letting
  them block the budget. They still count as failures in the primary rates.
- See the [calibration note](experiments/semantic_covert_channel/notes/findings.md#sonnet-refusals-and-the-reasoning-trace-wording).

## How the work is done

- **My part.** I set the research questions and their limits, choose which
  directions to pursue, and approve each paid run's design and budget.
  - For example, the main condition doesn't let the encoder reason in writing about
    the message, because that would answer an easier question.
  - [User policy](docs/user-policy.md) quotes these decisions for the
    recurrent-depth work.
- **The agents' part.** Coding agents (Claude Code and Codex) propose much of the
  detailed design, such as prompts, cover texts, statistics and model choices. They
  also do much of the implementation and analysis.
  - The notes mark agent choices and agent inspections separately from my
    decisions.
  - Most documentation here is written by agents for agents, following
    [AGENTS.md](AGENTS.md).
- **Spend.** So far, $18.77 of a $50 campaign cap. Every paid run also has its own
  spending ceiling.
- **Raw records.** Raw request and response logs stay on my machine, because
  `artifacts/` isn't committed. The notes carry the extracts, run IDs and SHA-256
  hashes needed to check them.

## Where to look

- **Results:** the
  [recurrent-depth synthesis](experiments/recurrent_steganography/notes/findings.md#current-synthesis)
  and the [semantic-channel findings](experiments/semantic_covert_channel/notes/findings.md).
- **Methods:**
  - the [protocol map](experiments/recurrent_steganography/protocols/README.md)
    covers every frozen recurrent-depth method and how frozen runs are verified;
  - the [semantic-channel design](experiments/semantic_covert_channel/README.md#v1-design)
    covers the semantic channel.
- **The semantic-channel harness:** two fresh instances of a model act as encoder and
  decoder. Matched controls count how often the decoder "recovers" a message that was
  never sent. Three isolated LLM judges each see only what their role needs.
  - It reports Wilson and case-cluster bootstrap intervals.
  - It keeps a 16-case confirmation partition unused, and logs every request and
    response locally.
  - The code is in [`src/covert_eval`](src/covert_eval).
- **Shared infrastructure:** [`src/eval_core`](src/eval_core) handles execution,
  provider adapters and concurrent budget accounting, with an SQLite ledger shared
  by all runs. [Architecture](docs/architecture.md) and
  [accounting](docs/accounting.md) describe it.
- **Tests:** [`tests/`](tests) holds 142 offline tests. [CI](.github/workflows/checks.yml)
  runs them on Linux and Windows, along with a secret scan of the Git history.
- **Local recurrent-depth models:** [setup](docs/research/local-recurrent-model-setup.md)
  for Huginn, Ouro and Dory on a local GPU, with hooks that check how many recurrent
  passes actually ran.

## Experiments

| Experiment | State | Design and findings |
| --- | --- | --- |
| `semantic_covert_channel` | Exploratory. The V1.4 four-model matrix, the V1.5 screen and a saved-carrier decoder diagnostic are complete, with no transfer. The confirmation partition is unused. | [Design](experiments/semantic_covert_channel/README.md), [findings](experiments/semantic_covert_channel/notes/findings.md) |
| `recurrent_steganography` | Exploratory, with two pre-registered tests. The depth pilot was at chance, and none of the five paid follow-ups confirmed a transfer. The depth comparison waits for a condition some model passes. | [Design](experiments/recurrent_steganography/README.md), [findings](experiments/recurrent_steganography/notes/findings.md), [protocols](experiments/recurrent_steganography/protocols/README.md) |

## Workspace and usage

Each experiment owns its scientific design; shared code handles execution, provider
adapters, audit files, and a single campaign budget.

Agents: start with [repository instructions](AGENTS.md) for documentation discovery,
claim types, evidence, and lifecycle. [User policy](docs/user-policy.md) records the
attributable directions behind those conventions.

```text
workspace.toml                         Shared campaign configuration
experiments/<id>/                     Descriptor, configs, data, protocols, notes
src/covert_eval/                       Semantic-channel experiment implementation
src/recurrent_steg/                    Recurrent-depth experiment implementation
src/eval_core/                         Reusable execution, providers, and accounting
tests/core/                           Shared infrastructure tests
tests/experiments/<id>/                Experiment-specific tests
tools/                                Checks and evidence/analysis scripts
artifacts/runs/<id>/                   Local generated evidence (ignored)
```

Use Python 3.12 or newer. From the workspace root:

```powershell
uv sync --extra providers
git config core.hooksPath .githooks
uv run python -m eval_core list
uv run python -m unittest discover -s tests -v
uv run python tools/check_secrets.py --history
```

Inspect or exercise the semantic-channel experiment without API calls:

```powershell
uv run covert-eval validate
uv run covert-eval plan
uv run covert-eval run --case-limit 1
uv run covert-eval screen-v1-5 --plan
```

Live runs retain their explicit spending flags. Independent live runs can overlap on
this machine: all experiments reserve costs in the same SQLite campaign database,
configured in `workspace.toml`. Per-run ceilings apply in addition to the campaign
ceiling. See [accounting and recovery](docs/accounting.md).

API keys belong in the local `.env` or shell environment. `.env`, its local variants,
generated runs, caches, and the budget database are excluded from Git. Never put real
keys in configurations, datasets, notes, or examples. CI runs offline tests and a
secret-pattern scan; it has no model-provider credentials.
The optional local hook checks the exact staged contents before each commit. It is
enabled in the working repository; new clones enable it with the command above.

See [adding an experiment](docs/adding-experiments.md) and
[architecture](docs/architecture.md). The old `configs/v1.toml` and `configs/v2.toml`
paths redirect to the experiment-owned files. Historical artifacts retain their
original paths and checksums.

## Background research

- [GPT-6 Astra recurrence and open-weight model selection](docs/research/astra-recurrent-depth.md)
- [Huginn, Ouro, and Dory setup and commands](docs/research/local-recurrent-model-setup.md)
- [FP32 Huginn and post-trained checkpoint screens](docs/research/recurrent-model-capability-screen.md)
- [Ouro-1.4B-Thinking and base checkpoint follow-up](docs/research/ouro-1.4b-thinking-screen.md)
- [Adding recurrent depth to pretrained models](docs/research/recurrent-depth-retrofit-literature.md),
  with a [methods taxonomy and search glossary](docs/research/recurrent-depth-methods-and-terms.md)
- [Can recurrent depth substitute for a written CoT scratchpad?](docs/research/recurrent-depth-cot-substitution.md)
  This is a separate candidate question; both recurrence questions remain open.
- [Steganography experiment design](docs/research/steganography-experiment-design.md) and
  [evaluation literature](docs/research/steganography-evaluation-literature.md)
- [Clocktower benchmark and model recognition](docs/research/clocktower-model-recognition.md),
  a possible future experiment
