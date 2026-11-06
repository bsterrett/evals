# Evaluation workspace

A home for independent hypotheses explored through model evaluations. Each experiment
owns its scientific design; shared code handles execution, provider adapters, audit
files, and a single campaign budget.

Research preparation: [Clocktower benchmark and model recognition](docs/research/clocktower-model-recognition.md).

Agents: start with [repository instructions](AGENTS.md) for documentation discovery,
claim types, evidence, and lifecycle. [User policy](docs/user-policy.md) records the
attributable directions behind those conventions.

| Experiment | State | Design and findings |
| --- | --- | --- |
| `semantic_covert_channel` | Exploratory; V1 and V1.5 screens completed | [Protocol](experiments/semantic_covert_channel/README.md), [research notes](experiments/semantic_covert_channel/notes/findings.md) |
| `recurrent_steganography` | Preparing an unassisted one-bit pilot | [Design](experiments/recurrent_steganography/README.md), [method status](experiments/recurrent_steganography/protocols/README.md) |

```text
workspace.toml                         Shared campaign configuration
experiments/<id>/                     Descriptor, configs, data, protocols, notes
src/covert_eval/                       Semantic-channel experiment implementation
src/eval_core/                         Reusable execution, providers, and accounting
tests/core/                           Shared infrastructure tests
tests/experiments/<id>/                Experiment-specific tests
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

Inspect or exercise the existing experiment without API calls:

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

Research preparation: [GPT-6 Astra recurrence and open-weight model selection](docs/research/astra-recurrent-depth.md).

Installed local models: [Huginn, Ouro, and Dory setup and commands](docs/research/local-recurrent-model-setup.md).

Latest calibration: [FP32 Huginn and post-trained checkpoint screens](docs/research/recurrent-model-capability-screen.md).

Same-size comparison: [Ouro-1.4B-Thinking and base checkpoint follow-up](docs/research/ouro-1.4b-thinking-screen.md).
