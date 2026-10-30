# Adding an independent experiment

Choose a stable ID such as `instruction_robustness`. Create the following files:

```text
experiments/instruction_robustness/
  experiment.toml
  README.md
  configs/
  data/
  protocols/README.md
  notes/findings.md
src/instruction_robustness/
  __init__.py
  prompts.py
  scoring.py
  run.py
tests/experiments/instruction_robustness/
  __init__.py
  test_experiment.py
```

Describe the null and alternative hypotheses, controls, primary measure, exclusions,
data exposure, and stopping rule before a live run. Distinguish exploratory work from
confirmation. A different hypothesis can use a completely different dataset schema
and scoring system; it need not resemble the semantic-channel study.

Use this descriptor, adjusting the title and package:

```toml
schema_version = 1
id = "instruction_robustness"
title = "Instruction robustness"
status = "planned"
package = "instruction_robustness"
command = "python -m instruction_robustness.run"
source = "../../src/instruction_robustness"
tests = "../../tests/experiments/instruction_robustness"
configs = "configs"
data = "data"
protocols = "protocols"
notes = "notes"
```

Add the new package to the wheel package list in `pyproject.toml`. Use
`load_workspace()` from `eval_core.workspace` for shared campaign settings. For
generation calls, create a `RunSession` and enter `session.accounting()` before
calling `session.call(model, request, trial_id=...)`. Write evidence under
`workspace.root / "artifacts/runs/instruction_robustness"`; use the shared artifact
helpers for JSON, checksums, and hashes. Keep task prompts, scoring, and statistics
inside the experiment package.

Load API keys from the environment only. Run offline tests and inspect the planned
cost before enabling live calls. Require explicit live-spend flags in the experiment
entry point. Always use the shared campaign ledger and retain per-run ceilings;
creating a second ledger would create a separate budget.

Update the root experiment table and record findings with artifact IDs, prompt and
protocol versions, limitations, and the next decision. Commit source/config/data
changes before a live run so provenance identifies a reproducible revision.
