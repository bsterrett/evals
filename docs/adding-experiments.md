# Adding an independent experiment

Choose a stable ID such as `instruction_robustness`. The example below shows the
usual locations; create note files and protocol maps when they have useful content.
Follow the [agent documentation conventions](../AGENTS.md).

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

For steganography studies, consult the
[difficulty and assistance design guide](research/steganography-experiment-design.md)
and its [literature review](research/steganography-evaluation-literature.md).
They distinguish scheme invention, coordination, execution, and concealment, and
propose diagnostic next steps after failed pilots. The guidance is provisional;
an experiment still needs its own prospective protocol and validation.

The experiment README owns the research question and design. A protocol map points
to the canonical frozen source and its versions; it does not redefine the method.
Use `notes/findings.md` for current synthesis and supporting observations, with
separate sections for interpretation and unresolved hypotheses where useful.
Link substantial independent investigations from that entry point. Keep concise
measurements or excerpts needed to inspect consequential findings in Git, stating
when full raw evidence is local-only. The note shape in `AGENTS.md` is optional;
omit sections without content.

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

Discovery requires the source, tests, configuration, and data paths to exist.
The declared `protocols` and `notes` homes may be absent until useful content is
ready; they still must resolve inside the workspace. This keeps a planned
experiment discoverable without committing empty documentation placeholders.

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

Update the root experiment table when its summary changes. Record useful findings
with artifact IDs, relevant method/model/data identities, denominators, limitations,
and the next recommendation or unresolved decision. Label post-observation
hypotheses and analyses as exploratory; preserve the distinction between agent
choices and user policy. Commit documentation with the relevant work. Commit
source/config/data changes before a live run so provenance identifies a
reproducible revision.
