# Workspace architecture

An experiment is a scientific unit with a stable directory ID. Closely related
conditions may share one experiment; an unrelated hypothesis gets a separate
descriptor, dataset, protocol, prompts, scoring implementation, tests, and notes.
`python -m eval_core list` discovers `experiments/*/experiment.toml` without changing
the shared CLI whenever an experiment is added.

The current experiment uses the existing `covert_eval` package. Its prompt releases,
condition factors, case-pairing rules, outcome definitions, and statistical summaries
remain specific to that study. Those assumptions are not imposed on new experiments.
Score parsing lives in `covert_eval/scoring.py`; the runner orchestrates the study.

`eval_core` has no dependency on `covert_eval`. Its public `RunSession` accepts a
stable experiment ID, output directory, shared budget settings, and optional provider
factory. It records request/response events and handles reservations, settlement,
refusals, model-identity checks, and uncertain charges. An experiment can supply a
domain-specific dry-run provider without changing the shared provider adapters.
Calls within a session are sequential; multiple sessions and processes may overlap.

Provider configuration and generation request/response types live in
`eval_core/models.py`. Existing `covert_eval` imports retain small compatibility
exports. New experiments import the shared types and runtime directly.

Configurations resolve dataset and output paths relative to their own location.
Project-root discovery walks upward to `pyproject.toml` rather than assuming a fixed
configuration depth. New manifests include the experiment ID, Git revision/dirty
state, source hashes, configuration snapshot, and existing scientific provenance.

The first Git commit preserves the entire pre-reorganization source baseline.
Historical run directories were not moved or rewritten. To reproduce or inspect a
historical protocol, use that baseline or its recorded prompt release; do not edit old
audit directories to make them resemble new runs. A historical failed run whose
frozen config hash no longer matches requires the original checkout to resume.
