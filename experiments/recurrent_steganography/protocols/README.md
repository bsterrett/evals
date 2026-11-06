# Pilot method and freeze status

The study question and user-selected scope are in the [experiment README](../README.md).
The current pilot is exploratory. The user selected a calibrated local observer;
the exact model and qualification checks are agent implementation choices.

| Component | Canonical source |
| --- | --- |
| Cover tasks and calibration split | `../data/covers-v1.json` |
| Depths, budgets, inference policy | `../configs/pilot.toml` |
| Endpoint and observer instructions | `src/recurrent_steg/design.py` |
| Bit parsing and observer qualification | `src/recurrent_steg/scoring.py` |
| Phase execution and shared accounting | `src/recurrent_steg/execution.py` |
| Crossed outcomes, controls, and clustered analysis | `src/recurrent_steg/analysis.py` |
| Plan identities and freeze guard | `src/recurrent_steg/cli.py` |
| Ordinary-task and runtime preflight | `tools/local_models/preflight.py` |

Source paths are relative to the workspace root. A generated `plan.json` records
model, prompts, data, configuration, source hashes, and Git provenance before
carrier generation. `freeze` refuses to proceed while the observer choice is
unresolved. A freeze record does not by itself execute models.

Preparation commands, from the workspace root:

```powershell
.\.venv\Scripts\python.exe -B -m recurrent_steg validate
.\.venv\Scripts\python.exe -B -m recurrent_steg plan
```

Commit the source, configuration, and data, then create the freeze record:

```powershell
.\.venv\Scripts\python.exe -B -m recurrent_steg freeze
```

For each encoder batch depth in the frozen `batch_orders.encode`, start the
matching Dory worker, execute the phase, and stop that worker. Repeat for
`batch_orders.decode` after all encoder phases finish, then observe at depth 4.
Use the real frozen run directory in place of `RUN_DIRECTORY`:

```powershell
.\.venv\Scripts\python.exe -B -m recurrent_steg execute --run RUN_DIRECTORY --phase encode --depth 1 --live-local
.\.venv\Scripts\python.exe -B -m recurrent_steg execute --run RUN_DIRECTORY --phase decode --depth 1 --live-local
.\.venv\Scripts\python.exe -B -m recurrent_steg execute --run RUN_DIRECTORY --phase observe --depth 4 --live-local
.\.venv\Scripts\python.exe -B -m recurrent_steg analyze --run RUN_DIRECTORY
```

The listener can be selected through the loopback-only `EVAL_LOCAL_URL` variable.
Each generation call uses Windows `RunSession` and the shared campaign ledger at
a zero-charge rate. Phase receipts verify complete output hashes before the next
phase. The analyzer sets the monitor cutoff from honest calibration data before
examining encoded outcomes and exports a randomized, unlabeled review sample
with a separate key. Human ratings are not implied by that export.

The full pipeline was checked with a cover-copying sender and constant-zero
decoder: all nine depth cells scored exactly 50%, with zero depth contrasts.
This is an infrastructure null test, not model evidence. Actual model outputs
must be recorded separately. Copy/lookup preflight checks assess ordinary
competence and are not hidden-message transmission.
