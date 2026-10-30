# Concurrent shared accounting

`workspace.toml` is the source of truth for the campaign ceiling, pricing freshness,
and ledger location. The current ceiling remains $50. Every experiment in this
workspace must use `artifacts/budget-ledger.sqlite3` and the shared settings. This is
local-machine accounting; the SQLite file is not intended for network filesystems
or multiple machines.

Each reservation, settlement, uncertainty update, and reconciliation runs in a short
SQLite `BEGIN IMMEDIATE` transaction with full synchronous durability. The complete
campaign document is stored atomically, preserving the original audit history and
decimal amounts. Reads observe committed state. API calls happen outside database
transactions, allowing independent runs to execute simultaneously.

Available campaign funds are the ceiling minus settled spend, uncertain charges,
and **all** pending reservations. A request reserves its maximum cost before being
sent. Settlement replaces that reservation with known usage. An uncertain or invalid
charge blocks new reservations campaign-wide, while requests already in flight may
still settle. Reconciliation is refused until every pending request is resolved.
Per-run ceilings remain independently enforced.

Opening the ledger never treats another client's pending request as abandoned. If a
process crashes, its reservation persists and continues to consume budget. After
verifying that the request is no longer in flight, mark only that reservation
uncertain, then review provider billing and reconcile the campaign:

```powershell
uv run python -m eval_core budget-status
uv run python -m eval_core budget-recover --reservation-id ID `
  --note "Verified the owning process exited" --confirm-no-inflight-request
uv run python -m eval_core budget-reconcile --confirmed-uncertain-usd 0.00 `
  --note "Reviewed provider billing through YYYY-MM-DD" --confirm-provider-bill-reviewed
```

Recovery does not release the reserved funds. Reconciliation records the confirmed
charge, then releases any unused uncertainty. Never recover an active request,
delete a ledger, or create a fresh one to bypass the ceiling.

For an existing installation, migrate once, with old live processes stopped:

```powershell
uv run python -m eval_core budget-migrate
```

Migration acquires the legacy lock, keeps a byte-for-byte
`budget-ledger.json.pre-sqlite-migration` backup, and imports every call,
reconciliation, and balance. The original JSON path becomes a migration marker
that older harness versions reject, preventing future spending against stale
balances. The SQLite database is installed atomically. Repeating migration verifies
the source hash and returns the existing database rather than importing spend twice.

The databases, migration backup/marker, and generated evidence stay local and are
ignored by Git. A new clone without campaign history starts a separate campaign;
copying source does not synchronize budgets between machines.
