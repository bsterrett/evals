from __future__ import annotations

import hashlib
import json
import sqlite3
import os
import threading
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterator

from eval_core.artifacts import write_json


class BudgetExceeded(RuntimeError):
    pass


class BudgetUncertain(RuntimeError):
    pass


class CampaignBlocked(BudgetExceeded):
    pass


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000001"), rounding=ROUND_CEILING), "f")


@dataclass(frozen=True)
class Reservation:
    id: str
    amount: Decimal


class CampaignLock:
    """Legacy JSON migration lock. Live SQLite runs do not hold this lock."""

    def __init__(self, ledger_path: Path) -> None:
        self.path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
        self._handle: Any = None

    def __enter__(self) -> "CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise BudgetExceeded(
                f"Another live run appears to hold the campaign ledger lock: {self.path}"
            ) from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


class BudgetLedger:
    """A shared campaign ledger; each mutation is one SQLite transaction.

    Pending reservations from other runs remain pending. They are never assumed
    abandoned merely because a second process opens the ledger.
    """

    def __init__(
        self, path: Path, maximum: Decimal, *, persist: bool = True,
        _initial_data: dict[str, Any] | None = None,
    ) -> None:
        if not maximum.is_finite() or maximum <= 0:
            raise ValueError("Budget maximum must be positive and finite")
        self.path = Path(path)
        self.maximum = maximum
        self.persist = persist
        self._thread_lock = threading.RLock()
        self._local = threading.local()
        self._data = _initial_data or self._new_data()
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path = self.path.with_suffix(".json")
            if (
                _initial_data is None and self.path.suffix == ".sqlite3"
                and legacy_path.exists()
                and (not self.path.exists() or self.path.stat().st_size == 0)
            ):
                raise ValueError("Existing JSON history or unfinished migration: run eval-workspace budget-migrate first")
            if self.path.exists():
                with self.path.open("rb") as handle:
                    signature = handle.read(16)
                if signature and signature != b"SQLite format 3\x00":
                    raise ValueError("Legacy JSON ledger: run eval-workspace budget-migrate first")
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS campaign_state "
                        "(id INTEGER PRIMARY KEY CHECK (id = 1), document TEXT NOT NULL)"
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO campaign_state VALUES (1, ?)",
                        (json.dumps(self._data),),
                    )
                    self._data = self._read(connection)
                    self._validate(self._data)
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    def _new_data(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "campaign_limit_usd": _money(self.maximum),
            "spent_usd": _money(Decimal()),
            "uncertain_usd": _money(Decimal()),
            "blocked": False, "blocked_reason": None, "blocked_at": None,
            "reservations": {}, "calls": [], "reconciliations": [],
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _read(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute("SELECT document FROM campaign_state WHERE id = 1").fetchone()
        if row is None:
            raise ValueError("Campaign database has no ledger state")
        return json.loads(row[0])

    def _validate(self, data: dict[str, Any]) -> None:
        if data.get("schema_version") != 3:
            raise ValueError("Unsupported budget ledger schema")
        if Decimal(data["campaign_limit_usd"]) != self.maximum:
            raise ValueError("Budget ledger ceiling differs from workspace configuration")
        for key in ("spent_usd", "uncertain_usd"):
            amount = Decimal(data[key])
            if not amount.is_finite() or amount < 0:
                raise ValueError(f"Invalid ledger {key}")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._thread_lock:
            before = json.loads(json.dumps(self._data))
            if not self.persist:
                try:
                    yield
                except BudgetUncertain:
                    raise
                except BaseException:
                    self._data = before
                    raise
                return
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._data = self._read(connection)
                self._validate(self._data)
                self._local.active = True
                try:
                    try:
                        yield
                    except BudgetUncertain:
                        # A known possible charge must survive the raised error.
                        connection.execute("UPDATE campaign_state SET document = ? WHERE id = 1",
                                           (json.dumps(self._data),))
                        connection.commit()
                        raise
                    else:
                        connection.execute("UPDATE campaign_state SET document = ? WHERE id = 1",
                                           (json.dumps(self._data),))
                        connection.commit()
                except BaseException:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                finally:
                    self._local.active = False

    def export(self) -> dict[str, Any]:
        with self._thread_lock:
            if not self.persist or getattr(self._local, "active", False):
                return json.loads(json.dumps(self._data))
            with closing(self._connect()) as connection:
                return self._read(connection)

    @property
    def spent(self) -> Decimal:
        return Decimal(self.export()["spent_usd"])

    @property
    def uncertain(self) -> Decimal:
        return Decimal(self.export()["uncertain_usd"])

    @property
    def pending(self) -> Decimal:
        return self._pending(self.export())

    @staticmethod
    def _pending(data: dict[str, Any]) -> Decimal:
        return sum((Decimal(item["amount_usd"]) for item in data["reservations"].values()), Decimal())

    @property
    def remaining(self) -> Decimal:
        data = self.export()
        return self.maximum - Decimal(data["spent_usd"]) - Decimal(data["uncertain_usd"]) - self._pending(data)

    @property
    def blocked(self) -> bool:
        return bool(self.export()["blocked"])

    @property
    def blocked_reason(self) -> str | None:
        return self.export()["blocked_reason"]

    def _ensure_active_locked(self) -> None:
        if self.blocked:
            raise CampaignBlocked(
                "Campaign ledger is blocked pending billing reconciliation: "
                f"{self.blocked_reason or 'reason not recorded'}"
            )

    def reserve(self, amount: Decimal, call: dict[str, Any]) -> Reservation:
        if not amount.is_finite() or amount <= 0:
            raise ValueError("Reservation must be positive")
        amount = Decimal(_money(amount))
        with self._transaction():
            self._ensure_active_locked()
            if amount > self.remaining:
                raise BudgetExceeded(
                    f"Next request needs up to ${amount:.6f}, but only ${self.remaining:.6f} remains"
                )
            reservation = Reservation(uuid.uuid4().hex, amount)
            self._data["reservations"][reservation.id] = {
                "amount_usd": _money(amount),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "call": call,
            }
            return reservation

    def settle(self, reservation: Reservation, actual: Decimal, call: dict[str, Any]) -> None:
        if not actual.is_finite() or actual < 0:
            raise ValueError("Actual cost cannot be negative")
        with self._transaction():
            stored = self._data["reservations"].pop(reservation.id, None)
            if stored is None:
                raise ValueError("Unknown or already settled reservation")
            if actual > reservation.amount:
                self._mark_uncertain_locked(
                    stored,
                    reservation,
                    call,
                    reason="actual exceeded reservation",
                    amount=actual,
                )
                raise BudgetUncertain("Actual charge exceeded its reservation; campaign is blocked")
            self._data["spent_usd"] = _money(self.spent + actual)
            self._data["calls"].append(
                {
                    **call,
                    "status": "settled",
                    "reserved_usd": _money(reservation.amount),
                    "actual_usd": _money(actual),
                    "settled_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def settle_and_block(
        self,
        reservation: Reservation,
        actual: Decimal,
        call: dict[str, Any],
        reason: str,
    ) -> None:
        if not actual.is_finite() or actual < 0:
            raise ValueError("Actual cost cannot be negative")
        with self._transaction():
            stored = self._data["reservations"].pop(reservation.id, None)
            if stored is None:
                raise ValueError("Unknown or already settled reservation")
            if actual > reservation.amount:
                self._mark_uncertain_locked(
                    stored,
                    reservation,
                    call,
                    reason=f"{reason}; actual exceeded reservation",
                    amount=actual,
                )
                return
            self._data["spent_usd"] = _money(self.spent + actual)
            self._data["calls"].append(
                {
                    **call,
                    "status": "settled_blocked",
                    "reserved_usd": _money(reservation.amount),
                    "actual_usd": _money(actual),
                    "reason": reason,
                    "settled_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._block_locked(reason)

    def mark_uncertain(
        self,
        reservation: Reservation,
        call: dict[str, Any],
        reason: str,
        *,
        amount: Decimal | None = None,
    ) -> None:
        with self._transaction():
            stored = self._data["reservations"].pop(reservation.id, None)
            if stored is None:
                raise ValueError("Unknown or already reconciled reservation")
            self._mark_uncertain_locked(
                stored,
                reservation,
                call,
                reason=reason,
                amount=reservation.amount if amount is None else amount,
            )

    def _block_locked(self, reason: str) -> None:
        self._data["blocked"] = True
        self._data["blocked_reason"] = reason
        self._data["blocked_at"] = datetime.now(timezone.utc).isoformat()

    def _mark_uncertain_locked(
        self,
        stored: dict[str, Any],
        reservation: Reservation,
        call: dict[str, Any],
        *,
        reason: str,
        amount: Decimal,
    ) -> None:
        if not amount.is_finite() or amount < 0:
            raise ValueError("Uncertain charge cannot be negative")
        self._data["uncertain_usd"] = _money(self.uncertain + amount)
        self._data["calls"].append(
            {
                **call,
                "status": "uncertain",
                "reserved_usd": stored["amount_usd"],
                "uncertain_usd": _money(amount),
                "reason": reason,
                "marked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._block_locked(reason)

    def reconcile(self, confirmed_uncertain: Decimal, note: str) -> None:
        if not confirmed_uncertain.is_finite() or confirmed_uncertain < 0:
            raise ValueError("Confirmed uncertain charge cannot be negative")
        if not note.strip():
            raise ValueError("A reconciliation note is required")
        with self._transaction():
            if self._data["reservations"]:
                raise CampaignBlocked("Cannot reconcile while a request reservation is pending")
            if not self.blocked and self.uncertain == 0:
                raise ValueError("Ledger is not blocked and has no uncertain charges")
            previous_uncertain = self.uncertain
            self._data["spent_usd"] = _money(self.spent + confirmed_uncertain)
            self._data["uncertain_usd"] = _money(Decimal())
            self._data["blocked"] = False
            self._data["blocked_reason"] = None
            self._data["blocked_at"] = None
            self._data["reconciliations"].append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "previous_uncertain_usd": _money(previous_uncertain),
                    "confirmed_uncertain_usd": _money(confirmed_uncertain),
                    "note": note.strip(),
                }
            )

    def recover_pending(self, reservation_id: str, note: str) -> None:
        """Explicitly mark an abandoned reservation uncertain; never releases money."""
        if not note.strip():
            raise ValueError("A recovery note is required")
        with self._transaction():
            stored = self._data["reservations"].pop(reservation_id, None)
            if stored is None:
                raise ValueError("Unknown or already settled reservation")
            amount = Decimal(stored["amount_usd"])
            self._mark_uncertain_locked(
                stored, Reservation(reservation_id, amount), stored.get("call", {}),
                reason=f"Abandoned request: {note}", amount=amount,
            )

    def status(self) -> dict[str, Any]:
        data = self.export()
        pending = self._pending(data)
        return {
            "campaign_limit_usd": _money(self.maximum),
            "spent_usd": data["spent_usd"], "uncertain_usd": data["uncertain_usd"],
            "pending_usd": _money(pending),
            "remaining_usd": _money(self.maximum - Decimal(data["spent_usd"]) - Decimal(data["uncertain_usd"]) - pending),
            "blocked": data["blocked"], "blocked_reason": data["blocked_reason"],
            "pending_reservations": data["reservations"],
        }


def migrate_json_ledger(source: Path, destination: Path) -> BudgetLedger:
    """Preserve history, fence legacy writers, then atomically install SQLite state."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("Source and destination ledger paths must differ")
    with CampaignLock(source):
        raw = source.read_bytes()
        data = json.loads(raw)
        backup = source.with_name(source.name + ".pre-sqlite-migration")
        if data.get("schema_version") == "migrated-to-sqlite":
            if data["destination"] != str(destination):
                raise ValueError("Legacy ledger was migrated to a different destination")
            raw = backup.read_bytes()
            data = json.loads(raw)
            if destination.exists():
                ledger = BudgetLedger(destination, Decimal(data["campaign_limit_usd"]))
                if ledger.export().get("legacy_import", {}).get("sha256") != hashlib.sha256(raw).hexdigest():
                    raise ValueError("Destination does not match the original ledger")
                return ledger
        elif destination.exists():
            raise ValueError("Destination already exists; refusing to replace campaign history")
        if data.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported legacy ledger schema")
        if data.get("reservations"):
            raise CampaignBlocked("Settle or reconcile old pending requests before migration")
        if backup.exists() and backup.read_bytes() != raw:
            raise ValueError("Migration backup differs from the source ledger")
        if not backup.exists():
            with backup.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        # Legacy versions reject this marker rather than spend against stale balances.
        write_json(source, {
            "schema_version": "migrated-to-sqlite", "blocked": True,
            "destination": str(destination), "backup": str(backup),
        })
        data.update({
            "schema_version": 3,
            "legacy_import": {"sha256": hashlib.sha256(raw).hexdigest(), "backup": str(backup)},
        })
        data.setdefault("blocked", Decimal(data.get("uncertain_usd", "0")) > 0)
        if Decimal(data.get("uncertain_usd", "0")) > 0:
            data["blocked"] = True
        data.setdefault("blocked_reason", None)
        data.setdefault("blocked_at", None)
        data.setdefault("reconciliations", [])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        BudgetLedger(temporary, Decimal(data["campaign_limit_usd"]), _initial_data=data)
        os.replace(temporary, destination)
        return BudgetLedger(destination, Decimal(data["campaign_limit_usd"]))
