from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class BudgetUncertain(RuntimeError):
    pass


class CampaignBlocked(BudgetExceeded):
    pass


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000001")), "f")


@dataclass(frozen=True)
class Reservation:
    id: str
    amount: Decimal


class CampaignLock:
    """Prevent two local processes from spending against the same ledger at once."""

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
    """A process-safe-enough persistent ledger for sequential V1 runs.

    It writes a reservation before each model generation. If a request fails after
    submission, the reservation becomes uncertain and remains charged against the
    campaign ceiling until a person reconciles the provider bill.
    """

    def __init__(self, path: Path, maximum: Decimal, *, persist: bool = True) -> None:
        if maximum <= 0:
            raise ValueError("Budget maximum must be positive")
        self.path = path
        self.maximum = maximum
        self.persist = persist
        self._lock = threading.Lock()
        self._data = self._load_or_create()

    def _new_data(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "campaign_limit_usd": _money(self.maximum),
            "spent_usd": _money(Decimal()),
            "uncertain_usd": _money(Decimal()),
            "blocked": False,
            "blocked_reason": None,
            "blocked_at": None,
            "reservations": {},
            "calls": [],
            "reconciliations": [],
        }

    def _load_or_create(self) -> dict[str, Any]:
        if self.persist and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            schema = data.get("schema_version")
            if schema not in {1, 2}:
                raise ValueError("Unsupported budget ledger schema")
            if Decimal(data["campaign_limit_usd"]) != self.maximum:
                raise ValueError(
                    "Budget ledger ceiling differs from config; reconcile or use a new ledger path"
                )
            changed = False
            if schema == 1:
                data["schema_version"] = 2
                data.setdefault("blocked", False)
                data.setdefault("blocked_reason", None)
                data.setdefault("blocked_at", None)
                data.setdefault("reconciliations", [])
                changed = True
            reservations = data.get("reservations", {})
            if reservations:
                recovered = sum(
                    (Decimal(item["amount_usd"]) for item in reservations.values()), Decimal()
                )
                data["uncertain_usd"] = _money(Decimal(data["uncertain_usd"]) + recovered)
                for reservation_id, item in reservations.items():
                    data["calls"].append(
                        {
                            **item.get("call", {}),
                            "status": "recovered_pending_as_uncertain",
                            "reservation_id": reservation_id,
                            "reserved_usd": item["amount_usd"],
                            "marked_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                data["reservations"] = {}
                data["blocked"] = True
                data["blocked_reason"] = "Unsettled reservation recovered after process exit"
                data["blocked_at"] = datetime.now(timezone.utc).isoformat()
                changed = True
            if Decimal(data.get("uncertain_usd", "0")) > 0 and not data.get("blocked"):
                data["blocked"] = True
                data["blocked_reason"] = "Ledger contains unreconciled uncertain charges"
                data["blocked_at"] = datetime.now(timezone.utc).isoformat()
                changed = True
            if changed:
                self._write(data)
            return data
        data = self._new_data()
        if self.persist:
            self._write(data)
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.path)

    @property
    def spent(self) -> Decimal:
        return Decimal(self._data["spent_usd"])

    @property
    def uncertain(self) -> Decimal:
        return Decimal(self._data["uncertain_usd"])

    @property
    def pending(self) -> Decimal:
        return sum(
            (Decimal(item["amount_usd"]) for item in self._data["reservations"].values()),
            Decimal(),
        )

    @property
    def remaining(self) -> Decimal:
        return self.maximum - self.spent - self.uncertain - self.pending

    @property
    def blocked(self) -> bool:
        return bool(self._data.get("blocked", False))

    @property
    def blocked_reason(self) -> str | None:
        value = self._data.get("blocked_reason")
        return str(value) if value else None

    def _ensure_active_locked(self) -> None:
        if self.blocked:
            raise CampaignBlocked(
                "Campaign ledger is blocked pending billing reconciliation: "
                f"{self.blocked_reason or 'reason not recorded'}"
            )
        if self._data["reservations"]:
            raise CampaignBlocked("Campaign ledger already has an unsettled request")

    def reserve(self, amount: Decimal, call: dict[str, Any]) -> Reservation:
        if amount <= 0:
            raise ValueError("Reservation must be positive")
        with self._lock:
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
            if self.persist:
                self._write(self._data)
            return reservation

    def settle(self, reservation: Reservation, actual: Decimal, call: dict[str, Any]) -> None:
        if actual < 0:
            raise ValueError("Actual cost cannot be negative")
        with self._lock:
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
            if self.persist:
                self._write(self._data)

    def settle_and_block(
        self,
        reservation: Reservation,
        actual: Decimal,
        call: dict[str, Any],
        reason: str,
    ) -> None:
        if actual < 0:
            raise ValueError("Actual cost cannot be negative")
        with self._lock:
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
            if self.persist:
                self._write(self._data)

    def mark_uncertain(
        self,
        reservation: Reservation,
        call: dict[str, Any],
        reason: str,
        *,
        amount: Decimal | None = None,
    ) -> None:
        with self._lock:
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
        if amount < 0:
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
        if self.persist:
            self._write(self._data)

    def reconcile(self, confirmed_uncertain: Decimal, note: str) -> None:
        if confirmed_uncertain < 0:
            raise ValueError("Confirmed uncertain charge cannot be negative")
        if not note.strip():
            raise ValueError("A reconciliation note is required")
        with self._lock:
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
            if self.persist:
                self._write(self._data)

    def status(self) -> dict[str, Any]:
        return {
            "campaign_limit_usd": _money(self.maximum),
            "spent_usd": _money(self.spent),
            "uncertain_usd": _money(self.uncertain),
            "pending_usd": _money(self.pending),
            "remaining_usd": _money(self.remaining),
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
        }

