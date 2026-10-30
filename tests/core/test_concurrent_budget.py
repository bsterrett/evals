from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from eval_core.budget import BudgetExceeded, BudgetLedger, CampaignBlocked, migrate_json_ledger

ROOT = Path(__file__).resolve().parents[2]


def reserve_in_process(path, barrier, release, messages, index):
    try:
        ledger = BudgetLedger(Path(path), Decimal("1"))
        barrier.wait(timeout=30)
        try:
            reservation = ledger.reserve(Decimal("0.20"), {"run_id": f"run-{index}"})
        except BudgetExceeded:
            messages.put(("rejected", index))
            return
        messages.put(("reserved", index))
        if not release.wait(timeout=30):
            raise RuntimeError("Timed out waiting for test settlement")
        ledger.settle(reservation, Decimal("0.10"), {"run_id": f"run-{index}"})
    except BaseException as exc:
        messages.put(("error", repr(exc)))
        raise


class ConcurrentBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)

    def test_unmigrated_history_cannot_be_replaced_with_a_zero_balance(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            old = Path(directory) / "campaign.json"
            old.write_text('{"schema_version": 2, "spent_usd": "0.75"}')
            with self.assertRaisesRegex(ValueError, "migration"):
                BudgetLedger(old.with_suffix(".sqlite3"), Decimal("1"))
            self.assertFalse(old.with_suffix(".sqlite3").exists())

    def test_separate_processes_cannot_over_reserve_campaign(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            path = Path(directory) / "campaign.sqlite3"
            ledger = BudgetLedger(path, Decimal("1"))
            context = multiprocessing.get_context("spawn")
            barrier, release, messages = context.Barrier(8), context.Event(), context.Queue()
            workers = [context.Process(target=reserve_in_process, args=(str(path), barrier, release, messages, i)) for i in range(8)]
            try:
                for worker in workers:
                    worker.start()
                outcomes = [messages.get(timeout=40) for _ in workers]
                self.assertEqual(sum(kind == "reserved" for kind, _ in outcomes), 5, outcomes)
                self.assertEqual(sum(kind == "rejected" for kind, _ in outcomes), 3, outcomes)
                self.assertEqual(ledger.pending, Decimal("1"))
                self.assertEqual(ledger.remaining, Decimal("0"))
            finally:
                release.set()
                for worker in workers:
                    if worker.pid is not None:
                        worker.join(timeout=15)
                        if worker.is_alive():
                            worker.terminate()
                            worker.join()
                messages.close()
                messages.join_thread()
            self.assertTrue(all(worker.exitcode == 0 for worker in workers))
            self.assertEqual(ledger.spent, Decimal("0.50"))
            self.assertEqual(ledger.pending, Decimal("0"))
            self.assertEqual(ledger.remaining, Decimal("0.50"))

    def test_blocking_one_run_preserves_other_inflight_reservation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            path = Path(directory) / "campaign.sqlite3"
            first, second = BudgetLedger(path, Decimal("1")), BudgetLedger(path, Decimal("1"))
            a, b = first.reserve(Decimal(".2"), {}), second.reserve(Decimal(".3"), {})
            first.mark_uncertain(a, {}, "timeout")
            with self.assertRaises(CampaignBlocked):
                second.reserve(Decimal(".1"), {})
            with self.assertRaises(CampaignBlocked):
                first.reconcile(Decimal(".1"), "Bill reviewed")
            second.settle(b, Decimal(".15"), {})
            first.reconcile(Decimal(".10"), "Bill reviewed")
            self.assertEqual(second.spent, Decimal(".25"))
            self.assertFalse(second.blocked)

    def test_migration_preserves_history_and_fences_legacy_writers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as directory:
            source, target = Path(directory) / "old.json", Path(directory) / "new.sqlite3"
            data = BudgetLedger(target, Decimal("1"), persist=False).export()
            data.update(schema_version=2, spent_usd="0.250000000")
            data["calls"] = [{"status": "settled", "actual_usd": ".25"}]
            raw = json.dumps(data).encode()
            source.write_bytes(raw)
            ledger = migrate_json_ledger(source, target)
            self.assertEqual(ledger.spent, Decimal(".25"))
            self.assertEqual(ledger.export()["calls"], data["calls"])
            self.assertEqual(source.with_name(source.name + ".pre-sqlite-migration").read_bytes(), raw)
            self.assertEqual(json.loads(source.read_text())["schema_version"], "migrated-to-sqlite")
            with self.assertRaises(ValueError):
                BudgetLedger(source, Decimal("1"))
            self.assertEqual(migrate_json_ledger(source, target).spent, Decimal(".25"))
