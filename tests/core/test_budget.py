from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from eval_core.budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetUncertain,
    CampaignBlocked,
    CampaignLock,
)


ROOT = Path(__file__).resolve().parents[2]
TEST_TMP = ROOT / ".test-tmp"


class BudgetTests(unittest.TestCase):
    def test_reserve_settle_and_persist(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, Decimal("1.00"))
            reservation = ledger.reserve(Decimal("0.20"), {"stage": "encoder"})
            ledger.settle(reservation, Decimal("0.12"), {"stage": "encoder"})
            self.assertEqual(ledger.spent, Decimal("0.120000000"))
            reloaded = BudgetLedger(path, Decimal("1.00"))
            self.assertEqual(reloaded.spent, Decimal("0.120000000"))
            saved = reloaded.export()
            self.assertEqual(saved["calls"][0]["status"], "settled")

    def test_budget_refuses_overspend(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", Decimal("0.10"))
            with self.assertRaises(BudgetExceeded):
                ledger.reserve(Decimal("0.11"), {"stage": "judge"})

    def test_uncertain_reservation_remains_charged(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json", Decimal("1.00"))
            reservation = ledger.reserve(Decimal("0.25"), {"stage": "decoder"})
            ledger.mark_uncertain(reservation, {"stage": "decoder"}, "timeout")
            self.assertEqual(ledger.uncertain, Decimal("0.250000000"))
            self.assertEqual(ledger.remaining, Decimal("0.750000000"))
            self.assertTrue(ledger.blocked)
            with self.assertRaises(CampaignBlocked):
                ledger.reserve(Decimal("0.01"), {"stage": "encoder"})

    def test_actual_over_reservation_is_retained_and_blocks_reload(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, Decimal("5.00"))
            reservation = ledger.reserve(Decimal("0.10"), {"stage": "judge"})
            with self.assertRaises(BudgetUncertain):
                ledger.settle(reservation, Decimal("2.00"), {"stage": "judge"})
            self.assertEqual(ledger.uncertain, Decimal("2.000000000"))
            reloaded = BudgetLedger(path, Decimal("5.00"))
            self.assertTrue(reloaded.blocked)
            with self.assertRaises(CampaignBlocked):
                reloaded.reserve(Decimal("0.01"), {"stage": "encoder"})

    def test_pending_request_survives_other_clients_until_explicit_recovery(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, Decimal("1.00"))
            reservation = ledger.reserve(Decimal("0.20"), {"stage": "decoder"})
            reloaded = BudgetLedger(path, Decimal("1.00"))
            self.assertEqual(reloaded.pending, Decimal("0.20"))
            self.assertFalse(reloaded.blocked)
            self.assertEqual(reloaded.remaining, Decimal("0.80"))
            reloaded.recover_pending(reservation.id, "Verified the original process exited")
            self.assertEqual(reloaded.pending, Decimal("0"))
            self.assertEqual(reloaded.uncertain, Decimal("0.200000000"))
            self.assertTrue(reloaded.blocked)

    def test_reconciliation_moves_confirmed_charge_to_spent_and_unblocks(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, Decimal("1.00"))
            reservation = ledger.reserve(Decimal("0.20"), {"stage": "decoder"})
            ledger.mark_uncertain(reservation, {"stage": "decoder"}, "timeout")
            ledger.reconcile(Decimal("0.07"), "Provider bill reviewed")
            self.assertFalse(ledger.blocked)
            self.assertEqual(ledger.uncertain, Decimal("0.000000000"))
            self.assertEqual(ledger.spent, Decimal("0.070000000"))
            ledger.reserve(Decimal("0.10"), {"stage": "encoder"})

    def test_campaign_lock_rejects_second_process_handle(self) -> None:
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / "ledger.json"
            with CampaignLock(path):
                with self.assertRaises(BudgetExceeded):
                    with CampaignLock(path):
                        self.fail("Second campaign lock unexpectedly succeeded")


if __name__ == "__main__":
    unittest.main()

