"""Regression checks for authority, temporal compatibility, conflicts and rollback."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from editguard.models import CorrectionCreate, Fact
from editguard.store import DomainError, Store


FACT = "nimbus-basic-v2-devices"
DATE = "2026-10-11"


class StoreTests(unittest.TestCase):
    def setUp(self):
        # Windows app sandboxes can expose a temp path SQLite cannot write to.
        # Keep disposable database files inside the writable checkout instead.
        self.directory = tempfile.TemporaryDirectory(prefix="editguard-test-", dir=Path(__file__).resolve().parent)
        self.path = Path(self.directory.name) / "editguard.sqlite3"
        self.store = Store(self.path)
        self.store.seed()

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def correction(self, **overrides):
        return CorrectionCreate(**{
            "fact_id": FACT, "original_value": "5", "corrected_value": "3",
            "source": "Fictional handbook amendment EG-001",
            "reason": "Reviewed reduction to Basic device capacity",
            "effective_from": DATE, **overrides,
        })

    def approved(self, **overrides):
        edit = self.store.create_correction(self.correction(**overrides))
        return self.store.approve(edit["edit_id"])

    def value(self, fact_id=FACT, as_of=DATE):
        return self.store.resolve_fact(fact_id, as_of)["value"]

    def test_seed_is_idempotent_and_does_not_restore_withdrawn_edits(self):
        edit = self.approved()
        self.store.withdraw(edit["edit_id"])
        revision = self.store.revision()
        self.store.seed()
        self.assertEqual(self.store.revision(), revision)
        self.assertEqual(len(self.store.facts()), 10)
        self.assertEqual(self.value(), "5")
        self.assertEqual(self.store.edits()[0]["status"], "withdrawn")

    def test_pending_edit_has_no_authority_and_approval_is_required(self):
        edit = self.store.create_correction(self.correction())
        self.assertEqual(self.value(), "5")
        self.assertEqual(self.store.resolve_fact(FACT, DATE)["edits"], [])
        self.store.approve(edit["edit_id"])
        self.assertEqual(self.value(), "3")
        self.assertEqual(self.store.resolve_fact(FACT, DATE)["edits"][0]["source"], edit["source"])

    def test_scope_is_copied_from_fact_and_other_scopes_are_retained(self):
        edit = self.approved()
        self.assertEqual((edit["entity_id"], edit["relation"], edit["plan"], edit["version"]),
                         ("nimbus", "device_limit", "Basic", "v2"))
        for fact_id, expected in [
            ("nimbus-basic-v2-price", "9"), ("nimbus-pro-v2-devices", "12"),
            ("nimbus-basic-v1-devices", "5"), ("nimbus-mini-basic-v2-devices", "2"),
        ]:
            with self.subTest(fact_id=fact_id):
                self.assertEqual(self.value(fact_id), expected)

    def test_effective_dates_are_half_open(self):
        self.approved(effective_to="2026-11-01")
        self.assertEqual(self.value(as_of="2026-10-10"), "5")
        self.assertEqual(self.value(as_of=DATE), "3")
        self.assertEqual(self.value(as_of="2026-10-31"), "3")
        self.assertEqual(self.value(as_of="2026-11-01"), "5")

    def test_supersession_preserves_history_and_rollback_restores_predecessor(self):
        first = self.approved()
        second = self.approved(original_value="3", corrected_value="2", effective_from="2026-11-01", supersedes=first["edit_id"])
        self.assertEqual(self.value(as_of="2026-10-10"), "5")
        self.assertEqual(self.value(as_of="2026-10-31"), "3")
        self.assertEqual(self.value(as_of="2026-11-01"), "2")
        with self.assertRaises(DomainError) as blocked:
            self.store.withdraw(first["edit_id"])
        self.assertEqual(blocked.exception.code, 409)
        self.store.withdraw(second["edit_id"])
        self.assertEqual(self.value(as_of="2026-11-01"), "3")
        self.store.withdraw(first["edit_id"])
        self.assertEqual(self.value(as_of="2026-11-01"), "5")
        self.assertEqual([event["event_type"] for event in self.store.events()][-2:],
                         ["correction_withdrawn", "correction_withdrawn"])

    def test_temporary_successor_restores_parent_after_its_exclusive_end(self):
        first = self.approved()
        self.approved(original_value="3", corrected_value="2", effective_from="2026-11-01",
                      effective_to="2026-12-01", supersedes=first["edit_id"])
        self.assertEqual(self.value(as_of="2026-11-01"), "2")
        self.assertEqual(self.value(as_of="2026-12-01"), "3")

    def test_unapproved_successor_does_not_change_predecessor(self):
        first = self.approved()
        pending = self.store.create_correction(self.correction(original_value="3", corrected_value="2",
                            effective_from="2026-11-01", supersedes=first["edit_id"]))
        self.assertEqual(self.value(as_of="2026-11-01"), "3")
        self.assertEqual(self.store.conflicts(), [])
        with self.assertRaises(DomainError):
            self.store.withdraw(first["edit_id"])
        self.store.withdraw(pending["edit_id"])
        self.store.withdraw(first["edit_id"])
        self.assertEqual(self.value(as_of="2026-11-01"), "5")

    def test_competing_drafts_block_only_affected_scope_and_overlapping_date(self):
        one = self.store.create_correction(self.correction(effective_to="2026-11-01"))
        two = self.store.create_correction(self.correction(corrected_value="4", effective_from="2026-10-20"))
        conflicts = self.store.conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0]["edit_ids"]), {one["edit_id"], two["edit_id"]})
        self.assertEqual(conflicts[0]["effective_from"], "2026-10-20")
        self.assertEqual(conflicts[0]["effective_to"], "2026-11-01")
        self.assertEqual(self.value(as_of=DATE), "5")
        self.assertIsNone(self.value(as_of="2026-10-20"))
        self.assertEqual(self.value(as_of="2026-11-01"), "5")
        self.assertEqual(self.value("nimbus-pro-v2-devices", "2026-10-20"), "12")
        with self.assertRaises(DomainError) as refused:
            self.store.approve(one["edit_id"])
        self.assertEqual(refused.exception.code, 409)
        self.assertTrue(all(edit["status"] == "draft" for edit in self.store.edits()))
        self.assertEqual(self.store.events()[-1]["event_type"], "approval_rejected")
        self.store.withdraw(two["edit_id"])
        self.store.approve(one["edit_id"])
        self.assertEqual(self.value(as_of="2026-10-20"), "3")

    def test_competing_draft_blocks_approved_authority_until_resolved(self):
        first = self.approved()
        competitor = self.store.create_correction(self.correction(original_value="3", corrected_value="4"))
        resolved = self.store.resolve_fact(FACT, DATE)
        self.assertIsNone(resolved["value"])
        self.assertEqual(resolved["edits"][0]["edit_id"], first["edit_id"])
        with self.assertRaises(DomainError):
            self.store.approve(competitor["edit_id"])
        self.store.withdraw(competitor["edit_id"])
        self.assertEqual(self.value(), "3")

    def test_disjoint_drafts_are_not_conflicts(self):
        first = self.store.create_correction(self.correction(effective_to="2026-11-01"))
        second = self.store.create_correction(self.correction(corrected_value="4", effective_from="2026-11-01"))
        self.assertEqual(self.store.conflicts(), [])
        self.store.approve(first["edit_id"])
        self.store.approve(second["edit_id"])
        self.assertEqual(self.value(as_of="2026-10-31"), "3")
        self.assertEqual(self.value(as_of="2026-11-01"), "4")

    def test_wrong_original_is_rejected_without_partial_mutation(self):
        revision = self.store.revision()
        with self.assertRaises(DomainError):
            self.store.create_correction(self.correction(original_value="12"))
        self.assertEqual(self.store.revision(), revision)
        self.assertEqual(self.store.edits(), [])

    def test_approval_revalidates_original_after_authority_changes(self):
        pending = self.store.create_correction(self.correction())
        first = self.approved(effective_to="2026-10-20")
        # Same endorsed value is not a contradictory conflict, but the old
        # original value is stale and must still be reviewed again.
        with self.assertRaises(DomainError) as refused:
            self.store.approve(pending["edit_id"])
        self.assertEqual(refused.exception.code, 409)
        self.assertEqual(self.store.edits()[0]["status"], "draft")
        self.store.withdraw(first["edit_id"])
        self.store.approve(pending["edit_id"])
        self.assertEqual(self.value(), "3")

    def test_supersession_requires_same_scope_approved_parent_and_later_date(self):
        pending = self.store.create_correction(self.correction())
        with self.assertRaises(DomainError):
            self.store.create_correction(self.correction(original_value="3", corrected_value="2",
                                        effective_from="2026-11-01", supersedes=pending["edit_id"]))
        approved = self.store.approve(pending["edit_id"])
        for changes in [
            {"effective_from": DATE},
            {"fact_id": "nimbus-pro-v2-devices", "original_value": "12"},
            {"fact_id": "nimbus-basic-v1-devices", "original_value": "5"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(DomainError):
                self.store.create_correction(self.correction(**{
                    "original_value": "3", "corrected_value": "2", "effective_from": "2026-11-01",
                    "supersedes": approved["edit_id"], **changes,
                }))

    def test_successor_must_stay_inside_parent_effective_interval(self):
        first = self.approved(effective_to="2026-12-01")
        with self.assertRaises(DomainError):
            self.store.create_correction(self.correction(original_value="3", corrected_value="2",
                                        effective_from="2026-11-01", supersedes=first["edit_id"]))

    def test_only_administrator_can_create_approve_or_withdraw(self):
        with self.assertRaises(DomainError) as refused:
            self.store.create_correction(self.correction(), actor="customer")
        self.assertEqual(refused.exception.code, 403)
        edit = self.store.create_correction(self.correction())
        for action in (self.store.approve, self.store.withdraw):
            with self.assertRaises(DomainError):
                action(edit["edit_id"], actor="customer")
        self.assertEqual(self.value(), "5")

    def test_withdrawal_is_idempotent_and_reapproval_is_forbidden(self):
        edit = self.approved()
        self.store.withdraw(edit["edit_id"])
        revision = self.store.revision()
        self.store.withdraw(edit["edit_id"])
        self.assertEqual(self.store.revision(), revision)
        with self.assertRaises(DomainError):
            self.store.approve(edit["edit_id"])
        self.assertEqual(self.value(), "5")

    def test_parallel_approval_and_withdrawal_preserve_revision_and_history(self):
        edit = self.store.create_correction(self.correction())
        with ThreadPoolExecutor(max_workers=4) as executor:
            approved = list(executor.map(lambda _: self.store.approve(edit["edit_id"]), range(8)))
        self.assertTrue(all(item["status"] == "approved" for item in approved))
        self.assertEqual(sum(event["event_type"] == "correction_approved" for event in self.store.events()), 1)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: self.store.withdraw(edit["edit_id"]), range(8)))
        self.assertEqual(sum(event["event_type"] == "correction_withdrawn" for event in self.store.events()), 1)
        revisions = [event["revision"] for event in self.store.events()]
        self.assertEqual(revisions, list(range(1, self.store.revision() + 1)))
        self.assertEqual(self.value(), "5")

    def test_failed_mutation_rolls_back_status_event_and_revision_together(self):
        edit = self.store.create_correction(self.correction())
        original_event = self.store._event

        def fail_after_event(*args, **kwargs):
            original_event(*args, **kwargs)
            raise RuntimeError("Simulated failure before transaction commit")

        revision = self.store.revision()
        events = self.store.events()
        with patch.object(self.store, "_event", side_effect=fail_after_event), self.assertRaises(RuntimeError):
            self.store.approve(edit["edit_id"])
        self.assertEqual(self.store.edits()[0]["status"], "draft")
        self.assertEqual(self.store.revision(), revision)
        self.assertEqual(self.store.events(), events)
        self.assertEqual(self.value(), "5")
        self.store.approve(edit["edit_id"])
        revision = self.store.revision()
        with patch.object(self.store, "_event", side_effect=fail_after_event), self.assertRaises(RuntimeError):
            self.store.withdraw(edit["edit_id"])
        self.assertEqual(self.store.edits()[0]["status"], "approved")
        self.assertEqual(self.store.revision(), revision)
        self.assertEqual(self.value(), "3")

    def test_snapshot_keeps_state_and_revision_coherent_across_another_writer(self):
        writer = Store(self.path)
        try:
            initial_revision = self.store.revision()

            def publish():
                edit = writer.create_correction(self.correction())
                writer.approve(edit["edit_id"])
                return writer.revision()

            with self.store.snapshot():
                # The first read establishes SQLite's WAL snapshot. A separate
                # connection may commit meanwhile without changing this view.
                self.assertEqual(self.store.revision(), initial_revision)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    committed_revision = executor.submit(publish).result(timeout=5)
                self.assertGreater(committed_revision, initial_revision)
                with self.store.snapshot():
                    exported = {"facts": self.store.facts(), "edits": self.store.edits(),
                                "events": self.store.events(), "revision": self.store.revision()}
                self.assertEqual(exported["revision"], initial_revision)
                self.assertEqual(exported["edits"], [])
                self.assertEqual(exported["events"][-1]["revision"], initial_revision)
                self.assertEqual(self.value(), "5")
            self.assertEqual(self.store.revision(), committed_revision)
            self.assertEqual(self.value(), "3")
            self.assertEqual(self.store.events()[-1]["revision"], committed_revision)
        finally:
            writer.close()

    def test_nested_snapshot_does_not_commit_its_mutation_caller(self):
        original_event = self.store._event
        initial_revision = self.store.revision()

        def inspect_and_fail(*args, **kwargs):
            original_event(*args, **kwargs)
            with self.store.snapshot():
                self.assertEqual(len(self.store.edits()), 1)
                self.assertEqual(self.store.revision(), initial_revision + 1)
            raise RuntimeError("Outer transaction must roll back")

        with patch.object(self.store, "_event", side_effect=inspect_and_fail), self.assertRaises(RuntimeError):
            self.store.create_correction(self.correction())
        self.assertEqual(self.store.edits(), [])
        self.assertEqual(self.store.revision(), initial_revision)

    def test_reopen_preserves_approved_values_and_audit_events(self):
        self.approved()
        events = self.store.events()
        self.store.close()
        self.store = Store(self.path)
        self.store.seed()
        self.assertEqual(self.value(), "3")
        self.assertEqual(self.store.events(), events)

    def test_invalid_dates_numeric_values_and_blank_provenance_rejected(self):
        for changes in [
            {"source": " "}, {"reason": " "}, {"corrected_value": "NaN"},
            {"corrected_value": "-3"}, {"corrected_value": "3 devices"},
            {"effective_from": "20261011"}, {"effective_to": DATE},
            {"corrected_value": "5"}, {"unknown_scope": "Pro"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.correction(**changes)
        for changes in [{"corrected_value": "3.5"}, {"effective_from": "2024-01-01"}]:
            with self.subTest(changes=changes), self.assertRaises(DomainError):
                self.store.create_correction(self.correction(**changes))
        with self.assertRaises(DomainError):
            self.store.resolve_fact(FACT, "2024-12-31")

    def test_fact_validation_and_canonical_values(self):
        fact = self.store.resolve_fact(FACT, DATE)["fact"]
        self.assertEqual(Fact.model_validate({**fact, "value": "05.00"}).value, "5")
        with self.assertRaises(ValidationError):
            Fact.model_validate({**fact, "relation": "support_hours", "value": "25"})


if __name__ == "__main__":
    unittest.main()
