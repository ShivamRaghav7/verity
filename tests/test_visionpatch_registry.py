"""Exact-state, approval and transaction regressions for the local workbench."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from visionpatch.editor import fit_projection_patch, projection_sha256
    from visionpatch.registry import PatchRegistry, RegistryConflictError, RegistryIntegrityError


@unittest.skipIf(torch is None, "Optional visual registry tests require PyTorch")
class PatchRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.db_path = self.directory / "workbench.sqlite3"
        self.base = torch.eye(3, dtype=torch.float32)
        self.registry = PatchRegistry(self.db_path, self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def fit(self, weight=None, column=0, value=2.0):
        if weight is None:
            weight = self.registry.get_current_weight()
        key = torch.eye(3)[column:column + 1]
        target = key @ weight.T
        target[0, column] = value
        return fit_projection_patch(weight, key, target, torch.eye(3),
                                    metadata={"label": f"column-{column}"})

    def propose(self, patch=None):
        return self.registry.propose(patch or self.fit(), actor="reviewer", reason="Test proposal")

    def activate(self, record):
        patch_id = record["patch_id"]
        self.registry.approve(patch_id, actor="admin", reason="Reviewed development evidence")
        return self.registry.activate(patch_id, actor="admin", reason="Activate reviewed correction")

    def test_candidate_preview_and_approval_never_change_live_state(self):
        before = self.registry.state()
        candidate = self.propose()
        preview = self.registry.load_patch(candidate["patch_id"]).edited_weight()
        self.assertNotEqual(projection_sha256(preview), before["current_sha256"])
        self.assertEqual(self.registry.state(), before)
        with self.assertRaisesRegex(RegistryConflictError, "approved"):
            self.registry.activate(candidate["patch_id"], actor="admin", reason="Unapproved attempt")
        self.registry.approve(candidate["patch_id"], actor="admin", reason="Approve only")
        self.assertEqual(self.registry.state(), before)
        self.assertEqual(self.registry.get_patch(candidate["patch_id"])["status"], "approved")

    def test_restart_persists_active_chain_and_exact_rollback(self):
        candidate = self.propose()
        edited = self.activate(candidate)
        restarted = PatchRegistry(self.db_path, self.base)
        self.assertTrue(torch.equal(restarted.get_current_weight(), edited))
        self.assertEqual(restarted.state()["active_patch_ids"], [candidate["patch_id"]])
        restored = restarted.rollback(actor="admin", reason="Restore exact original")
        self.assertTrue(torch.equal(restored, self.base))
        self.assertEqual(restarted.state()["revision"], 2)
        self.assertEqual(restarted.get_patch(candidate["patch_id"])["status"], "approved")
        self.assertEqual([e["action"] for e in restarted.events()],
                         ["initialize", "propose", "approve", "activate", "rollback"])

    def test_two_sequential_edits_restore_parent_then_base_exactly(self):
        first = self.propose()
        first_weight = self.activate(first)
        second = self.propose(self.fit(column=1, value=3.0))
        self.assertEqual(second["parent_patch_id"], first["patch_id"])
        self.activate(second)
        self.assertEqual(self.registry.state()["active_patch_ids"],
                         [first["patch_id"], second["patch_id"]])
        self.assertTrue(torch.equal(self.registry.rollback(actor="admin", reason="Pop second"),
                                    first_weight))
        self.assertTrue(torch.equal(self.registry.rollback(actor="admin", reason="Pop first"),
                                    self.base))
        with self.assertRaisesRegex(RegistryConflictError, "no active"):
            self.registry.rollback(actor="admin", reason="Empty rollback")

    def test_ancestor_restore_rejects_nonancestor_and_can_restore_base(self):
        first = self.propose()
        first_weight = self.activate(first)
        self.activate(self.propose(self.fit(column=1, value=3.0)))
        self.activate(self.propose(self.fit(column=2, value=4.0)))
        state = self.registry.state()
        with self.assertRaisesRegex(RegistryConflictError, "ancestor"):
            self.registry.rollback(actor="admin", reason="Bad target", to_patch_id="unknown")
        self.assertEqual(state, self.registry.state())
        restored = self.registry.rollback(actor="admin", reason="Keep first only",
                                          to_patch_id=first["patch_id"])
        self.assertTrue(torch.equal(restored, first_weight))
        self.assertEqual(self.registry.state()["active_patch_ids"], [first["patch_id"]])
        self.assertTrue(torch.equal(self.registry.rollback(actor="admin", reason="Reset all",
                                                          to_patch_id="base"), self.base))

    def test_stale_sibling_conflict_has_no_database_or_tensor_mutation(self):
        first = self.propose(self.fit(value=2.0))
        stale = self.propose(self.fit(value=3.0))
        self.activate(first)
        self.registry.approve(stale["patch_id"], actor="admin", reason="Approve branch evidence")
        before, events = self.registry.state(), self.registry.events()
        with self.assertRaisesRegex(RegistryConflictError, "stale patch parent"):
            self.registry.activate(stale["patch_id"], actor="admin", reason="Reject stale parent")
        self.assertEqual(self.registry.state(), before)
        self.assertEqual(self.registry.events(), events)
        self.registry.rollback(actor="admin", reason="Restore branch parent")
        self.registry.activate(stale["patch_id"], actor="admin", reason="Now exact parent matches")
        self.assertEqual(self.registry.state()["current_sha256"], stale["after_sha256"])

    def test_concurrent_same_revision_activations_have_one_winner(self):
        records = [self.propose(self.fit(value=value)) for value in (2.0, 3.0)]
        for record in records:
            self.registry.approve(record["patch_id"], actor="admin", reason="Reviewed independently")

        def attempt(record):
            try:
                self.registry.activate(record["patch_id"], actor="admin", reason="Concurrent request",
                                       expected_revision=0)
                return "activated"
            except RegistryConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, records))
        self.assertCountEqual(outcomes, ["activated", "conflict"])
        self.assertEqual(self.registry.state()["revision"], 1)
        self.assertEqual(len(self.registry.state()["active_patch_ids"]), 1)
        self.assertEqual(sum(e["action"] == "activate" for e in self.registry.events()), 1)

    def test_wrong_base_unknown_source_and_artifact_tampering_fail_closed(self):
        wrong_base = self.base.clone()
        wrong_base[0, 0] = 9
        with self.assertRaisesRegex(RegistryIntegrityError, "base projection"):
            PatchRegistry(self.db_path, wrong_base)
        before = self.registry.state()
        with self.assertRaisesRegex(RegistryIntegrityError, "not a known"):
            self.propose(self.fit(weight=wrong_base))
        self.assertEqual(self.registry.state(), before)
        record = self.propose()
        self.registry.approve(record["patch_id"], actor="admin", reason="Reviewed artifact")
        path = self.registry.artifacts_dir / f"{record['patch_id']}.pt"
        payload = torch.load(path, weights_only=True)
        payload["metadata_json"] = '{"unreviewed": true}'
        torch.save(payload, path)
        with self.assertRaisesRegex(RegistryIntegrityError, "file hash"):
            self.registry.activate(record["patch_id"], actor="admin", reason="Tampered artifact")
        self.assertEqual(self.registry.state(), before)

    def test_base_artifact_tampering_is_detected_on_read_and_restart(self):
        torch.save(self.base + 1, self.registry.artifacts_dir / "base.pt")
        with self.assertRaises(RegistryIntegrityError):
            self.registry.get_current_weight()
        with self.assertRaises(RegistryIntegrityError):
            PatchRegistry(self.db_path, self.base)

    def test_database_failure_rolls_back_chain_revision_and_audit(self):
        record = self.propose()
        self.registry.approve(record["patch_id"], actor="admin", reason="Reviewed artifact")
        before, events = self.registry.state(), self.registry.events()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("CREATE TRIGGER reject_activation BEFORE INSERT ON audit_events "
                               "WHEN NEW.action = 'activate' BEGIN "
                               "SELECT RAISE(ABORT, 'injected storage failure'); END")
            connection.commit()
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected storage failure"):
            self.registry.activate(record["patch_id"], actor="admin", reason="Storage fault test")
        self.assertEqual(self.registry.state(), before)
        self.assertEqual(self.registry.events(), events)
        self.assertTrue(torch.equal(self.registry.get_current_weight(), self.base))

    def test_rollback_storage_failure_preserves_active_state(self):
        record = self.propose()
        expected = self.activate(record)
        before, events = self.registry.state(), self.registry.events()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("CREATE TRIGGER reject_rollback BEFORE INSERT ON audit_events "
                               "WHEN NEW.action = 'rollback' BEGIN "
                               "SELECT RAISE(ABORT, 'injected rollback failure'); END")
            connection.commit()
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected rollback failure"):
            self.registry.rollback(actor="admin", reason="Storage fault test")
        self.assertEqual(self.registry.state(), before)
        self.assertEqual(self.registry.events(), events)
        self.assertTrue(torch.equal(self.registry.get_current_weight(), expected))

    def test_active_artifact_corruption_blocks_rollback_and_preserves_sql_state(self):
        record = self.propose()
        self.activate(record)
        before, events = self.registry.state(), self.registry.events()
        path = self.registry.artifacts_dir / f"{record['patch_id']}.pt"
        path.write_bytes(b"corrupt")
        with self.assertRaises(RegistryIntegrityError):
            self.registry.rollback(actor="admin", reason="Corrupt rollback must fail closed")
        self.assertEqual(self.registry.events(), events)
        with closing(sqlite3.connect(self.db_path)) as connection:
            current, revision = connection.execute(
                "SELECT current_sha256, revision FROM registry_state").fetchone()
            active = connection.execute("SELECT patch_id FROM active_chain").fetchall()
        self.assertEqual((current, revision), (before["current_sha256"], before["revision"]))
        self.assertEqual(active, [(record["patch_id"],)])

    def test_withdrawal_requires_rollback_and_keeps_history(self):
        record = self.propose()
        self.activate(record)
        with self.assertRaisesRegex(RegistryConflictError, "roll back"):
            self.registry.withdraw(record["patch_id"], actor="admin", reason="Cannot remove active")
        self.registry.rollback(actor="admin", reason="Stop using patch")
        withdrawn = self.registry.withdraw(record["patch_id"], actor="admin", reason="Retired patch")
        self.assertEqual(withdrawn["status"], "withdrawn")
        self.assertTrue((self.registry.artifacts_dir / f"{record['patch_id']}.pt").exists())
        with self.assertRaises(RegistryConflictError):
            self.registry.activate(record["patch_id"], actor="admin", reason="Retired patch blocked")
        with self.assertRaises(RegistryConflictError):
            self.registry.approve(record["patch_id"], actor="admin", reason="Retired patch blocked")

    def test_audit_is_append_only_and_revision_rollback_is_optimistic(self):
        self.activate(self.propose())
        before = self.registry.state()
        with self.assertRaisesRegex(RegistryConflictError, "revision changed"):
            self.registry.rollback(actor="admin", reason="Stale browser", expected_revision=0)
        self.assertEqual(self.registry.state(), before)
        with closing(sqlite3.connect(self.db_path)) as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute("DELETE FROM audit_events")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute("UPDATE audit_events SET actor = 'rewritten'")

    def test_returned_tensors_are_independent_and_metadata_is_validated(self):
        tensor = self.registry.get_current_weight()
        tensor[0, 0] = 123
        self.assertTrue(torch.equal(self.registry.get_current_weight(), self.base))
        for kwargs in ({"actor": "", "reason": "reason"},
                       {"actor": "admin", "reason": " "},
                       {"actor": "admin", "reason": "reason", "metadata": {"x": float("nan")}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.registry.propose(self.fit(), **kwargs)
        self.assertEqual(self.registry.list_patches(), [])


if __name__ == "__main__":
    unittest.main()
