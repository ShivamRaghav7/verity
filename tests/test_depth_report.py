"""Depth export contracts using explicit software fixtures, never model measurements."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard_benchmark.data import digest
from editguard_benchmark.metrics import summarize
from scripts import build_depth_report as exporter


class DepthReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="editguard-depth-report-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.receipts = [self.root / f"final-{model}.json" for model in ("reasonke", "instruction")]
        self.protocol = self.root / "protocol.json"
        self.output = self.root / "derived"
        self.records = []
        categories = {"direct": 1, "direct_paraphrase": 1, "two_hop": 2, "two_hop_paraphrase": 2,
                      "three_hop": 3, "three_hop_paraphrase": 3,
                      "wrong_version_or_tier": 0, "same_entity_other_relation": 0}
        for model, path in zip(("reasonke", "instruction"), self.receipts):
            prefix = "R" if model == "reasonke" else "G"
            rows = []
            for index, (category, depth, lineage) in enumerate(
                    (category, depth, lineage) for category, depth in categories.items() for lineage in range(32)):
                rows.append(self.record(f"{prefix}-RAG:fixture-{category}-{lineage}", prefix+"-RAG",
                                        category, depth, lineage, index))
            for index in range(16):
                rows.append(self.record(f"{prefix}-ORACLE:fixture-three_hop-{index}", prefix+"-ORACLE",
                                        "three_hop", 3, index, index))
            path.write_text(json.dumps({"explicit_software_fixture": True,
                "plan": {"model": model}, "records": rows}), encoding="utf-8")
            self.records += rows
        inputs = {}
        for phase in ("profile", "development"):
            for model in ("reasonke", "instruction"):
                path = self.root / f"{phase}-{model}.json"
                path.write_text('{"explicit_software_fixture":true}', encoding="utf-8")
                inputs[path.name] = digest(path)
        self.protocol.write_text(json.dumps({"explicit_software_fixture": True,
                                            "input_receipts": inputs}), encoding="utf-8")
        self.report = {"schema": "editguard-depth-results-v1", "phase": "final",
            "execution_status": "incomplete", "assigned_records": 544, "independent_lineages": 32,
            "conditions": {}, "paired_differences": {}, "model_measurements": {}, "human_reviewed": False}
        for name in exporter.ORDER:
            rows = [row for row in self.records if row["condition"] == name]
            depths = sorted({str(row["question"]["depth"]) for row in rows})
            self.report["conditions"][name] = {
                "overall": summarize(rows),
                "by_depth": {d: summarize([row for row in rows if str(row["question"]["depth"]) == d]) for d in depths},
                "by_category": {c: summarize([row for row in rows if row["question"]["category"] == c])
                                for c in sorted({row["question"]["category"] for row in rows})},
                "retrieval_by_depth": {d: self.retrieval([row for row in rows if str(row["question"]["depth"]) == d]) for d in depths}}
        self.report["receipt_audit"] = [{"receipt": str(p), "sha256": digest(p)} for p in self.receipts]

    @staticmethod
    def record(identifier, condition, category, depth, lineage, index):
        outcome = ("wrong", "abstain", "execution_error", "unexecuted")[index] if index < 4 else "correct"
        row = {"id": identifier, "condition": condition, "outcome": outcome,
            "question": {"id": identifier.split(":", 1)[1], "lineage_id": f"fixture-{lineage}",
                         "depth": depth, "category": category, "expected_answer": "Explicit software fixture"},
            "correct": outcome == "correct"}
        if outcome != "unexecuted":
            row["pipeline_seconds"] = .5 + index * .01
            row["retrieval"] = {"found": max(depth, 1), "complete": True}
        if outcome == "execution_error":
            row["error"] = {"type": "FixtureError", "message": "Explicit CPU software fixture"}
        elif outcome != "unexecuted":
            row["response"] = {"content": "Explicit generated-output fixture", "software_fixture": True}
        return row

    @staticmethod
    def retrieval(rows):
        return {"assigned": len(rows), "found": sum(r.get("retrieval", {}).get("found", 0) for r in rows),
            "required": sum(max(r["question"]["depth"], 1) for r in rows),
            "complete": sum(r.get("retrieval", {}).get("complete", False) for r in rows)}

    @staticmethod
    def render_fixture(report, records, output):
        (output / "explicit-software-fixture.svg").write_text("Not a measurement or real figure", encoding="utf-8")
        return {"matplotlib_version": "explicit-software-fixture", "figure_files": ["explicit-software-fixture.svg"]}

    def generate_fixture(self, **kwargs):
        with patch.object(exporter, "audit_and_aggregate", return_value=copy.deepcopy(self.report)) as audit:
            with patch.object(exporter, "_render", side_effect=self.render_fixture):
                result = exporter.generate(self.receipts, self.output, protocol=self.protocol, **kwargs)
        audit.assert_called_once_with(self.receipts, config_path=kwargs.get("config_path"), protocol=self.protocol)
        return result

    def test_audit_is_mandatory_before_output_or_plotting(self):
        with patch.object(exporter, "audit_and_aggregate", side_effect=ValueError("Changed frozen inputs")), \
                patch.object(exporter, "_render") as render:
            with self.assertRaisesRegex(ValueError, "Changed frozen"):
                exporter.generate(self.receipts, self.output, protocol=self.protocol)
            render.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".derived.staging-*")), [])
        with patch.object(exporter, "audit_and_aggregate") as audit:
            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                exporter.generate(self.receipts, self.output, protocol=self.root / "missing.json")
            audit.assert_not_called()

    def test_counts_and_denominators_survive_incomplete_outcomes(self):
        self.generate_fixture()
        with (self.output / "scorecard.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        main = next(r for r in rows if r["condition"] == "R-RAG" and r["population"] == "overall")
        self.assertEqual([main[k] for k in ("assigned", "correct", "wrong", "abstain", "execution_error", "unexecuted")],
                         ["256", "252", "1", "1", "1", "1"])
        self.assertEqual(float(main["accuracy"]), 252/256)
        self.assertEqual(float(main["coverage"]), 253/256)
        self.assertEqual(main["generated_latency_n"], "254")
        self.assertEqual(main["failed_latency_n"], "1")
        oracle = next(r for r in rows if r["condition"] == "R-ORACLE" and r["population"] == "overall")
        self.assertEqual((oracle["assigned"], oracle["correct"]), ("16", "12"))
        raw = json.loads((self.output / "results.json").read_text())
        self.assertEqual(raw["execution_status"], "incomplete")
        self.assertEqual(raw["assigned_records"], 544)
        with (self.output / "latency.csv").open(encoding="utf-8") as stream:
            latency = list(csv.DictReader(stream))
        self.assertEqual(len(latency), 544)
        self.assertEqual(next(r for r in latency if r["outcome"] == "unexecuted")["pipeline_seconds"], "")
        text = (self.output / "REPORT.md").read_text()
        self.assertIn("252/256", text)
        self.assertIn("Execution: **incomplete**", text)
        self.assertIn("not ordinary retrieval accuracy", text)
        self.assertIn("not verify generated reasoning", text)

    def test_host_gate_and_figure_hashes_are_acyclic_and_portable(self):
        manifest = self.generate_fixture()
        gate = json.loads((self.output / "host-audit.json").read_text())
        self.assertEqual(gate["schema"], "editguard-depth-host-audit-v1")
        self.assertEqual((gate["status"], gate["model_calls"]), ("passed", 0))
        expected = {f"{phase}-{model}.json" for phase in ("profile", "development", "final")
                    for model in ("reasonke", "instruction")} | {"protocol.json", "results.json"}
        self.assertEqual(set(gate["files"]), expected)
        self.assertEqual(gate["results_sha256"], digest(self.output / "results.json"))
        self.assertEqual(gate["protocol_sha256"], digest(self.protocol))
        self.assertEqual(gate["figure_manifest_sha256"], digest(self.output / "figure-manifest.json"))
        self.assertNotIn("host-audit.json", manifest["files"])
        self.assertNotIn("figure-manifest.json", manifest["files"])
        self.assertFalse(manifest["model_inference_executed"])
        for name, checksum in manifest["files"].items():
            self.assertEqual(digest(self.output / name), checksum)

    def test_rejects_old_or_partial_populations_without_creating_output(self):
        for key, value in (("schema", "editguard-operations-results-v1"), ("phase", "development"),
                           ("assigned_records", 512), ("independent_lineages", 256)):
            report = copy.deepcopy(self.report); report[key] = value
            with self.subTest(key=key), patch.object(exporter, "audit_and_aggregate", return_value=report), \
                    patch.object(exporter, "_render") as render:
                with self.assertRaises(ValueError):
                    exporter.generate(self.receipts, self.output, protocol=self.protocol)
                self.assertFalse(self.output.exists()); render.assert_not_called()
        changed = copy.deepcopy(self.report)
        changed["conditions"]["R-RAG"]["overall"]["counts"]["unexecuted"] = 0
        with patch.object(exporter, "audit_and_aggregate", return_value=changed):
            with self.assertRaisesRegex(ValueError, "assigned denominator"):
                exporter.generate(self.receipts, self.output, protocol=self.protocol)

    def test_existing_report_is_preserved_and_staged_failure_is_cleaned(self):
        self.output.mkdir(); (self.output / "keep.txt").write_text("original")
        with patch.object(exporter, "audit_and_aggregate") as audit:
            with self.assertRaisesRegex(ValueError, "Preserve"):
                exporter.generate(self.receipts, self.output, protocol=self.protocol)
            audit.assert_not_called()
        self.assertEqual((self.output / "keep.txt").read_text(), "original")
        target = self.root / "failed-render"
        with patch.object(exporter, "audit_and_aggregate", return_value=copy.deepcopy(self.report)), \
                patch.object(exporter, "_render", side_effect=RuntimeError("Explicit renderer fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "renderer fixture"):
                exporter.generate(self.receipts, target, protocol=self.protocol)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".failed-render.staging-*")), [])

    def test_receipt_or_bound_input_changes_cannot_publish_a_report(self):
        def alter_final(report, records, output):
            self.receipts[0].write_text('{"changed":true}')
            return self.render_fixture(report, records, output)
        with patch.object(exporter, "audit_and_aggregate", return_value=copy.deepcopy(self.report)), \
                patch.object(exporter, "_render", side_effect=alter_final):
            with self.assertRaisesRegex(ValueError, "changed during export"):
                exporter.generate(self.receipts, self.output, protocol=self.protocol)
        self.assertFalse(self.output.exists())

    def test_profile_input_change_during_render_prevents_publication(self):
        def alter_profile(report, records, output):
            (self.root / "profile-reasonke.json").write_text('{"changed":true}')
            return self.render_fixture(report, records, output)
        with patch.object(exporter, "audit_and_aggregate", return_value=copy.deepcopy(self.report)), \
                patch.object(exporter, "_render", side_effect=alter_profile):
            with self.assertRaisesRegex(ValueError, "changed during export"):
                exporter.generate(self.receipts, self.output, protocol=self.protocol)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting runtime not installed")
    def test_real_renderer_produces_five_png_svg_pairs_from_software_fixtures(self):
        target = self.root / "software-fixture-render"
        target.mkdir()
        metadata = exporter._render(self.report, self.records, target)
        self.assertEqual(len(metadata["figure_files"]), 10)
        for path in target.iterdir():
            if path.suffix == ".png":
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            else:
                self.assertIn("<svg", path.read_text(encoding="utf-8"))
        self.assertEqual({p.stem for p in target.iterdir()}, {"depth-accuracy", "category-accuracy",
                         "outcome-composition", "retrieval-and-answer", "latency"})


if __name__ == "__main__":
    unittest.main()
