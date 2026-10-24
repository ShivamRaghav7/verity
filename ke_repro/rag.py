"""Updated one-correction visual RAG with validation-selected cosine scope.

Uses the released BLIP visual prefix and an official pinned MiniLM CPU encoder.
No retriever training, editor meta-training or test-dependent tuning is performed.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .smoke import Resources, digest, preflight, write_json

TEXT_WEIGHT = .5
VALIDATION_INDICES = tuple(range(2, 10))
RECIPE = {"text_weight": TEXT_WEIGHT, "text_max_length": 256,
          "visual_prefix_tokens": 32, "visual_prefix_prompt": "Image:",
          "text_pooling": "attention-mask mean; L2 normalized; CPU eval no_grad",
          "visual_pooling": "Qformer-to-OPT prefix mean; L2 normalized",
          "text_only_policy": "reject", "selection_rule": "max F1; then precision; then threshold"}


def calibration_code_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ("ke_repro/rag.py", "ke_repro/generation.py", "ke_repro/bridge.py",
             "ke_repro/smoke.py", "ke_repro/__init__.py", "scripts/prepare_vlkeb_rag.py",
             "scripts/prepare_vlkeb_assets.py", "scripts/prepare_vlkeb_eval.py",
             "requirements-vlkeb.txt")
    return {path: digest(root / path) for path in paths}


def calibration_base_receipt(args, verification):
    from . import UPSTREAM_COMMIT
    from scripts.prepare_vlkeb_assets import load_manifest
    manifest = load_manifest(args.manifest)
    if (verification.get("assets_verified") is not True
            or verification.get("upstream_revision") != UPSTREAM_COMMIT
            or verification.get("asset_manifest_sha256") != digest(args.manifest)):
        raise ValueError("RAG base receipt requires a successful matching base preflight")
    checkpoints = {}
    for asset in manifest["assets"]:
        if asset.get("group") == "core" and Path(asset["path"]).suffix in (".bin", ".pth", ".pt"):
            algorithm = "sha256" if asset.get("sha256") else "md5"
            checkpoints[asset["path"]] = {"algorithm": algorithm, "checksum": asset[algorithm],
                                          "size_bytes": asset["size_bytes"]}
    return {"assets_verified": True, "base_asset_manifest_sha256": digest(args.manifest),
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_config_sha256": digest(args.upstream / "hparams/MEND/blip2.yaml"),
            "checkpoint_hashes": checkpoints,
            "checkpoint_hash_provenance": "Published manifest checksums verified against actual bytes by controller preflight"}


def validate_calibration_rows(rows):
    categories = ("direct", "paraphrase", "alternate_image", "text_locality", "visual_locality")
    expected = {(f"train-{i:04d}", category, category in categories[:3])
                for i in VALIDATION_INDICES for category in categories}
    if (not isinstance(rows, list) or len(rows) != 40
            or any(not isinstance(row, dict) or not isinstance(row.get("applicable"), bool) for row in rows)
            or {(r.get("edit_id"), r.get("category"), r.get("applicable")) for r in rows} != expected):
        raise ValueError("RAG calibration has wrong validation examples/labels")
    for row in rows:
        components = [row.get(key) for key in ("score", "text_cosine", "visual_cosine")]
        if row["category"] == "text_locality":
            if any(value is not None for value in components):
                raise ValueError("Text-only validation must have null score and components")
            continue
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or not -1. <= value <= 1. for value in components):
            raise ValueError("RAG validation cosines and scores must be finite in [-1,1]")
        score, text, visual = components
        if not math.isclose(score, TEXT_WEIGHT * text + (1 - TEXT_WEIGHT) * visual, rel_tol=0., abs_tol=1e-7):
            raise ValueError("RAG calibration score differs from weighted cosine recipe")
        if row["category"] == "direct" and any(not math.isclose(value, 1., rel_tol=0., abs_tol=1e-6) for value in components):
            raise ValueError("Direct validation must compare the correction with itself")


class MiniLMEncoder:
    def __init__(self, directory):
        from scripts.prepare_vlkeb_rag import verify_minilm
        verify_minilm(Path(directory).parent)
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(directory), local_files_only=True, use_safetensors=False).cpu().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("MiniLM requires nonempty text")
        torch = self.torch
        with torch.no_grad():
            tokens = self.tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)[0].tolist()


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Cosine vectors must be nonempty and same-sized")
    if any(not math.isfinite(float(x)) for x in (*left, *right)):
        raise ValueError("Cosine vectors must be finite")
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if denominator == 0:
        raise ValueError("Cosine vector has zero norm")
    return max(-1., min(1., sum(x*y for x, y in zip(left, right)) / denominator))


def image_feature(backend, image):
    import torch
    from .generation import blip2_prefix_embeddings
    with torch.no_grad():
        embeddings, _ = blip2_prefix_embeddings(backend.model, backend._image(image), "Image:")
        if backend.model.query_tokens.shape[1] != 32 or embeddings.shape[1] < 32:
            raise ValueError("RAG recipe requires the released32-token visual prefix")
        mean = embeddings[0, :32].float().mean(0)
        return torch.nn.functional.normalize(mean, p=2, dim=0).cpu().tolist()


class VisualCorrectionRetriever:
    def __init__(self, backend, encoder, threshold, *, visual_encoder=None):
        if not math.isfinite(float(threshold)):
            raise ValueError("Scope threshold must be finite")
        self.backend, self.encoder, self.threshold = backend, encoder, float(threshold)
        self.visual_encoder = visual_encoder or (lambda image: image_feature(backend, image))
        self.memory = None
        self.text_cache, self.visual_cache = {}, {}

    def _text(self, text):
        if text not in self.text_cache:
            self.text_cache[text] = self.encoder.encode(text)
        return self.text_cache[text]

    def _visual(self, image):
        # Evaluation uses immutable, verified file paths. Never silently cache
        # arbitrary mutable image tensors under their string representation.
        key = str(image) if isinstance(image, (str, Path)) else None
        if key is None:
            return self.visual_encoder(image)
        if key not in self.visual_cache:
            self.visual_cache[key] = self.visual_encoder(image)
        return self.visual_cache[key]

    def set_correction(self, image, edit_src, target, edit_id):
        if image is None or not all(isinstance(s, str) and s.strip() for s in (edit_src, target, edit_id)):
            raise ValueError("One approved visual correction needs image, question, target and identifier")
        self.memory = {"edit_id": edit_id, "edit_src": edit_src, "target": target,
                       "reference_image": str(image), "text_features": self._text(edit_src),
                       "visual_features": self._visual(image)}

    def retrieve(self, image, question):
        if self.memory is None:
            raise RuntimeError("Set the episode's current approved correction first")
        if image is None:
            return {"accepted": False, "score": None, "text_cosine": None, "visual_cosine": None,
                    "threshold": self.threshold, "edit_id": self.memory["edit_id"],
                    "reason": "text_only_out_of_scope_for_visual_edit", "score_is_probability": False}
        text = cosine(self._text(question), self.memory["text_features"])
        visual = cosine(self._visual(image), self.memory["visual_features"])
        score = TEXT_WEIGHT * text + (1 - TEXT_WEIGHT) * visual
        return {"accepted": score >= self.threshold, "score": score, "text_cosine": text,
                "visual_cosine": visual, "threshold": self.threshold, "edit_id": self.memory["edit_id"],
                "reason": "score_at_or_above_threshold" if score >= self.threshold else "below_threshold",
                "score_is_probability": False}

    def prompt(self, question, routing):
        if self.memory is None or routing.get("edit_id") != self.memory["edit_id"]:
            raise ValueError("Routing must belong to the current correction")
        return (f"Question: {self.memory['edit_src']} Answer: {self.memory['target']}\nQuestion: {question} Answer:"
                if routing["accepted"] else f"Question: {question} Answer:")


def threshold_metrics(rows, threshold):
    predictions = [r["score"] is not None and r["score"] >= threshold for r in rows]
    tp = sum(p and r["applicable"] for p, r in zip(predictions, rows))
    fp = sum(p and not r["applicable"] for p, r in zip(predictions, rows))
    fn = sum(not p and r["applicable"] for p, r in zip(predictions, rows))
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp+fp) if tp+fp else 0., "recall": tp / (tp+fn) if tp+fn else 0.,
            "f1": 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0., "denominator": len(rows)}


def select_threshold(rows):
    if not rows or any(not isinstance(r.get("applicable"), bool) or (r.get("score") is not None and not math.isfinite(r["score"])) for r in rows):
        raise ValueError("Threshold selection requires finite validation scores and Boolean labels")
    scores = sorted({r["score"] for r in rows if r["score"] is not None})
    if not scores:
        raise ValueError("No scored visual validation examples")
    candidates = scores + [math.nextafter(max(scores), math.inf)]
    return max((threshold_metrics(rows, threshold) for threshold in candidates),
               key=lambda metric: (metric["f1"], metric["precision"], metric["threshold"]))


def calibration_operation(backend, encoder, fixture, assets):
    from scripts.prepare_vlkeb_assets import relative_path
    retriever = VisualCorrectionRetriever(backend, encoder, threshold=0.)
    rows = []
    for item in fixture["records"]:
        record = item["record"]
        image = lambda key: str(relative_path(assets / "images", record[key]))
        edit_id = f"train-{item['source_index']:04d}"
        retriever.set_correction(image("image"), record["src"], record["alt"], edit_id)
        cases = (("direct", image("image"), record["src"], True),
                 ("paraphrase", image("image"), record["rephrase"], True),
                 ("alternate_image", image("image_rephrase"), record["src"], True),
                 ("visual_locality", image("m_loc"), record["m_loc_q"], False),
                 ("text_locality", None, record["loc"], False))
        for category, query_image, question, applicable in cases:
            routing = retriever.retrieve(query_image, question)
            rows.append({"edit_id": edit_id, "category": category, "applicable": applicable,
                         "score": routing["score"], "text_cosine": routing["text_cosine"], "visual_cosine": routing["visual_cosine"]})
    return {"rows": rows, "selection": select_threshold(rows), "probability_calibrated": False,
            "score_recipe": "0.5*MiniLM question/edit-question cosine +0.5*visual-prefix cosine; text-only query rejected",
            "selection_rule": "Maximum validation F1; ties higher precision, then higher threshold",
            "validation_row_count": len(rows), "validation_source_indices": list(VALIDATION_INDICES),
            "training_executed": False, "evaluation_split_opened": False,
            "limitations": "Eight fixed nonrandom upstream training rows, possibly repeated entity lineages and seen during published MEND training; this is our threshold-validation subset only. Cosine scores are not probabilities."}


def calibration_worker(args):
    backend, resources, torch_module = None, None, None
    report = {"execution_status": "failed", "training_executed": False}
    code = 1
    try:
        import torch
        torch_module = torch
        from .bridge import build_backend
        from scripts.prepare_vlkeb_rag import validate_calibration_fixture
        fixture, evidence = validate_calibration_fixture(args.assets, args.rag_root, args.manifest)
        preflight_path = args.output / "base-preflight.json"
        if digest(preflight_path) != args.expected_preflight_sha256:
            raise ValueError("Calibration base preflight receipt changed after controller verification")
        base_receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        with Resources() as resources:
            encoder = MiniLMEncoder(args.rag_root / "minilm")
            start = time.perf_counter()
            backend = build_backend(args.upstream, args.assets, device=args.device)
            report["load_seconds"] = time.perf_counter() - start
            report.update(calibration_operation(backend, encoder, fixture, args.assets))
        validate_calibration_rows(report["rows"])
        report.update(execution_status="completed", resources=resources.summary(torch),
                      sources={**evidence, "base": base_receipt, "recipe": RECIPE},
                      code_sha256=calibration_code_hashes())
        code = 0
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if resources is not None and hasattr(resources, "elapsed") and torch_module is not None:
            report["resources"] = resources.summary(torch_module)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                report.update(cleanup_error=str(error), execution_status="failed")
                code = 1
        write_json(args.output / "calibration.json", report)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-preflight-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--assets", type=Path, default=Path("artifacts/vlkeb/assets"))
    parser.add_argument("--rag-root", type=Path, default=Path("artifacts/vlkeb/rag"))
    parser.add_argument("--upstream", type=Path, default=Path("tmp/vlkeb-upstream"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/vlkeb-assets.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vlkeb/rag-calibration"))
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args._worker and not args.expected_preflight_sha256:
        parser.error("Private calibration worker needs its verified base-preflight checksum")
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("Use a finite cap from1 to1800 seconds")
    for key in ("assets", "rag_root", "upstream", "manifest", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.fixture = None
    if not (args.calibrate or args._worker):
        print(json.dumps({"mode": "plan", "models_loaded": False, "training_requested": False,
            "validation_indices": list(VALIDATION_INDICES), "validation_case_count": 40,
            "test_data_requested": False, "maximum_seconds": args.timeout_seconds,
            "runtime_estimate_seconds": None, "estimate_note": "One fresh backend plus40 scope scores; profile before a larger run. Timeout is a cap.",
            "expected_artifacts": ["calibration.json", "calibration.log"],
            "stopping_condition": "Eight fixed train rows, five cases each, one finite threshold selection; no optimizer/generation"}, indent=2))
        return 0
    try:
        if args._worker:
            return calibration_worker(args)
        if args.output.exists():
            raise FileExistsError("Use a fresh calibration output directory")
        from scripts.prepare_vlkeb_rag import validate_calibration_fixture
        validate_calibration_fixture(args.assets, args.rag_root, args.manifest)
        verification = preflight(args)
        base_receipt = calibration_base_receipt(args, verification)
        args.output.mkdir(parents=True)
        preflight_path = args.output / "base-preflight.json"
        write_json(preflight_path, base_receipt)
        command = [sys.executable, "-m", "ke_repro.rag", "--_worker", "--assets", str(args.assets),
            "--rag-root", str(args.rag_root), "--upstream", str(args.upstream), "--manifest", str(args.manifest),
            "--output", str(args.output), "--device", args.device,
            "--expected-preflight-sha256", digest(preflight_path)]
        timed_out = False
        with (args.output / "calibration.log").open("w", encoding="utf-8") as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}).returncode
            except subprocess.TimeoutExpired:
                timed_out, code = True, None
        report_path = args.output / "calibration.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if code != 0:
            report.update(execution_status="timed_out" if timed_out else "process_failed")
        report["worker_return_code"] = code
        write_json(report_path, report)
        if code == 0 and report.get("execution_status") == "completed":
            canonical_report = args.rag_root / "calibration.json"
            if canonical_report.exists():
                raise FileExistsError("Refusing to replace the frozen canonical calibration report")
            write_json(canonical_report, report)
        print(json.dumps({"execution_status": report.get("execution_status"), "selection": report.get("selection")}, indent=2))
        return 0 if code == 0 and report.get("execution_status") == "completed" else 1
    except Exception as error:
        print(json.dumps({"execution_status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
