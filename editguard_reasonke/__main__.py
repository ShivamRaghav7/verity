"""Explicit CUDA loading, a five-case diagnostic, and authenticated serving."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from .runtime import ReasonKERuntime, RuntimeErrorBase, validate_request
from .server import make_server


def write_receipt(path, receipt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reasonke-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cases(path):
    raw = Path(path).read_bytes()
    if len(raw) > 256 * 1024:
        raise RuntimeErrorBase("The five-case diagnostic file exceeds 256 KiB")
    value = json.loads(raw)
    cases = value.get("cases") if isinstance(value, dict) else value
    if not isinstance(cases, list) or len(cases) != 5:
        raise RuntimeErrorBase("This diagnostic is fixed to exactly five predefined cases")
    ids = set()
    checked = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise RuntimeErrorBase("Each diagnostic case needs a unique nonempty ID")
        ids.add(case["id"])
        request = {key: case[key] for key in ("question", "facts", "history", "max_new_tokens") if key in case}
        validate_request(request)
        expected = case.get("expected_answer")
        if expected is not None and not (isinstance(expected, str) or
                isinstance(expected, list) and expected and all(isinstance(item, str) for item in expected)):
            raise RuntimeErrorBase("expected_answer must be a string or a list of reviewed aliases")
        checked.append({"id": case["id"], "request": request, "expected_answer": expected})
    return checked, hashlib.sha256(raw).hexdigest()


def run_profile(runtime, cases_path, output_path, *, resume=False):
    cases, cases_sha = load_cases(cases_path)
    output_path = Path(output_path)
    plan = {"cases_sha256": cases_sha, "manifest_sha256": runtime.manifest["manifest_sha256"]}
    if output_path.exists():
        if not resume:
            raise RuntimeErrorBase("Receipt already exists; choose a new output or explicitly use --resume")
        receipt = json.loads(output_path.read_text(encoding="utf-8"))
        if receipt.get("plan") != plan:
            raise RuntimeErrorBase("Resume inputs differ from the saved diagnostic plan")
    else:
        receipt = {"schema": "editguard-reasonke-profile-v1", "plan": plan,
            "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "cases": [],
            "purpose": "Finite deployment diagnostic, not a held-out benchmark or general accuracy estimate",
            "disclosures": ["Author checkpoint; not trained by this project",
                "Exact checkpoint identity behind published 95.48% is unverified",
                "4-bit inference is a deployment adaptation", "All five outcomes including errors remain in this receipt"]}
    completed = {item["id"] for item in receipt["cases"]}
    started = time.perf_counter()
    receipt["runtime"] = runtime.status()
    write_receipt(output_path, receipt)
    for case in cases:
        if case["id"] in completed:
            continue
        row = {"id": case["id"], "request": case["request"], "expected_answer": case["expected_answer"]}
        try:
            row["response"] = runtime.generate(case["request"])
            row["status"] = "generated"
            expected = case["expected_answer"]
            aliases = [expected] if isinstance(expected, str) else expected
            row["exact_alias_match"] = None if aliases is None else row["response"]["content"].strip() in [item.strip() for item in aliases]
        except Exception as exc:
            row.update(status="error", error={"type": type(exc).__name__, "message": str(exc)}, exact_alias_match=False)
        receipt["cases"].append(row)
        write_receipt(output_path, receipt)
        print(json.dumps({"case": row["id"], "status": row["status"], "exact_alias_match": row["exact_alias_match"]}), flush=True)
    receipt.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
        this_session_seconds=time.perf_counter() - started,
        generated_count=sum(row["status"] == "generated" for row in receipt["cases"]),
        error_count=sum(row["status"] == "error" for row in receipt["cases"]))
    receipt["runtime"] = runtime.status()
    write_receipt(output_path, receipt)
    return receipt


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(root / "configs/reasonke-model.json"))
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--profile", metavar="CASES_JSON")
    parser.add_argument("--output", default=str(root / "artifacts/reasonke/profile.json"))
    parser.add_argument("--resume", action="store_true", help="Skip cases already recorded under an identical input plan")
    args = parser.parse_args(argv)
    if not args.serve and not args.profile:
        parser.error("Choose --serve, --profile CASES_JSON, or both")
    token = os.environ.get("EDITGUARD_REASONKE_TOKEN", "")
    if args.serve and (len(token) < 16 or "\r" in token or "\n" in token):
        parser.error("Set EDITGUARD_REASONKE_TOKEN to at least 16 characters before serving")
    try:
        runtime = ReasonKERuntime(args.manifest)
        # Validate plan/output before downloading or loading a large model.
        if args.profile:
            _, cases_sha = load_cases(args.profile)
            if Path(args.output).exists() and not args.resume:
                raise RuntimeErrorBase("Receipt already exists; choose a new output or explicitly use --resume")
            if Path(args.output).exists() and args.resume:
                existing = json.loads(Path(args.output).read_text(encoding="utf-8"))
                plan = {"cases_sha256": cases_sha, "manifest_sha256": runtime.manifest["manifest_sha256"]}
                if existing.get("plan") != plan:
                    raise RuntimeErrorBase("Resume inputs differ from the saved diagnostic plan")
        runtime.load()
        if args.profile:
            receipt = run_profile(runtime, args.profile, args.output, resume=args.resume)
            print(json.dumps({"receipt": str(Path(args.output).resolve()), "generated": receipt["generated_count"],
                              "errors": receipt["error_count"]}), flush=True)
            if receipt["error_count"] and not args.serve:
                return 1
        if args.serve:
            server = make_server(runtime, token, args.host, args.port)
            print(f"ReasonKE worker listening on {args.host}:{args.port}; bearer authentication required.", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ReasonKE worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
