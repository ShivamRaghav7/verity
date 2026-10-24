"""Finite opt-in acceptance rehearsal with the real local model, not a benchmark.

Creates its own conversation and one clearly synthetic correction. Preserves all
outputs, including failures, and withdraws its correction when finished.
"""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/chat-studio-local-1/live-acceptance.json"))
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"execution_requested": False, "question": "Does normal conversation, publish/compare/withdraw, and optional image generation work with the real local model?",
            "bounded_work": "At most 10 model generations; each capped at 384 new tokens with a 240-second request timeout. No training.",
            "inputs": "Prepared Ollama models and EditGuard at 127.0.0.1:8004",
            "artifacts": [str(args.output)], "limitation": "Synthetic development acceptance cases; not a frozen accuracy evaluation."}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError("Preserve the existing live receipt and choose another output path")
    base, token, records = "http://127.0.0.1:8004", None, []
    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-EditGuard-Token"] = token
        req = urllib.request.Request(base + path, data=data, headers=headers)
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=260) as response:
            result = json.load(response)
        # Never persist the local session token.
        result.pop("token", None)
        result.pop("csrf_token", None)
        records.append({"path": path, "seconds": time.perf_counter() - started, "response": result})
        return result
    with urllib.request.urlopen(base + "/api/status", timeout=10) as response:
        status = json.load(response)
    token = status.pop("token")
    assert status["model"]["ready"] and status["model"]["embedding_ready"]
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "kind": "own_real_local_model_development_acceptance_not_a_benchmark",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_commit_is_complete_identity": not bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
        "source_file_sha256": {str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted(Path("editguard_chat").glob("*.py")), Path("configs/chat-models.json"), Path(__file__)]},
        "model_status": status, "records": records, "checks": {}, "errors": [],
        "limitations": "One synthetic edit and generated geometric image; automated checks and qualitative inspection, no independent human review or representative chat accuracy estimate. Earlier generic-tag abstention is preserved separately."}
    correction_id = None
    try:
        chat_id = request("/api/chats", {})["chat"]["id"]
        def ask(message, **extra):
            return request("/api/chat", {"chat_id": chat_id, "message": message, "mode": "edited", **extra})
        hello = ask("Explain in two sentences why leaves usually look green.")
        follow = ask("Explain that for a ten-year-old in one sentence.")
        question = "What is the access code for Aster Demo Registry?"
        before = ask(question)
        published = request("/api/corrections", {"question": question, "answer": "The access code for Aster Demo Registry is LUMEN-73.",
            "scope": "Aster Demo Registry / access code / synthetic demo", "source": "Controlled synthetic acceptance fixture, 25 October 2026; not a real credential."})
        correction_id = published["correction"]["id"]
        direct = ask(question)
        paraphrase = ask("Tell me Aster Demo Registry's access code. Keep the answer short.")
        unrelated = ask("What is the capital of Japan? Answer in one sentence.")
        compared = request("/api/compare", {"chat_id": chat_id, "message_id": before["user"]["id"]})
        withdrawn = request("/api/corrections/" + correction_id + "/withdraw", {})
        after = ask(question)
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (600, 300), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((55, 65, 215, 225), fill="red")
        draw.ellipse((365, 65, 525, 225), fill="blue")
        output = io.BytesIO()
        image.save(output, format="PNG")
        picture = ask("What color is the square on the left? Answer in one sentence.", image_base64=base64.b64encode(output.getvalue()).decode("ascii"), image_mime="image/png")
        report["synthetic_image_sha256"] = hashlib.sha256(output.getvalue()).hexdigest()
        report["checks"] = {
            "normal_text_generation_nonempty": bool(hello["assistant"]["content"]),
            "followup_generation_nonempty": bool(follow["assistant"]["content"]),
            "new_chat_history_persisted": len(request("/api/chats/" + chat_id)["messages"]) == 16,
            "explicit_publish_active": published["correction"]["status"] == "active",
            "direct_routed_to_published_correction": [c["id"] for c in direct["assistant"]["evidence"]["corrections"]] == [correction_id],
            "direct_generation_nonempty": bool(direct["assistant"]["content"]),
            "paraphrase_generation_nonempty": bool(paraphrase["assistant"]["content"]),
            "unrelated_question_not_routed": not unrelated["assistant"]["evidence"]["corrections"],
            "comparison_original_has_no_correction_context": not compared["original"]["evidence"]["corrections"],
            "comparison_edited_has_published_correction": [c["id"] for c in compared["edited"]["evidence"]["corrections"]] == [correction_id],
            "withdrawal_removed_correction": withdrawn["correction"]["status"] == "withdrawn" and not after["assistant"]["evidence"]["corrections"],
            "revision_changed_after_withdrawal": after["revision"] > direct["revision"],
            "image_generated_answer_nonempty": bool(picture["assistant"]["content"]),
        }
        report["answers_for_qualitative_review"] = {name: row["assistant"]["content"] for name, row in
            (("normal", hello), ("followup", follow), ("before", before), ("direct", direct), ("paraphrase", paraphrase), ("unrelated", unrelated), ("withdrawn", after), ("image", picture))}
        def normalize(text):
            return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
        code_aliases = {normalize("The access code for Aster Demo Registry is LUMEN-73."), normalize("LUMEN-73.")}
        report["quality_checks"] = {
            "direct_endorses_fixture_code_whole_answer": normalize(direct["assistant"]["content"]) in code_aliases,
            "paraphrase_endorses_fixture_code_whole_answer": normalize(paraphrase["assistant"]["content"]) in code_aliases,
            "paraphrase_routed_to_published_correction": [c["id"] for c in paraphrase["assistant"]["evidence"]["corrections"]] == [correction_id],
            "withdrawn_answer_does_not_endorse_fixture_code": normalize(after["assistant"]["content"]) not in code_aliases,
            "unrelated_answer_matches_reviewed_alias": normalize(unrelated["assistant"]["content"]) in {normalize("Tokyo"), normalize("The capital of Japan is Tokyo")},
            "image_answer_matches_reviewed_alias": normalize(picture["assistant"]["content"]) in {normalize("Red"), normalize("The square on the left is red")},
        }
        report["scoring_notice"] = "Six narrow development-fixture assertions with explicit reviewed whole-answer aliases. They are not representative accuracy or proof that all unendorsed alternatives are correct."
        report["execution_status"] = "completed" if all(report["checks"].values()) and all(report["quality_checks"].values()) else "completed_with_failed_checks"
    except Exception as exc:
        report["execution_status"] = "failed"
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if correction_id:
            try:
                rows = request("/api/corrections")["corrections"]
                if next(row for row in rows if row["id"] == correction_id)["status"] == "active":
                    request("/api/corrections/" + correction_id + "/withdraw", {})
            except Exception as exc:
                report["errors"].append("Cleanup: " + str(exc))
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"execution_status": report["execution_status"], "checks": report["checks"], "quality_checks": report.get("quality_checks"), "errors": report["errors"], "output": str(args.output)}, indent=2))
    return 0 if report["execution_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
