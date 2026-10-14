"""Train a finite reviewed auxiliary run, or an explicitly synthetic smoke test.

No training starts on import. Production mode requires a preflight-valid reviewed
train/validation run card. Smoke mode uses cached pinned pretrained weights and
labels artifacts smoke_test; the application must refuse them as trained models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.training import file_hash, json_file_bytes, run_training, synthetic_rows
from scripts.preflight_training import validate_run_card
from scripts.prepare_data import load_reviewed_rows


def reviewed_inputs(card_path):
    """Always run the existing independent preflight before loading ML packages."""
    preflight = validate_run_card(card_path)
    card = json.loads(Path(card_path).read_text(encoding="utf-8"))
    profile = json.loads(Path(card["profile_path"]).read_text(encoding="utf-8"))["profile"]
    declared_device = card.get("device", profile["device"])
    if declared_device != profile["device"]:
        raise ValueError("Declared training device differs from the measured profile")
    card["device"] = declared_device
    train = load_reviewed_rows(Path(card["train_path"]), Path(card["reviews_path"]), review_policy=card.get("review_policy", "human"))
    validation = load_reviewed_rows(Path(card["validation_path"]), Path(card["reviews_path"]), review_policy=card.get("review_policy", "human"))
    return card, train, validation, preflight


def smoke_card(args):
    if args.run_card:
        raise ValueError("Smoke fixtures and reviewed run cards are mutually exclusive")
    if args.task not in ("scope", "answer") or args.output is None:
        raise ValueError("Smoke mode requires --task and --output")
    max_steps = 4 if args.max_steps is None else args.max_steps
    if not 1 <= max_steps <= 4:
        raise ValueError("Developer smoke is capped at four optimizer steps")
    configuration = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    model_id = args.model_id or configuration[f"{args.task}_candidate"]
    revision = args.model_revision or configuration["model_revisions"][args.task]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Smoke model revision must pin a 40-character commit SHA")
    return {"schema_version": 1, "task": args.task, "purpose": "smoke_test",
            "question": "Does a bounded optimizer/checkpoint/resume execution work on explicit synthetic fixtures?",
            "dataset_id": "synthetic-developer-fixture-not-reviewed", "model_id": model_id,
            "model_revision": revision, "output_dir": str(args.output.resolve()),
            "device": args.device or "cpu", "seed": 42 if args.seed is None else args.seed, "batch_size": 2,
            "input_length": 192, "target_length": 48 if args.task == "answer" else None,
            "precision": "float32", "scheduler": "linear", "max_steps": max_steps,
            "checkpoint_every_steps": 1, "validation_every_steps": 1, "keep_last_checkpoints": 2,
            "learning_rate": 2e-5, "weight_decay": 0.01, "max_grad_norm": 1.0, "cpu_threads": 2,
            "manifest_sha256": None, "reviews_sha256": None,
            "review_policy": "No human reviews exist or are fabricated for this developer fixture."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-card", type=Path, help="Preflight-valid reviewed production run card")
    parser.add_argument("--resume", type=Path, help="Existing exact-state checkpoint inside the run output")
    parser.add_argument("--stop-after-steps", type=int, help="Absolute early checkpoint step; does not change max_steps/schedule")
    parser.add_argument("--smoke", action="store_true", help="Explicit synthetic-fixture smoke; never publishes a trained model")
    parser.add_argument("--task", choices=["scope", "answer"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--max-steps", type=int, help="Smoke-only finite cap, default four, at most four")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Smoke-only, default cpu; reviewed runs use measured profile device")
    parser.add_argument("--seed", type=int, help="Smoke-only seed, default 42")
    args = parser.parse_args()
    try:
        if args.smoke:
            card = smoke_card(args)
            train, validation = synthetic_rows(card["task"])
            purpose, card_hash = "smoke_test", hashlib.sha256(json_file_bytes(card)).hexdigest()
            print("Developer smoke: synthetic fixture; purpose=smoke_test; no trained-model or human-review claim.", flush=True)
        else:
            if args.run_card is None:
                raise ValueError("Supply --run-card for reviewed training, or explicitly select --smoke")
            if (args.task or args.output or args.model_id or args.model_revision
                    or args.max_steps is not None or args.device is not None or args.seed is not None):
                raise ValueError("Reviewed run choices must come from the exact run card")
            card, train, validation, preflight = reviewed_inputs(args.run_card)
            purpose, card_hash = "trained", file_hash(args.run_card)
            print(json.dumps({"preflight": preflight, "trainer_implementation": "available", "question": card["question"]}, indent=2), flush=True)
        manifest = run_training(card, train_rows=train, validation_rows=validation, purpose=purpose,
                                run_card_sha256=card_hash,
                                resume_checkpoint=args.resume or card.get("resume_checkpoint"),
                                stop_after_steps=args.stop_after_steps,
                                run_card_source=args.run_card if not args.smoke else None)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": manifest["status"], "purpose": purpose, "task": card["task"],
                      "completed_steps": manifest["completed_steps"], "selection": manifest["selection"],
                      "manifest": str(Path(card["output_dir"]) / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
