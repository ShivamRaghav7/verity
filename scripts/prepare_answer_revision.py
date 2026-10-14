"""Prepare one profiled, finite answer-objective revision; never execute training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.neural import validate_artifact
from editguard.training import file_hash, json_file_bytes, loss_objective
from scripts.prepare_pilot_runs import make_card
from scripts.preflight_training import validate_run_card

POLICY = "comparison_first_token_8x"


def revision_card(dataset, reviews, profile, initial, output):
    dataset, reviews, profile, initial, output = (
        Path(path).resolve() for path in (dataset, reviews, profile, initial, output))
    original = validate_artifact(initial, "answer")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    objective = loss_objective("answer", POLICY)
    measured = json.loads(profile.read_text(encoding="utf-8"))["profile"]
    if (manifest.get("pilot_version") != 2
            or manifest.get("dataset_id") != original.get("dataset_id")
            or file_hash(dataset / "manifest.json") != original.get("manifest_sha256")
            or file_hash(reviews) != original.get("reviews_sha256")
            or measured.get("loss_policy") != POLICY or measured.get("loss_objective") != objective):
        raise ValueError("Answer revision must preserve the initial model's exact reviewed data and bind the measured objective.")
    if output.exists() or output == initial or output in initial.parents or initial in output.parents:
        raise ValueError("Use a fresh answer revision output separate from the initial model.")
    card = make_card("answer", dataset, reviews, profile, output, "automated_development")
    if card["max_steps"] != 240 or card["checkpoint_every_steps"] != 48:
        raise ValueError("The answer revision retains the 240-step, five-pass initial budget.")
    card.update(purpose="targeted_development_answer_revision", revision_number=1,
        maximum_targeted_revisions=2, loss_policy=POLICY, loss_objective=objective,
        question="Does giving the opening Yes/No token equal extra weight for both comparison classes improve generated comparison polarity on the unchanged held-out edits, while preserving complete factual answers?",
        initial_answer_artifact=str(initial), initial_answer_manifest_sha256=file_hash(initial / "manifest.json"),
        initialization="Same pinned pretrained FLAN-T5-small, seed, data, targets, optimizer, learning rate, token shape and finite cap as initial run; only the comparison decision-token objective changes.",
        selection_metric="validation_weighted_token_loss",
        finite_stopping_condition="Exactly 240 optimizer steps (five passes); no automatic retry, extension or final-test tuning.",
        comparison_plan="Keep both exports. Measure unweighted token loss and first-token Yes/No performance on the same train/validation rows; run oracle generation on all48 validation examples and matched full v3 validation with the fixed scope policy. Retain all failures and abstentions. Weighted loss is not directly comparable with initial uniform loss.")
    return card


def publish(card, path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("Use a fresh run-card path; evidence is not overwritten.")
    for key in ("data_dir", "reviews_path", "profile_path", "initial_answer_artifact", "output_dir"):
        protected = Path(card[key]).resolve()
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError("Run-card output must remain separate from input and model artifacts.")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.pending-{uuid4().hex}")
    try:
        with staging.open("xb") as stream:
            stream.write(json_file_bytes(card))
            stream.flush()
            os.fsync(stream.fileno())
        result = validate_run_card(staging)
        with path.open("xb") as stream:
            stream.write(staging.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        return result
    finally:
        if staging.exists():
            staging.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v2-reviews.automated.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "reports/profiles/guarded-answer-polarity.json")
    parser.add_argument("--initial", type=Path, default=ROOT / "artifacts/answer/pilot-v2-initial")
    parser.add_argument("--model-output", type=Path, default=ROOT / "artifacts/answer/pilot-v2-polarity1")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/run_cards/targeted-revisions/answer.json")
    args = parser.parse_args()
    try:
        card = revision_card(args.dataset, args.reviews, args.profile, args.initial, args.model_output)
        result = publish(card, args.output)
        print(json.dumps({"preflight": result, "run_card": str(args.output), "training_execution_requested": False}, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
