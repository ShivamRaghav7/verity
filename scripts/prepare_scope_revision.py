"""Prepare only bounded scope revision 1; never train or change prior evidence."""

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
from editguard.training import file_hash, json_file_bytes
from scripts.prepare_pilot_runs import make_card
from scripts.preflight_training import validate_run_card


def revision_card(dataset, reviews, profile, initial, output):
    dataset, reviews, profile, initial, output = (
        Path(path).resolve() for path in (dataset, reviews, profile, initial, output))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    prior = validate_artifact(initial, "scope")
    if (manifest.get("pilot_version") != 3 or manifest.get("revision_version") != 1
            or manifest.get("scope_labels_changed") is not False
            or manifest.get("structured_fact_truth_changed") is not False
            or manifest.get("positive_questions_changed") is not False
            or manifest.get("answer_inputs_or_targets_changed") is not False
            or manifest.get("counts") != {"scope": {"train": 240, "validation": 120},
                                           "answer": {"train": 96, "validation": 48}}
            or manifest.get("origin_dataset_id") != prior.get("dataset_id")
            or manifest.get("origin_hashes", {}).get("manifest.json") != prior.get("manifest_sha256")):
        raise ValueError("Revision 1 must bind the completed initial scope model and unchanged pilot facts/populations.")
    if (output.exists() or output == initial or initial in output.parents or output in initial.parents):
        raise ValueError("Use a fresh revision model directory separate from initial artifacts.")
    for split in ("train", "validation"):
        styles = manifest.get("scope_style_balance", {}).get(split, {})
        positive, negative = (24, 36) if split == "train" else (12, 18)
        if set(styles) != {"direct", "paraphrase", "comparison_yes", "comparison_no"} or any(
                value != {"positive": positive, "negative": negative, "positive_prior": .4}
                for value in styles.values()):
            raise ValueError("Each question style must have the matched 40% applicable prior.")
    card = make_card("scope", dataset, reviews, profile, output, "automated_development")
    card.update(
        purpose="targeted_development_scope_revision",
        question="Does balancing the existing negative question forms improve direct-edit detection on held-out pilot entities, without increasing false correction applications or sacrificing paraphrase/comparison recall?",
        revision_number=1,
        maximum_targeted_revisions=2,
        initial_scope_manifest_sha256=file_hash(initial / "manifest.json"),
        initial_scope_artifact=str(initial),
        initialization="Pinned pretrained DistilBERT; same seed, optimizer, shape and finite cap as initial run; no continuation of the initial checkpoint.",
        comparison_plan="Select checkpoint by validation loss, calibrate with the declared validation cost, then compare on revised validation and separately report the unchanged 48 applicable questions. Revised negatives differ from initial; do not present the full 120-question scores as an identical paired population.",
        finite_stopping_condition="Exactly 360 optimizer steps (three passes); no automatic retries, extensions or final-test tuning.")
    if card["max_steps"] != 360 or card["checkpoint_every_steps"] != 120:
        raise ValueError("Revision must retain the initial finite step and checkpoint plan.")
    return card


def publish(card, path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("Use a fresh run-card path; never overwrite experiment evidence.")
    for key in ("data_dir", "reviews_path", "profile_path", "initial_scope_artifact", "output_dir"):
        protected = Path(card[key]).resolve()
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError("Run-card output must remain separate from data, review, profile and model artifacts.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending-{uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(json_file_bytes(card))
            stream.flush()
            os.fsync(stream.fileno())
        result = validate_run_card(temporary)
        # An exclusive final creation avoids replacing an intervening writer.
        with path.open("xb") as stream:
            stream.write(temporary.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        return result
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v3-reviews.automated.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "reports/pilot-v3-scope-profile.json")
    parser.add_argument("--initial", type=Path, default=ROOT / "artifacts/scope/pilot-v2-initial")
    parser.add_argument("--model-output", type=Path, default=ROOT / "artifacts/scope/pilot-v3-revision1")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/run_cards/pilot-v3/scope.json")
    args = parser.parse_args()
    try:
        card = revision_card(args.dataset, args.reviews, args.profile, args.initial, args.model_output)
        result = publish(card, args.output)
        print(json.dumps({"preflight": result, "run_card": str(args.output),
                          "revision_number": 1, "training_execution_requested": False}, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
