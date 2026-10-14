"""Prepare the second and final scope revision; never execute training."""

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
from editguard.scope_inputs import scope_policy_identity
from editguard.training import file_hash, json_file_bytes
from scripts.prepare_pilot_runs import make_card
from scripts.preflight_training import validate_run_card

POLICY = "entity_neutral_v1"


def revision_card(dataset, reviews, profile, prior, output):
    dataset, reviews, profile, prior, output = (
        Path(path).resolve() for path in (dataset, reviews, profile, prior, output))
    original = validate_artifact(prior, "scope")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    measured = json.loads(profile.read_text(encoding="utf-8"))["profile"]
    policy = scope_policy_identity(POLICY)
    if (manifest.get("pilot_version") != 3 or manifest.get("revision_version") != 1
            or manifest.get("counts") != {"scope": {"train": 240, "validation": 120},
                                           "answer": {"train": 96, "validation": 48}}
            or manifest.get("dataset_id") != original.get("dataset_id")
            or file_hash(dataset / "manifest.json") != original.get("manifest_sha256")
            or file_hash(reviews) != original.get("reviews_sha256")
            or any(measured.get(key) != value for key, value in policy.items())):
        raise ValueError("Scope revision 2 must preserve revision 1's exact reviewed data and bind the measured input adaptation.")
    if output.exists() or output == prior or output in prior.parents or prior in output.parents:
        raise ValueError("Use a fresh scope revision output separate from earlier models.")
    card = make_card("scope", dataset, reviews, profile, output, "automated_development")
    if card["max_steps"] != 360 or card["checkpoint_every_steps"] != 120:
        raise ValueError("The final scope revision retains the 360-step, three-pass budget.")
    card.update(policy)
    card.update(purpose="targeted_development_scope_revision", revision_number=2,
        maximum_targeted_revisions=2,
        question="Does canonicalizing matching product/plan/version literals and removing opaque provenance from classifier inputs improve routing on the unchanged held-out edits and the disclosed Nimbus development workflow, without increasing false applications?",
        prior_scope_artifact=str(prior), prior_scope_manifest_sha256=file_hash(prior / "manifest.json"),
        initialization="Same pinned pretrained DistilBERT, seed, reviewed raw rows, labels, optimizer, learning rate, token shape and finite cap as revision 1; only the versioned classifier input adaptation changes. No continuation of prior weights.",
        selection_metric="validation_loss",
        finite_stopping_condition="Exactly 360 optimizer steps (three passes); second and final permitted scope revision. No automatic retry, extension or final-test tuning.",
        comparison_plan="Keep both exports and original metadata. Recalibrate on validation using the declared guard-aware routing cost; evaluate raw scope and end-to-end decisions separately on all120 unchanged v3 questions and the Nimbus development workflow. Report deterministic matching and transformed-input overlap as adaptations/limitations, not learned entity verification. No test rows may influence this revision.")
    return card


def publish(card, path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("Use a fresh run-card path; evidence is not overwritten.")
    for key in ("data_dir", "reviews_path", "profile_path", "prior_scope_artifact", "output_dir"):
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
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v3-reviews.automated.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "reports/profiles/guarded-scope-neutral.json")
    parser.add_argument("--prior", type=Path, default=ROOT / "artifacts/scope/pilot-v3-revision1")
    parser.add_argument("--model-output", type=Path, default=ROOT / "artifacts/scope/pilot-v3-neutral2")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/run_cards/targeted-revisions/scope.json")
    args = parser.parse_args()
    try:
        card = revision_card(args.dataset, args.reviews, args.profile, args.prior, args.model_output)
        result = publish(card, args.output)
        print(json.dumps({"preflight": result, "run_card": str(args.output), "training_execution_requested": False}, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
