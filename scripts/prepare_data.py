"""Prepare an independent fictional dataset; generated labels need human review.

No third-party packages, network calls, or model-generated labels are used here.
Training consumes reviewed rows only, through load_reviewed_rows().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RELATIONS = ("device_limit", "price_monthly", "storage_gb", "support_hours")
UNITS = {
    "device_limit": "devices",
    "price_monthly": "USD/month",
    "storage_gb": "GB",
    "support_hours": "hours/day",
}
NEGATIVE_TYPES = (
    "same_entity_different_property", "different_entity_same_property",
    "similar_name", "wrong_tier", "wrong_version", "ordinary",
)
AS_OF = "2026-10-11"
EFFECTIVE_FROM = "2026-09-01"
VALUE_POOLS = {
    "device_limit": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    "price_monthly": [7, 9, 12, 14, 17, 19, 22, 24, 29, 32, 39, 44],
    "storage_gb": [20, 25, 40, 50, 60, 75, 100, 120, 150, 180, 200, 240],
    "support_hours": [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20],
}
UNSEEN_CHALLENGE_VALUES = {
    "device_limit": "17", "price_monthly": "53",
    "storage_gb": "512", "support_hours": "24",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def family_splits(seed: int = 42, family_count: int = 80) -> dict[str, str]:
    """Assign the entire entity family before constructing any question."""
    if family_count < 20:
        raise ValueError("At least 20 families are needed for distinct within-split negative entities.")
    families = [f"family-{index:03d}" for index in range(family_count)]
    random.Random(seed).shuffle(families)
    train_end = int(family_count * 0.70)
    validation_end = train_end + int(family_count * 0.15)
    return {family: ("train" if index < train_end else
                     "validation" if index < validation_end else "test")
            for index, family in enumerate(families)}


def _base_value(relation: str, rng: random.Random) -> str:
    return str(rng.choice(VALUE_POOLS[relation]))


def _corrected_values(relation: str, original: str) -> list[str]:
    return [str(value) for value in VALUE_POOLS[relation] if str(value) != original]


def _subject(fact: dict, alias: bool = False) -> str:
    name = fact["aliases"][0] if alias else fact["entity_name"]
    return f"{name} {fact['plan']} version {fact['version']}"


def question_for(fact: dict, style: int = 0, comparison: int | None = None) -> str:
    subject = _subject(fact, alias=style == 1)
    direct = {
        "device_limit": f"How many devices does {subject} support?",
        "price_monthly": f"What is the monthly price of {subject}?",
        "storage_gb": f"How much storage is included with {subject}?",
        "support_hours": f"How many support hours per day does {subject} include?",
    }
    paraphrase = {
        "device_limit": f"What is the device allowance on {subject}?",
        "price_monthly": f"How much would I pay each month for {subject}?",
        "storage_gb": f"What is the storage quota for {subject}?",
        "support_hours": f"For how many hours each day is support available on {subject}?",
    }
    if style == 0:
        return direct[fact["relation"]]
    if style == 1:
        return paraphrase[fact["relation"]]
    if comparison is None:
        raise ValueError("A comparison question requires a quantity.")
    comparisons = {
        "device_limit": f"Can I connect {comparison} devices on {subject}?",
        "price_monthly": f"Does {subject} cost at most {comparison} USD per month?",
        "storage_gb": f"Can I store {comparison} GB on {subject}?",
        "support_hours": f"Does {subject} include at least {comparison} support hours per day?",
    }
    return comparisons[fact["relation"]]


def answer_for(fact: dict, value: str, polarity: str | None = None) -> str:
    subject = _subject(fact)
    answers = {
        "device_limit": f"{subject} supports {value} devices.",
        "price_monthly": f"{subject} costs {value} USD per month.",
        "storage_gb": f"{subject} includes {value} GB of storage.",
        "support_hours": f"{subject} includes {value} support hours per day.",
    }
    prefix = f"{polarity.capitalize()}. " if polarity else ""
    return prefix + answers[fact["relation"]]


def correction_text(edit: dict) -> str:
    return (f"Correction for {edit['entity_name']} (aliases: "
            f"{', '.join(edit['entity_aliases'])}), {edit['plan']}, "
            f"version {edit['version']}: {edit['relation']} changed from "
            f"{edit['original_value']} to {edit['corrected_value']} "
            f"{edit['unit']}, effective from {edit['effective_from']}. "
            f"Source: {edit['source']}.")


def _expected_fact(fact: dict, value: str | None = None) -> dict:
    return {"fact_id": fact["fact_id"], "entity_id": fact["entity_id"],
            "relation": fact["relation"], "value": value or fact["value"],
            "unit": fact["unit"], "plan": fact["plan"],
            "version": fact["version"], "as_of": AS_OF}


def _context(fact: dict) -> dict:
    return {"entity_id": fact["entity_id"], "plan": fact["plan"],
            "version": fact["version"], "as_of": AS_OF}


def build_dataset(seed: int = 42, family_count: int = 80) -> dict:
    splits = family_splits(seed, family_count)
    facts: list[dict] = []
    index: dict[tuple, dict] = {}
    for family in sorted(splits):
        rng = random.Random(f"{seed}:{family}")
        serial = family.split("-")[1]
        # Near-name products, versions and tiers share the assigned family.
        for product, plan, version in (("core", "Basic", "v2"),
                                       ("core", "Pro", "v2"),
                                       ("core", "Basic", "v1"),
                                       ("plus", "Basic", "v2")):
            entity_id = f"aster-{serial}-{product}"
            entity_name = f"Aster {serial}" + (" Plus" if product == "plus" else "")
            for relation in RELATIONS:
                fact = {
                    "fact_id": f"{entity_id}:{plan}:{version}:{relation}",
                    "family_id": family, "split": splits[family],
                    "entity_id": entity_id, "entity_name": entity_name,
                    "aliases": [f"A{serial}" + ("+" if product == "plus" else "")],
                    "relation": relation, "value": _base_value(relation, rng),
                    "unit": UNITS[relation], "plan": plan, "version": version,
                    "effective_from": "2025-01-01", "effective_to": None,
                    "source": f"controlled://handbook/{family}/{product}/{version}",
                    "lineage_id": f"{entity_id}:{plan}:{relation}",
                    "reviewed": False, "review_status": "pending",
                }
                facts.append(fact)
                index[(family, product, plan, version, relation)] = fact
    edits, scope_rows, answer_rows, challenges = [], [], [], []
    families_by_split = {split: sorted(family for family in splits if splits[family] == split)
                         for split in ("train", "validation", "test")}
    for family in sorted(splits):
        split = splits[family]
        split_families = families_by_split[split]
        other_family = split_families[(split_families.index(family) + 1) % len(split_families)]
        for relation_index, relation in enumerate(RELATIONS):
            fact = index[(family, "core", "Basic", "v2", relation)]
            values = _corrected_values(relation, fact["value"])
            # Alternate counterfactual worlds remain in the same lineage/split.
            variant_count = 2 if relation_index % 2 == 0 else 1
            for variant in range(variant_count):
                value = values[(int(family.split("-")[1]) + variant) % len(values)]
                edit = {
                    "edit_id": f"edit:{fact['fact_id']}:variant-{variant}",
                    "fact_id": fact["fact_id"], "family_id": family, "split": split,
                    "entity_id": fact["entity_id"], "entity_name": fact["entity_name"],
                    "entity_aliases": fact["aliases"], "relation": relation,
                    "original_value": fact["value"], "corrected_value": value,
                    "unit": fact["unit"], "plan": fact["plan"], "version": fact["version"],
                    "effective_from": EFFECTIVE_FROM, "effective_to": None,
                    "source": f"controlled://corrections/{family}/{relation}/{variant}",
                    "approval_status": "approved_in_simulated_scenario",
                    "lineage_id": fact["lineage_id"], "supersedes": None,
                    "scenario_id": f"scenario:{fact['fact_id']}:{variant}",
                    "reviewed": False, "review_status": "pending",
                }
                edits.append(edit)
                common = {"split": split, "family_id": family,
                          "edit_id": edit["edit_id"], "lineage_id": fact["lineage_id"],
                          "correction": edit, "correction_text": correction_text(edit),
                          "reviewed": False, "review_status": "pending"}
                for style in range(4):
                    # Every scenario has both polarities; relation or variant is
                    # never a shortcut for the comparison label.
                    comparison = int(value)
                    if style == 3:
                        comparison += -1 if relation == "price_monthly" else 1
                    polarity = None
                    if style >= 2:
                        truth = (int(value) <= comparison if relation == "price_monthly"
                                 else comparison <= int(value))
                        polarity = "yes" if truth else "no"
                    question = question_for(fact, style, comparison)
                    expected = _expected_fact(fact, value)
                    scope_rows.append({**common, "id": f"scope:{edit['edit_id']}:p{style}",
                                       "question": question, "request_context": _context(fact),
                                       "label": 1, "negative_type": None,
                                       "expected_fact": expected,
                                       "question_type": ("direct", "paraphrase", "comparison", "comparison")[style],
                                       "expected_polarity": polarity})
                    supporting = [index[(family, "core", "Basic", "v2", rel)]
                                  for rel in RELATIONS if rel != relation][:2]
                    support_text = " ".join(answer_for(row, row["value"]) for row in supporting)
                    answer_rows.append({**common, "id": f"answer:{edit['edit_id']}:p{style}",
                                        "question": question, "request_context": _context(fact),
                                        "supporting_context": supporting,
                                        "input_text": (f"answer correction: {correction_text(edit)} "
                                                       f"Question: {question} As of: {AS_OF}. "
                                                       f"Unchanged context: {support_text}"),
                                        "target_text": answer_for(fact, value, polarity),
                                        "expected_fact": expected, "expected_polarity": polarity,
                                        "question_type": ("direct", "paraphrase", "comparison", "comparison")[style]})
                next_relation = RELATIONS[(relation_index + 1) % len(RELATIONS)]
                negative_facts = (
                    index[(family, "core", "Basic", "v2", next_relation)],
                    index[(other_family, "core", "Basic", "v2", relation)],
                    index[(family, "plus", "Basic", "v2", relation)],
                    index[(family, "core", "Pro", "v2", relation)],
                    index[(family, "core", "Basic", "v1", relation)],
                    index[(other_family, "plus", "Basic", "v2", next_relation)],
                )
                for negative_type, negative_fact in zip(NEGATIVE_TYPES, negative_facts):
                    scope_rows.append({**common, "id": f"scope:{edit['edit_id']}:n:{negative_type}",
                                       "question": question_for(negative_fact),
                                       "request_context": _context(negative_fact), "label": 0,
                                       "negative_type": negative_type,
                                       "expected_fact": _expected_fact(negative_fact),
                                       "question_family_id": negative_fact["family_id"],
                                       "question_type": "out_of_scope"})
            if split == "test":
                # Draft challenge wording is intentionally held outside augmentation.
                wording = {
                    "device_limit": "I already have two devices attached; what's the total cap?",
                    "price_monthly": "What's the recurring bill, excluding any annual discounts?",
                    "storage_gb": "How much space do I get before I hit the quota?",
                    "support_hours": "For how long in a day can I reach the support team?",
                }
                challenges.append({"id": f"challenge:{fact['fact_id']}", "split": "test",
                                   "family_id": family, "lineage_id": fact["lineage_id"],
                                   "question": f"For {_subject(fact)}, {wording[relation]}",
                                   "request_context": _context(fact),
                                   "scenario_id": f"scenario:{fact['fact_id']}:0",
                                   "expected_fact": _expected_fact(fact, values[int(family.split('-')[1]) % len(values)]),
                                   "challenge_type": "wording_draft",
                                   "authorship": "agent_authored_draft_requires_independent_human_revision",
                                   "reviewed": False, "review_status": "pending"})
                novel_edit = {**edit, "edit_id": f"challenge-edit:{fact['fact_id']}:unseen-value",
                              "corrected_value": UNSEEN_CHALLENGE_VALUES[relation],
                              "scenario_id": f"challenge-scenario:{fact['fact_id']}:unseen-value",
                              "source": f"controlled://challenge-corrections/{family}/{relation}/unseen-value"}
                challenges.append({"id": f"challenge-unseen-value:{fact['fact_id']}",
                                   "split": "test", "family_id": family,
                                   "lineage_id": fact["lineage_id"],
                                   "question": f"For {_subject(fact)}, {wording[relation]}",
                                   "request_context": _context(fact),
                                   "scenario_id": novel_edit["scenario_id"],
                                   "correction": novel_edit,
                                   "expected_fact": _expected_fact(fact, novel_edit["corrected_value"]),
                                   "challenge_type": "unseen_value_draft",
                                   "authorship": "agent_authored_draft_requires_independent_human_revision",
                                   "reviewed": False, "review_status": "pending"})
    return {"facts": facts, "edits": edits, "scope": scope_rows,
            "answer": answer_rows, "challenges": challenges, "family_splits": splits}


def dataset_files(dataset: dict) -> dict[str, bytes]:
    files = {"catalogue.json": _json_bytes(dataset["facts"]),
             "edits.json": _json_bytes(dataset["edits"]),
             "split_assignments.json": _json_bytes(dataset["family_splits"]),
             "challenge_drafts.jsonl": _jsonl_bytes(dataset["challenges"])}
    for task in ("scope", "answer"):
        for split in ("train", "validation", "test"):
            files[f"{task}_{split}.jsonl"] = _jsonl_bytes(
                [row for row in dataset[task] if row["split"] == split])
    return files


def _validate_review(row: dict, record: dict | None) -> bool:
    if not isinstance(record, dict) or record.get("decision") != "accept":
        return False
    reviewer = record.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or record.get("row_sha256") != content_hash(row):
        return False
    try:
        timestamp = datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return False
    except (KeyError, ValueError, AttributeError):
        return False
    return True


def load_reviewed_rows(path: Path, registry_path: Path, *, require_all: bool = True) -> list[dict]:
    """Gate experiments on human decisions bound to exact row content.

    Rejected rows may be excluded only with a documented review decision.
    Pending, malformed, or stale decisions are fatal by default.
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = registry.get("rows", {})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    accepted, unresolved = [], []
    for row in rows:
        record = records.get(row["id"])
        if _validate_review(row, record):
            accepted.append({**row, "reviewed": True, "review_status": "accepted"})
        elif (isinstance(record, dict) and record.get("decision") == "reject"
              and isinstance(record.get("reason"), str) and record["reason"].strip()
              and isinstance(record.get("reviewer"), str) and record["reviewer"].strip()
              and record.get("row_sha256") == content_hash(row)):
            # Apply the same timestamp/provenance checks as accepted rows.
            if _validate_review(row, {**record, "decision": "accept"}):
                continue
            unresolved.append(row["id"])
        else:
            unresolved.append(row["id"])
    if require_all and unresolved:
        raise ValueError(f"{len(unresolved)} rows lack valid human review in {path.name}.")
    if not accepted:
        raise ValueError(f"No accepted reviewed rows in {path.name}.")
    return accepted


def write_dataset(output: Path, *, seed: int = 42, family_count: int = 80,
                  overwrite: bool = False) -> dict:
    dataset = build_dataset(seed, family_count)
    files = dataset_files(dataset)
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    lock_path = output / "test_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if any(hashes.get(name) != digest for name, digest in lock["files"].items()):
            raise ValueError("Frozen test artifacts differ. Create a new dataset version; do not overwrite.")
    # Preflight every output before changing any existing file.
    for name, contents in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != contents and not overwrite:
            raise ValueError(f"{name} differs; use a new version or explicit --overwrite before freezing.")
    counts = {task: dict(Counter(row["split"] for row in dataset[task]))
              for task in ("scope", "answer")}
    manifest = {"dataset_id": "editguard-controlled-v1", "generator_version": 2,
                "seed": seed, "family_count": family_count,
                "split_unit": "entity_family_before_augmentation", "counts": counts,
                "facts": len(dataset["facts"]), "edit_scenarios": len(dataset["edits"]),
                "challenge_drafts": len(dataset["challenges"]), "files": hashes,
                "review_status": "unreviewed_generated_draft",
                "eligible_for_training": False, "eligible_for_final_evaluation": False,
                "test_frozen": lock_path.exists(),
                "warning": "Simulated approvals are not administrator-approved production corrections."}
    output.mkdir(parents=True, exist_ok=True)
    files["manifest.json"] = _json_bytes(manifest)
    for name, contents in files.items():
        path = output / name
        if not path.exists() or path.read_bytes() != contents:
            path.write_bytes(contents)
    return manifest


def freeze_test(output: Path, registry_path: Path, evaluation_config: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    names = ("scope_test.jsonl", "answer_test.jsonl", "challenge_drafts.jsonl")
    for name in names:
        actual_hash = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if actual_hash != manifest["files"][name]:
            raise ValueError(f"{name} differs from manifest; finalize the reviewed dataset first.")
        load_reviewed_rows(output / name, registry_path)
    evaluation = json.loads(evaluation_config.read_text(encoding="utf-8"))
    if evaluation.get("status") != "frozen":
        raise ValueError("Evaluation configuration must have status=frozen before test locking.")
    lock = {"dataset_id": manifest["dataset_id"],
            "files": {name: manifest["files"][name] for name in names},
            "review_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "evaluation_config_sha256": hashlib.sha256(evaluation_config.read_bytes()).hexdigest(),
            "policy": "No test-guided tuning. A changed test set creates a new declared benchmark."}
    lock_path = output / "test_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Existing test lock differs; create a new declared dataset version.")
    lock_path.write_bytes(_json_bytes(lock))
    manifest["test_frozen"] = True
    (output / "manifest.json").write_bytes(_json_bytes(manifest))
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/controlled-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--freeze-test", action="store_true")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--evaluation-config", type=Path, default=ROOT / "configs/evaluation.json")
    args = parser.parse_args()
    if args.freeze_test:
        if args.reviews is None:
            parser.error("--freeze-test requires --reviews with human review decisions.")
        result = freeze_test(args.output, args.reviews, args.evaluation_config)
    else:
        result = write_dataset(args.output, seed=args.seed, family_count=args.families,
                               overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
