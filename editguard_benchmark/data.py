"""Author isolated fictional fact graphs, split lineages before wording augmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

CATEGORIES = ("direct", "direct_paraphrase", "two_hop", "two_hop_paraphrase", "same_entity_other_relation",
              "similar_entity_same_relation", "wrong_version_or_tier", "unrelated")
SOURCE = "Controlled fictional internal operations handbook. This record defines the fact in its stated release scope."
PREFIXES = ("Aster", "Beryl", "Cairn", "Dahlia", "Ember", "Fable", "Garnet", "Harbor", "Ibis", "Juniper", "Kestrel", "Larch")
SYSTEMS = ("Relay", "Gateway", "Registry", "Scheduler")
TEAM_WORDS = ("Cedar", "Silver", "Indigo", "Saffron", "Amber", "Violet", "Copper", "Willow")
PEOPLE_FIRST = ("Nira", "Tovan", "Serin", "Mara", "Ilan", "Veya", "Orin", "Lina")
PEOPLE_LAST = ("Voss", "Vale", "Moss", "Arden", "Quill", "Solen", "Dane", "Reeve", "Fen", "Lorne", "Hale", "Wren")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(seed=42):
    assigned = list(range(48))
    random.Random(seed).shuffle(assigned)
    development = set(assigned[:16])
    # Partition lineage identifiers before any question/wording construction.
    lineages, questions = [], []
    for index in range(48):
        uid = f"ops-{index:03d}"
        split = "development" if index in development else "final"
        service = f"{PREFIXES[index % 12]} {SYSTEMS[index // 12]}"
        other = service + " Pro"
        old = f"{TEAM_WORDS[index % 8]} unit {index + 11}"
        new = f"{TEAM_WORDS[(index + 3) % 8]} unit {index + 71}"
        neighbor = f"{TEAM_WORDS[(index + 5) % 8]} unit {index + 131}"
        historic = f"{TEAM_WORDS[(index + 6) % 8]} unit {index + 191}"
        old_person = f"{PEOPLE_FIRST[index % 8]} {PEOPLE_LAST[index // 8]}"
        person = f"{PEOPLE_FIRST[(index + 3) % 8]} {PEOPLE_LAST[index // 8 + 6]}"
        region = ("North sector", "South sector", "East sector", "West sector")[index % 4]
        archive = f"{PREFIXES[(index + 5) % 12]} records office {index + 1}"
        city = ("Lyris", "Velora", "Meren", "Kelm", "Tirren", "Olven")[index % 6]

        def fact(number, entity, relation, scope, question, answer):
            return {"id": f"{uid}-f{number}", "entity_id": f"{uid}-{entity}", "relation_id": relation,
                    "scope_id": scope, "question": question, "answer": answer,
                    "scope": f"{question.split('?')[0]}; {scope}", "source": SOURCE}

        initial = [
            fact(1, "service", "owner", "release-2", f"Which team owns {service} in release 2?", old),
            fact(2, "old-team", "on_call_contact", "current", f"Who is the on-call contact for {old}?", old_person),
            fact(3, "new-team", "on_call_contact", "current", f"Who is the on-call contact for {new}?", person),
            fact(4, "service", "region", "release-2", f"Which region hosts {service} in release 2?", region),
            fact(5, "pro-service", "owner", "release-2", f"Which team owns {other} in release 2?", neighbor),
            fact(6, "service", "owner", "release-1", f"Which team owns {service} in release 1?", historic),
            fact(7, "archive", "city", "current", f"In which city is {archive}?", city),
            fact(8, "neighbor-team", "charter", "current", f"What is the function of {neighbor}?", "Archive synchronization"),
            fact(9, "historic-team", "charter", "current", f"What is the function of {historic}?", "Telemetry processing"),
        ]
        current = [dict(row) for row in initial]
        current[0].update(answer=new, source=SOURCE + " Approved ownership revision 2.")
        # Stable shuffled order avoids always putting the requested fact first.
        random.Random(seed + index).shuffle(initial)
        random.Random(seed + index).shuffle(current)
        rows = {row["id"]: row for row in current}
        cases = [
            ("direct", f"Which team owns {service} in release 2?", new, old, [1]),
            ("direct_paraphrase", f"For release 2 of {service}, name the team responsible for ownership.", new, old, [1]),
            ("two_hop", f"Who is the on-call contact for the team that owns {service} in release 2?", person, old_person, [1, 3]),
            ("two_hop_paraphrase", f"In release 2, identify {service}'s owning team and give only that team's on-call person's name.", person, old_person, [1, 3]),
            ("same_entity_other_relation", f"Which region hosts {service} in release 2?", region, region, [4]),
            ("similar_entity_same_relation", f"Which team owns {other} in release 2?", neighbor, neighbor, [5]),
            ("wrong_version_or_tier", f"Which team owns {service} in release 1?", historic, historic, [6]),
            ("unrelated", f"In which city is {archive}?", city, city, [7]),
        ]
        # A few declared wording patterns; shared templates limit language-generalization claims.
        if index % 2:
            cases[1] = ("direct_paraphrase", f"What is the name of the team maintaining ownership of {service} for release 2?", new, old, [1])
        for category, query, expected, before, support in cases:
            applicable = category in CATEGORIES[:4]
            questions.append({"id": uid + "-" + category, "lineage_id": uid, "split": split,
                "category": category, "applicable": applicable, "question": query,
                "expected_answer": expected, "expected_aliases": [expected], "initial_answer": before,
                "requested_entity": f"{uid}-" + ("pro-service" if category == "similar_entity_same_relation" else "archive" if category == "unrelated" else "service"),
                "requested_relation": "owner_contact" if category in CATEGORIES[2:4] else "region" if category == "same_entity_other_relation" else "city" if category == "unrelated" else "owner",
                "requested_scope": "release-1" if category == "wrong_version_or_tier" else "current" if category == "unrelated" else "release-2",
                "required_fact_ids": [f"{uid}-f{number}" for number in support],
                "inapplicable_edit_aliases": [] if applicable else [new, person],
                "review": {"policy": "deterministic_graph_audit", "human_reviewed": False}})
        lineages.append({"id": uid, "split": split, "entity": service, "aliases": [service],
            "initial_facts": initial, "current_facts": current,
            "edit": {"id": uid + "-edit", "fact_id": uid + "-f1", "entity_id": uid + "-service",
                "relation_id": "owner", "scope_id": "release-2", "original_value": old, "corrected_value": new,
                "approval_status": "approved", "source": SOURCE + " Approved ownership revision 2.",
                "supersedes": None, "lineage_id": uid},
            "ground_truth": {"old_owner": old, "new_owner": new, "old_contact": old_person, "new_contact": person}})
    return lineages, questions


def audit(lineages, questions):
    assert len(lineages) == 48 and len(questions) == 384
    ids = {row["id"] for row in lineages}
    assert len(ids) == 48 and len({row["id"] for row in questions}) == 384
    checks = 0
    names_by_split = {split: set() for split in ("development", "final")}
    for lineage in lineages:
        cases = [row for row in questions if row["lineage_id"] == lineage["id"]]
        assert {row["category"] for row in cases} == set(CATEGORIES)
        initial = {row["id"]: row for row in lineage["initial_facts"]}
        current = {row["id"]: row for row in lineage["current_facts"]}
        assert len(current) <= 20 and sum(len(str(value)) for row in current.values() for value in row.values()) <= 12000
        changed = [key for key in initial if initial[key]["answer"] != current[key]["answer"]]
        assert changed == [lineage["edit"]["fact_id"]]
        names_by_split[lineage["split"]].update([lineage["entity"], *lineage["ground_truth"].values()])
        for row in cases:
            assert row["split"] == lineage["split"]
            supports = [current[key] for key in row["required_fact_ids"]]
            assert row["expected_answer"] == supports[-1]["answer"]
            if len(supports) == 2:
                assert supports[0]["answer"] in supports[1]["question"]
            assert row["expected_answer"] not in row["inapplicable_edit_aliases"]
            checks += 1
    counts = {split: sum(row["split"] == split for row in questions) for split in ("development", "final")}
    assert counts == {"development": 128, "final": 256}
    assert names_by_split["development"].isdisjoint(names_by_split["final"])
    return {"status": "passed", "checked_question_paths": checks, "question_counts": counts,
            "review_policy": "deterministic_graph_audit", "human_reviewed_data": False,
            "limitations": ["Fictional controlled catalogue", "Shared question templates", "No independent human wording review",
                           "Foundation pretraining or upstream fine-tuning disjointness is not established"]}


def write_dataset(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset output already exists; preserve it and choose another version")
    lineages, questions = build()
    review = audit(lineages, questions)
    output.mkdir(parents=True)
    for name, rows in (("lineages.jsonl", lineages), ("questions.jsonl", questions)):
        (output / name).write_bytes(b"".join(encoded(row) for row in rows))
    (output / "audit.json").write_bytes(encoded(review))
    manifest = {"schema": "editguard-operations-benchmark-v1", "status": "authored_not_final_evaluation_frozen",
                "dataset_id": "editguard-operations-v1", "seed": 42, "lineages": 48,
                "split_unit": "edit_lineage_before_wording_augmentation", "questions": 384,
                "development_lineages": 16, "final_lineages": 32, "categories": CATEGORIES,
                "files": {name: digest(output / name) for name in ("lineages.jsonl", "questions.jsonl", "audit.json")},
                "generator_sha256": digest(__file__), "generated_model_answers": 0,
                "truth_source": "Curated deterministic fictional graph; automated audit explicitly disclosed"}
    (output / "manifest.json").write_bytes(encoded(manifest))
    return manifest


def load_dataset(path, split):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(path / name) != expected:
            raise ValueError("Dataset input checksum mismatch: " + name)
    lineages = [json.loads(line) for line in (path / "lineages.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [json.loads(line) for line in (path / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    audit(lineages, questions)
    return manifest, {row["id"]: row for row in lineages if row["split"] == split}, [row for row in questions if row["split"] == split]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/reasonke/operations-v1")
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
