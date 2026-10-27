"""Author Indian-context fictional graphs with one-, two- and three-hop paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from editguard_benchmark.data import digest, encoded


CATEGORIES = ("direct", "direct_paraphrase", "two_hop", "two_hop_paraphrase",
              "three_hop", "three_hop_paraphrase", "wrong_version_or_tier",
              "same_entity_other_relation")
SOURCE = ("Controlled fictional Indian-context operations handbook. Service, team, "
          "person and office assignments are invented for this experiment; city "
          "names are illustrative labels, not claims about real organizations.")
RIVERS = ("Kaveri", "Narmada", "Godavari", "Sutlej", "Krishna", "Tapi",
          "Mahanadi", "Sabarmati", "Periyar", "Brahmaputra", "Mandovi", "Vaigai")
SYSTEMS = ("Gateway", "Registry", "Scheduler", "Relay")
TEAMS = ("Sahyadri", "Konkan", "Malwa", "Vindhya", "Deccan", "Aravalli", "Nilgiri", "Satpura")
FIRST = ("Aarav", "Diya", "Kabir", "Meera", "Arjun", "Ananya", "Rohan", "Ishita",
         "Vihaan", "Kavya", "Aditya", "Nandini")
LAST = ("Mehta", "Iyengar", "Rao", "Sen", "Desai", "Nair", "Kapoor", "Menon",
        "Shah", "Pillai", "Verma", "Bose", "Joshi", "Reddy", "Kulkarni", "Sethi")
CITIES = ("Bengaluru", "Mumbai", "Pune", "Hyderabad", "Chennai", "Kolkata", "Jaipur", "Kochi",
          "Lucknow", "Bhopal", "Ahmedabad", "Surat", "Nagpur", "Indore", "Patna", "Thiruvananthapuram")

# Different question constructions, not just substituted entity names, are
# reserved for the final partition. Canonical fact syntax and relations remain
# shared: this is a bounded wording challenge, not unrestricted language.
TEMPLATES = {
    "development": {
        "direct": ("Which team owns {service} in release 2?", "What is the owning team for {service}, release 2?", "Name the owner team of {service} in release 2."),
        "direct_paraphrase": ("For release 2 of {service}, give only the responsible team's name.", "Who maintains team ownership of {service} for release 2?", "Under release 2, what team is responsible for {service}?"),
        "two_hop": ("Who is the on-call contact for the team that owns {service} in release 2?", "Name the on-call person of {service}'s release-2 owner team.", "For {service} in release 2, who is on call for its owning team?"),
        "two_hop_paraphrase": ("Look up {service}'s release-2 owner, then return only that team's on-call person's name.", "Find the team responsible for {service} in release 2. Who is that team's on-call contact?", "For release 2 of {service}, give the person on call for its owner, not the team name."),
        "three_hop": ("In which city is the office of the on-call contact for the team that owns {service} in release 2?", "For {service} in release 2, name the office city of its owner's on-call person.", "What city hosts the office of the person on call for {service}'s release-2 owning team?"),
        "three_hop_paraphrase": ("Find {service}'s release-2 owning team and that team's on-call contact. Give only the contact's office city.", "Follow ownership and on-call responsibility for {service} in release 2, then report the person's office city.", "For release 2 of {service}, where is the office of the person reached through its owning team's on-call contact? Give the city."),
        "wrong_version_or_tier": ("Which team owns {service} in release 1?", "Name the owning team for release 1 of {service}.", "What is the release-1 owner team of {service}?"),
        "same_entity_other_relation": ("Which hosting zone serves {service} in release 2?", "What is the hosting zone of {service}, release 2?", "Name the release-2 hosting zone for {service}."),
    },
    "final": {
        "direct": ("Return the team listed as owner of release 2 of {service}.", "For {service}'s second release, which team holds ownership?", "Whose team owns the release-2 service called {service}? Give the team name."),
        "direct_paraphrase": ("An ownership record is needed for {service}, release 2. Supply the team name only.", "Identify the team to which ownership of {service} belongs in release 2.", "For the release-2 edition of {service}, what team should the owner field contain?"),
        "two_hop": ("For release 2 of {service}, give the name of the person on call for the team holding ownership.", "Identify {service}'s release-2 owning team and give only its on-call contact's name.", "Which on-call individual belongs to the team recorded as {service}'s owner in release 2?"),
        "two_hop_paraphrase": ("If I follow {service}'s release-2 ownership entry to the owning team's on-call entry, whose name do I reach?", "I need the on-call person, not a team: start with the owner of {service} under release 2.", "Who would I find by joining the release-2 owner of {service} to that team's on-call contact record?"),
        "three_hop": ("Starting with {service} in release 2, follow owner team, on-call contact and office location. Which city is reached?", "Give the city in the office record of the on-call person belonging to {service}'s release-2 owner.", "Which city is linked to {service} in release 2 through its owner team's on-call person's office?"),
        "three_hop_paraphrase": ("I have {service}'s release-2 ownership record. Use its owning team's on-call contact to locate the person's office, and answer with only the city.", "Which city would I travel to for the office of the on-call contact attached to the owner of {service} in release 2?", "Join {service}'s release-2 owner to the team's on-call person and then to that person's office. Return the city, not the person or team."),
        "wrong_version_or_tier": ("For the first release of {service}, what team belongs in its owner field?", "Use the historical release-1 entry for {service}: which team has ownership?", "Ignore release 2 when naming the owning team for {service}'s release 1."),
        "same_entity_other_relation": ("For release 2 of {service}, report the hosting-zone value rather than ownership.", "What zone is recorded for hosting {service}'s second release?", "Read {service}'s release-2 hosting entry and return its zone."),
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _label_records(questions):
    return [{key: row[key] for key in ("id", "expected_answer", "expected_aliases", "initial_answer",
                                       "required_fact_ids", "initial_required_fact_ids")} for row in questions]


def _fact_inputs(worlds):
    return [{"id": row["id"], "initial_facts": row["initial_facts"], "current_facts": row["current_facts"]} for row in worlds]


def build(seed=43):
    indices = list(range(48))
    random.Random(seed).shuffle(indices)
    development = set(indices[:16])  # Partition before constructing wording.
    lineages, questions = [], []
    for index in range(48):
        uid = f"depth-{index:03d}"
        split = "development" if index in development else "final"
        service = f"{RIVERS[index % 12]} {SYSTEMS[index // 12]}"
        people = [f"{FIRST[index % 12]} {LAST[index // 12 + role * 4]}" for role in range(4)]
        teams = [f"{TEAMS[(index + role * 2) % 8]} unit {index + 10 + role * 100}" for role in range(4)]
        cities = [CITIES[(index + offset) % len(CITIES)] for offset in (0, 5, 9, 12)]
        zone = CITIES[(index + 3) % len(CITIES)] + " hosting zone"
        role_names = ("old", "new", "neighbor", "historic")

        def fact(number, entity, relation, scope_id, subject, question, answer, object_entity):
            return {"id": f"{uid}-f{number}", "entity_id": f"{uid}-{entity}", "relation_id": relation,
                    "scope_id": scope_id, "entity_aliases": [subject], "question": question,
                    "answer": answer, "scope": question.removesuffix("?") + "; " + scope_id,
                    "source": SOURCE, "subject_entity_id": f"{uid}-{entity}",
                    "object_entity_id": f"{uid}-{object_entity}"}

        facts = [fact(1, "service", "owner", "release-2", service,
                      f"Which team owns {service} in release 2?", teams[0], "old-team")]
        for role, number in ((0, 2), (1, 3), (2, 9), (3, 11)):
            name = role_names[role]
            facts.append(fact(number, name + "-team", "on_call_contact", "current", teams[role],
                              f"Who is the on-call contact for {teams[role]}?", people[role], name + "-person"))
        for role, number in ((0, 4), (1, 5), (2, 10), (3, 12)):
            name = role_names[role]
            facts.append(fact(number, name + "-person", "office_city", "current", people[role],
                              f"In which city is {people[role]}'s office?", cities[role], name + "-office-city"))
        facts.extend([
            fact(6, "service", "hosting_zone", "release-2", service,
                 f"Which hosting zone serves {service} in release 2?", zone, "hosting-zone"),
            fact(7, "pro-service", "owner", "release-2", service + " Pro",
                 f"Which team owns {service} Pro in release 2?", teams[2], "neighbor-team"),
            fact(8, "service", "owner", "release-1", service,
                 f"Which team owns {service} in release 1?", teams[3], "historic-team"),
        ])
        initial = sorted(facts, key=lambda row: int(row["id"].rsplit("f", 1)[1]))
        current = [dict(row) for row in initial]
        current[0].update(answer=teams[1], object_entity_id=f"{uid}-new-team",
                          source=SOURCE + " Approved fictional ownership revision 2.")
        random.Random(seed + index).shuffle(initial)
        random.Random(seed + index).shuffle(current)
        world = {"id": uid, "split": split, "entity": service, "aliases": [service],
                 "initial_facts": initial, "current_facts": current,
                 "edit": {"id": uid + "-edit", "fact_id": uid + "-f1", "entity_id": uid + "-service",
                          "relation_id": "owner", "scope_id": "release-2", "original_value": teams[0],
                          "corrected_value": teams[1], "approval_status": "approved",
                          "source": SOURCE + " Approved fictional ownership revision 2.", "supersedes": None,
                          "lineage_id": uid},
                 "ground_truth": {"old_owner": teams[0], "new_owner": teams[1],
                                  "old_contact": people[0], "new_contact": people[1],
                                  "old_city": cities[0], "new_city": cities[1]}}
        specifications = ((1, teams[1], teams[0], [1], "owner"), (1, teams[1], teams[0], [1], "owner"),
                          (2, people[1], people[0], [1, 3], "owner_contact"),
                          (2, people[1], people[0], [1, 3], "owner_contact"),
                          (3, cities[1], cities[0], [1, 3, 5], "owner_contact_office_city"),
                          (3, cities[1], cities[0], [1, 3, 5], "owner_contact_office_city"),
                          (0, teams[3], teams[3], [8], "owner"),
                          (0, zone, zone, [6], "hosting_zone"))
        for category, (depth, expected, before, support, relation) in zip(CATEGORIES, specifications):
            template_index = index % len(TEMPLATES[split][category])
            wording = TEMPLATES[split][category][template_index]
            applicable = depth > 0
            questions.append({"id": uid + "-" + category, "lineage_id": uid, "split": split,
                "category": category, "depth": depth, "applicable": applicable,
                "wording_family": split, "template_id": f"{split}:{category}:{template_index}",
                "question": wording.format(service=service), "expected_answer": expected,
                "expected_aliases": [expected], "initial_answer": before,
                "requested_entity": uid + "-service", "requested_relation": relation,
                "requested_scope": "release-1" if category == "wrong_version_or_tier" else "release-2",
                "required_fact_ids": [f"{uid}-f{number}" for number in support],
                "initial_required_fact_ids": ([uid + "-f1", uid + "-f2", uid + "-f4"][:depth]
                                              if applicable else [f"{uid}-f{number}" for number in support]),
                "inapplicable_edit_aliases": [] if applicable else [teams[1], people[1], cities[1]],
                "review": {"policy": "deterministic_graph_path_audit", "human_reviewed": False}})
        lineages.append(world)
    return lineages, questions


def audit(lineages, questions):
    _require(len(lineages) == 48 and len(questions) == 384, "Expected 48 worlds and 384 questions")
    _require(len({row["id"] for row in lineages}) == 48, "Duplicate world identity")
    _require(len({row["id"] for row in questions}) == 384, "Duplicate question identity")
    aliases, canonical = {s: set() for s in TEMPLATES}, {s: set() for s in TEMPLATES}
    _require(set(TEMPLATES["development"]) == set(CATEGORIES) == set(TEMPLATES["final"]), "Template categories differ")
    development_templates = {template for templates in TEMPLATES["development"].values() for template in templates}
    final_templates = {template for templates in TEMPLATES["final"].values() for template in templates}
    _require(development_templates.isdisjoint(final_templates), "Wording template leakage")
    checked = 0
    for world in lineages:
        split, uid = world["split"], world["id"]
        _require(split in TEMPLATES, "Unknown world split")
        initial = {row["id"]: row for row in world["initial_facts"]}
        current = {row["id"]: row for row in world["current_facts"]}
        _require(len(initial) == len(current) == 12 and set(initial) == set(current), "Expected identical twelve fact identities")
        _require(set(current) == {f"{uid}-f{number}" for number in range(1, 13)}, "Fact identity escapes its lineage")
        changed = [key for key in initial if initial[key] != current[key]]
        _require(changed == [world["edit"]["fact_id"]], "Only the release-2 ownership fact may change")
        edited_before, edited_after = initial[uid + "-f1"], current[uid + "-f1"]
        _require(edited_before["answer"] == world["edit"]["original_value"] != world["edit"]["corrected_value"] == edited_after["answer"], "Edit values differ from ownership records")
        _require(edited_before["object_entity_id"] == uid + "-old-team" and edited_after["object_entity_id"] == uid + "-new-team", "Edited ownership identities differ")
        _require(all(edited_before[key] == edited_after[key] for key in edited_before if key not in ("answer", "object_entity_id", "source")), "Edit changes ownership scope or subject")
        _require(world["aliases"] == edited_after["entity_aliases"] == [world["entity"]], "Service alias differs")
        _require(world["edit"]["approval_status"] == "approved" and world["edit"]["lineage_id"] == uid, "Edit lineage/approval differs")
        alias_owners = {}
        for row in current.values():
            for alias in row["entity_aliases"]:
                folded = alias.casefold()
                _require(folded not in alias_owners or alias_owners[folded] == row["entity_id"], "Ambiguous alias inside world")
                alias_owners[folded] = row["entity_id"]
            aliases[split].update(alias.casefold() for alias in row["entity_aliases"])
            key = (row["entity_id"], row["relation_id"], row["scope_id"])
            _require(key not in canonical[split], "Duplicate scoped relation")
            canonical[split].add(key)
            _require(row["subject_entity_id"] == row["entity_id"], "Fact subject identity differs")
        expected_specs = {
            "direct": (1, [1], [1], "owner", ["owner"]),
            "direct_paraphrase": (1, [1], [1], "owner", ["owner"]),
            "two_hop": (2, [1, 3], [1, 2], "owner_contact", ["owner", "on_call_contact"]),
            "two_hop_paraphrase": (2, [1, 3], [1, 2], "owner_contact", ["owner", "on_call_contact"]),
            "three_hop": (3, [1, 3, 5], [1, 2, 4], "owner_contact_office_city", ["owner", "on_call_contact", "office_city"]),
            "three_hop_paraphrase": (3, [1, 3, 5], [1, 2, 4], "owner_contact_office_city", ["owner", "on_call_contact", "office_city"]),
            "wrong_version_or_tier": (0, [8], [8], "owner", ["owner"]),
            "same_entity_other_relation": (0, [6], [6], "hosting_zone", ["hosting_zone"]),
        }
        cases = [row for row in questions if row["lineage_id"] == uid]
        _require(len(cases) == 8 and {row["category"] for row in cases} == set(CATEGORIES), "World question categories differ")
        for question in cases:
            _require(question["split"] == split and question["wording_family"] == split, "Lineage/wording split mismatch")
            family, category, template_index = question["template_id"].split(":")
            _require(family == split and category == question["category"], "Template identity mismatch")
            expected_wording = TEMPLATES[family][category][int(template_index)].format(service=world["entity"])
            _require(question["question"] == expected_wording, "Question differs from declared template")
            depth, after_ids, before_ids, relation, relations = expected_specs[category]
            _require(question["depth"] == depth and question["applicable"] == (depth > 0), "Depth or edit applicability differs")
            _require(question["requested_entity"] == uid + "-service" and question["requested_relation"] == relation, "Requested subject/relation differs")
            _require(question["required_fact_ids"] == [f"{uid}-f{number}" for number in after_ids], "Wrong current subject path")
            _require(question["initial_required_fact_ids"] == [f"{uid}-f{number}" for number in before_ids], "Wrong initial subject path")
            paths = ((current, "required_fact_ids", "expected_answer"), (initial, "initial_required_fact_ids", "initial_answer"))
            for view, ids_key, answer_key in paths:
                supports = [view[key] for key in question[ids_key]]
                _require(len(supports) == (question["depth"] or 1), "Path depth differs")
                _require(supports[0]["entity_id"] == question["requested_entity"] and supports[0]["scope_id"] == question["requested_scope"], "Path starts from wrong subject/scope")
                _require([row["relation_id"] for row in supports] == relations, "Path relation sequence differs")
                _require(supports[-1]["answer"] == question[answer_key], "Terminal fact does not endorse target")
                for first, second in zip(supports, supports[1:]):
                    _require(first["object_entity_id"] == second["subject_entity_id"], "Broken entity join")
                    _require(first["answer"] in second["entity_aliases"], "Joined display labels differ")
            _require(question["expected_aliases"] == [question["expected_answer"]], "Unexpected answer alias")
            _require(question["expected_answer"] not in question["inapplicable_edit_aliases"], "Locality target equals edited value")
            _require(question["requested_scope"] == ("release-1" if category == "wrong_version_or_tier" else "release-2"), "Requested scope differs")
            checked += 1
    _require(aliases["development"].isdisjoint(aliases["final"]), "Subject/entity alias leakage")
    _require(canonical["development"].isdisjoint(canonical["final"]), "Scoped entity/relation leakage")
    counts = {s: sum(row["split"] == s for row in questions) for s in TEMPLATES}
    _require(counts == {"development": 128, "final": 256}, "Question split counts differ")
    return {"status": "passed", "checked_question_paths": checked, "checked_current_and_initial_paths": checked * 2,
            "question_counts": counts, "entity_aliases_disjoint": True, "scoped_relation_keys_disjoint": True,
            "question_template_banks_disjoint": True, "review_policy": "deterministic_graph_path_audit",
            "human_reviewed_data": False,
            "limitations": ["Controlled fictional Indian-context assignments", "No independent human semantic review",
                           "Within each split a small template bank repeats", "City values and relation/version vocabulary repeat across splits",
                           "Pretraining/upstream fine-tuning disjointness is not established", "Three hops do not represent arbitrary graph depth"]}


def write_dataset(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset output exists; preserve it and choose another version")
    worlds, questions = build()
    review = audit(worlds, questions)
    output.mkdir(parents=True)
    for name, rows in (("lineages.jsonl", worlds), ("questions.jsonl", questions)):
        (output / name).write_bytes(b"".join(encoded(row) for row in rows))
    (output / "audit.json").write_bytes(encoded(review))
    labels = _label_records(questions)
    inputs = _fact_inputs(worlds)
    manifest = {"schema": "editguard-depth-dataset-v1", "status": "authored_awaiting_separate_protocol_freeze",
                "dataset_id": "editguard-depth-v1", "seed": 43, "lineages": 48, "questions": 384,
                "development_lineages": 16, "final_lineages": 32, "facts_per_world": 12,
                "categories": list(CATEGORIES), "question_depths": [0, 1, 2, 3],
                "split_unit": "edit_lineage_before_wording_augmentation",
                "wording_split": "disjoint_template_banks_development_and_final",
                "files": {name: digest(output / name) for name in ("lineages.jsonl", "questions.jsonl", "audit.json")},
                "factual_inputs_sha256": hashlib.sha256(encoded(inputs)).hexdigest(),
                "labels_sha256": hashlib.sha256(encoded(labels)).hexdigest(),
                "wording_templates_sha256": hashlib.sha256(encoded(TEMPLATES)).hexdigest(),
                "generator_sha256": digest(__file__), "generated_model_answers": 0,
                "truth_source": SOURCE, "human_reviewed_data": False}
    (output / "manifest.json").write_bytes(encoded(manifest))
    return manifest


def load_dataset(path, split):
    _require(split in TEMPLATES, "Choose development or final split")
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "editguard-depth-dataset-v1" and manifest.get("dataset_id") == "editguard-depth-v1", "Unknown dataset schema/identity")
    _require(set(manifest["files"]) == {"lineages.jsonl", "questions.jsonl", "audit.json"}, "Unknown dataset file set")
    for name, expected in manifest["files"].items():
        _require(digest(path / name) == expected, "Dataset input checksum mismatch: " + name)
    worlds = [json.loads(line) for line in (path / "lineages.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [json.loads(line) for line in (path / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    audit(worlds, questions)
    _require(manifest["generator_sha256"] == digest(__file__), "Dataset generator source differs")
    _require(manifest["labels_sha256"] == hashlib.sha256(encoded(_label_records(questions))).hexdigest(), "Label identity differs")
    _require(manifest["factual_inputs_sha256"] == hashlib.sha256(encoded(_fact_inputs(worlds))).hexdigest(), "Fact-view identity differs")
    _require(manifest["wording_templates_sha256"] == hashlib.sha256(encoded(TEMPLATES)).hexdigest(), "Wording bank identity differs")
    return manifest, {row["id"]: row for row in worlds if row["split"] == split}, [row for row in questions if row["split"] == split]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/reasonke/depth-v1")
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
