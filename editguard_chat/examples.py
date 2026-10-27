"""Explicit operator-approved fictional example catalogue, never automatic memory."""
from editguard_benchmark.data import load_dataset
from .store import ChatStore


def catalogue(root, *, locale="legacy"):
    if locale == "india":
        return india_catalogue()
    if locale != "legacy":
        raise ValueError("Unknown example locale")
    manifest, worlds, questions = load_dataset(root / "data/reasonke/operations-v1", "development")
    world = worlds[sorted(worlds)[0]]
    return {"schema": "editguard-fictional-operations-example-v1", "world": world,
            "questions": [row["question"] for row in questions if row["lineage_id"] == world["id"]
                          and row["category"] in ("direct", "two_hop", "wrong_version_or_tier", "same_entity_other_relation")],
            "notice": "Fictional example handbook. Approving it adds nine versioned facts; no model is trained and no answer is generated.",
            "dataset_id": manifest["dataset_id"]}


def india_catalogue():
    """A display-only fictional handbook; never an alteration of benchmark truth."""
    uid = "india-demo-001"
    service = "Kaveri Payment Gateway"
    old, new = "Bengaluru Platform Team", "Pune Reliability Team"
    old_person, person = "Meera Iyer", "Arjun Deshmukh"
    source = ("Controlled fictional India-themed operations handbook. Service, teams, people and "
              "assignments are invented for this demonstration. Indian city names are illustrative "
              "locations; this is not a claim about any real company or government service.")

    def fact(number, entity, relation, scope, question, answer):
        return {"id": f"{uid}-f{number}", "entity_id": f"{uid}-{entity}",
                "relation_id": relation, "scope_id": scope, "question": question,
                "answer": answer, "scope": question.removesuffix("?") + "; " + scope,
                "source": source}

    initial = [
        fact(1, "service", "owner", "release-2", f"Which team owns {service} in release 2?", old),
        fact(2, "old-team", "on_call_contact", "current", f"Who is the on-call contact for {old}?", old_person),
        fact(3, "new-team", "on_call_contact", "current", f"Who is the on-call contact for {new}?", person),
        fact(4, "service", "hosting_city", "release-2", f"Which Indian city hosts {service} in release 2?", "Bengaluru"),
        fact(5, "pro-service", "owner", "release-2", f"Which team owns {service} Pro in release 2?", "Hyderabad Integration Team"),
        fact(6, "service", "owner", "release-1", f"Which team owns {service} in release 1?", "Chennai Legacy Team"),
        fact(7, "archive", "city", "current", "In which Indian city is Konkan Records Office?", "New Delhi"),
        fact(8, "old-contact", "office_city", "current", f"In which Indian city is {old_person}'s office?", "Chennai"),
        fact(9, "new-contact", "office_city", "current", f"In which Indian city is {person}'s office?", "Hyderabad"),
    ]
    current = [dict(row) for row in initial]
    current[0].update(answer=new, source=source + " Approved demonstration ownership revision 2.")
    world = {"id": uid, "split": "display_only", "entity": service,
             "aliases": [service], "initial_facts": initial, "current_facts": current,
             "edit": {"id": uid + "-edit", "fact_id": uid + "-f1", "entity_id": uid + "-service",
                      "relation_id": "owner", "scope_id": "release-2", "original_value": old,
                      "corrected_value": new, "approval_status": "approved",
                      "source": current[0]["source"], "supersedes": None, "lineage_id": uid},
             "ground_truth": {"old_owner": old, "new_owner": new,
                              "old_contact": old_person, "new_contact": person,
                              "old_contact_office": "Chennai", "new_contact_office": "Hyderabad"}}
    return {"schema": "editguard-fictional-india-example-v1", "world": world,
            "questions": [initial[0]["question"],
                          f"Who is the on-call contact for the team that owns {service} in release 2?",
                          f"In which Indian city is the office of the on-call contact for the team that owns {service} in release 2?",
                          initial[3]["question"], initial[5]["question"]],
            "notice": "Fictional India-themed display example, separate from benchmark results. "
                      "Approving it adds nine versioned facts; no model is trained and no answer is generated.",
            "dataset_id": "editguard-india-display-v1", "display_only": True}


def approved_snapshot(example):
    source = ChatStore(":memory:")
    try:
        rows = {}
        for fact in example["world"]["initial_facts"]:
            rows[fact["id"]] = source.publish_correction(fact["question"], fact["answer"], fact["source"], fact["scope"],
                identity={key: fact[key] for key in ("entity_id", "relation_id", "scope_id")})
        owner = example["world"]["edit"]["fact_id"]
        current = next(row for row in example["world"]["current_facts"] if row["id"] == owner)
        suggestion = {**rows[owner], "answer": current["answer"], "source": current["source"],
                      "supersedes": rows[owner]["id"]}
        return source.export_snapshot(), suggestion
    finally:
        source.close()
