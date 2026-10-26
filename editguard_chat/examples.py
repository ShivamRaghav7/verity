"""Explicit operator-approved fictional example catalogue, never automatic memory."""
from editguard_benchmark.data import load_dataset
from .store import ChatStore


def catalogue(root):
    manifest, worlds, questions = load_dataset(root / "data/reasonke/operations-v1", "development")
    world = worlds[sorted(worlds)[0]]
    return {"schema": "editguard-fictional-operations-example-v1", "world": world,
            "questions": [row["question"] for row in questions if row["lineage_id"] == world["id"]
                          and row["category"] in ("direct", "two_hop", "wrong_version_or_tier", "same_entity_other_relation")],
            "notice": "Fictional example handbook. Approving it adds nine versioned facts; no model is trained and no answer is generated.",
            "dataset_id": manifest["dataset_id"]}


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
