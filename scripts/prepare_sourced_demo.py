"""Create a fresh source-backed product-demo workspace. No model calls or benchmark changes."""
import argparse
import hashlib
import json
from pathlib import Path

from editguard_chat.store import ChatStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/demos/air-india-handover.json"


def load_fixture():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if fixture["schema"] != "verity-sourced-demonstration-v1":
        raise ValueError("Unknown demo fixture schema")
    for source in fixture["sources"].values():
        if not source["url"].startswith(("https://www.pib.gov.in/", "https://www.tata.com/")):
            raise ValueError("The curated fixture requires an official government or Tata source")
    return fixture


def publication(row, fixture):
    source = fixture["sources"][row["source_id"]]
    dated = source.get("date", "accessed " + source.get("accessed", fixture["verified_on"]))
    return {"question": row["question"], "answer": row["answer"], "scope": row["scope"],
            "source": source["title"] + " (" + dated + "). " + source["url"],
            "identity": row["identity"]}


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new workspace directory; existing user memory is preserved")
    fixture = load_fixture()
    output.mkdir(parents=True)
    store = ChatStore(output / "chat.sqlite3")
    try:
        records = {}
        for row in fixture["facts"]:
            approved = store.publish_correction(**publication(row, fixture))
            records[row["key"]] = approved["id"]
        replacement = publication(fixture["replacement"], fixture)
        replacement["supersedes"] = records["selected_owner"]
        report = {"schema": "verity-sourced-demo-workspace-v1", "fixture": str(FIXTURE.relative_to(ROOT)),
                  "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                  "purpose": fixture["purpose"], "notice": fixture["notice"],
                  "initial_revision": store.get_revision(), "fact_ids": records,
                  "suggested_replacement": replacement, "questions": fixture["questions"],
                  "new_model_calls": 0, "benchmark_observations": 0,
                  "storage": str(output.resolve())}
        (output / "sourced-demo-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        return report
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"prepared": True, "storage": result["storage"], "initial_revision": result["initial_revision"],
                      "model_calls": 0, "benchmark_observations": 0, "notice": result["notice"]}, indent=2))
