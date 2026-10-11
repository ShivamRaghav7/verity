"""Human review/export and closed-family pilots; never auto-accept labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_data import canonical_bytes, content_hash
except ModuleNotFoundError:
    from prepare_data import canonical_bytes, content_hash


DEVELOPMENT_SPLITS = ("train", "validation")
TASKS = ("scope", "answer")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def checked_rows(data_dir: Path, task: str, split: str) -> list[dict]:
    if task not in TASKS or split not in DEVELOPMENT_SPLITS:
        raise ValueError("Review tooling only reads scope/answer train/validation files.")
    manifest = read_json(data_dir / "manifest.json")
    name = f"{task}_{split}.jsonl"
    path = data_dir / name
    if manifest.get("files", {}).get(name) != file_hash(path):
        raise ValueError(f"{name} content differs from its manifest.")
    rows = [json.loads(line, object_pairs_hook=_unique_object)
            for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = set()
    for row in rows:
        if row.get("split") != split or row["id"] in ids:
            raise ValueError("Dataset contains a split mismatch or duplicate row ID.")
        ids.add(row["id"])
    return rows


def _catalogue(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "catalogue.json"
    if not path.exists():
        return {}
    manifest = read_json(data_dir / "manifest.json")
    if manifest.get("files", {}).get("catalogue.json") != file_hash(path):
        raise ValueError("Catalogue content differs from its manifest.")
    return {fact["fact_id"]: fact for fact in read_json(path)
            if fact.get("split") in DEVELOPMENT_SPLITS}


def integrity_flags(row: dict, catalogue: dict[str, dict] | None = None) -> list[str]:
    """Independent structured/numeric checks, never a human review decision."""
    catalogue = catalogue or {}
    edit, expected, context = row["correction"], row["expected_fact"], row["request_context"]
    flags = []
    for key in ("entity_id", "plan", "version"):
        if context.get(key) != expected.get(key):
            flags.append(f"request_ground_truth_{key}_mismatch")
    same_scope = all(edit.get(key) == expected.get(key)
                     for key in ("entity_id", "relation", "plan", "version"))
    try:
        as_of = date.fromisoformat(context["as_of"])
        effective = date.fromisoformat(edit["effective_from"])
        end = date.fromisoformat(edit["effective_to"]) if edit.get("effective_to") else None
        if end is not None and end <= effective:
            flags.append("invalid_effective_interval")
        applicable = same_scope and effective <= as_of and (end is None or as_of < end)
    except (KeyError, ValueError, TypeError):
        applicable = False
        flags.append("invalid_or_missing_date")
    if "label" in row and (type(row["label"]) is not int or row["label"] not in (0, 1)
                           or row["label"] != int(applicable)):
        flags.append("scope_label_disagrees_with_structured_scope")
    if "target_text" in row and not applicable:
        flags.append("answer_row_has_inapplicable_correction")
    if applicable and expected.get("value") != edit.get("corrected_value"):
        flags.append("applicable_ground_truth_value_mismatch")
    if edit.get("original_value") == edit.get("corrected_value"):
        flags.append("correction_does_not_change_value")
    original = catalogue.get(edit.get("fact_id"))
    if original and original.get("value") != edit.get("original_value"):
        flags.append("correction_original_value_mismatch")
    expected_original = catalogue.get(expected.get("fact_id"))
    if not applicable and expected_original and expected.get("value") != expected_original.get("value"):
        flags.append("out_of_scope_ground_truth_value_mismatch")
    for support in row.get("supporting_context", []):
        if all(support.get(key) == edit.get(key) for key in ("entity_id", "relation", "plan", "version")):
            flags.append("support_contains_edited_relation")
        known = catalogue.get(support.get("fact_id"))
        if known and known.get("value") != support.get("value"):
            flags.append("support_value_differs_from_catalogue")
        if support.get("split") not in (None, row.get("split")):
            flags.append("support_split_leakage")
    polarity = row.get("expected_polarity")
    if row.get("question_type") == "comparison":
        patterns = {"device_limit": r"connect (\d+(?:\.\d+)?) devices",
                    "price_monthly": r"at most (\d+(?:\.\d+)?) USD",
                    "storage_gb": r"store (\d+(?:\.\d+)?) GB",
                    "support_hours": r"at least (\d+(?:\.\d+)?) support hours"}
        match = re.search(patterns.get(expected["relation"], r"(?!)"), row["question"], re.I)
        if not match:
            flags.append("comparison_requires_manual_semantic_check")
        else:
            try:
                amount, value = float(match[1]), float(expected["value"])
                truth = value <= amount if expected["relation"] == "price_monthly" else amount <= value
                if polarity != ("yes" if truth else "no"):
                    flags.append("comparison_polarity_mismatch")
            except (TypeError, ValueError):
                flags.append("comparison_value_not_numeric")
        if "target_text" in row and not row["target_text"].lower().startswith(f"{polarity}."):
            flags.append("target_polarity_mismatch")
    if "target_text" in row:
        target = row["target_text"]
        value = re.escape(str(expected["value"]))
        predicates = {"device_limit": rf"supports {value} devices\.",
                      "price_monthly": rf"costs {value} USD per month\.",
                      "storage_gb": rf"includes {value} GB of storage\.",
                      "support_hours": rf"includes {value} support hours per day\."}
        subject = f"{edit['entity_name']} {expected['plan']} version {expected['version']}"
        if subject not in target or not re.search(predicates.get(expected["relation"], r"(?!)"), target):
            flags.append("target_subject_relation_or_value_mismatch")
    return sorted(set(flags))


def selected_rows(data_dir: Path, task: str, split: str, families: list[str] | None = None) -> list[dict]:
    rows = checked_rows(data_dir, task, split)
    if not families:
        return rows
    available = {row["family_id"] for row in rows}
    if set(families) - available:
        raise ValueError("Requested family is not present in the selected development split.")
    return [row for row in rows if row["family_id"] in set(families)]


def validate_decision(row: dict, decision: dict) -> None:
    if not isinstance(decision, dict) or decision.get("decision") not in ("accept", "reject"):
        raise ValueError("A human decision must be accept or reject.")
    if not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip():
        raise ValueError("A named human reviewer is required.")
    try:
        stamp = datetime.fromisoformat(decision["reviewed_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Timestamp has no timezone.")
    except (KeyError, ValueError, AttributeError, TypeError):
        raise ValueError("reviewed_at must be an ISO timestamp with an explicit timezone.") from None
    if decision.get("row_sha256") != content_hash(row):
        raise ValueError(f"Review hash is stale or incorrect for {row['id']}.")
    if decision["decision"] == "reject" and (not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
        raise ValueError("A rejected row requires a reason.")


def import_decisions(data_dir: Path, incoming_path: Path, registry_path: Path) -> dict:
    incoming = read_json(incoming_path)
    if not isinstance(incoming, dict) or not isinstance(incoming.get("rows"), dict):
        raise ValueError("Decision file must be an object with a rows object.")
    manifest = read_json(data_dir / "manifest.json")
    if incoming.get("dataset_id") not in (None, manifest["dataset_id"]):
        raise ValueError("Incoming decisions refer to a different dataset.")
    if incoming.get("origin_manifest_sha256") not in (None, file_hash(data_dir / "manifest.json")):
        raise ValueError("Incoming export was created from a different manifest snapshot.")
    rows = {row["id"]: row for task in TASKS for split in DEVELOPMENT_SPLITS
            if (data_dir / f"{task}_{split}.jsonl").exists()
            for row in checked_rows(data_dir, task, split)}
    existing = read_json(registry_path) if registry_path.exists() else {"dataset_id": manifest["dataset_id"], "rows": {}}
    if existing.get("dataset_id") not in (None, manifest["dataset_id"]):
        raise ValueError("Existing registry belongs to a different dataset; choose the correct registry.")
    if not isinstance(existing.get("rows"), dict):
        raise ValueError("Existing review registry rows must be an object.")
    merged = {**existing, "rows": dict(existing["rows"])}
    added = 0
    for row_id, decision in incoming["rows"].items():
        if row_id not in rows:
            raise ValueError(f"Unknown or non-development row ID: {row_id}. Test changes are prohibited.")
        validate_decision(rows[row_id], decision)
        if row_id in merged["rows"]:
            if merged["rows"][row_id] != decision:
                raise ValueError(f"Conflicting existing decision for {row_id}; no silent overwrite.")
        else:
            merged["rows"][row_id] = decision
            added += 1
    lock_path = data_dir / "test_lock.json"
    if added and lock_path.exists() and registry_path.exists():
        lock = read_json(lock_path)
        if file_hash(registry_path) == lock.get("review_registry_sha256"):
            raise ValueError("The frozen test review registry is immutable; use a separate development registry.")
    if not added and registry_path.exists():
        return {"added_decisions": 0, "total_decisions": len(merged["rows"]), "registry_path": str(registry_path)}
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if not registry_path.exists() or registry_path.read_bytes() != payload:
        temporary = registry_path.with_name(registry_path.name + ".pending")
        if temporary.exists():
            raise ValueError("Pending registry write already exists; inspect it before retrying.")
        temporary.write_bytes(payload)
        temporary.replace(registry_path)
    return {"added_decisions": added, "total_decisions": len(merged["rows"]), "registry_path": str(registry_path)}


HTML_TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EditGuard human dataset review</title><style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}main{max-width:1080px;margin:auto;padding:24px}
article{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:20px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px}
label{display:block;margin:10px 0}input,select,button{font:inherit;padding:8px;margin:4px}button{cursor:pointer;border:1px solid #64748b;border-radius:6px;background:#fff}
.notice{border-left:5px solid #2563eb;padding:12px;background:#eff6ff}.flags{color:#9a3412}h2{overflow-wrap:anywhere}a{color:#1d4ed8}
</style></head><body><main><h1>EditGuard human dataset review</h1>
<p class="notice">No row is accepted automatically. Rule checks are integrity flags, not semantic review. Inspect question, correction, scope, truth, polarity and supporting facts before deciding. Export decisions and import them with the CLI to enforce hashes and preserve the authoritative registry.</p>
<p id="summary"></p><label>Your reviewer name <input id="reviewer" autocomplete="name" required></label>
<button id="export" type="button">Export explicit decisions as JSON</button><p id="status" role="status" aria-live="polite"></p><nav id="navigation" aria-label="Review rows"></nav><div id="cards"></div>
</main><script type="application/json" id="review-data">__DATA__</script><script>
"use strict";
const payload=JSON.parse(document.getElementById("review-data").textContent),decisions={};
document.getElementById("summary").textContent=payload.dataset_id+" · "+payload.task+" · "+payload.split+" · "+payload.rows.length+" rows. These records remain pending until explicit human decisions are imported.";
const cards=document.getElementById("cards"),navigation=document.getElementById("navigation");
for(const [index,item] of payload.rows.entries()){
 const row=item.row,article=document.createElement("article");article.id="row-"+index;
 const heading=document.createElement("h2");heading.textContent=(index+1)+". "+row.id;article.append(heading);
 const question=document.createElement("p");question.textContent=row.question;article.append(question);
 const flags=document.createElement("p");flags.className="flags";flags.textContent="Automated integrity flags: "+(item.integrity_flags.join(", ")||"none; human semantic review still required");article.append(flags);
 const pre=document.createElement("pre");pre.textContent=JSON.stringify(row,null,2);article.append(pre);
 const label=document.createElement("label");label.textContent="Decision for row "+(index+1)+" ";
 const select=document.createElement("select");for(const [value,text] of [["","Pending"],["accept","Accept after inspection"],["reject","Reject with reason"]]){const option=document.createElement("option");option.value=value;option.textContent=text;select.append(option)}label.append(select);article.append(label);
 const reasonLabel=document.createElement("label");reasonLabel.textContent="Rejection reason or optional acceptance note ";const reason=document.createElement("input");reason.type="text";reason.size=60;reasonLabel.append(reason);article.append(reasonLabel);
 const save=document.createElement("button");save.type="button";save.textContent="Record this explicit decision";save.addEventListener("click",()=>{
 const reviewer=document.getElementById("reviewer").value.trim();if(!reviewer||!select.value||(select.value==="reject"&&!reason.value.trim())){document.getElementById("status").textContent="Provide your name, an explicit decision and a rejection reason when needed.";return}
 const record={decision:select.value,reviewer,reviewed_at:new Date().toISOString(),row_sha256:item.row_sha256};if(reason.value.trim())record.reason=reason.value.trim();
 if(decisions[row.id]&&!window.confirm("Replace your unsaved decision for this row? The authoritative CLI rejects conflicts with previously imported decisions."))return;
 decisions[row.id]=record;document.getElementById("status").textContent="Explicit decisions recorded in this page: "+Object.keys(decisions).length+". Export them before closing; no automatic save.";
 });article.append(save);cards.append(article);
 if(index<20){const link=document.createElement("a");link.href="#row-"+index;link.textContent="Row "+(index+1)+" ";navigation.append(link)}
}
document.getElementById("export").addEventListener("click",()=>{
 if(!Object.keys(decisions).length){document.getElementById("status").textContent="No explicit decisions to export.";return}
 const result={dataset_id:payload.dataset_id,origin_manifest_sha256:payload.manifest_sha256,rows:decisions};
 const blob=new Blob([JSON.stringify(result,null,2)+"\\n"],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="review-decisions.json";a.click();URL.revokeObjectURL(url);
});
</script></body></html>'''


def export_html(data_dir: Path, output: Path, task: str, split: str,
                families: list[str] | None = None, limit: int | None = None) -> dict:
    rows = selected_rows(data_dir, task, split, families)
    if limit is not None:
        if limit <= 0:
            raise ValueError("Review limit must be positive.")
        rows = rows[:limit]
    catalogue = _catalogue(data_dir)
    payload = {"dataset_id": read_json(data_dir / "manifest.json")["dataset_id"],
               "manifest_sha256": file_hash(data_dir / "manifest.json"), "task": task, "split": split,
               "rows": [{"row": row, "row_sha256": content_hash(row),
                         "integrity_flags": integrity_flags(row, catalogue)} for row in rows]}
    # JSON in a script element must not contain an HTML closing-tag sequence.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    rendered = HTML_TEMPLATE.replace("__DATA__", encoded)
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError("HTML output differs; choose a new filename instead of silently overwriting.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return {"rows": len(rows), "output": str(output), "autoaccepted": 0}


def interactive_review(data_dir: Path, registry_path: Path, task: str, split: str,
                       families: list[str] | None = None) -> dict:
    reviewer = input("Your actual reviewer name: ").strip()
    if not reviewer:
        raise ValueError("Reviewer name is required.")
    catalogue = _catalogue(data_dir)
    decisions = {}
    for row in selected_rows(data_dir, task, split, families):
        print(json.dumps({"row": row, "automated_integrity_flags": integrity_flags(row, catalogue)}, indent=2))
        choice = input("[a]ccept, [r]eject, [s]kip pending, [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice not in ("a", "r"):
            continue
        reason = input("Rejection reason: ").strip() if choice == "r" else ""
        if choice == "r" and not reason:
            print("No rejection recorded: a reason is required.")
            continue
        record = {"decision": "accept" if choice == "a" else "reject", "reviewer": reviewer,
                  "reviewed_at": datetime.now(timezone.utc).isoformat(), "row_sha256": content_hash(row)}
        if reason:
            record["reason"] = reason
        decisions[row["id"]] = record
    temporary = registry_path.with_name(registry_path.name + ".incoming")
    if temporary.exists():
        raise ValueError("Incoming decision file already exists; inspect it before retrying.")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps({"rows": decisions}, indent=2), encoding="utf-8")
    result = import_decisions(data_dir, temporary, registry_path)
    temporary.unlink()
    return result


def build_pilot(source: Path, output: Path, *, task: str = "both", train_families: int = 4,
                validation_families: int = 2, seed: int = 42,
                maximum_families_per_split: int = 16, policy: str = "reroute") -> dict:
    """Finite whole-family pilot, with declared derived foreign negatives.

    The default reroutes only foreign negatives among selected same-split
    families. Optional closure retains original rows but may expand cyclic graphs.
    """
    if task not in (*TASKS, "both") or min(train_families, validation_families) < 2:
        raise ValueError("Pilot needs at least two families per development split.")
    if maximum_families_per_split <= 0 or policy not in ("reroute", "closure"):
        raise ValueError("Pilot task/counts must be valid and positive.")
    tasks = TASKS if task == "both" else (task,)
    source_manifest = read_json(source / "manifest.json")
    catalogue = _catalogue(source)
    entity_family = {fact["entity_id"]: fact["family_id"] for fact in catalogue.values()}
    source_hashes = {"manifest.json": file_hash(source / "manifest.json")}
    if (source / "catalogue.json").exists():
        source_hashes["catalogue.json"] = file_hash(source / "catalogue.json")
    selected, rows_out, requested, derivations = {}, {}, {}, []
    for split, count in (("train", train_families), ("validation", validation_families)):
        task_rows = {name: checked_rows(source, name, split) for name in tasks}
        for name in tasks:
            source_hashes[f"{name}_{split}.jsonl"] = file_hash(source / f"{name}_{split}.jsonl")
        rows = [row for records in task_rows.values() for row in records]
        available = sorted({row["family_id"] for row in rows})
        if count > len(available):
            raise ValueError(f"Requested {count} families exceeds {split} availability.")
        rng = random.Random(f"{seed}:{split}")
        rng.shuffle(available)
        chosen = set(available[:count])
        requested[split] = sorted(chosen)
        while policy == "closure":
            dependencies = set()
            for row in rows:
                if row["family_id"] not in chosen:
                    continue
                if row.get("question_family_id"):
                    dependencies.add(row["question_family_id"])
                reference = entity_family.get(row["expected_fact"]["entity_id"])
                if reference:
                    dependencies.add(reference)
            if dependencies - set(available):
                raise ValueError("Pilot row references a family outside its declared split.")
            expanded = chosen | dependencies
            if len(expanded) > maximum_families_per_split:
                raise ValueError(f"Whole-family {split} negative closure exceeds the {maximum_families_per_split}-family pilot budget; source references form a large connected component.")
            if expanded == chosen:
                break
            chosen = expanded
        if len(chosen) > maximum_families_per_split:
            raise ValueError("Requested family count exceeds the declared pilot budget.")
        selected[split] = sorted(chosen)
        for name, records in task_rows.items():
            rows_out[(name, split)] = []
            for original in records:
                if original["family_id"] not in chosen:
                    continue
                row = {**original, "id": f"pilot-v1:{seed}:{original['id']}",
                       "reviewed": False, "review_status": "pending"}
                referenced_family = original.get("question_family_id") or entity_family.get(original["expected_fact"]["entity_id"])
                rerouted = False
                if referenced_family and referenced_family not in chosen:
                    if policy != "reroute" or original.get("label") != 0:
                        raise ValueError("Only foreign negative questions may be rerouted.")
                    old_fact = catalogue.get(original["expected_fact"]["fact_id"])
                    if not old_fact:
                        raise ValueError("Foreign negative fact is absent from source catalogue.")
                    target_families = sorted(chosen - {original["family_id"]})
                    offset = int(hashlib.sha256(original["id"].encode()).hexdigest()[:8], 16) % len(target_families)
                    target_family = target_families[offset]
                    role = old_fact["entity_id"].rsplit("-", 1)[-1]
                    targets = [fact for fact in catalogue.values()
                               if fact["family_id"] == target_family
                               and fact["entity_id"].rsplit("-", 1)[-1] == role
                               and all(fact[key] == old_fact[key] for key in ("relation", "plan", "version"))]
                    if len(targets) != 1:
                        raise ValueError("Foreign negative role cannot be rerouted uniquely.")
                    replacement = targets[0]
                    if old_fact["entity_name"] not in original["question"]:
                        raise ValueError("Foreign question needs manual derivation; canonical source name missing.")
                    row["question"] = original["question"].replace(old_fact["entity_name"], replacement["entity_name"])
                    row["request_context"] = {**original["request_context"], "entity_id": replacement["entity_id"]}
                    row["expected_fact"] = {**original["expected_fact"], **{key: replacement[key]
                                            for key in ("fact_id", "entity_id", "relation", "value", "unit", "plan", "version")}}
                    row["question_family_id"] = target_family
                    rerouted = True
                rows_out[(name, split)].append(row)
                derivations.append({"id": row["id"], "source_row_id": original["id"],
                                    "source_row_sha256": content_hash(original),
                                    "new_row_sha256": content_hash(row), "foreign_negative_rerouted": rerouted})
    if set(selected["train"]) & set(selected["validation"]):
        raise ValueError("Pilot families overlap across train/validation.")
    files = {f"{name}_{split}.jsonl": b"".join(canonical_bytes(row) + b"\n" for row in rows_out[(name, split)])
             for name in tasks for split in DEVELOPMENT_SPLITS}
    selected_all = set(selected["train"] + selected["validation"])
    pilot_catalogue = [fact for fact in catalogue.values() if fact["family_id"] in selected_all]
    files["catalogue.json"] = (json.dumps(pilot_catalogue, indent=2, sort_keys=True) + "\n").encode()
    selection = {"seed": seed, "requested_families": requested, "selected_families": selected,
                 "policy": policy, "rerouted_rows": sum(record["foreign_negative_rerouted"] for record in derivations),
                 "description": ("whole source families; foreign negatives rerouted inside selected same-split families"
                                 if policy == "reroute" else "transitive same-split question/expected-entity family closure")}
    files["selection.json"] = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
    files["derivation.json"] = (json.dumps(derivations, indent=2, sort_keys=True) + "\n").encode()
    assignment = {family: split for split, families in selected.items() for family in families}
    files["split_assignments.json"] = (json.dumps(assignment, indent=2, sort_keys=True) + "\n").encode()
    selection_id = content_hash({"selected": selected, "tasks": tasks, "policy": policy})[:8]
    manifest = {"dataset_id": f"{source_manifest['dataset_id']}-pilot-v1-{seed}-{selection_id}",
                "schema_version": 1, "pilot_version": 1, "tasks": list(tasks), "purpose": "development_pilot_not_final_evaluation",
                "origin_dataset_id": source_manifest["dataset_id"], "origin_hashes": source_hashes,
                "counts": {name: {split: len(rows_out[(name, split)]) for split in DEVELOPMENT_SPLITS} for name in tasks},
                "family_counts": {split: len(selected[split]) for split in DEVELOPMENT_SPLITS},
                "review_status": "pending_independent_human_decisions", "eligible_for_training": False,
                "test_frozen": False, "test_files_included": False,
                "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if (output / "test_lock.json").exists():
        raise ValueError("Cannot change a frozen dataset directory into a pilot.")
    if source.resolve() == output.resolve() or output.resolve() in source.resolve().parents:
        raise ValueError("Pilot output must not overwrite or contain its source dataset.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Pilot output must be a new or empty directory; no silent overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        (output / name).write_bytes(contents)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "export-html", "interactive"):
        child = sub.add_parser(command)
        child.add_argument("--data-dir", type=Path, required=True)
        child.add_argument("--task", choices=TASKS, required=True)
        child.add_argument("--split", choices=DEVELOPMENT_SPLITS, required=True)
        child.add_argument("--family", action="append")
        if command == "export-html":
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--limit", type=int)
        if command == "interactive":
            child.add_argument("--registry", type=Path, required=True)
    child = sub.add_parser("import-decisions")
    child.add_argument("--data-dir", type=Path, required=True)
    child.add_argument("--incoming", type=Path, required=True)
    child.add_argument("--registry", type=Path, required=True)
    child = sub.add_parser("pilot")
    child.add_argument("--data-dir", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--task", choices=(*TASKS, "both"), default="both")
    child.add_argument("--train-families", type=int, default=4)
    child.add_argument("--validation-families", type=int, default=2)
    child.add_argument("--seed", type=int, default=42)
    child.add_argument("--maximum-families-per-split", type=int, default=16)
    child.add_argument("--policy", choices=("reroute", "closure"), default="reroute")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            catalogue = _catalogue(args.data_dir)
            rows = selected_rows(args.data_dir, args.task, args.split, args.family)
            flags = Counter(flag for row in rows for flag in integrity_flags(row, catalogue))
            result = {"rows": len(rows), "family_counts": dict(Counter(row["family_id"] for row in rows)),
                      "automated_integrity_flags": dict(flags), "human_decisions_created": 0}
        elif args.command == "export-html":
            result = export_html(args.data_dir, args.output, args.task, args.split, args.family, args.limit)
        elif args.command == "interactive":
            result = interactive_review(args.data_dir, args.registry, args.task, args.split, args.family)
        elif args.command == "import-decisions":
            result = import_decisions(args.data_dir, args.incoming, args.registry)
        else:
            result = build_pilot(args.data_dir, args.output, task=args.task,
                                 train_families=args.train_families, validation_families=args.validation_families,
                                 seed=args.seed, maximum_families_per_split=args.maximum_families_per_split,
                                 policy=args.policy)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
