"""Independently review all 504 pilot-v3 development rows using explicit facts.

Standard library only. No generator, formatter, repairer, scorer, ML package,
existing review decision, test or challenge corpus is imported or read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROW_FILES = tuple(f"{task}_{split}.jsonl" for task in ("scope", "answer") for split in ("train", "validation"))
SOURCE_FILES = (*ROW_FILES, "catalogue.json", "derivation.json", "selection.json", "split_assignments.json")
UNITS = {"device_limit": "devices", "price_monthly": "USD/month", "storage_gb": "GB", "support_hours": "hours/day"}
NEGATIVES = ("same_entity_different_property", "different_entity_same_property", "similar_name", "wrong_tier", "wrong_version", "ordinary")
MUTABLE = {"id", "question", "question_type", "expected_polarity"}
QUESTION_FORMS = (
    ("devices_direct", "device_limit", "direct", r"How many devices does (?P<subject>.+) support\?", None),
    ("devices_paraphrase", "device_limit", "paraphrase", r"What is the device allowance on (?P<subject>.+)\?", None),
    ("devices_comparison", "device_limit", "comparison", r"Can I connect (?P<quantity>\d+) (?P<device_word>devices?) on (?P<subject>.+)\?", "allowance"),
    ("price_direct", "price_monthly", "direct", r"What is the monthly price of (?P<subject>.+)\?", None),
    ("price_paraphrase", "price_monthly", "paraphrase", r"How much would I pay each month for (?P<subject>.+)\?", None),
    ("price_comparison", "price_monthly", "comparison", r"Does (?P<subject>.+) cost at most (?P<quantity>\d+) USD per month\?", "budget"),
    ("storage_direct", "storage_gb", "direct", r"How much storage is included with (?P<subject>.+)\?", None),
    ("storage_paraphrase", "storage_gb", "paraphrase", r"What is the storage quota for (?P<subject>.+)\?", None),
    ("storage_comparison", "storage_gb", "comparison", r"Can I store (?P<quantity>\d+) GB on (?P<subject>.+)\?", "allowance"),
    ("support_direct", "support_hours", "direct", r"How many support hours per day does (?P<subject>.+) include\?", None),
    ("support_paraphrase", "support_hours", "paraphrase", r"For how many hours each day is support available on (?P<subject>.+)\?", None),
    ("support_comparison", "support_hours", "comparison", r"Does (?P<subject>.+) include at least (?P<quantity>\d+) support (?P<hour_word>hours?) per day\?", "allowance"),
)
ASSERTION_FORMS = (
    ("device_limit", r"(?P<subject>.+) supports (?P<value>\d+) (?P<device_word>devices?)\."),
    ("price_monthly", r"(?P<subject>.+) costs (?P<value>\d+) USD per month\."),
    ("storage_gb", r"(?P<subject>.+) includes (?P<value>\d+) GB of storage\."),
    ("support_hours", r"(?P<subject>.+) includes (?P<value>\d+) support (?P<hour_word>hours?) per day\."),
)


def reject_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def read_rows(path):
    return [json.loads(line, object_pairs_hook=reject_duplicate_keys)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def content_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective(record, as_of):
    moment = date.fromisoformat(as_of)
    return date.fromisoformat(record["effective_from"]) <= moment and (
        not record.get("effective_to") or moment < date.fromisoformat(record["effective_to"]))


def noun_agreement(match, quantity):
    fields = match.groupdict()
    return ((not fields.get("device_word") or fields["device_word"] == ("device" if quantity == 1 else "devices"))
            and (not fields.get("hour_word") or fields["hour_word"] == ("hour" if quantity == 1 else "hours")))


def assertion(text, subjects):
    prefix = re.match(r"^(Yes|No)\. ", text)
    polarity = prefix[1].lower() if prefix else None
    body = text[prefix.end():] if prefix else text
    hits = [(relation, re.fullmatch(pattern, body)) for relation, pattern in ASSERTION_FORMS]
    hits = [(relation, match) for relation, match in hits if match]
    if len(hits) != 1:
        return None
    relation, match = hits[0]
    candidates = subjects[(match["subject"], relation)]
    if len(candidates) != 1:
        return None
    fact = candidates[0]
    value = Decimal(match["value"])
    return {key: fact[key] for key in ("fact_id", "entity_id", "relation", "plan", "version", "unit")} | {
        "value": str(value), "polarity": polarity, "noun_agreement": noun_agreement(match, value)}


def negative_semantics(kind, correction, fact):
    same = {key: correction[key] == fact[key] for key in ("entity_id", "relation", "plan", "version")}
    return {
        "same_entity_different_property": same["entity_id"] and not same["relation"] and same["plan"] and same["version"],
        "different_entity_same_property": not same["entity_id"] and same["relation"] and same["plan"] and same["version"] and correction["family_id"] != fact["family_id"],
        "similar_name": not same["entity_id"] and same["relation"] and same["plan"] and same["version"] and correction["family_id"] == fact["family_id"] and fact["entity_name"].startswith(correction["entity_name"] + " Plus"),
        "wrong_tier": same["entity_id"] and same["relation"] and not same["plan"] and same["version"],
        "wrong_version": same["entity_id"] and same["relation"] and same["plan"] and not same["version"],
        "ordinary": not same["entity_id"] and not same["relation"] and correction["family_id"] != fact["family_id"],
    }.get(kind, False)


def preflight_directory(directory, version):
    directory = directory.resolve()
    # Inventory names only; refuse unsupported/test artifacts before reading any.
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != set(SOURCE_FILES) | {"manifest.json"}:
        raise ValueError("Only exact pilot development files are accepted; no test/challenge/review or unknown artifacts")
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("pilot_version") != version or manifest.get("input_format_version") != "editguard-input-v2"
            or manifest.get("purpose") != "development_pilot_not_final_evaluation"
            or manifest.get("test_frozen") is not False or manifest.get("test_files_included") is not False
            or set(manifest.get("files", {})) != set(SOURCE_FILES)):
        raise ValueError("Expected an unfrozen, complete pilot-v2/v3 development manifest")
    hashes = {name: file_hash(directory / name) for name in (*SOURCE_FILES, "manifest.json")}
    if any(hashes[name] != manifest["files"][name] for name in SOURCE_FILES):
        raise ValueError("Manifest file checksum mismatch")
    return directory, manifest, hashes


def audit(dataset, origin, reviewed_at):
    dataset, manifest, hashes = preflight_directory(dataset, 3)
    origin, original_manifest, origin_hashes = preflight_directory(origin, 2)
    if dataset == origin or manifest["origin_hashes"] != origin_hashes:
        raise ValueError("Pilot-v3 must bind a distinct, unchanged pilot-v2 origin")
    if manifest["origin_dataset_id"] != original_manifest["dataset_id"]:
        raise ValueError("Origin dataset identity mismatch")
    rows = [(name, row) for name in ROW_FILES for row in read_rows(dataset / name)]
    originals = {row["id"]: row for name in ROW_FILES for row in read_rows(origin / name)}
    facts = read_json(dataset / "catalogue.json")
    catalogue = {fact["fact_id"]: fact for fact in facts}
    assignments = read_json(dataset / "split_assignments.json")
    derivations = read_json(dataset / "derivation.json")
    derivation = {item["id"]: item for item in derivations}
    by_id = {row["id"]: row for _, row in rows}
    subjects = defaultdict(list)
    for fact in facts:
        for alias in set([fact["entity_name"], *fact.get("aliases", [])]):
            subjects[(f"{alias} {fact['plan']} version {fact['version']}", fact["relation"])].append(fact)
    totals = defaultdict(lambda: {"denominator": 0, "passed": 0, "failed": 0, "failed_ids": []})
    decisions = []
    for filename, row in rows:
        checks, issues = {}, []
        result = {"id": row["id"], "file": filename, "split": row["split"], "family_id": row["family_id"],
                  "edit_id": row["edit_id"], "lineage_id": row["lineage_id"], "row_sha256": content_hash(row),
                  "reviewer": "EditGuard independent automated development reviewer", "reviewer_type": "automated_development",
                  "reviewed_at": reviewed_at, "question": row["question"], "computed_applicable": None,
                  "computed_scope_label": None, "computed_polarity": None, "parsed_question_fact": None, "parsed_target": None}

        def check(name, value):
            value = bool(value)
            checks[name] = value
            totals[name]["denominator"] += 1
            totals[name]["passed" if value else "failed"] += 1
            if not value:
                totals[name]["failed_ids"].append(row["id"])
                issues.append(name)
            return value

        try:
            correction, truth, context = row["correction"], row["expected_fact"], row["request_context"]
            hits = [(template, relation, style, re.fullmatch(pattern, row["question"]), operator)
                    for template, relation, style, pattern, operator in QUESTION_FORMS]
            hits = [hit for hit in hits if hit[3]]
            if not check("one_question_form", len(hits) == 1):
                raise ValueError("Question has no unique supported interpretation")
            template, relation, style, match, operator = hits[0]
            candidates = subjects[(match["subject"], relation)]
            if not check("one_scoped_catalogue_fact", len(candidates) == 1):
                raise ValueError("Question has no unique catalogue subject/relation")
            fact = candidates[0]
            result["template_id"] = template
            result["parsed_question_fact"] = {key: fact[key] for key in (
                "fact_id", "entity_id", "entity_name", "relation", "plan", "version", "value", "unit")} | {"as_of": context["as_of"]}
            applies = all(correction[key] == fact[key] for key in ("entity_id", "relation", "plan", "version")) and effective(correction, context["as_of"])
            value = Decimal(correction["corrected_value"] if applies else fact["value"])
            expected = {key: fact[key] for key in ("fact_id", "entity_id", "relation", "plan", "version", "unit")} | {"value": str(value), "as_of": context["as_of"]}
            result.update(computed_applicable=applies, computed_scope_label=int(applies), computed_expected_fact=expected)
            check("question_scope_and_effective_date", all(context[key] == fact[key] for key in ("entity_id", "plan", "version")) and effective(fact, context["as_of"]))
            check("applicability_label_or_answer", (type(row["label"]) is int and row["label"] == int(applies)) if "label" in row else applies)
            check("expected_identity_unit_date", all(truth[key] == expected[key] for key in expected if key != "value") and truth["unit"] == UNITS[relation])
            check("independently_selected_numeric_value", Decimal(truth["value"]) == value)
            base = catalogue.get(correction["fact_id"])
            check("edit_original_and_scoped_metadata", base is not None and all(correction[key] == base[key] for key in (
                "entity_id", "entity_name", "relation", "plan", "version", "unit", "family_id", "split", "lineage_id"))
                and correction["entity_aliases"] == base["aliases"] and Decimal(correction["original_value"]) == Decimal(base["value"])
                and Decimal(correction["corrected_value"]) != Decimal(base["value"])
                and row["lineage_id"] == correction["lineage_id"] and row["edit_id"] == correction["edit_id"])
            check("all_references_within_split", row["split"] == filename.split("_")[1].split(".")[0]
                and assignments[row["family_id"]] == row["split"] and fact["split"] == row["split"]
                and correction["split"] == row["split"] and assignments[fact["family_id"]] == row["split"]
                and all(support["split"] == row["split"] for support in row.get("supporting_context", [])))
            check("source_flags_pending_and_simulated", row.get("reviewed") is False and row.get("review_status") == "pending"
                and correction.get("reviewed") is False and correction.get("review_status") == "pending"
                and correction["approval_status"] == "approved_in_simulated_scenario")
            check("input_format_v2", row.get("input_format_version") == "editguard-input-v2")
            description = re.fullmatch(r"Correction for (?P<name>.+) \(aliases: (?P<aliases>.+)\), (?P<plan>.+), version (?P<version>v\d+): (?P<relation>\w+) changed from (?P<old>\d+) to (?P<new>\d+) (?P<unit>[^,]+), effective from (?P<date>\d{4}-\d{2}-\d{2})\. Source: (?P<source>.+)\.", row["correction_text"])
            display_unit = correction["unit"]
            if Decimal(correction["corrected_value"]) == 1:
                display_unit = {"devices": "device", "hours/day": "hour/day"}.get(display_unit, display_unit)
            check("edit_text_matches_metadata_and_noun_agreement", description is not None
                and description["name"] == correction["entity_name"] and description["aliases"].split(", ") == correction["entity_aliases"]
                and all(description[key] == correction[key] for key in ("plan", "version", "relation", "source"))
                and Decimal(description["old"]) == Decimal(correction["original_value"])
                and Decimal(description["new"]) == Decimal(correction["corrected_value"])
                and description["unit"] == display_unit and description["date"] == correction["effective_from"])
            polarity = None
            if operator:
                quantity = Decimal(match["quantity"])
                polarity = "yes" if (value <= quantity if operator == "budget" else quantity <= value) else "no"
                result.update(comparison_quantity=str(quantity), comparison_operator="price<=budget" if operator == "budget" else "request<=allowance")
                check("comparison_noun_agreement", noun_agreement(match, quantity))
            result["computed_polarity"] = polarity
            check("independent_comparison_polarity", row.get("expected_polarity") == polarity)
            check("question_type", row["question_type"] == style)
            actual_style = "comparison_" + polarity if style == "comparison" else style
            result["computed_question_style"] = actual_style
            check("scope_style_matches_computed_polarity", actual_style in ("direct", "paraphrase", "comparison_yes", "comparison_no"))
            if row.get("label") == 0:
                check("negative_type_semantics", not applies and negative_semantics(row.get("negative_type"), correction, fact))
            if filename.startswith("answer"):
                target = assertion(row["target_text"], subjects)
                result["parsed_target"] = target
                check("target_endorsed_subject_relation_value_unit_polarity", target is not None
                    and all(target[key] == expected[key] for key in expected if key != "as_of")
                    and target["polarity"] == polarity and target["noun_agreement"])
                support = row["supporting_context"]
                check("two_unchanged_other_property_support_facts", len(support) == 2 and len({item["relation"] for item in support}) == 2
                    and all(item == catalogue.get(item["fact_id"]) and all(item[key] == fact[key] for key in ("entity_id", "plan", "version"))
                            and item["relation"] != relation and effective(item, context["as_of"]) for item in support))
                parsed_input = re.fullmatch(r"answer correction: (?P<correction>.+) Question: (?P<question>.+) As of: (?P<date>\d{4}-\d{2}-\d{2})\. Unchanged context: (?P<context>.+)", row["input_text"])
                sentences = re.findall(r"[^.]+\.", parsed_input["context"]) if parsed_input else []
                good = len(sentences) == len(support)
                for sentence, item in zip(sentences, support):
                    endorsed = assertion(sentence.strip(), subjects)
                    good = good and endorsed is not None and all(endorsed[key] == item[key] for key in (
                        "fact_id", "entity_id", "relation", "plan", "version", "unit", "value")) and endorsed["polarity"] is None and endorsed["noun_agreement"]
                check("input_exact_edit_question_date_support", parsed_input is not None
                    and parsed_input["correction"] == row["correction_text"] and parsed_input["question"] == row["question"]
                    and parsed_input["date"] == context["as_of"] and good)
                pair = by_id.get(row["id"].replace(":answer:", ":scope:", 1))
                check("positive_scope_counterpart", pair is not None and pair["label"] == 1
                    and all(pair.get(key) == row.get(key) for key in ("question", "correction", "request_context", "expected_fact", "expected_polarity", "question_type", "split", "lineage_id", "input_format_version")))
            provenance = derivation.get(row["id"])
            original = originals.get(row["id"].replace("pilot-v3:", "pilot-v2:", 1))
            check("v3_row_derivation_hash", provenance is not None and provenance["new_row_sha256"] == result["row_sha256"]
                and original is not None and provenance["source_row_id"] == original["id"]
                and provenance["source_row_sha256"] == content_hash(original)
                and provenance["lineage_id"] == provenance["source_lineage_id"] == row["lineage_id"])
            immutable = {key: val for key, val in row.items() if key not in MUTABLE}
            check("v2_truth_scope_lineage_and_answer_preserved", original is not None
                and immutable == {key: val for key, val in original.items() if key not in MUTABLE}
                and provenance["immutable_fields_sha256"] == content_hash(immutable)
                and (row.get("label") == 0 or all(row.get(key) == original.get(key) for key in MUTABLE - {"id"})))
            check("derivation_changed_fields_and_style", provenance is not None
                and provenance["changed_fields"] == sorted(key for key in MUTABLE - {"id"}
                    if original.get(key) != row.get(key) or (key in original) != (key in row))
                and provenance["question_style"] == (actual_style if filename.startswith("scope") else None))
        except (KeyError, TypeError, ValueError, ArithmeticError) as error:
            issues.append("review_exception:" + str(error))
        result.update(decision="flagged" if issues else "accept", checks=checks, issues=issues,
            reason="Independent parsed subject/relation, original/corrected value, applicability, comparison polarity, target/support endorsement and origin lineage are consistent under controlled demonstration semantics."
                   if not issues else "Independent automated review flags the listed ambiguity or inconsistency for resolution.")
        decisions.append(result)
    global_checks = {
        "exact504_unique_rows": len(rows) == len(by_id) == len(originals) == 504,
        "partition_counts": dict(Counter(name for name, _ in rows)) == {"scope_train.jsonl": 240, "scope_validation.jsonl": 120, "answer_train.jsonl": 96, "answer_validation.jsonl": 48},
        "v3_development_only": manifest["pilot_version"] == 3 and manifest["test_files_included"] is False,
        "unique_catalogue_and_derivations": len(catalogue) == len(facts) and len(derivation) == len(derivations) == 504,
        "origin_catalogue_selection_and_splits_unchanged": all(hashes[name] == origin_hashes[name] for name in ("catalogue.json", "selection.json", "split_assignments.json")),
        "four_train_two_validation_families": Counter(assignments.values()) == {"train": 4, "validation": 2},
    }
    for name in SOURCE_FILES:
        global_checks["manifest_hash:" + name] = hashes[name] == manifest["files"][name]
    split_sets = {split: {"family": set(), "entity": set(), "lineage": set()} for split in ("train", "validation")}
    for _, row in rows:
        values = split_sets[row["split"]]
        values["family"].add(row["family_id"])
        for item in (row["correction"], catalogue[row["expected_fact"]["fact_id"]], *row.get("supporting_context", [])):
            values["family"].add(item["family_id"])
            values["entity"].add(item["entity_id"])
            values["lineage"].add(item["lineage_id"])
    for key in ("family", "entity", "lineage"):
        global_checks["disjoint:" + key] = not split_sets["train"][key] & split_sets["validation"][key]
    balance, worlds = {}, defaultdict(list)
    for split in ("train", "validation"):
        pairs = [decision for decision in decisions if decision["file"] == f"scope_{split}.jsonl"]
        balance[split] = {style: {"positive": sum(d.get("computed_question_style") == style and d.get("computed_scope_label") == 1 for d in pairs),
                                  "negative": sum(d.get("computed_question_style") == style and d.get("computed_scope_label") == 0 for d in pairs)}
                          for style in ("direct", "paraphrase", "comparison_yes", "comparison_no")}
        wanted = {"positive": 24 if split == "train" else 12, "negative": 36 if split == "train" else 18}
        global_checks["matched_style_prior:" + split] = all(value == wanted for value in balance[split].values())
        global_checks["style_manifest_matches:" + split] = all(
            {key: value for key, value in item.items() if key != "positive_prior"} == balance[split][style]
            and item["positive_prior"] == .4 for style, item in manifest["scope_style_balance"][split].items())
    for filename, row in rows:
        if filename.startswith("scope"):
            worlds[row["edit_id"]].append(row)
    global_checks["complete36_edit_worlds"] = len(worlds) == 36 and all(
        len(group) == 10 and Counter(row["label"] for row in group) == {0: 6, 1: 4}
        and Counter(row["negative_type"] for row in group if row["label"] == 0) == Counter(NEGATIVES) for group in worlds.values())
    variants = defaultdict(dict)
    for _, row in rows:
        variants[row["lineage_id"]][row["edit_id"]] = row["correction"]
    for lineage, edits in variants.items():
        global_checks["counterfactual:" + lineage] = (len({edit["corrected_value"] for edit in edits.values()}) == len(edits)
            and len({edit["scenario_id"] for edit in edits.values()}) == len(edits)
            and len({edit["split"] for edit in edits.values()}) == 1)
    if not all(global_checks.values()):
        for decision in decisions:
            decision["decision"] = "flagged"
            decision["issues"].append("global_dataset_consistency")
            decision["reason"] = "Global hash/split/scenario/style checks require resolution before acceptance."
    unchanged = all(file_hash(dataset / name) == value for name, value in hashes.items()) and all(file_hash(origin / name) == value for name, value in origin_hashes.items())
    if not unchanged:
        raise ValueError("Dataset or origin bytes changed during review; no decisions published")
    source = Path(__file__).read_text(encoding="utf-8")
    accepted = sum(item["decision"] == "accept" for item in decisions)
    return {"schema_version": 1, "review_id": "pilot-v3-independent-automated-development-v1",
            "reviewer_type": "automated_development", "reviewer": "EditGuard independent automated development reviewer",
            "reviewed_at": reviewed_at, "authorization_basis": "User explicitly authorized automated review of the 504 development labels and continued the project; pilot-v3 is a bounded revision of those same rows.",
            "dataset_id": manifest["dataset_id"], "input_format_version": manifest["input_format_version"],
            "status": "all504_accepted_for_automated_development" if accepted == 504 else "flagged_rows_require_resolution",
            "human_review_performed": False, "human_approval_claimed": False, "registry_written": False,
            "dataset_modified": False, "test_or_challenge_rows_read": False, "model_or_gpu_used": False,
            "total_rows": len(decisions), "decision_counts": {"accept": accepted, "flagged": len(decisions) - accepted},
            "by_file": {name: dict(Counter(item["decision"] for item in decisions if item["file"] == name)) for name in ROW_FILES},
            "source_hashes": {"data/generated/pilot-v3/" + name: value for name, value in hashes.items()},
            "origin_source_hashes": {"data/generated/pilot-v2/" + name: value for name, value in origin_hashes.items()},
            "sources_unchanged_after_review": unchanged,
            "implementation": {"source": source, "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "script": "scripts/audit_pilot_v3.py", "script_file_sha256": file_hash(Path(__file__)),
                "independence": "Standard-library-only independent anchored question and assertion parsing; no generator, formatter, repair, scorer, model or previous review decision is used."},
            "checks": dict(totals), "global_checks": global_checks, "scope_style_balance": balance,
            "class_prior": {"train": {"positive": 96, "total": 240, "value": .4}, "validation": {"positive": 48, "total": 120, "value": .4}},
            "templates": [{"id": name, "relation": relation, "style": style, "independent_regex": regex,
                           "rows": sum(item.get("template_id") == name for item in decisions)}
                          for name, relation, style, regex, _ in QUESTION_FORMS],
            "limitations": ["Machine-only consistency review of controlled fictional facts, not independent human semantic review.",
                "Four bounded styles and six families do not establish natural-language generalization.",
                "All comparisons use total allowance or at-most price; no usage, tax, hardware or time-of-day conditions.",
                "Audit computes dataset labels and targets only; no generated-model accuracy or post-revision improvement measured."],
            "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--origin", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pilot-v3-independent-review.json")
    parser.add_argument("--reviewed-at", default=datetime.now(timezone.utc).isoformat(), help="Explicit ISO timestamp for exact decision reproduction")
    args = parser.parse_args()
    try:
        stamp = datetime.fromisoformat(args.reviewed_at.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Review timestamp must include a timezone")
        if args.output.exists() or args.output.resolve() in (args.dataset.resolve(), args.origin.resolve()):
            raise ValueError("A fresh output file is required")
        if any(source.resolve() in args.output.resolve().parents or args.output.resolve() in source.resolve().parents
               for source in (args.dataset, args.origin)):
            raise ValueError("Review output must be separate from source datasets")
        result = audit(args.dataset, args.origin, args.reviewed_at)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output.resolve()), "rows": result["total_rows"],
            "decision_counts": result["decision_counts"], "global_checks_passed": all(result["global_checks"].values()),
            "audit_sha256": file_hash(args.output), "human_review_performed": False, "test_or_challenge_rows_read": False}, indent=2))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
