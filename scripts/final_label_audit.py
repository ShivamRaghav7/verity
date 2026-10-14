"""Independent deterministic consistency review of a prepared final packet.

No generator, formatter, scorer, model or human review decisions are consulted.
Anchored question/assertion grammars and structured catalogue facts are checked.
This is a machine review of the controlled benchmark, not human semantic review.
"""

from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
import re

from scripts.audit_pilot_v3 import (QUESTION_FORMS, UNITS, assertion, content_hash,
                                   effective, negative_semantics)

CHALLENGE_FORMS = (
    ("device_limit", r"For (?P<subject>.+), I already have two devices attached; what's the total cap\?"),
    ("price_monthly", r"For (?P<subject>.+), What's the recurring bill, excluding any annual discounts\?"),
    ("storage_gb", r"For (?P<subject>.+), How much space do I get before I hit the quota\?"),
    ("support_hours", r"For (?P<subject>.+), For how long in a day can I reach the support team\?"),
)
CORRECTION = re.compile(r"Correction for (?P<name>.+) \(aliases: (?P<aliases>.+)\), (?P<plan>.+), version (?P<version>v\d+): (?P<relation>\w+) changed from (?P<old>\d+) to (?P<new>\d+) (?P<unit>[^,]+), effective from (?P<date>\d{4}-\d{2}-\d{2})\. Source: (?P<source>.+)\.")


def audit_rows(facts, populations, assignments, reviewed_at):
    stamp = datetime.fromisoformat(reviewed_at)
    if stamp.utcoffset() is None:
        raise ValueError("Automated final review timestamp must include a timezone")
    catalogue = {fact["fact_id"]: fact for fact in facts}
    if len(catalogue) != len(facts):
        raise ValueError("Final catalogue contains duplicate fact IDs")
    subjects = defaultdict(list)
    for fact in facts:
        for alias in [fact["entity_name"], *fact.get("aliases", [])]:
            subjects[(f"{alias} {fact['plan']} version {fact['version']}", fact["relation"])].append(fact)
    decisions, totals = {}, defaultdict(lambda: Counter(passed=0, failed=0, denominator=0))
    all_rows = [row for rows in populations.values() for row in rows]
    by_id = {row["id"]: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("Final populations contain duplicate row IDs")
    for population, rows in populations.items():
        for row in rows:
            checks, issues = {}, []

            def check(name, condition):
                passed = bool(condition)
                checks[name] = passed
                totals[name]["denominator"] += 1
                totals[name]["passed" if passed else "failed"] += 1
                if not passed:
                    issues.append(name)
                return passed

            result = {"id": row["id"], "population": population, "row_sha256": content_hash(row),
                      "reviewer_type": "automated_final", "reviewer": "EditGuard deterministic final consistency audit",
                      "reviewed_at": reviewed_at}
            try:
                correction, truth, context = row["correction"], row["expected_fact"], row["request_context"]
                if population == "challenge":
                    hits = [(relation, "challenge", re.fullmatch(pattern, row["question"]), None)
                            for relation, pattern in CHALLENGE_FORMS]
                else:
                    hits = [(relation, style, re.fullmatch(pattern, row["question"]), operator)
                            for _, relation, style, pattern, operator in QUESTION_FORMS]
                hits = [hit for hit in hits if hit[2] is not None]
                if not check("unique_question_interpretation", len(hits) == 1):
                    raise ValueError("Question does not have one supported independent interpretation")
                relation, style, match, operator = hits[0]
                candidates = subjects[(match["subject"], relation)]
                if not check("unique_scoped_catalogue_fact", len(candidates) == 1):
                    raise ValueError("Question subject/relation is ambiguous or absent")
                fact = candidates[0]
                applies = all(correction[key] == fact[key] for key in ("entity_id", "relation", "plan", "version")) and effective(correction, context["as_of"])
                expected = {key: fact[key] for key in ("fact_id", "entity_id", "relation", "plan", "version", "unit")}
                expected.update(value=correction["corrected_value"] if applies else fact["value"], as_of=context["as_of"])
                value = Decimal(expected["value"])
                check("question_context_scope_date", all(context[key] == fact[key] for key in ("entity_id", "plan", "version")) and effective(fact, context["as_of"]))
                check("explicit_applicability_label", type(row["label"]) is int and row["label"] == int(applies))
                check("expected_fact_identity_value_unit_date", truth == expected and truth["unit"] == UNITS[relation])
                base = catalogue.get(correction["fact_id"])
                check("original_correction_fact_scope", base is not None
                      and all(correction[key] == base[key] for key in ("entity_id", "entity_name", "relation", "plan", "version", "unit", "family_id", "split", "lineage_id"))
                      and correction["entity_aliases"] == base["aliases"]
                      and Decimal(correction["original_value"]) == Decimal(base["value"])
                      and Decimal(correction["corrected_value"]) != Decimal(base["value"])
                      and row["lineage_id"] == correction["lineage_id"])
                check("test_only_references", row["split"] == fact["split"] == correction["split"] == "test"
                      and assignments[row["family_id"]] == assignments[fact["family_id"]] == assignments[correction["family_id"]] == "test"
                      and all(item["split"] == "test" and assignments[item["family_id"]] == "test" for item in row.get("supporting_context", [])))
                check("simulated_approval_disclosed", correction["approval_status"] == "approved_in_simulated_scenario"
                      and correction.get("reviewed") is False and correction.get("review_status") == "pending"
                      and row.get("reviewed") is False and row.get("review_status") == "pending")
                check("canonical_input_version", row["input_format_version"] == "editguard-input-v2")
                description = CORRECTION.fullmatch(row["correction_text"])
                unit = correction["unit"]
                if Decimal(correction["corrected_value"]) == 1:
                    unit = {"devices": "device", "hours/day": "hour/day"}.get(unit, unit)
                check("correction_text_matches_metadata", description is not None
                      and description["name"] == correction["entity_name"] and description["aliases"].split(", ") == correction["entity_aliases"]
                      and all(description[key] == correction[key] for key in ("plan", "version", "relation", "source"))
                      and Decimal(description["old"]) == Decimal(correction["original_value"])
                      and Decimal(description["new"]) == Decimal(correction["corrected_value"])
                      and description["unit"] == unit and description["date"] == correction["effective_from"])
                polarity = None
                if operator:
                    quantity = Decimal(match["quantity"])
                    polarity = "yes" if (value <= quantity if operator == "budget" else quantity <= value) else "no"
                check("independent_comparison_polarity", row.get("expected_polarity") == polarity)
                check("question_type", row["question_type"] == ("out_of_scope" if row["label"] == 0 else style))
                if row["label"] == 0:
                    check("negative_scope_semantics", not applies and negative_semantics(row.get("negative_type"), correction, fact))
                if population in ("answer", "challenge"):
                    check("oracle_correction_applicable", applies)
                if population == "challenge":
                    check("challenge_draft_authorship_disclosed", row["authorship"] == "agent_authored_draft_requires_independent_human_revision"
                          and row["challenge_type"] in ("wording_draft", "unseen_value_draft"))
                    check("challenge_scenario_reference", row["scenario_id"] == correction["scenario_id"])
                if population == "answer":
                    target = assertion(row["target_text"], subjects)
                    check("complete_target_subject_relation_value_unit_polarity", target is not None
                          and all(target[key] == expected[key] for key in expected if key != "as_of")
                          and target["polarity"] == polarity and target["noun_agreement"])
                    support = row["supporting_context"]
                    check("unchanged_supporting_context", len(support) == 2 and len({item["relation"] for item in support}) == 2
                          and all(item == catalogue.get(item["fact_id"]) and all(item[key] == fact[key] for key in ("entity_id", "plan", "version"))
                                  and item["relation"] != relation and effective(item, context["as_of"]) for item in support))
                    parsed = re.fullmatch(r"answer correction: (?P<correction>.+) Question: (?P<question>.+) As of: (?P<date>\d{4}-\d{2}-\d{2})\. Unchanged context: (?P<context>.+)", row["input_text"])
                    sentences = re.findall(r"[^.]+\.", parsed["context"]) if parsed else []
                    good = len(sentences) == len(support)
                    for sentence, item in zip(sentences, support):
                        endorsed = assertion(sentence.strip(), subjects)
                        good = good and endorsed is not None and all(endorsed[key] == item[key] for key in ("fact_id", "entity_id", "relation", "plan", "version", "unit", "value")) and endorsed["polarity"] is None and endorsed["noun_agreement"]
                    check("input_has_only_correction_question_date_and_unchanged_support", parsed is not None
                          and parsed["correction"] == row["correction_text"] and parsed["question"] == row["question"]
                          and parsed["date"] == context["as_of"] and good)
                    counterpart = by_id.get(row["id"].replace("answer:", "scope:", 1))
                    check("positive_scope_counterpart", counterpart is not None and counterpart["label"] == 1
                          and all(counterpart.get(key) == row.get(key) for key in ("question", "correction", "request_context", "expected_fact", "expected_polarity", "question_type", "split", "lineage_id")))
                result.update(computed_scope_label=int(applies), computed_polarity=polarity,
                              computed_expected_fact=expected, parsed_question_fact_id=fact["fact_id"])
            except (KeyError, ValueError, TypeError, ArithmeticError) as error:
                issues.append("review_exception:" + str(error))
            result.update(decision="flagged" if issues else "accept", checks=checks, issues=issues,
                          reason="Independent anchored question and structured scope/value consistency checks passed."
                          if not issues else "Unresolved consistency issue; no row may be removed or label tuned against model output.")
            decisions[row["id"]] = result
    return {"review_policy": "automated_final", "human_reviewed_data": False, "human_approval_claimed": False,
        "human_semantic_review": False, "model_or_gpu_used": False, "reviewed_at": reviewed_at,
        "total_rows": len(all_rows), "accepted": sum(item["decision"] == "accept" for item in decisions.values()),
        "flagged": sum(item["decision"] == "flagged" for item in decisions.values()),
        "checks": {key: dict(value) for key, value in totals.items()}, "rows": decisions,
        "limitations": ["This audit establishes controlled-format consistency only, not arbitrary natural-language truth.",
                        "Challenge wording is agent-authored draft wording, not independent human-authored questions.",
                        "Machine review does not turn simulated approvals into administrator-approved production information."]}
