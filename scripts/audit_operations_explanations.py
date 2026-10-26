"""Audit bounded, explicit ReasonKE subanswer claims; never certify all prose.

No generation, training, primary-score mutation or LLM judge is used. Unknown
wording and residual prose remain visible as unassessable material.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from editguard_benchmark.data import digest, encoded, load_dataset
from editguard_benchmark.run import request_for, source_identity, verify_protocol

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("R-RAG", "R-ALL")
LABELS = ("supported", "contradicted", "unsupported", "unassessable")
INDEX = re.compile(r"\[Sub question\s+(\d+)\]", re.I)
BOX = re.compile(r"\\boxed\{([^{}\n]{1,200})\}")
REFERENCE = re.compile(r"\[sub answer\s+(\d+)\]", re.I)
KNOWN_QUERY = re.compile(r"(?:Which team owns .+ in release \d+|Who is the on-call contact for .+)\?", re.I)
NEGATION = re.compile(r"\b(?:not|never|reject(?:ed)?|incorrect|example|possibility)\b|n't|rather than|instead of", re.I)


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().rstrip(" .?").split())


def indexed_sections(value):
    matches = list(INDEX.finditer(value))
    rows = {}
    for index, match in enumerate(matches):
        body = value[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(value)].strip()
        rows.setdefault(int(match.group(1)), []).append(body)
    return rows


def resolve_query(question, answers, index):
    dependencies = [int(match.group(1)) for match in REFERENCE.finditer(question)]
    if any(key >= index or key not in answers for key in dependencies):
        return None, "Unresolved or forward subanswer reference"
    query = REFERENCE.sub(lambda match: answers[int(match.group(1))], question)
    # One finite, declared anaphora rule for the controlled two-hop format.
    if re.fullmatch(r"Who is the on-call contact for (?:that|this) team\?", query, re.I):
        if index != 2 or 1 not in answers:
            return None, "The single declared team-anaphora rule cannot resolve this question"
        query = "Who is the on-call contact for " + answers[1] + "?"
    return query, "Indexed-reference substitution; that/this team resolves only at step 2 to boxed step 1"


def assess(question, answer, facts):
    if not isinstance(question, str) or not isinstance(answer, str):
        return {"label": "unassessable", "reason": "No single endorsed boxed answer and resolved question"}
    matching = [row for row in facts if normalize(row["question"]) == normalize(question)]
    if not matching:
        recognized = bool(KNOWN_QUERY.fullmatch(question.strip()))
        return {"label": "unsupported" if recognized else "unassessable",
                "reason": "Recognized factual query has no fact in this view" if recognized else "Question wording is outside exact known fact questions and declared controlled grammar",
                "matching_fact_ids": []}
    values = {normalize(row["answer"]) for row in matching}
    if len(values) != 1:
        return {"label": "unassessable", "reason": "Fact view contains competing answers", "matching_fact_ids": [row["id"] for row in matching]}
    supported = normalize(answer) in values
    return {"label": "supported" if supported else "contradicted",
            "reason": "Exact resolved fact-question and answer match" if supported else "Known fact-question has a different authoritative answer",
            "matching_fact_ids": [row["id"] for row in matching],
            "fact_answers": [row["answer"] for row in matching]}


def audit_trace(row, current_facts):
    response = row.get("response")
    raw = response.get("raw_output") if isinstance(response, dict) else None
    result = {"id": row["id"], "question_id": row["question"]["id"], "lineage_id": row["question"]["lineage_id"],
              "condition": row["condition"], "primary_outcome_unchanged": row["outcome"],
              "final_answer_unchanged": response.get("content") if isinstance(response, dict) else None,
              "claims": [], "unassessable_segments": [], "raw_output": raw,
              "trace_fully_verified": False}
    if not isinstance(raw, str) or not raw.strip():
        result["trace_status"] = row["outcome"] if row["outcome"] in ("unexecuted", "execution_error") else "no_raw_trace"
        result["unassessable_segments"].append({"reason": "No generated raw trace available", "text": ""})
        return result
    result["trace_status"] = "inspected"
    result["raw_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    decomposition = re.findall(r"<decompose>(.*?)</decompose>", raw, re.S | re.I)
    actions = re.findall(r"<action>(.*?)</action>", raw, re.S | re.I)
    if len(decomposition) != 1 or len(actions) != 1:
        result["unassessable_segments"].append({"reason": "Expected exactly one decompose and one action block; no claims guessed", "text": raw})
        return result
    questions, statements = indexed_sections(decomposition[0]), indexed_sections(actions[0])
    first_marker = INDEX.search(actions[0])
    if first_marker and actions[0][:first_marker.start()].strip():
        result["unassessable_segments"].append({"reason": "Action preamble retained without semantic certification",
                                               "text": actions[0][:first_marker.start()].strip()})
    answers = {}
    for index in sorted(questions.keys() | statements.keys()):
        question_rows, action_rows = questions.get(index, []), statements.get(index, [])
        question = question_rows[0] if len(question_rows) == 1 else None
        statement = action_rows[0] if len(action_rows) == 1 else None
        boxes = BOX.findall(statement) if statement else []
        answer = boxes[0].strip() if len(boxes) == 1 and not NEGATION.search(statement) else None
        resolved, resolution = resolve_query(question, answers, index) if question else (None, "Missing or duplicated indexed subquestion")
        claim = {"subquestion_index": index, "question": question, "resolved_question": resolved,
                 "resolution_rule": resolution, "boxed_answer": answer, "raw_action_statements": action_rows,
                 "supplied_fact_support": assess(resolved, answer, row.get("supplied_facts", [])),
                 "current_fact_correctness": assess(resolved, answer, current_facts)}
        if answer:
            answers[index] = answer
        result["claims"].append(claim)
        # The indexed boxed proposition is checked; surrounding generated prose
        # may contain additional factual assertions and is NOT certified by it.
        for text in action_rows:
            residual = BOX.sub("[boxed subanswer audited separately]", text)
            result["unassessable_segments"].append({"reason": "Surrounding action prose is retained without semantic claim certification", "text": residual})
    # Preserve ALL raw material outside the action block as unassessable. This
    # includes acknowledgement, decomposition, narrative and final-tag prose.
    residual = re.sub(r"<action>.*?</action>", "[indexed action block audited separately]", raw, flags=re.S | re.I)
    result["unassessable_segments"].append({"reason": "Raw text outside action block is retained; not semantically audited", "text": residual})
    if not result["claims"]:
        result["unassessable_segments"].append({"reason": "No indexed claim was extractable", "text": actions[0]})
    return result


def audit_receipt(receipt_path, *, protocol=None, config_path=None):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    config_path = Path(config_path or ROOT / "configs/operations-evaluation.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = receipt["plan"]
    phase = plan["phase"]
    if plan["model"] != "reasonke" or phase not in ("development", "final"):
        raise ValueError("Use a ReasonKE matched development or final receipt")
    if plan["sources"] != source_identity(config_path):
        raise ValueError("Receipt source/config/data identity differs")
    manifest, worlds, questions = load_dataset(ROOT / config["dataset"], "final" if phase == "final" else "development")
    if receipt["dataset"] != manifest:
        raise ValueError("Dataset receipt identity differs")
    if phase == "final":
        if protocol is None:
            raise ValueError("The final explanation audit requires its frozen protocol")
        frozen = verify_protocol(protocol, config_path)
        if plan["protocol_sha256"] != digest(protocol):
            raise ValueError("Final receipt identifies a different protocol")
        declaration = frozen.get("explanation_audit", {})
        if declaration.get("conditions") != list(CONDITIONS) or declaration.get("assigned_raw_traces") != 32:
            raise ValueError("The frozen explanation-audit population differs")
        selected = sorted((row for row in questions if row["category"] in ("two_hop", "two_hop_paraphrase")),
                          key=lambda row: hashlib.sha256(("oracle-42:" + row["id"]).encode()).hexdigest())[:16]
        selected_ids = {row["id"] for row in selected}
    else:
        selected_ids = {row["question"]["id"] for row in receipt["records"]
                        if row["condition"] in CONDITIONS and row["question"]["category"] in ("two_hop", "two_hop_paraphrase")}
    indexed = {row["id"]: row for row in receipt["records"]}
    required_ids = [condition + ":" + key for condition in CONDITIONS for key in sorted(selected_ids)]
    if len(indexed) != len(receipt["records"]) or any(key not in indexed for key in required_ids):
        raise ValueError("Receipt has duplicate/missing assigned audit traces")
    truths = {row["id"]: row for row in questions}
    traces = []
    for key in required_ids:
        row = indexed[key]
        if row["question"] != truths[row["question"]["id"]]:
            raise ValueError("Question identity differs in audit receipt")
        if row["outcome"] != "unexecuted":
            expected, retrieval = request_for(row, worlds, config)
            if (row.get("supplied_facts") != expected["facts"] or row.get("retrieval") != retrieval
                    or row.get("request_sha256") != hashlib.sha256(encoded(expected)).hexdigest()):
                raise ValueError("Supplied context or request digest differs in audit receipt")
        traces.append(audit_trace(row, worlds[row["question"]["lineage_id"]]["current_facts"]))
    by_condition = {}
    for condition in CONDITIONS:
        rows = [row for row in traces if row["condition"] == condition]
        claims = [claim for row in rows for claim in row["claims"]]
        by_condition[condition] = {"assigned_traces": len(rows), "trace_status_counts": dict(Counter(row["trace_status"] for row in rows)),
            "extracted_indexed_claims": len(claims),
            "supplied_support_counts": {label: sum(claim["supplied_fact_support"]["label"] == label for claim in claims) for label in LABELS},
            "current_fact_counts": {label: sum(claim["current_fact_correctness"]["label"] == label for claim in claims) for label in LABELS},
            "traces_with_current_fact_contradictions": sum(any(claim["current_fact_correctness"]["label"] == "contradicted" for claim in row["claims"]) for row in rows),
            "unassessable_segments_retained": sum(len(row["unassessable_segments"]) for row in rows),
            "fully_verified_traces": 0}
    return {"schema": "editguard-explanation-support-audit-v1", "phase": phase, "execution_status": receipt["status"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "receipt_sha256": digest(receipt_path), "protocol_sha256": digest(protocol) if protocol else None,
            "audit_script_sha256": digest(__file__), "assigned_traces": len(traces), "by_condition": by_condition, "traces": traces,
            "model_generations": 0, "human_reviewed": False, "primary_scores_modified": False,
            "population": "The sixteen preselected oracle question IDs crossed with R-RAG/R-ALL" if phase == "final" else "All assigned development two-hop question IDs crossed with R-RAG/R-ALL; separate parser diagnostic",
            "denominators": "Assigned traces include errors/unexecuted/no-raw-trace outcomes. Claim labels cover only extracted indexed boxed propositions; unparsed prose is not silently credited.",
            "limitations": ["Automated narrow-format support audit, not independent human semantic review",
                "Exact resolved fact-question/answer matching with one declared team-anaphora rule",
                "Surrounding action prose and all text outside action blocks remain unassessable",
                "Unsupported means no fact for a recognized query in that view, not proof the claim is false",
                "Supported explicit subanswers do not certify the complete explanation or internal reasoning faithfulness"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing audit; choose another output")
    result = audit_receipt(args.receipt, protocol=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("schema", "phase", "assigned_traces", "by_condition", "model_generations", "human_reviewed")}, indent=2))


if __name__ == "__main__":
    main()
