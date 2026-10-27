"""Bounded answer-frontier lexical retrieval for the separate depth challenge.

The query alone selects one or two seed facts with the existing BM25-style
``rank`` function. At each of at most three expansion rounds, the newest
selected facts' answer text becomes the search frontier. Select the highest
ranked unseen fact, then advance to that fact's answer. Fill any unused context
slots from the last frontier's ranking. Ranking ties use the existing fact-ID
order. Only approved fact id/question/scope/answer strings are inspected; hidden
question targets, required support IDs, structured relation labels and depth
metadata are never inputs to retrieval.

The predeclared development family is seed_count in (1, 2), rounds=3 and cap in
(4, 5, 6). Choose the smallest cap with complete support on every development
question, breaking a configuration tie by fewer seeds. ``development_comparison``
replays this finite family on the development partition only. Its diagnostic is
retrieval recall, not generated-answer accuracy. Extra context may hurt a model.
This module never modifies the completed operations-study retriever or results.

Recorded development diagnostic (16 lineages, 128 questions): all six candidates
retrieved complete support for 128/128 questions and 224/224 required facts.
The initial packet-only choice used one seed. Before any neural profiling or
freeze, the separate India display-development starter exposed a hosting-city
seed distractor. The same six candidates were then checked on both display
memory states as well. The selected configuration uses two seeds and four facts:
complete support for 128/128 packet questions plus both display states, 230/230
required fact occurrences total. Packet-only depth 0/1/2/3 each remains 32/32
complete. The original two-stage k4 packet baseline completed 96/128 questions
and found 192/224 facts; it completed 0/32 three-hop questions. These are retrieval
diagnostics, not model quality or additional independent held-out observations.
"""
from pathlib import Path

from editguard_benchmark.data import digest
from editguard_benchmark.retrieval import rank, retrieve as retrieve_original


DEFAULT_K = 4
SEED_COUNT = 2
EXPANSION_ROUNDS = 3
MAX_K = 6
MAX_FACTS = 200
MAX_FACT_CHARACTERS = 1_000_000
CANDIDATES = tuple((seeds, cap) for cap in (4, 5, 6) for seeds in (1, 2))


def _validate(query, facts, k, seed_count):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Retrieval query must be nonempty text")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ValueError("Iterative retrieval k must be an integer from 1 to 6")
    if seed_count not in (1, 2):
        raise ValueError("Choose one or two lexical seeds")
    if not isinstance(facts, (list, tuple)) or len(facts) > MAX_FACTS:
        raise ValueError("Iterative retrieval accepts at most 200 fact records")
    ids, characters = set(), 0
    for row in facts:
        try:
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Fact IDs must be nonempty, unique strings")
            ids.add(identifier)
            for key in ("question", "scope", "answer"):
                value = row[key]
                if not isinstance(value, str):
                    raise ValueError("Lexical fact fields must be text")
                characters += len(value)
        except (KeyError, TypeError) as exc:
            raise ValueError("Each fact requires id, question, scope and answer") from exc
    if characters > MAX_FACT_CHARACTERS:
        raise ValueError("Lexical fact text exceeds the one-million-character bound")


def _retrieve_candidate(query, facts, *, k, seed_count):
    _validate(query, facts, k, seed_count)
    if not facts:
        return []
    selected = rank(query, facts)[:min(seed_count, k)]
    ids = {row["id"] for row in selected}
    frontier = " ".join(row["answer"] for row in selected)
    for _ in range(EXPANSION_ROUNDS):
        remaining = [row for row in facts if row["id"] not in ids]
        if len(selected) >= k or not remaining:
            break
        next_fact = rank(frontier, remaining)[0]
        selected.append(next_fact)
        ids.add(next_fact["id"])
        frontier = next_fact["answer"]
    if len(selected) < k:
        remaining = [row for row in facts if row["id"] not in ids]
        selected.extend(rank(frontier, remaining)[:k - len(selected)])
    return selected


def retrieve_iterative(query, facts, k=DEFAULT_K):
    """Return at most k original fact records, with no generation or gold inputs."""
    return _retrieve_candidate(query, facts, k=k, seed_count=SEED_COUNT)


def _recall(worlds, questions, retrieve):
    groups = {str(depth): {"assigned": 0, "complete": 0, "found": 0, "required": 0}
              for depth in (0, 1, 2, 3)}
    for question in questions:
        # Ground truth is used by this evaluator, never by the retriever above.
        selected = retrieve(question["question"], worlds[question["lineage_id"]]["current_facts"])
        required = set(question["required_fact_ids"])
        found = required.intersection(row["id"] for row in selected)
        group = groups[str(question["depth"])]
        group["assigned"] += 1
        group["complete"] += int(found == required)
        group["found"] += len(found)
        group["required"] += len(required)
    return {"assigned": len(questions), "complete": sum(row["complete"] for row in groups.values()),
            "found": sum(row["found"] for row in groups.values()),
            "required": sum(row["required"] for row in groups.values()), "by_depth": groups}


def development_comparison(dataset_path):
    """Replay the six predeclared retrieval-only candidates; never evaluate final."""
    from .data import load_dataset
    from editguard_chat.examples import india_catalogue
    manifest, worlds, questions = load_dataset(Path(dataset_path), "development")
    if not questions or any(row["split"] != "development" for row in questions):
        raise ValueError("Retrieval selection requires development questions only")
    display = india_catalogue()
    display_worlds, display_questions = {}, []
    for view, suffixes in (("initial_facts", (1, 2, 8)), ("current_facts", (1, 3, 9))):
        uid = "display-development-" + view
        display_worlds[uid] = {"current_facts": display["world"][view]}
        display_questions.append({"id": uid + "-three-hop", "lineage_id": uid, "depth": 3,
                                  "question": display["questions"][2],
                                  "required_fact_ids": ["india-demo-001-f" + str(index) for index in suffixes]})
    candidates = []
    for seed_count, cap in CANDIDATES:
        retriever = lambda query, facts: _retrieve_candidate(query, facts, k=cap, seed_count=seed_count)
        recall = _recall(worlds, questions,
                        retriever)
        display_recall = _recall(display_worlds, display_questions, retriever)
        candidates.append({"seed_count": seed_count, "expansion_rounds": EXPANSION_ROUNDS,
                           "cap": cap, **recall, "display_development": display_recall,
                           "all_development": {key: recall[key] + display_recall[key]
                                               for key in ("assigned", "complete", "found", "required")}})
    eligible = [row for row in candidates if row["all_development"]["complete"] == row["all_development"]["assigned"]]
    chosen = min(eligible, key=lambda row: (row["cap"], row["seed_count"])) if eligible else None
    return {"schema": "editguard-depth-retrieval-selection-v1", "split": "development",
            "selection_revision": 2,
            "revision_reason": "Before neural profiling/freezing, an India display-development starter exposed a one-seed distractor. The same six candidates were reassessed with both explicit display states.",
            "dataset_id": manifest["dataset_id"], "independent_lineages": len(worlds),
            "dataset_manifest_sha256": digest(Path(dataset_path) / "manifest.json"),
            "method_source_sha256": digest(__file__),
            "shared_rank_source_sha256": digest(Path(__file__).resolve().parents[1] / "editguard_benchmark/retrieval.py"),
            "method": "bounded_iterative_answer_frontier_lexical",
            "baseline_two_stage_k4": _recall(worlds, questions, lambda query, facts: retrieve_original(query, facts, 4)),
            "baseline_display_development": _recall(display_worlds, display_questions, lambda query, facts: retrieve_original(query, facts, 4)),
            "display_fixture": {"dataset_id": display["dataset_id"], "purpose": "Separate display-development control, not a held-out result",
                                "source_sha256": digest(Path(__file__).resolve().parents[1] / "editguard_chat/examples.py"),
                                "worlds": display_worlds, "questions": display_questions},
            "selection_populations": {"depth_packet_development_questions": len(questions), "display_memory_states": 2,
                                      "selection_paths": len(questions) + 2,
                                      "display_states_are_independent_held_out_questions": False},
            "candidate_family": candidates,
            "selection_rule": "Complete support on all128 packet development questions and both display states; smallest cap, then fewest seeds",
            "selected": None if chosen is None else {key: chosen[key] for key in ("seed_count", "expansion_rounds", "cap")},
            "neural_model_calls": 0, "human_reviewed": False,
            "used_final_labels_for_selection": False, "used_final_model_outputs": False,
            "loader_note": "The dataset loader audits the whole packet, then returns development rows only; no final questions are evaluated for this selection.",
            "limitations": ["Development-only retrieval diagnostic, not answer-model accuracy",
                            "Fact ID ties and lexical text matching; no semantic scope oracle",
                            "Additional supplied facts can harm generated answers",
                            "No held-out final retrieval or generation outcome used for selection"]}
