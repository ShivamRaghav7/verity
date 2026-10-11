"""Fixed-denominator evaluation and paired family-level uncertainty (stdlib only)."""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict

from editguard.scoring import canonical_endorsement

METRICS_VERSION = "editguard-fixed-denominators-v2"
_ANSWERED = {"answered", "ok", "success", "generated"}
_ABSTAINED = {"abstained", "abstention", "clarification", "conflict"}
_ERRORS = {"error", "failed", "timeout", "missing_record"}


def rate(numerator, denominator) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _binary(label):
    if isinstance(label, (bool, int)) and label in (0, 1):
        return bool(label)
    raise ValueError("Labels must be explicit binary values, not inferred from answers.")


def scope_metrics(labels, probabilities, threshold=.5, *, bins=10) -> dict:
    """Raw pair metrics. Missing probabilities abstain, never silently disappear.

    Full-population calibration is unavailable if any probability is missing.
    Observed-only Brier/ECE are exposed as incomplete diagnostics with coverage.
    This is distinct from deterministic scope gating and end-to-end routing.
    """
    labels, probabilities = list(labels), list(probabilities)
    if len(labels) != len(probabilities):
        raise ValueError("Each fixed labeled pair must have a probability or explicit None.")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one.")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("Calibration bins must be a positive predeclared integer.")
    truth = [_binary(label) for label in labels]
    valid = []
    tp = fp = tn = fn = missing = 0
    for label, probability in zip(truth, probabilities):
        if probability is None:
            missing += 1
            if label:
                fn += 1
            continue
        if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilities must be finite numbers in [0,1] or explicit None.")
        prediction = probability >= threshold
        valid.append((label, float(probability)))
        if label and prediction:
            tp += 1
        elif prediction:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    positives = sum(truth)
    negatives = len(truth) - positives
    squared_error = sum((probability - int(label)) ** 2 for label, probability in valid)
    reliability = []
    weighted_gap = 0.0
    for index in range(bins):
        pairs = [(label, probability) for label, probability in valid
                 if min(int(probability * bins), bins - 1) == index]
        count = len(pairs)
        confidence = sum(probability for _, probability in pairs) / count if count else None
        accuracy = sum(label for label, _ in pairs) / count if count else None
        gap = abs(confidence - accuracy) if count else 0.0
        weighted_gap += gap * count
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "upper_inclusive": index == bins - 1, "count": count,
                            "mean_probability": confidence, "empirical_applicability": accuracy})
    observed_brier = rate(squared_error, len(valid))
    observed_ece = rate(weighted_gap, len(valid))
    complete = missing == 0
    return {"population": "raw_labeled_pairs", "threshold": float(threshold), "bins": bins,
            "counts": {"total": len(truth), "positive": positives, "negative": negatives,
                       "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                       "missing_probability": missing},
            "precision": rate(tp, tp + fp), "recall": rate(tp, positives),
            "false_positive_rate": rate(fp, negatives),
            "specificity": rate(tn, negatives), "accuracy": rate(tp + tn, len(truth)),
            "probability_coverage": rate(len(valid), len(truth)),
            "calibration_complete": complete,
            "brier": observed_brier if complete else {**rate(squared_error, len(truth)), "value": None, "reason": "Missing probabilities; full-population calibration unavailable."},
            "ece": observed_ece if complete else {**rate(weighted_gap, len(truth)), "value": None, "reason": "Missing probabilities; full-population calibration unavailable."},
            "observed_only_brier": observed_brier, "observed_only_ece": observed_ece,
            "reliability_bins": reliability}


def _status(record):
    return record.get("status", "answered")


def _answerable(record):
    if "answerable" in record:
        return _binary(record["answerable"])
    if "expected_fact" in record:
        return bool(record["expected_fact"])
    # Minimal dictionaries are useful for standalone paired uncertainty fixtures.
    return True


def _answered(record):
    return _status(record) in _ANSWERED and isinstance(record.get("answer"), str) and bool(record["answer"].strip())


def _correct(record):
    score = record.get("score") or {}
    correct = score.get("correct", record.get("correct"))
    return _answered(record) and correct is True


def _summary(rows):
    answerable = [row for row in rows if _answerable(row)]
    answered = [row for row in answerable if _answered(row)]
    correct = sum(_correct(row) for row in answerable)
    model_answered = [row for row in answered if row.get("answer_kind") == "model_generated"]
    model_correct = sum(_correct(row) for row in model_answered)
    return {"count": len(rows), "answerable_count": len(answerable),
            "accuracy": rate(correct, len(answerable)), "coverage": rate(len(answered), len(answerable)),
            "selective_accuracy": rate(correct, len(answered)),
            "model_generated_answer_accuracy": rate(model_correct, len(answerable)),
            "model_generated_answer_coverage": rate(len(model_answered), len(answerable)),
            "automatic_correct": sum(_correct(row) and not bool((row.get("score") or {}).get("human_adjudicated")) for row in answerable),
            "human_adjudicated_correct": sum(_correct(row) and bool((row.get("score") or {}).get("human_adjudicated")) for row in answerable),
            "human_adjudicated_count": sum(bool((row.get("score") or {}).get("human_adjudicated")) for row in rows),
            "pending_review": sum(bool((row.get("score") or {}).get("needs_review")) for row in rows),
            "abstentions": sum(_status(row) in _ABSTAINED for row in rows),
            "errors": sum(_status(row) in _ERRORS for row in rows),
            "unavailable": sum(_status(row) == "unavailable" for row in rows)}


def _truth_key(row):
    return json.dumps({key: row.get(key) for key in ("expected_fact", "expected_polarity", "expected_status", "expected_handling", "question", "request_context", "applicable", "question_type", "answerable")}, sort_keys=True, ensure_ascii=False)


def _index(rows):
    indexed = {}
    for row in rows:
        identity = row.get("id")
        family = row.get("family_id")
        if not isinstance(identity, str) or not identity or not isinstance(family, str) or not family:
            raise ValueError("Every evaluation record needs a nonempty question id and family_id.")
        if identity in indexed:
            raise ValueError(f"Duplicate question id: {identity}")
        indexed[identity] = row
    return indexed


def paired_cluster_bootstrap(left, right, seed=42, resamples=2000) -> dict:
    """Paired right-minus-left fixed-denominator accuracy, sampled by family.

    Every row in a sampled family travels together, preserving paraphrases and
    counterfactual variants. Unmatched rows are disclosed, never paired by order.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("A finite positive number of bootstrap resamples is required.")
    left_index, right_index = _index(left), _index(right)
    shared = sorted(left_index.keys() & right_index.keys())
    clusters = defaultdict(lambda: [0, 0])
    answer_kinds = {"left": Counter(), "right": Counter()}
    for identity in shared:
        a, b = left_index[identity], right_index[identity]
        if a["family_id"] != b["family_id"]:
            raise ValueError(f"Paired family differs for {identity}.")
        if _truth_key(a) != _truth_key(b):
            raise ValueError(f"Paired ground truth or question differs for {identity}.")
        if not _answerable(a):
            continue
        clusters[a["family_id"]][0] += int(_correct(b)) - int(_correct(a))
        clusters[a["family_id"]][1] += 1
        answer_kinds["left"][a.get("answer_kind", "unspecified")] += 1
        answer_kinds["right"][b.get("answer_kind", "unspecified")] += 1
    count = sum(size for _, size in clusters.values())
    difference = sum(value for value, _ in clusters.values()) / count if count else None
    result = {"metric": "fixed_denominator_answer_accuracy", "direction": "right_minus_left",
              "difference": difference, "difference_percentage_points": difference * 100 if difference is not None else None,
              "matched_ids": len(shared), "matched_answerable": count,
              "left_unmatched": len(left_index.keys() - right_index.keys()),
              "right_unmatched": len(right_index.keys() - left_index.keys()),
              "family_count": len(clusters), "seed": seed, "resamples": resamples,
              "confidence_level": .95, "ci95": None,
              "answer_kind_counts": {key: dict(value) for key, value in answer_kinds.items()}}
    if len(clusters) < 2:
        result["interval_status"] = "At least two paired families are required for a meaningful cluster interval."
        return result
    rng = random.Random(seed)
    cluster_values = [clusters[key] for key in sorted(clusters)]
    draws = []
    for _ in range(resamples):
        sample = rng.choices(cluster_values, k=len(cluster_values))
        draws.append(sum(value for value, _ in sample) / sum(size for _, size in sample))
    result["ci95"] = [percentile(draws, .025), percentile(draws, .975)]
    result["interval_status"] = "Paired percentile family-cluster bootstrap; limited family counts constrain precision."
    return result


def _latency(rows):
    observed = [float(row["latency_ms"]) for row in rows
                if not isinstance(row.get("latency_ms"), bool) and isinstance(row.get("latency_ms"), (float, int))
                and math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0]
    complete = len(observed) == len(rows)
    return {"unit": "ms", "total_count": len(rows), "sample_count": len(observed),
            "missing_count": len(rows) - len(observed), "complete": complete,
            "p50_ms": percentile(observed, .5) if complete else None,
            "p95_ms": percentile(observed, .95) if complete else None,
            "observed_only_p50_ms": percentile(observed, .5),
            "observed_only_p95_ms": percentile(observed, .95)}


def _same_endorsement(left, right):
    a = (left.get("score") or {}).get("endorsed_fact")
    b = (right.get("score") or {}).get("endorsed_fact")
    return (_answered(left) and _answered(right) and isinstance(a, dict) and bool(a)
            and isinstance(b, dict) and bool(b)
            and canonical_endorsement(a, polarity=a.get("polarity")) == canonical_endorsement(b, polarity=b.get("polarity")))


def aggregate_predictions(records, *, bootstrap_seed=42, bootstrap_resamples=2000) -> dict:
    """Aggregate the same question population across explicitly present configs.

    A missing row in a present configuration becomes an unavailable result using
    the union question population. Entirely unrequested configurations are not
    invented. Duplicate IDs or incompatible paired truth are data errors.
    """
    rows = list(records)
    groups = defaultdict(list)
    question_index = {}
    scope_truth = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Prediction records must be dictionaries.")
        config = row.get("configuration")
        if not isinstance(config, str) or not config:
            raise ValueError("Every record needs an explicit configuration name.")
        _index([row])
        if "applicable" in row:
            _binary(row["applicable"])
        if row.get("scope_probability") is not None and row.get("scope_label") is None:
            raise ValueError("A scope probability requires explicit labeled pair truth.")
        identity = row["id"]
        if row.get("scope_label") is not None:
            label = _binary(row["scope_label"])
            if identity in scope_truth and scope_truth[identity] != label:
                raise ValueError(f"Explicit scope-pair truth differs for question {identity}.")
            scope_truth[identity] = label
        if identity in question_index:
            previous = question_index[identity]
            if previous["family_id"] != row["family_id"] or _truth_key(previous) != _truth_key(row):
                raise ValueError(f"Configuration conditions differ for question {identity}.")
        else:
            question_index[identity] = row
        groups[config].append(dict(row))
    for config, config_rows in groups.items():
        present = _index(config_rows)
        labeled = [row for row in config_rows if row.get("scope_label") is not None]
        thresholds = {float(row.get("scope_threshold", .5)) for row in labeled}
        if len(thresholds) > 1:
            raise ValueError(f"Configuration {config} mixes operating thresholds; report separate configurations.")
        threshold = next(iter(thresholds), .5)
        for identity in sorted(question_index.keys() - present.keys()):
            original = question_index[identity]
            placeholder = {key: original[key] for key in ("id", "family_id", "question", "request_context", "question_type", "expected_fact", "expected_polarity", "expected_status", "expected_handling", "applicable", "answerable") if key in original}
            placeholder.update(configuration=config, status="missing_record", answer=None,
                               answer_kind="unavailable", score={"correct": False, "needs_review": False, "endorsed_fact": None},
                               retrieval_hits={}, latency_ms=None)
            config_rows.append(placeholder)
        # A scope-diagnostic configuration retains explicitly known pair truth
        # for failed/missing predictions. Never derive pair truth from applicable.
        if labeled:
            for row in config_rows:
                if row.get("scope_label") is None and row["id"] in scope_truth:
                    row["scope_label"] = scope_truth[row["id"]]
                    row["scope_probability"] = None
                    row["scope_threshold"] = threshold
    baseline = _index(groups.get("unchanged_base", []))
    configurations = {}
    for config in sorted(groups):
        config_rows = groups[config]
        statuses = Counter(_status(row) for row in config_rows)
        kinds = sorted({row.get("answer_kind", "unspecified") for row in config_rows})
        answerable = [row for row in config_rows if _answerable(row)]
        out_of_scope = [row for row in answerable if row.get("applicable") is False or row.get("applicable") == 0]
        query_types = sorted({row.get("question_type", "unspecified") for row in config_rows})
        subsets = {name: _summary([row for row in config_rows if row.get("question_type", "unspecified") == name]) for name in query_types}
        subsets["out_of_scope"] = _summary(out_of_scope)
        for name in ("direct", "paraphrase", "comparison"):
            subsets.setdefault(name, _summary([]))
        subsets["direct_corrected"] = _summary([row for row in answerable if row.get("applicable") in (True, 1) and row.get("question_type") == "direct"])
        retrieval_keys = sorted({str(key) for row in config_rows for key in (row.get("retrieval_hits") or {})})
        applicable_rows = [row for row in answerable if row.get("applicable") in (True, 1)]
        retrieval = {}
        for key in retrieval_keys:
            hits = sum(any(str(k) == key and value is True for k, value in (row.get("retrieval_hits") or {}).items()) for row in applicable_rows)
            retrieval[key] = rate(hits, len(applicable_rows))
        pair_rows = [row for row in config_rows if row.get("scope_label") is not None]
        thresholds = {float(row.get("scope_threshold", .5)) for row in pair_rows}
        if len(thresholds) > 1:
            raise ValueError(f"Configuration {config} mixes operating thresholds; report separate configurations.")
        scope = scope_metrics([row["scope_label"] for row in pair_rows], [row.get("scope_probability") for row in pair_rows], threshold=next(iter(thresholds), .5)) if pair_rows else None
        retention_rows = [row for row in out_of_scope if row["id"] in baseline and _correct(baseline[row["id"]])]
        preservation_rows = [row for row in out_of_scope if row["id"] in baseline]
        handling = [row for row in config_rows if not _answerable(row)]
        handling_correct = sum(row.get("handling_correct") is True for row in handling)
        correction_application = rate(sum(bool(row.get("selected_edit_id")) for row in out_of_scope), len(out_of_scope))
        if config in {"unchanged_base", "updated_rag", "prompt_only"}:
            correction_application.update(value=None, reason="This configuration has no explicit editor routing decision; correction influence in generated prose is not measured by selected_edit_id.")
        summary = _summary(config_rows)
        configurations[config] = {**summary,
            "counts": {"total": len(config_rows), "input_records": len(config_rows) - statuses["missing_record"],
                       "missing_records": statuses["missing_record"], "answerable": len(answerable),
                       "answered": sum(_answered(row) for row in answerable), "correct": sum(_correct(row) for row in answerable),
                       "pending_review": summary["pending_review"], "abstentions": summary["abstentions"],
                       "errors": summary["errors"], "unavailable": summary["unavailable"],
                       "missing_score": sum(_answered(row) and not isinstance(row.get("score"), dict) and "correct" not in row for row in config_rows)},
            "status_counts": dict(statuses), "subsets": subsets,
            "by_answer_kind": {kind: _summary([row for row in config_rows if row.get("answer_kind", "unspecified") == kind]) for kind in kinds},
            "retrieval_recall": retrieval,
            "false_correction_application_rate": correction_application,
            "false_correction_application_definition": "Out-of-scope questions with a selected_edit_id / all answerable out-of-scope questions. This is an audited editor routing event; it does not infer correction influence in generated RAG or prompt-only prose.",
            "baseline_correct_retention": rate(sum(_correct(row) for row in retention_rows), len(retention_rows)),
            "output_preservation": rate(sum(_same_endorsement(row, baseline[row["id"]]) for row in preservation_rows), len(preservation_rows)),
            "preservation_parsed_pair_count": sum(bool((row.get("score") or {}).get("endorsed_fact")) and bool((baseline[row["id"]].get("score") or {}).get("endorsed_fact")) for row in preservation_rows),
            "baseline_available": bool(baseline),
            "handling_accuracy": rate(handling_correct, len(handling)),
            "handling_labels_available": sum(isinstance(row.get("handling_correct"), bool) for row in handling),
            "scope": scope, "scope_labeled_records": len(pair_rows),
            "scope_label_coverage": rate(len(pair_rows), len(config_rows)) if pair_rows else None,
            "latency": _latency(config_rows)}
    pairs = []
    for left_name, right_name in itertools.combinations(sorted(groups), 2):
        paired = paired_cluster_bootstrap(groups[left_name], groups[right_name],
                                          seed=bootstrap_seed, resamples=bootstrap_resamples)
        pairs.append({"left": left_name, "right": right_name, **paired})
    return {"schema_version": 1, "metrics_version": METRICS_VERSION,
            "record_count": len(rows), "question_count": len(question_index),
            "disclaimer": "Uncredited abstentions, unavailable/error outputs and pending human reviews remain in answerable accuracy denominators. Model-generated answer accuracy uses the entire answerable population and excludes deterministic answers from its numerator. Calibration/latency with missing measurements is explicitly incomplete; absent scope labels are disclosed rather than inferred.",
            "configurations": configurations, "paired_comparisons": pairs}
