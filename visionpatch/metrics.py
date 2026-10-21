"""Count repairs and collateral errors without hiding their denominators."""

from __future__ import annotations

import random


def ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def summarize(rows: list[dict], method: str, *, distractor: str = "camera") -> dict:
    by_source = {}
    for row in rows:
        variants = by_source.setdefault(row["image_id"], {})
        if row["variant"] in variants:
            raise ValueError("Duplicate variant for an original source image")
        if variants and next(iter(variants.values()))["class_id"] != row["class_id"]:
            raise ValueError("A source image has inconsistent ground-truth classes")
        variants[row["variant"]] = row
    if any(set(variants) != {"clean", "blank", "distractor"} for variants in by_source.values()):
        raise ValueError("Every source must have exactly three declared variants")
    summary = {}
    for variant in ["clean", "blank", "distractor"]:
        selected = [variants[variant] for variants in by_source.values()]
        summary[f"{variant}_accuracy"] = ratio(sum(r[method] == r["class_id"] for r in selected), len(selected))
        base_correct = [r for r in selected if r["base"] == r["class_id"]]
        summary[f"{variant}_new_errors"] = ratio(sum(r[method] != r["class_id"] for r in base_correct), len(base_correct))
    eligible, induced, repaired = [], [], []
    for variants in by_source.values():
        clean, attacked = variants["clean"], variants["distractor"]
        if clean["class_id"] != distractor and clean["base"] == clean["class_id"]:
            eligible.append(attacked)
            if attacked["base"] != attacked["class_id"]:
                induced.append(attacked)
                if attacked[method] == attacked["class_id"]:
                    repaired.append(attacked)
    summary["baseline_attack_induced_errors"] = ratio(len(induced), len(eligible))
    summary["repair_rate"] = ratio(len(repaired), len(induced))
    summary["repairs_per_all_sources"] = ratio(len(repaired), len(by_source))
    genuine = [v["clean"] for v in by_source.values() if v["clean"]["class_id"] == distractor]
    summary["genuine_distractor_clean_accuracy"] = ratio(sum(r[method] == r["class_id"] for r in genuine), len(genuine))
    summary["sources"] = len(by_source)
    return summary


def paired_accuracy_interval(rows: list[dict], method: str, *, seed: int = 42, samples: int = 2000) -> dict:
    """Percentile paired bootstrap of attacked accuracy differences by photo."""
    if type(samples) is not int or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    selected = [r for r in rows if r["variant"] == "distractor"]
    if len({r["image_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate attacked variant would inflate the independent-source count")
    if not selected:
        return {"sources": 0, "difference": None, "interval_95": None}
    differences = [int(r[method] == r["class_id"]) - int(r["base"] == r["class_id"]) for r in selected]
    generator = random.Random(seed)
    means = sorted(sum(generator.choices(differences, k=len(differences))) / len(differences)
                   for _ in range(samples))
    return {"sources": len(selected), "difference": sum(differences) / len(differences),
            "interval_95": [means[int(samples * .025)], means[min(samples - 1, int(samples * .975))]],
            "bootstrap_samples": samples, "seed": seed,
            "limitation": "Small development subset; source-grouped interval, not a deployment reliability guarantee"}
