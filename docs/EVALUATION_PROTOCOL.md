# Evaluation protocol

**Status (14 October 2026):** Development selection is complete. The selected
configuration is `configs/inference-final.json`: scope `pilot-v3-neutral2`, answer
`pilot-v2-polarity1`, fixed existing calibrated routing and 48 output tokens.
Primary scoring is `editguard-short-facts-v3`; secondary request-bound scoring is
`editguard-request-bound-short-qa-v1`. Stage G binds exact hashes before reading
the held-out question files and freezes the audited packet before inference.
No final result exists yet. The structured workflow is an engineering regression
demonstration, separate from measured model generation.

The user explicitly authorized an automated final audit. All final labels and
challenge questions therefore disclose `human_reviewed_data=false`. Independent
machine checks can detect inconsistencies; they do not establish human review,
natural language diversity or external validity.

## Truth and matched comparison

The reviewed fact store defines subject, relation, version, plan, interval, endorsed
value and unit. Each question has a scenario and explicit truth. The base reads the
initial handbook rather than assuming pretrained knowledge of fictional products.
Apply each mutually exclusive correction scenario independently; alternate worlds
are not simultaneously active contradictions.

Compare all five configurations on the same questions and scenario states:

| Configuration | Information available and behavior |
|---|---|
| Unchanged frozen base | Initial handbook only; diagnostic reference for stale facts and baseline errors. |
| Properly updated RAG | Initial handbook updated with every approved correction, including correct version/date scope and reindexing; same frozen generator. |
| Prompt-only correction | Initial handbook plus retrieved eligible approved correction descriptors in a fixed prompt budget; same frozen generator. |
| Similarity editor | Same approved memory and support context; validation-selected embedding similarity routing, then the same trained answer model used by the learned editor. |
| Trained scope editor | Same candidate memory and support context; calibrated trained scope routing, then the trained answer model. |

Updated configurations have the same approved corrections and necessary support
information. RAG must actually update its factual snapshot/index; a stale-only RAG
comparison is invalid. Shared administrator conflict, compatibility and rollback
rules apply across configurations. Also run raw scope-pair diagnostics and gate
ablations so deterministic protection is not attributed to the classifier.

Use compatible deterministic generation settings, equal maximum answer tokens
(initial candidate: 48), fixed context budgets, the same request metadata and
identical scorer. Tune each baseline only on validation. Record actual prompt/input
formats, retrieval k and thresholds. Generator weights differ between frozen and
trained configurations; disclose this intervention and resource difference. An
optional RAG-plus-trained-generator ablation can isolate generator training if the
core five-configuration pipeline is already working.

## Attribute each failure

Evaluate four layers separately:

1. Candidate retrieval: Was a truly applicable approved edit in top-k candidates,
   before and after deterministic compatibility filtering?
2. Scope: Given fixed candidate pairs, did the classifier apply the correct edit
   and reject ordinary/difficult negatives? Separate pair metrics from top-1 routes.
3. Oracle answer generation: Give the answer model the correct applicable edit and
   necessary context, regardless of retrieval/classifier performance.
4. End-to-end: Record the actual route, model output/abstention, selected edit,
   final endorsed fact, supporting sources, revision and full latency.

Failure tags include retrieval miss, scope false positive/negative, wrong subject,
wrong relation, wrong value/unit, incorrect comparison polarity, unsupported
generation, version/date resolution, unresolved conflict, and regression after
rollback. Retain failures and uncertainty; do not showcase only successful rows.

## Metric definitions and denominators

Every reported row includes integer numerator/denominator, percentage or score,
dataset subset, abstentions and missing/error responses. An empty denominator is
reported as not applicable, never zero or perfect performance.

| Metric | Numerator / denominator or definition |
|---|---|
| Applicable-edit retrieval recall@k | Queries containing at least one truly applicable edit in top-k / all answerable applicable-edit queries; include retrieval errors as misses. |
| Scope precision | True applicable pairs classified applicable / all pairs classified applicable. Pair confidence abstention is not a positive prediction. |
| Scope recall | True applicable pairs classified applicable / all applicable pairs, including abstentions. |
| False correction application rate | Out-of-scope questions routed to any inapplicable correction / all out-of-scope questions; disclose pair false-positive rate separately. |
| Direct corrected-answer accuracy | Correct endorsed fact on direct questions / all answerable direct applicable-edit questions, including abstentions and failures. |
| Held-out paraphrase accuracy | Correct endorsed fact / all answerable held-out paraphrase questions; template and independently authored challenges are distinct subsets. |
| Baseline-correct retention | Correctly retained post-edit answer / out-of-scope questions the unchanged base answered correctly; disclose this smaller selected population. |
| Out-of-scope factual accuracy | Correct factual answers / all answerable out-of-scope questions, including ones the base answered incorrectly. |
| Output preservation | Endorsed post-edit answer equals endorsed base answer / all eligible out-of-scope questions; preserving a wrong answer remains factually wrong. |
| Conflict handling | Correct explicit conflict/clarification outcome with no asserted conflicting fact / reviewed conflict cases; not folded into answer accuracy. |
| Rollback consistency | Restored scoped values/history behavior and unrelated-scope invariance / declared rollback assertions and end-to-end cases. |
| Answer coverage | Non-abstained answers / all answerable questions. |
| Selective accuracy | Correct answers / answered questions at a declared threshold, always alongside coverage and fixed-denominator accuracy. |
| Calibration | Brier score, validation-chosen ECE bins and reliability plot on all fixed labeled pairs; also report deployed-route population. |
| Latency | End-to-end p50/p95 across the full question set; separately warm/cold, answered/abstained, with hardware, candidate count and token lengths. |
| Resources | Actual wall time, examples/steps per second, peak allocated/reserved CUDA memory and host process memory for each training/inference run. |

On answerable questions, abstentions, parsing failures, timeouts and empty answers
count as incorrect in fixed-denominator accuracy. A correct clarification on a
reviewed ambiguous/unanswerable question is scored in its own handling metric.
Do not drop abstentions from edit success or drawdown summaries. Ambiguity tests
specify which missing product, tier, version or date makes clarification necessary.

## Endorsed-fact scoring

Do not use substring target matching. A model saying "five was the old limit;
three is wrong" must not receive credit for a three-device correction. Scoring
checks the requested subject, property, tier/version/date scope, endorsed value,
unit and expected yes/no polarity. Retrieve-and-template answers are classified
`deterministic_structured`; their correctness never counts as trained answer-model
generation accuracy, even when displayed beside a learned route.

Start with exact normalized short-answer forms and audited aliases. Apply numeric
normalization only with correct units; subject aliases are tied to canonical entity
IDs. For this release, unmatched prose remains pending and uncredited in the fixed
denominator; no human adjudication is claimed. The primary whole-assertion grammar
checks entity, plan, version, relation, unit, value and comparison polarity.
The narrow storage wording addition was resolved on development outputs and applied
equally to all configurations; historical v2 scores remain archived.
See [the alias rationale](SCORER_STORAGE_ALIAS.md).

The separately reported secondary request-bound score accepts whole quantities or
whole Yes/No outputs only for the existing twelve explicit controlled question
forms, after checking the question's subject, plan, version, relation and predicate
against structured truth. It never overrides a primary incorrect verdict and
does not prove a complete scoped fact from a fragment. Unsupported challenge
wording remains pending and uncredited. This rule is frozen before final inference,
even though it originated as a post-hoc development diagnostic. No LLM judge is
used as the sole truth source, and no automated prose review inflates accuracy.

For oracle-answer evaluation, score the original generated text without rewriting
it from the database. Structured fallback or verifier rejection is a separately
named configuration with its own coverage; it cannot inflate generator accuracy.

## Validation, calibration and uncertainty

All model/backbone choices, early stopping, thresholds, ambiguity margins,
temperature scaling, retrieval k and scoring rules use training/validation only.
Plot calibration and risk/coverage over validation thresholds; freeze the operating
point before test. Keep held-out test challenge wording out of debugging prompts.

Report paired differences because configurations answer the same questions.
Use a predeclared 2,000-resample bootstrap over entity families, preserving every
question, lineage and counterfactual variant in a sampled cluster. Use fixed seed
42 and 95% intervals. Report the number of families (12 initial test families),
which limits precision. Include raw aggregate sample counts; per-row independence
assumptions would exaggerate confidence. Multiple training seeds are optional and
must be named; one seed is disclosed as one seed rather than represented as robust.

## Final freeze and reporting

Before reading held-out questions, create an immutable final protocol binding the
original source-file hashes, deterministic packet transformation, selected model
checkpoint/config hashes, tokenizer versions, threshold/calibration artifacts,
scorer/source hashes, generation budgets and the final runner. Then derive and
audit a separate packet without modifying the original draft data, and freeze its
hashes and machine review registry before inference. The dedicated final harness
verifies protocol, packet, review, source and artifact hashes at startup.

The five matched configurations use all 720 scope test questions plus 96 challenge
questions, with separate cohort reports and a disclosed combined denominator of
816. Oracle generation uses the separate 288 answer examples plus 96 challenges,
reported by cohort and combined as 384; these partly overlapping answer examples
are not added to the matched headline population. All twelve test families remain
disjoint from training and development families. Challenge wording was agent
authored and has an independent automated audit, with no manual-authorship or
human-review claim. Mutually exclusive correction scenarios run separately using
one focal edit and at most eight foreign-product distractors, as in development;
this is not a dense same-product edit-memory stress test.

Run the final test once after freezing. Exact-bound recovery can continue an
interrupted run from its durable observation log without redoing completed tasks;
it cannot change the locked configuration or scoring rules. Preserve per-question outputs before
aggregating and export JSON/CSV, plots and the complete configuration. A reporting
bug may be corrected transparently from existing outputs with a new scorer version
and explanation; it cannot quietly tune model behavior using test observations.
A changed dataset/model prompted by test failures becomes a newly declared
benchmark with new held-out data and a documented reason.

Required scientific figures from real outputs: retrieval recall@k, classifier
precision/recall and hard-negative slices, reliability/calibration, risk versus
coverage, direct/paraphrase/out-of-scope accuracy with paired intervals, error
attribution, latency distributions and training/resource traces. Publish raw result
tables alongside figures. Until measured, show "not measured" rather than invented
bars or illustrative performance values presented as experiment results.
