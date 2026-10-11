# Evaluation protocol

**Status:** Development specification. Final configuration, trained checkpoint
identities, scorer version and test lock are unset. The first structured workflow
is a regression demonstration; it is not an ML benchmark result.

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

Start with exact normalized short-answer forms and reviewed aliases. Apply numeric
normalization only with correct units; subject aliases are tied to canonical entity
IDs. Contradictory, ambiguous or unmatched prose goes to blinded human semantic
review with two reviewers and recorded adjudication. Report the automatic/manual
counts and disagreement. An LLM judge can propose review candidates but is not the
sole truth source. Lock accepted forms and adjudication rules before test evaluation.

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

Before final evaluation, finalize review decisions, model checkpoint/config hashes,
memory snapshot, tokenizer versions, threshold/calibration artifacts, accepted answer
forms, scorer code hash, generation budgets and question set hashes. Set the final
evaluation configuration `status=frozen` and create `test_lock.json` through the
data preparation tool. The evaluation harness must verify these hashes at startup.

Run the final test once after freezing. Preserve per-question outputs before
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
