# Short-fact scoring and evaluation exports

The scorer evaluates the original answer text against explicit ground truth. It
does not repair generated answers, retrieve an answer for the model, or use an
LLM judge. These are development tools; their unit-test fixtures are synthetic
engineering cases and provide no model-performance result.

`editguard/scoring.py` declares scorer version `editguard-short-facts-v2`.
`editguard/metrics.py` declares metric version `editguard-fixed-denominators-v2`.
Final evaluation must freeze both source hashes with the question set and model
configuration. Changing accepted forms after inspecting test answers requires a
documented scorer revision and transparent rescoring of saved outputs.

## Accepted assertions

`score_answer(text, expected_fact, aliases=(), expected_polarity=None,
answer_kind='model_generated')` returns `correct` (`true`, `false`, or `null`),
`verdict`, `reason`, `endorsed_fact`, `needs_review`, `answer_kind` and
`scorer_version`. A `null` decision receives no automatic accuracy credit.

Ground truth supplies a canonical entity ID, reviewed names/aliases, relation,
value, unit, and applicable plan/version scope. The evaluation runner attaches
these fields after inference. The model never receives scoring truth as an
answer hint. Entity aliases belong to one expected entity; a near-name prefix
cannot identify another product. Exact short assertions require every supplied
plan/version field. Missing scope goes to review; recognized incompatible scope
is incorrect.

Accepted relation forms include:

| Relation | Example complete assertion |
|---|---|
| `device_limit` | `Nimbus Basic v2 supports up to 3 devices.` |
| `price_monthly` | `Nimbus Basic v2 costs 9 USD per month.` |
| `storage_gb` | `Nimbus Basic v2 includes 50 GB of storage.` |
| `support_hours` | `Nimbus Basic v2 includes 8 support hours per day.` |

The source enumerates additional verbs and relation-specific forms. Unicode
typography, whitespace and case are normalized. Numeric digits, decimals and
English integer words from zero to ninety-nine are canonicalized; `3.0` and
`three` denote the same value. Units are checked separately. `MB` does not match
`GB`, and an unspecified dollar currency requires review. Currency or storage
conversions are outside this initial grammar.

Comparison questions supply reviewed `expected_polarity='yes'` or `'no'`. The
answer must begin with the matching yes/no endorsement and also endorse the
correct factual limit. A correct value accompanied by the wrong comparison
polarity fails. Unsolicited yes/no prose on a non-comparison question needs review.

The scorer accepts one complete assertion. Additional or unfamiliar prose goes
to human review, even if it contains a target value. Explicit target rejection,
recognized wrong subjects/relations/values/units, incompatible complete
endorsements, and contradictory yes/no endorsements fail. A bare `Three.` needs
review because its subject and relation are not explicit. This conservative
policy can reduce automatic credit for semantically correct prose; the separate
review procedure addresses that limitation.

`as_of` in an endorsed fact records the question's evaluated scenario. It is not
evidence that the model explicitly uttered or independently verified a date.
Date/version compatibility and correction activation belong to the authoritative
scenario and routing rules. Extra temporal prose is not accepted by this grammar.
Retrieved provenance alone never verifies arbitrary generated prose.

## Human semantic review

The aggregation command exports `review_queue.jsonl` for unmatched prose. Each
item preserves the exact question, answer, truth and reviewed entity aliases, but
omits configuration and routing evidence. Its opaque review ID binds the complete
original prediction record and scorer source hash, including the exact output.

An adjudication registry has `schema_version: 1` and a `decisions` list. Each vote
contains `review_id`, `row_sha256`, a nonempty `reviewer` identifier,
`decision: 'correct' | 'incorrect'`, a substantive `reason`, and timezone-aware
ISO `reviewed_at`. Two or more independently identified, agreeing reviewers can
resolve a pending item. One vote remains pending. Any disagreement remains
uncredited until independently resolved through a documented review process.
Stale hashes, duplicate reviewer votes, naive timestamps and decisions addressing
automatically scored items are rejected.

The tool validates provenance structure; it cannot authenticate identities or
certify reviewer expertise. Reviewers must inspect endorsement, scope, relation,
unit and polarity, rather than awarding credit for a target mention. Correct
adjudications store a canonical fact so numeric formatting does not falsely
reduce output preservation. Incorrect unmatched prose has no invented parsed
endorsement; preservation therefore remains uncredited unless an actual
endorsement is established by a future, separately specified review extension.

The original automatic score and every vote are retained. Reports disclose
automatic/manual correct counts, pending reviews and disagreements. The final
report should present results before and after adjudication where material.

## Fixed denominators and matched conditions

`aggregate_predictions(records)` expects explicit question IDs and family IDs,
configuration, question/ground truth, answer kind, status, raw answer, score,
latency, and applicable-query labels. Optional retrieval hits use declared k keys.
Scope diagnostics require an independent `scope_label`, probability or explicit
missing probability, and a frozen operating threshold. Query applicability is
never substituted for a pair label.

Duplicate configuration/question IDs, incompatible paired truth, different
request context or handling expectations, conflicting scope-pair labels, and
mixed thresholds fail. A missing row in a present configuration is padded as
`missing_record` over the union of question IDs. It remains in the accuracy
denominator. Entirely absent configurations are not invented; the run manifest
must identify requested configurations and the runner must emit their unavailable
records. A question absent from every configuration cannot be reconstructed from
predictions alone, so the runner's declared question population must also be
checked.

| Output | Interpretation |
|---|---|
| `accuracy` | Correct factual answers / all answerable questions, including abstentions, errors, unavailable results and pending review. |
| `coverage` | Nonempty answered outputs / all answerable questions. A stale score with missing answer text receives no credit. |
| `selective_accuracy` | Correct answers / answered questions; always read alongside coverage and fixed-denominator accuracy. |
| `model_generated_answer_accuracy` | Correct model-generated answers / the entire answerable population. Structured answers never enter its numerator. |
| `model_generated_answer_coverage` | Answered model-generated outputs / the entire answerable population. |
| `by_answer_kind` | Separate subset summaries for model generation, structured answers, clarification and unavailable outputs. |
| `baseline_correct_retention` | Post-edit correctness on out-of-scope questions the unchanged base answered correctly. |
| `output_preservation` | Matching parsed endorsed facts / all paired answerable out-of-scope questions, including preservation of wrong answers. |
| `handling_accuracy` | Reviewed correct handling / all declared unanswerable cases; missing handling labels receive no credit. |

Every rate stores `numerator`, `denominator`, and `value`. Empty denominators yield
`null`, not zero or perfect performance. Direct corrected, paraphrase, comparison
and out-of-scope subsets are reported separately. Output-preservation reports the
parsed-pair count because unfamiliar prose cannot establish an endorsement
automatically. Preserving an incorrect baseline answer never establishes factual
correctness.

Applicable-edit retrieval recall counts missing/error retrieval as a miss, over
all applicable answerable questions. The k keys must be declared by the runner;
if no record declares a k, the aggregator cannot invent its retrieval population.
The false correction application metric counts out-of-scope queries with an
audited `selected_edit_id`. It measures editor routing events; it does not infer
whether RAG or prompt-only prose was influenced by a retrieved correction. Factual
accuracy captures incorrect answers in those configurations independently. This
route metric is unavailable for unchanged-base, updated-RAG and prompt-only
configurations, rather than reporting an unsupported zero.

Raw scope precision/recall/false-positive rate use explicit pair truth. Missing
probabilities are not positive predictions: missing applicable pairs reduce
recall, and missing negative pairs receive no specificity/accuracy credit. Known
pair labels survive padded failures, with the configuration's threshold.
Scope-label coverage is disclosed when the labeled population is incomplete.
Brier/ECE are unavailable for the full population if any probability is missing;
observed-only calibration diagnostics are separately labeled with probability
coverage. ECE uses declared equal-width probability bins; the last bin includes
probability one. These pair metrics do not establish end-to-end routing quality.

Latency covers measured answers, abstentions and failures. If any latency is
missing or invalid, full-population p50/p95 are unavailable; observed-only values
and missing counts remain visible. Cold start, model loading and warm inference
must be distinguished in the run manifest. No synthetic latency is invented.

## Paired uncertainty and exports

`paired_cluster_bootstrap(left, right, seed=42, resamples=2000)` pairs by question
ID and checks family/ground-truth compatibility. It resamples whole families with
replacement, carrying every paraphrase and counterfactual variant together.
The statistic is question-weighted right-minus-left fixed-denominator accuracy.
Unequal family sizes retain their actual denominators in each draw. The report
contains a paired percentile 95% interval, family count, matched/unmatched counts
and answer-kind counts. Fewer than two families yields no interval. Few families
can produce wide, unstable intervals; resampling does not create independent
evidence or demonstrate robustness across training seeds.

Use the aggregation command after preserving raw predictions:

```powershell
.\.venv\Scripts\python.exe scripts/aggregate_evaluation.py `
  --input reports/my-run/predictions.jsonl `
  --output-dir reports/my-run/aggregation
```

Optionally add `--adjudications path/to/decisions.json` after independent review.
The command executes no inference, training, model selection or downloads. It
exports `report.json`, long-form `metrics.csv` with denominators,
`scored_predictions.jsonl`, and the blinded `review_queue.jsonl`. Input and review
evidence cannot be overwritten by those output filenames. Report metadata stores
the input/scorer hashes, aggregation timestamp and adjudication counts.

Regression verification:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_scoring tests.test_metrics -v
```

These tests cover wrong subjects, relations, versions, units and polarity;
target rejection and contradiction; stale review evidence; missing/error outputs;
structured/generative separation; denominator retention; and paired family
resampling. Passing tests validates the harness rules, not a trained model's
accuracy.
