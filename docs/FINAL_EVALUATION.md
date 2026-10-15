# Frozen final evaluation

The frozen execution is sealed with 4,080 matched observations and 384 original
oracle observations. The five matched configurations completed with 4,009 answers,
71 clarifications and **zero runtime errors**. Updated RAG leads the frozen
standard-question request-bound metric: 649/720 (90.14%), versus the trained
editor's 613/720 (85.14%). The trained editor's useful direct-edit behavior does
not establish general language reliability or superiority to updated RAG.

The original oracle measurement is incomplete: 65 answers and 319 CUDA runtime
errors. Those errors remain in its denominator and are not model-generated
wrong answers. A separately named fixed-configuration recovery supplement has
completed all 319 selected cases after GPU availability was restored. It preserves
the original run and retains its 65 successful outputs. The original GPU
instance's final fingerprint is unavailable; a fresh
closure verified file hashes and freshly loaded weights, not that lost in-memory
state. Read the [runtime disclosure](../reports/final/runtime-failure-disclosure.json)
alongside the manifest. No model, threshold, generation setting, label or scorer
was changed using final results.

## Locked inputs

| Item | Active record |
|---|---|
| Protocol | `reports/final/protocol-v2.json`, SHA-256 `8a938b33055220fdb24fa86ed056405f186dc9b6c4b72fe136c76ff0dc3d86ab` |
| Inference configuration | `configs/inference-final.json`, SHA-256 `c86a11c2ef63f34b3b80976f0334b58e04b55b7088c0b013a3d0f5a3969e9134` |
| Scope export | `artifacts/scope/pilot-v3-neutral2`; manifest `c85e77193ff3cf5137baf0a569d142585cfac3a1734393fadf49ce95f9b88459` |
| Answer export | `artifacts/answer/pilot-v2-polarity1`; manifest `8e435ac9bcd5486bb967c7ca09c38bf1786b810c97a7c95a365c6250ceeb0a72` |
| Packet | `data/final/final-v2`; manifest `1013448a43bbbca322b73aedf269e91dc441031d6a7bf022ebbfa0de9f14b50c` |
| Packet lock | `reports/final/test_lock-v2.json` |
| Primary scorer | `editguard-short-facts-v3` |
| Separate request-bound scorer | `editguard-request-bound-short-qa-v1` |
| Review | `automated_final`, `human_reviewed_data=false` |
| Execution output / ledger | `reports/final/evaluation` / `runtime/final-evaluation-ledger.json` |

The protocol binds exact model files, source code, configuration, calibrations,
scorers, original dataset hashes and execution paths. Scope calibration is fixed
at temperature 1.0, threshold 0.97 and ambiguity margin 0.025. Similarity
calibration is likewise fixed from development data. Greedy generation uses one
beam, no sampling and 48 new tokens. Model loading requires the saved local
exports and pinned cached pretrained weights; metadata alone is insufficient.

## Populations and matched conditions

Twelve raw entity families are held out from training and development. There are
720 standard scope questions, 288 applicable-edit answer examples and 96
challenge questions. The automated final audit accepted all 1,104 unique labels
and flagged zero; this is controlled-format consistency review, not independent
human semantic review.

The five configurations answer the same 816 standard/challenge questions:

1. Unchanged base: diagnostic access to the original handbook.
2. Properly updated RAG: the same approved changes applied to retrieved facts.
3. Prompt-only correction: approved descriptors supplied to the frozen generator.
4. Similarity editor: embedding-based routing with the trained answer model.
5. Trained scope editor: eligible candidate retrieval, trained classifier routing
   and the same trained answer model.

Each alternate-world memory contains a focal correction and at most eight
foreign-product distractors. Updated baselines receive matched approved
corrections and supporting information. The unchanged base deliberately defines
the stale diagnostic reference; it is not the updated RAG baseline. The memory
is small and sparse, so candidate retrieval results do not demonstrate scaling
to a large correction repository.

Separate oracle generation supplies the applicable correction directly for 288
answer examples and the same 96 challenges: 384 outputs. That isolates generation
from retrieval and scope failures. The finite execution totals
`5 × 816 + 384 = 4,464` committed inference tasks, ordered by population and row
identifier, with no parameter search. The standard scope population supplies raw
pair classification metrics in addition to end-to-end routing observations.

Nine of 720 standard scope inputs normalize to same-label training pairs, none
matches validation, and 711 match neither. Results must separate these groups.
Raw products and lineages remain disjoint, while templates and logical forms are
shared. The 96 challenges comprise 48 agent-authored unfamiliar-wording drafts
and 48 unseen-value scenarios. No human-authored challenge claim is made.

## Scoring and denominators

The primary scorer requires a complete assertion that endorses the expected
subject, plan, version, relation, unit and value. Comparison responses must also
endorse the expected Yes/No answer. Mentioning the target value as an example,
negated claim or rejected possibility does not qualify. Unsupported prose remains
pending and uncredited; no invented human adjudication or sole LLM judge is used.

The separately reported request-bound metric recognizes a complete short numeric
or Yes/No response for a finite controlled question grammar. It can credit a
useful short answer without establishing a complete scoped fact assertion. It
does not override a primary incorrect verdict and does not replace the primary
metric. This metric was exploratory during development and explicitly frozen as
a separate measure before final question access. Unsupported challenge wording
may remain uncredited in this metric too.

All answerable questions remain in each accuracy denominator. Clarifications,
abstentions, unavailable/error outputs and pending semantic cases receive no
correct-answer credit. Selective accuracy and coverage are separate. Structured
database/template answers never enter the learned-generation numerator.

Retention measures preservation of answers the base already answered correctly;
out-of-scope factual accuracy measures truth for all out-of-scope questions.
Preserving an incorrect base answer is not a factual success. A zero eligible
baseline-correct denominator is undefined, not 100% retention. False edit
application is an observed selected-edit routing event; RAG/prompt-only prose
does not have that event and cannot be assigned an equivalent routing rate.

## Required completed exports

The [completed manifest](../reports/final/evaluation/manifest.json) binds the
following sealed exports. Preserve:

- `predictions.jsonl` and `oracle_predictions.jsonl`: raw text, routing, truth,
  provenance, latency and both scores.
- `metrics.json`: fixed-denominator primary metrics, raw scope, retrieval,
  out-of-scope correctness/retention and paired comparisons.
- `short_qa_metrics.json`: separately labeled request-bound results.
- `manifest.json`: exact bindings, resources, base weight checks and session
  history; `progress.json` and the execution ledger preserve recovery evidence.

Paired uncertainty uses 2,000 seeded family-cluster bootstrap resamples, 95%
intervals, seed 42. Twelve clusters support only bounded conclusions. Report
counts, missingness and uncertainty rather than treating correlated questions as
independent samples. Retrieval recall is identified as before/after eligibility
where available; raw classifier performance differs from complete routed-answer
performance.

Per-request p50/p95 include retrieval, scope and generation. First-use requests
remain in the sample and are labeled; model loading is recorded separately.
These serial CLI measurements do not include HTTP/browser rendering or represent
randomized-order load tests. The separate HTTP engineering benchmark below
measures a different repeated demonstration workload.
Resources measured in resumed sessions do not establish one uninterrupted
command's wall time or lifetime peak.

## Execution history and recovery

The first startup rejected a dependency inventory mismatch before packet loading,
model construction, ledger creation or inference. The fix unified the freeze and
runner inventory. The rejected protocol, packet and log remain archived;
[`startup-amendment.json`](../reports/final/startup-amendment.json) records zero
inference calls and unchanged models, threshold, scorer, questions and labels.
The active packet preserves identical question/truth/derivation bytes. This was
an engineering guard repair, not a test-result-based experiment change.

Execution later encountered Windows access-denied failures while replacing its
progress file at committed task prefixes 288 and 583. Both observation chains
remained intact. The lock owner/cause has not been established; do not attribute
it to antivirus, OneDrive or another process without evidence.

The declared IO-only recovery wrapper is
[`scripts/resume_final_io.py`](../scripts/resume_final_io.py), SHA-256
`2754c2830ce5528d4f937c54a1b6267009e6c8618f1962b2fa4188f4f178a302`.
Its [recovery declaration](../reports/final/io-recovery-declaration.json), SHA-256
`b4b2eff5db6cca221187ab5244857ac62e4bfd77bff507202d3bfce84db63b32`,
allows at most eight attempts and 1.95 seconds of retry delay for the identical
value on only two mutable files: progress and the ledger. It leaves the frozen
evaluator source unchanged and does not alter inference, scoring, inputs,
configuration, committed observations or final-run count. Retry waits occur
outside per-request route latency. Interrupted sessions and total command time
must be disclosed separately. Recovery events are retained in
`runtime/final-io-recovery-events.jsonl`.

The original execution is completed. Its single-use ledger refuses more
inference. Do not delete it or start a fresh final evaluation. During interruption
recovery, the following command resumed the exact same unfinished execution;
it is retained as historical reproduction evidence, not a request to rerun the
completed benchmark:

```powershell
$env:HF_HOME = (Join-Path (Get-Location) 'runtime/hf-cache')
$env:HF_HUB_OFFLINE = '1'
.\.venv-ml\Scripts\python.exe scripts/resume_final_io.py --declaration reports/final/io-recovery-declaration.json --declaration-sha256 b4b2eff5db6cca221187ab5244857ac62e4bfd77bff507202d3bfce84db63b32
```

Changed bound inputs are refused before model construction. Recovery validates
committed observation hashes and allows an interrupted, uncommitted inference
to replay; each committed question/configuration appears once. Completed outputs
refuse further inference. `--resume --export-only` derives missing exports only
when all observations and the final unchanged-base check are already committed;
it does not load models or start another inference run.

## Standard-question results: 720 held-out questions

The request-bound measure uses the frozen secondary rules, giving a useful
comparison of short outputs from all five configurations. It remains distinct
from complete scoped fact endorsement. All pending outputs and nonanswers stay
in the denominator.

| Configuration | Correct / 720 | Accuracy | Incorrect/nonanswer bucket | Pending, uncredited | Answers / 720 |
|---|---:|---:|---:|---:|---:|
| Unchanged base | 500 | 69.44% | 198 | 22 | 720 |
| Updated RAG | 649 | 90.14% | 49 | 22 | 720 |
| Prompt-only correction | 533 | 74.03% | 173 | 14 | 720 |
| Similarity editor | 632 | 87.78% | 54 | 34 | 720 |
| Trained scope editor | 613 | 85.14% | 74 | 33 | 697 |

The trained editor's 74 uncredited incorrect/nonanswer cases include 23
clarifications; these are not an additional exclusion. Its coverage is 697/720
(96.81%). All configurations have zero runtime errors or unavailable outputs on
this cohort. The editor gets 72/72 applicable direct questions, 39/72 applicable
template paraphrases and 97/144 applicable comparisons correct under this metric.
Updated RAG gets 72/72, 26/72 and 123/144 respectively. A narrower direct success
does not erase comparison and wording failures.

Out-of-scope request-bound correctness is 405/432 for the trained editor and
428/432 for updated RAG and the unchanged base. On the 428 questions that the base
already answers correctly, retention is 405/428 for the trained editor,
426/428 for similarity and 428/428 for updated RAG. This is distinct from overall
out-of-scope correctness. The editor makes zero observed false edit applications
out of 432 out-of-scope cases, while similarity makes 3/432. Zero observed routing
events is not a guarantee about future questions; 23 clarifications reduce coverage.

The primary complete-assertion scorer credits 207/720 for the trained editor,
206/720 for similarity and zero for the three frozen-generator configurations.
For the editor, 443 remain pending, 47 are incorrect model answers and 23 are
clarifications. Similarity has 466 pending and 48 incorrect answers. All 720
base/RAG/prompt responses remain pending because they omit the complete scoped
assertion required by that scorer. **Their zero primary credit is not zero
factual accuracy and cannot establish learned superiority.** These differences
are why both frozen metrics and raw outputs are reported.

Sources: [primary metrics](../reports/final/evaluation/metrics.json) and
[request-bound metrics](../reports/final/evaluation/short_qa_metrics.json).

## Paired uncertainty

The existing frozen family-bootstrap implementation aggregates the saved secondary
verdicts without new scoring, inference or tuning. It uses every standard
question, 12 families, 2,000 resamples and seed 42. Differences are right minus
left; the following intervals are percentage points.

| Difference | Estimate | Paired 95% family-cluster interval |
|---|---:|---:|
| Trained editor minus updated RAG | -5.00 | [-6.67, -3.06] |
| Similarity editor minus updated RAG | -2.36 | [-4.03, -0.56] |
| Trained editor minus unchanged base | +15.69 | [+14.17, +17.22] |

These descriptive intervals concern the controlled standard-question metric,
not arbitrary dialogue. Twelve correlated families and shared templates limit
external validity. See [all ten paired comparisons](../reports/final/secondary-paired-uncertainty.json).
To reproduce this aggregation into a fresh derived file:

```powershell
.\.venv\Scripts\python.exe scripts/paired_final_short_qa.py --run reports/final/evaluation --protocol reports/final/protocol-v2.json --protocol-sha256 8a938b33055220fdb24fa86ed056405f186dc9b6c4b72fe136c76ff0dc3d86ab --output runtime/secondary-paired-reproduction.json
```

## Retrieval and scope classification

For the trained editor, pre-eligibility applicable-edit retrieval is 283/288 at
rank 1 and 288/288 at rank 3. Updated RAG's corresponding retrieval is 280/288
and 288/288. These are small alternate-world memories, not a large-memory scale
benchmark.

At the fixed 0.97 threshold, raw pair classification has 288 TP, 19 FP, 413 TN
and zero FN: accuracy 701/720, precision 288/307 and recall 288/288. Raw FPR is
19/432; runtime's zero false applications additionally depends on eligibility
guards and uncertainty handling. Raw Brier score is 0.03195 and 10-bin ECE is
0.03218. Calibration is complete on this population but does not transfer
unchanged to unfamiliar wording.

The nine abstracted inputs shared with training are 9/9 correct; the 711 unshared
inputs are 692/711 correct, with 19 false positives. Raw identity/lineage
separation remains intact, but shared logical forms mean neither group proves
unrestricted understanding.

## Challenges: no established general-language accuracy

The trained editor clarifies 48/96 challenges and answers 48; none receives
primary complete-assertion credit, and 48 responses remain pending. Similarity
receives 40/96 primary credits and leaves 56 pending. Base and updated RAG leave
96 pending each; prompt-only leaves 95 pending and one incorrect. There are zero
runtime errors in this matched challenge cohort.

All 96 challenge question forms are unsupported by the frozen request-bound
parser, so its zero credits do not measure zero factual accuracy. Under those
rules similarity/base/RAG have 96 pending; prompt has 95 pending plus one
incorrect; the trained editor has 48 pending plus 48 uncredited clarifications.
The scorer was not expanded after seeing test questions. No unfamiliar-wording
success rate is established by these unreviewed semantic cases.

Raw challenge scope recall is 72/96, with 24 false negatives among 96 applicable
pairs; Brier 0.24834 and ECE 0.25476 are much worse than on standard forms.
Runtime coverage falls to 48/96. This is a measured generalization/coverage
failure, independently of how many unsupported generated phrases might be
semantically correct. The challenges are agent-authored and automatically
audited, not independently human-authored/reviewed.

## Oracle runtime failure and weight-integrity boundary

The original 384-case oracle export contains 65 generated answers, 319 runtime
errors, 45 primary credits, eight pending responses and 12 incorrect generated
answers. Its fixed-denominator credited result is 45/384; coverage is 65/384.
The 319 errors include 223/288 standard answers and all 96 challenges. They must
remain uncredited, and the 65 early answers are not a representative replacement
denominator. These values do not estimate complete answer-model capability.

CUDA reported an unknown runtime error from task 4,145 through task 4,463. A
fresh process subsequently confirmed CUDA availability and a simple GPU sum.
The original instance's post-inference weight fingerprint was unavailable.
The final closure read unchanged source/model files and reproduced the expected
base fingerprint in a fresh instance, committing no new predictions. Thus the
manifest's equal before/after fingerprints describe that available closure
evidence; they do not reconstruct the failed original instance's memory state.
No persistent model/data changes were made.

A separately named recovery measurement for the failed oracle cases is complete.
It uses the same locked configuration and preserves all original errors, while
identifying the new observations and runtime sessions. It does not overwrite the
primary export or represent the uninterrupted original oracle run.
The [recovery declaration](../reports/final/oracle-runtime-recovery-declaration.json),
SHA-256 `fba83b9215eb91f6e239b36d033c9045a22329e19624e139dc639e4885f6d355`,
selects exactly the 319 CUDA-error IDs and leaves the 65 original answers
untouched. Its finite cap is 319 generation calls. New CUDA failure stops that
supplement; reserved interrupted IDs cannot be silently retried or omitted.
The first supplement session stopped during model loading with `No CUDA GPUs are
available`, before making any generation call or reserving a selected ID. Its
[interrupted manifest](../reports/final/oracle-runtime-recovery/interrupted_manifest.json)
records zero committed observations, zero reservations and all 319 selected cases
not attempted at that interruption. After the user restored GPU availability and
a fresh RTX 4050 matrix-multiply probe passed, the same declared supplement
resumed from index 0. All 319 selected cases generated answers with zero new
runtime errors, interrupted reservations or not-attempted cases. Selection used
execution status only, not correctness. No model, scoring, input or setting changed.

The [completed recovery manifest](../reports/final/oracle-runtime-recovery/manifest.json)
binds new predictions and the combined 384-case view. That view retains the 65
original successful outputs and adds the 319 operationally recovered outputs.
It has 247/384 primary complete-assertion credits, 90 pending and 47 incorrect
generated answers, with 384/384 answered and zero runtime errors. It is a mixed
original/recovery measurement, not a new uninterrupted evaluation or an ML
improvement over the original 45/384 runtime-affected result.

| Combined oracle cohort | Primary correct | Primary pending | Incorrect generated | Request-bound correct | Request-bound pending |
|---|---:|---:|---:|---:|---:|
| Standard answer, 288 | 207/288 | 34 | 47 | 208/288 | 33 |
| Challenge, 96 | 40/96 | 56 | 0 | 0/96 | 96 |
| All 384 | 247/384 | 90 | 47 | 208/384 | 129 |

There are no abstentions or runtime errors in the combined oracle view. Its 47
wrong answers are standard comparisons; oracle routing does not solve answer
polarity. Standard primary strata are 71/72 direct, 39/72 paraphrase and 97/144
comparison. The secondary grammar credits one additional complete short direct
answer, making standard request-bound credit 208/288 (72.22%). Every challenge
question remains unsupported by that secondary parser, so its zero challenge
credit is not zero factual accuracy. Primary challenge credits show 40 complete
assertions with oracle routing, while 56 remain pending; this does not change the
matched editor's earlier 48/96 challenge coverage failure.

The healthy recovery session records matching before/after base fingerprint
`d52e6f6b8cf6ecdda796d5d6884879308bd64a5e5944b8168c88989a3deef2ee`.
That verifies the available healthy instance, not the unavailable memory state of
the original failed GPU instance. Combined latency mixes original and recovery
sessions and must not be presented as one uninterrupted timing benchmark.
Sources: [recovery primary metrics](../reports/final/oracle-runtime-recovery/metrics.json)
and [recovery request-bound metrics](../reports/final/oracle-runtime-recovery/short_qa_metrics.json).
See [runtime-failure-disclosure.json](../reports/final/runtime-failure-disclosure.json).

## Measured serial latency and resources

| Standard cohort, 720 requests each | p50 ms | p95 ms |
|---|---:|---:|
| Unchanged base | 167.58 | 290.50 |
| Updated RAG | 150.99 | 278.84 |
| Prompt-only | 154.26 | 285.27 |
| Similarity editor | 250.57 | 873.60 |
| Trained editor | 241.78 | 906.22 |

These include retrieval/routing/generation and the editor's clarifications.
They exclude model loading, scoring, checkpoint IO and HTTP/browser overhead.
No warmup requests were removed. Fixed ordering and shared caches can bias timing;
the lower challenge p50 for the editor includes many short clarifications.
Model construction was recorded separately at 101.97, 37.12, 33.84 and 27.07
seconds across four sessions, the last being zero-prediction closure. Available
session snapshots report about 933.07 MiB allocated and 1,002 MiB reserved at
measurement points, not a complete inference lifetime peak. The failed instance's
end resources are unavailable. Training resources remain in [MODEL_CARD.md](MODEL_CARD.md).

## Separate learned HTTP engineering measurement

After GPU restoration, the finite isolated HTTP benchmark completed 30 measured
requests, all HTTP 200 and `model_generated`, with zero errors or clarifications.
Warm client round-trip p50 is **230.42 ms** and p95 **372.04 ms**. One cold request
with lazy model loading took **68,781.35 ms**; it is reported separately. Three
warmup requests are excluded from the warm percentile population. The workload
uses concurrency 1, a new HTTP connection per request, and three repeated
development demonstration questions; it is neither the held-out test population
nor a load test.

Client timings include transport, JSON decoding, retrieval, learned routing and
generation, plus the API's separate before-answer generation. Shared model and
retrieval caches affect warm timings. The selected model/configuration bindings
remain fixed, all responses observe fixture revision 3, and the isolated fixture's
domain state is unchanged. The user's live application/database is not mutated.

Availability is not correctness: the four-device question still elicits
“Yes. Nimbus Basic version v2 supports 3 devices.” The response contradicts the
requested comparison even while copying the corrected limit. Thirty answered
requests are not 30 correct answers. The artifact explicitly records
`ml_accuracy=null`; no accuracy estimate is derived from its repetitions. See the
[completed HTTP evidence](../reports/final/neural-http.json).

## Release interpretation

Both trained models, an approved-edit workflow, honest baselines and reproducible
frozen artifacts now exist. The trained editor improves stale-base direct
answers, but updated RAG is stronger on overall standard short QA; unfamiliar
wording and original oracle runtime reliability remain limitations. GPU
availability is restored; the separately named oracle recovery and finite learned
HTTP measurement are complete. Reporting preserves the original runtime errors
and identifies the operational supplement. The finite implementation, training,
evaluation and presentation stages are complete; release-head CI remains the
final sign-off gate. No locked ML claim changes and no accuracy number is promised.

The [report PDF](../output/pdf/EditGuard-Report.pdf) and its six individual
landscape scientific figure pages have now been rendered and visually checked.
The current dashboard and its exact JSON/CSV exports are verified. Local release
checks pass 516 tests with 18 explicit optional skips; a separate 34-test synthetic
CPU numeric/checkpoint group also passes. These are engineering results, distinct
from the ML scores and from both the original oracle incident and its separate
completed operational recovery.
