# Failure analysis

The project now has two trained auxiliary models, matched development comparisons
and retained raw failures. The selected scope model substantially improves
development routing, while the answer generator still makes comparison and
paraphrase errors. Updated RAG leads the separate development short-answer
diagnostic. The frozen matched final comparison is now sealed and confirms that
updated RAG leads the standard short-answer metric. It also exposes weak
unfamiliar-wording coverage. The original oracle measurement is incomplete due
to 319 CUDA runtime errors; it is preserved separately from any recovery.
No final result was used to tune models, thresholds, labels or scoring rules.

## Learned component findings from development

| Finding | Evidence and consequence | Resolution or remaining limit |
|---|---|---|
| Initial scope learning exploited question-form imbalance | On the common pilot-v3 validation population, the initial model gets 56/120 raw pairs correct; direct positive edit questions are missed. | Balanced existing negative forms in a versioned pilot, preserving facts, labels, splits and size. Revision 1 gets 98/120 on the same pairs. |
| Revision 1 remains weak at identity/version discrimination | Raw TP 46, FP 20, TN 52, FN 2. Wrong-entity and wrong-version false positives remain despite recognizing all direct positives. | Runtime compatibility guards reject incompatible edits. These are deterministic safeguards, not evidence that the classifier learned identity resolution. |
| Irrelevant literals burden the compact classifier | Revision 2 uses declared `entity_neutral_v1` preprocessing and gets 119/120 raw validation pairs correct, with one false positive. | Keep the transformation explicit. Eight abstracted validation inputs overlap training; 111/112 unshared are correct. Shared templates still limit generalization. |
| Raw threshold optimization differs from runtime routing | The earlier operating-point selection scored raw pairs, while runtime gates reject incompatible edits first. | Calibrate the defined runtime decision using validation-only fixed scores. Selected temperature 1.0, threshold 0.97 and margin 0.025 yield 48 true applies, no false applies and one clarification on 120 selection pairs. These are not independent test results. |
| Token-average loss hides wrong comparison polarity | The original answer model's low uniform loss coexists with 0/12 correct opening No tokens under teacher forcing. Copying the correct fact can accompany an incorrect Yes endorsement. | Weight both comparison first tokens equally at 8. The revised teacher-forced No result is 4/12 and Yes is 11/12; teacher forcing is not free-running accuracy. |
| Weighted training helps some comparisons, not general arithmetic | In matched free generation with oracle routing, the selected model correctly endorses 5/12 No answers and fails seven. Complete assertions are correct on 34/48 total oracle questions. | Retain raw failures. Do not claim that loss weighting solves numerical reasoning or that supplying a correct correction verifies generated text. No final-test-driven revision is allowed. |
| Paraphrase can elicit the wrong relation or incomplete prose | Oracle complete-assertion credit is 6/12 on development paraphrases, versus 12/12 direct questions; five outputs remain pending semantic interpretation. | Score requested subject, relation, unit and endorsement. Incomplete/pending cases remain uncredited in the full denominator. |
| Conservative complete-assertion scoring excludes useful fragments | Frozen baselines often output a bare number or Yes/No. All 120 outputs for each base/RAG/prompt baseline remain pending under primary scoring. | Report an independently labeled request-bound whole-answer metric, frozen before final question access. Primary zero credit is not proof of zero factual correctness. |
| A legitimate storage assertion was outside the parser grammar | “Supports N GB of storage” gives a full scoped storage-quota fact, but the earlier parser accepted only a different verb. Four correct revised oracle storage responses were uncredited. | Add a bounded whole-assertion alias under scorer v3, with negative/conditional/example/identity/unit regressions. Rescore every compared configuration in fresh artifacts. Preserve earlier scores and disclose the development-time change. |
| Scope gains do not guarantee end-to-end gains | The selected learned editor obtains 86/120 in the separate request-bound diagnostic; updated RAG obtains 91/120 under matched corrections and settings. | Report the stronger updated baseline honestly. Oracle generation isolates remaining answer-model errors; no learned-superiority claim follows from classifier improvement. |

The sources are [scope diagnosis](../reports/pilot-v3-neutral-scope-diagnosis/summary.json),
[teacher-forced answer diagnosis](../reports/pilot-v2-answer-revision-diagnosis/summary.json),
[matched complete-assertion metrics](../reports/targeted-candidate-v3-scored/metrics.json)
and [request-bound diagnostic](../reports/targeted-candidate-v3-short-qa/summary.json).
Counts come from development populations with automated review. Checkpoint and
calibration selection used these populations, so their results are optimistic
development estimates.

## Findings from the sealed final comparison

| Finding | Measured evidence | Interpretation |
|---|---|---|
| Overall updated RAG remains stronger | Standard request-bound QA: RAG 649/720 (90.14%), similarity 632/720, trained editor 613/720 (85.14%), prompt-only 533/720, stale base 500/720. | The editor improves stale-base behavior but does not beat a fairly updated retrieval baseline. Learned minus RAG is -5.00 percentage points, paired 95% family interval [-6.67, -3.06]. |
| Direct editing does not ensure paraphrase/comparison correctness | Trained editor: applicable direct 72/72, paraphrases 39/72, comparisons 97/144, using the request-bound metric. RAG: 72/72, 26/72, 123/144. | Retain all strata. The editor's paraphrase advantage does not erase its weaker comparisons or overall result. |
| Guards preserve scope at a coverage cost | Zero selected-edit false applications in 432 out-of-scope cases, versus similarity 3/432. Trained editor clarifies 23/720 standard questions. | A zero observed event rate does not guarantee universal reliability. Clarifications count as uncredited on answerable questions. |
| Avoiding an edit does not make the base answer true | Trained editor out-of-scope correctness 405/432; baseline-correct retention 405/428. Base and RAG correctness are 428/432. | Preserve the four already incorrect base cases as failures. Report retention and correctness separately. |
| Raw classification and complete routing differ | At fixed threshold 0.97, raw scope has TP 288, FP 19, TN 413, FN 0: 701/720 correct. Runtime has zero observed false applications after gates/margin. | The raw classifier's 19 false positives cannot be presented as zero learned errors merely because controls prevent application. |
| Standard calibration does not transfer to challenge wording | Standard Brier/ECE 0.03195/0.03218; challenge 0.24834/0.25476. Raw challenge scope recall is 72/96. | The small validation population and familiar templates underestimate generalization risk. No threshold was changed after final access. |
| High scores can still fall inside the abstention band | A held-out “already have two devices attached; what's the total cap?” challenge receives an applicable score around 0.987, yet runtime clarifies under the selected threshold/margin policy. | Literal normalization and high probability do not imply the chosen operating policy answers unfamiliar wording. This contributes to 48/96 challenge clarifications. |
| Challenge output quality is not fully measured by either grammar | Trained editor has zero primary credits, 48 pending responses and 48 clarifications; similarity has 40 complete-assertion credits and 56 pending. All 96 question forms are unsupported by the frozen request-bound parser. | Do not equate secondary zero credit with zero factual accuracy or retrofit a test-specific scorer. General-language success is not established. |
| Primary grammar gives a misleading headline if shown alone | Standard primary credits: trained 207/720, similarity 206/720, base/RAG/prompt zero with all 720 pending each. | Bare useful baseline answers do not satisfy complete-assertion grammar. Their zero is not factual failure and cannot justify learned superiority. |
| Oracle execution encountered infrastructure failure | Original 384: 65 answered, 319 CUDA errors, 45 primary credits, eight pending and 12 incorrect generated answers. All 96 oracle challenges are runtime errors. | Keep 45/384 and 65/384 coverage with explicit failure attribution; 319 errors are not wrong model answers, and 65 early answers do not form an unbiased capability estimate. |

The five matched configurations have 4,080 observations: 4,009 answered and 71
clarifications, **zero runtime errors**. Thus the oracle CUDA incident does not
erase the completed matched comparison. All standard/question and challenge
denominators remain intact. Sources are the sealed
[primary metrics](../reports/final/evaluation/metrics.json),
[request-bound metrics](../reports/final/evaluation/short_qa_metrics.json),
[paired secondary intervals](../reports/final/secondary-paired-uncertainty.json)
and [runtime failure disclosure](../reports/final/runtime-failure-disclosure.json).
Twelve controlled families and automated-only review limit external validity.

## Data and measurement findings

| Finding | Control and honest boundary |
|---|---|
| Early draft numeric supports/polarity permitted shortcuts | Original and corrected values now share per-relation pools, excluding equality; both comparison polarities occur per scenario. Versioned archives preserve the rejected drafts. |
| Shared templates remain after entity splitting | Split families before augmentation and report challenge wording separately. Entity-heldout results do not establish unfamiliar-language understanding. |
| Literal abstraction reintroduces exact scope-input overlap | Final audit finds nine same-label test/train overlaps, zero test/validation overlaps and 711 unshared scope inputs. Keep separate denominators; raw identity/lineage disjointness alone is insufficient. |
| Machine audit is weaker than independent semantic review | The user authorized automated final audit. Accepted 1,104/1,104 controlled labels are consistency evidence, with no human-review or human-authored-challenge claim. |
| Two validation families give weak uncertainty evidence | Development comparisons do not support population superiority. Final paired intervals cluster by the 12 held-out families and disclose their limited number. |
| Runtime measurements depend on boundary | Training loop measurements exclude complete startup/export time. Serial inference latency excludes browser/HTTP rendering and is not a randomized load test. Resumed-session resources do not establish uninterrupted lifetime peaks. |

## Governance and application findings

| Finding | Implemented control |
|---|---|
| Unknown tiers or unsupported versions could inherit a selected context | Explicit product/plan/version vocabulary and compatibility checks request clarification. Arbitrary reference resolution remains outside the bounded parser. |
| A short alias appears inside a similar product name | Span-aware alias matching preserves the longest contained name and detects separately mentioned products. Multi-product ambiguity clarifies. |
| Overlapping contradictory edits could silently win by recency | No latest-write-wins rule. Conflicting scopes/intervals block and require administrator resolution. |
| Supersession or withdrawal could destroy historical behavior | Original-value checks, exact-scope lineage, effective intervals and ancestor restoration run transactionally. Historical queries use applicable earlier facts. |
| Independent reads could combine revisions | SQLite snapshots and revision-aware state keep facts, history and audit export coherent; cross-instance regression checks state consistency. |
| Cached answers could survive edits or rollback | Revision-aware keys invalidate stale responses when state changes. Failed mutations leave revision and facts unchanged. |
| Retrieved provenance can look like answer verification | UI/API distinguish `model_generated`, structured facts and clarification. Sources support inspection; they do not certify arbitrary generated prose. |

The isolated selected-model workflow passes 31/31 engineering checks, covering
administrator edits, scope guards, conflict handling, supersession, historical
facts and rollback. These checks validate state/routing contracts, not 31 correct
neural answers. Its 13 handpicked factual observations and repeated requests are
not an independent statistical benchmark. See
[`targeted-candidate-learned-workflow.json`](../reports/targeted-candidate-learned-workflow.json).

## Final execution incidents

An initial freeze/runner dependency-inventory mismatch caused a startup rejection
before model construction, packet loading, ledger creation or inference. The
source inventory was unified and the handoff covered by regression. Original
protocol/packet/logs are retained. The active packet's question/truth/derivation
bytes remain identical; models, threshold and scorers did not change.

A later Windows access-denied failure on atomic progress replacement interrupted
execution at committed prefixes 288 and 583. Both committed chains remain intact;
the lock owner/cause has not been established. A separately declared IO-only
wrapper retries the same progress/ledger value at most eight times with 1.95
seconds total delay, preserving the frozen evaluator, inference, scorer, inputs
and observations. It resumes only the same locked execution. Retry waits are
outside route latency; session history and recovery events remain available.
Do not diagnose antivirus, OneDrive or a second process without evidence,
discard successful observations, start a new benchmark or use partial results
for tuning.
After all matched comparisons completed, CUDA reported unknown runtime errors
from task 4,145 through 4,463, affecting 319 oracle observations. The failed GPU
instance could not produce its final weight fingerprint. A fresh process
confirmed CUDA availability and performed a simple GPU sum; a zero-prediction
closure verified unchanged source/model files and freshly loaded base weights.
These checks do not reconstruct the original instance's lost memory state.
Although the completed manifest contains equal before/after base hashes, reporting
must include this fingerprint-availability boundary. No persistent model/data
changes were made.

A separate fixed-configuration supplement for exactly the 319 failed oracle cases
was attempted, but model loading stopped with `No CUDA GPUs are available` before
any generation. Its interrupted manifest records zero committed/reserved IDs and
319 not-attempted cases; that interrupted session produced no new answer. The
same declared supplement retained those original errors, prompts, configuration,
scorers and successful observations. GPU availability was subsequently restored
and the same supplement resumed, answering all 319 selected cases with zero new
runtime errors or interrupted reservations. Its combined view retains the 65
original answers: 247/384 primary credits, 90 pending and 47 incorrect generated
answers; standard request-bound credit is 208/288. Oracle routing still leaves
47 wrong comparison endorsements. Challenge primary credit is 40/96 with 56
pending; secondary wording is unsupported, so zero secondary credit is not zero
factual accuracy. Healthy recovery before/after fingerprints match, without
retroactively proving the original failed instance's memory state. This is
operational recovery evidence, not a tuned model, ML improvement or replacement
original run.

The separate learned HTTP benchmark subsequently completes all 30 measured
requests with HTTP 200 and no errors/clarifications. Warm client p50/p95 are
230.42/372.04 ms; the 68,781.35 ms lazy-loading request and three warmups are
separate. It repeats three development questions, not held-out accuracy cases.
The four-device question still returns “Yes. Nimbus Basic version v2 supports
3 devices.” This is a comparison-polarity failure despite successful routing,
copied fact and HTTP 200. The benchmark's `ml_accuracy` is null; 30 answers cannot
be presented as 30 correct answers. All model/configuration bindings and the
isolated fixture state remain unchanged, and the live database is not mutated.
See [the completed engineering artifact](../reports/final/neural-http.json).
The complete incident trail and reporting status are in
[FINAL_EVALUATION.md](FINAL_EVALUATION.md).

## Failure record policy

Retain successes and failures with question ID, population, expected scoped fact,
candidates, probabilities, eligibility/routing decision, actual text, latency,
scorer version/verdict and component cause. Distinguish retrieval, scope,
generation, state and scorer errors. Abstentions and pending semantic cases stay
in denominators. Publish representative failures alongside the panel's successful
demonstration, without replacing model text with a structured answer or claiming
that preserved wrong baseline answers are factually correct.
