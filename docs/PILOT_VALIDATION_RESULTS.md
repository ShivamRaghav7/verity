# Initial pilot-v2 validation results

Both trained auxiliary models now run in the comparison pipeline. On this small
development population, the learned scope classifier missed all 12 applicable
direct questions, and the answer model generated an incorrect yes/no endorsement
for 12 comparison questions even when given the correct edit. These are useful
failure diagnoses. They do not establish that the learned editor outperforms RAG
or that the project has completed its final evaluation.

The strict complete-assertion scorer credited 19/120 generated answers from the
learned editor, 30/120 from the similarity editor, and 31/48 from the answer model
with oracle routing. Most other outputs require semantic adjudication. In
particular, **zero automatically credited baseline assertions is not zero factual
correctness**: the frozen FLAN-T5-small assistant frequently emitted short values
such as `1` rather than stating the requested subject, scope and relation.
The primary scorer conservatively leaves those outputs uncredited and marks them
`review_required`; it does not silently label them wrong.

## Population and comparison controls

The immutable evidence is in `reports/pilot-v2-validation-initial`: 120 validation
questions, each run through five configurations, producing 600 records. An
additional 48 applicable questions were run through the trained answer model
using the correct edit and intended supporting context. That oracle diagnostic
has a separate denominator and is not a sixth end-to-end configuration.

The 120 questions come from two entity families (`family-044`, `family-045`)
absent from the four-family training partition. They comprise 12 direct questions,
12 paraphrases, 24 numerical comparisons and 72 out-of-scope questions, with 48
applicable and 72 negative scope labels. Counterfactual edit variants remain
inside the same family split. Shared question templates were used across splits,
so this does **not** measure unfamiliar wording or independent natural language
coverage. No test or challenge partition was read.

The dataset is a controlled fictional support catalogue. Its 504 label decisions
were reviewed under the user-authorized `automated_development` policy; the model
and evaluation manifests explicitly record `human_reviewed_data: false`.
Dataset label acceptance is separate from human semantic adjudication of model
outputs. No output has been credited through human adjudication in this report.

The unchanged base uses the initial handbook. Updated RAG materializes approved
edits into that handbook before retrieving facts. Prompt-only receives approved
correction descriptors, and both editors receive the same approved memory and
share the same trained answer generator and frozen fallback. Per question,
memory contains one focal edit and at most eight foreign-product distractors;
counterfactual corrections are evaluated as separate scenarios rather than
activated simultaneously. This is bounded development evaluation, not a large
production memory benchmark. All configurations use the same requested scope,
questions, fact sources, 48-token answer budget and scoring rules. The base model
tensor fingerprint is unchanged before and after the run.

## Primary complete-assertion scores

| Configuration | Automatically credited / answerable | Explicitly incorrect | Semantic review pending | Answered / answerable |
|---|---:|---:|---:|---:|
| Unchanged base | 0/120 | 0 | 120 | 120/120 |
| Properly updated RAG | 0/120 | 0 | 120 | 120/120 |
| Prompt-only correction | 0/120 | 0 | 120 | 120/120 |
| Similarity editor | 30/120 (25.0%) | 12 | 78 | 120/120 |
| Trained scope editor | 19/120 (15.8%) | 12 | 89 | 120/120 |
| Oracle answer model; separate population | 31/48 (64.6%) | 12 | 5 | 48/48 |

Every row remains in the denominator. These runs contain zero abstentions,
errors or unavailable outputs, and every answer is `model_generated`. A
deterministic structured answer contributes no model-generation credit.
Pending semantic review is unsuccessful under the fixed-denominator primary
score, but its factual status is unresolved. The primary rule requires a single
complete assertion endorsing the requested entity, plan, version, relation,
value and unit, with the correct yes/no polarity where applicable.

| Applicable subset | Similarity editor | Trained scope editor | Oracle answer model |
|---|---:|---:|---:|
| Direct | 12/12 | 0/12; all 12 pending | 12/12 |
| Paraphrases | 6/12; 6 pending | 7/12; 5 pending | 7/12; 5 pending |
| Comparisons | 12/24; 12 incorrect | 12/24; 12 incorrect | 12/24; 12 incorrect |

The oracle model's direct questions succeeded under this narrow scoring rule.
Its paraphrase and comparison failures show why oracle generation quality must
be measured separately from retrieval and routing.

![Recorded assertion outcomes and CLI latency](../reports/pilot-v2-figures/validation-outcomes.png)

## Separate exploratory short-answer diagnostic

The primary report is preserved byte for byte. A separate **post-hoc** analysis
in `reports/pilot-v2-short-qa` examines the same recorded outputs without
retraining or inference. Its request-bound scorer supports only the controlled
dataset's 12 question forms. It independently parses the question's exact entity
alias, plan, version and relation, checks that they agree with the request and
ground truth, then accepts a whole quantity with a compatible unit or a whole
`Yes`/`No` for the requested comparison predicate. Unsupported prose remains
pending. Existing complete-assertion verdicts are retained.

For example, a whole `1` can satisfy the device-value question in that explicit
request context, while `Yes` on a comparison measures the predicate and does
not prove the exact device limit. The question supplies the omitted subject,
property and unit. Such short outputs **do not prove a complete scoped fact
endorsement**. No output is rewritten into a template assertion, no deterministic
answer receives model credit, and a retrieved source is not used as a substitute
for the generated text.

| Configuration | Requested QA correct / all cases | Incorrect | Pending and uncredited | Complete scoped facts established |
|---|---:|---:|---:|---:|
| Unchanged base | 85/120 (70.8%) | 31 | 4 | 0 |
| Updated RAG | 110/120 (91.7%) | 6 | 4 | 0 |
| Prompt-only | 89/120 (74.2%) | 29 | 2 | 0 |
| Similarity editor | 102/120 (85.0%) | 13 | 5 | 30 |
| Trained scope editor | 91/120 (75.8%) | 24 | 5 | 19 |
| Oracle answer; separate population | 31/48 (64.6%) | 12 | 5 | 31 |

This bounded diagnostic gives properly updated RAG the highest observed
requested-QA score; the learned editor does not beat it. Under the same narrow
rule, every configuration answers 72/72 out-of-scope requested values correctly
and preserves requested-QA correctness on those 72 baseline-correct cases.
Those observations apply to these two families and known templates, not arbitrary
questions or guaranteed retention. The diagnostic is neither independent human
semantic review nor a frozen final-test result. It was introduced after seeing
the validation fragments and must be labeled exploratory; it does not replace
the primary strict scores or justify a new final performance claim.

## Retrieval, scope and calibration

Applicable-edit retrieval recall is 47/48 at rank 1 and 48/48 at rank 3 for
the configurations that retrieve edit descriptors. These counts measure the
recorded raw memory candidate lists for applicable questions. Oracle routing
has no retrieval credit. The single-focal-pair similarity calibration report
must not be presented as this full-memory retrieval measurement.

The scope classifier's calibrated raw question–correction pairs have TP 36,
FP 0, TN 72 and FN 12 at threshold 0.11: precision 36/36, recall 36/48 (75.0%)
and false-positive rate 0/72. Raw pair classification is recorded independently
of deterministic plan/product/version compatibility gates. Actual end-to-end
editor routing selected the focal correction 36 times and fell back 84 times.
All 12 missed applicable questions used the direct question template; general
paraphrase robustness cannot be inferred from the higher aggregate pair score.

Inspection of only `scope_train.jsonl` and `scope_validation.jsonl` found a
question-form confound in the original pilot. Negative rows are tagged
`out_of_scope` in metadata, but their actual wording always matches one of the
four direct question templates. Positives use a direct template, a paraphrase,
and two comparison quantities per edit. The following counts classify actual
wording rather than treating `out_of_scope` as a linguistic style:

| Actual wording | Training positive / negative | Validation positive / negative |
|---|---:|---:|
| Direct | 24 / 144 | 12 / 72 |
| Paraphrase | 24 / 0 | 12 / 0 |
| Comparison; both quantity variants | 48 / 0 | 24 / 0 |
| All styles | 96 / 144 | 48 / 72 |

The overall applicable class prior is 40% in both partitions, while only
24/168 (14.3%) training and 12/84 (14.3%) validation direct-form questions are
applicable. Every paraphrase and comparison question is positive. This makes
question form an unintended shortcut. The classifier's 12 direct-form false
negatives are consistent with dependence on that shortcut, but these observations
do not establish causality or isolate the effect of wording from other features.
The inspected source hashes are
`f061e979322400b28d7cbad162c7f3d6e4ae6e69e6e97c70b4407f58b32b66cd`
(training) and
`e4060769f69916e4abdd4126c10fb9bbe4e8862f0948942c4df168f22571f7e3`
(validation). No test or challenge data was inspected for this diagnosis.

The planned bounded pilot-v3 correction rotates the existing six negative types
across the four question forms. It keeps the same families, splits, facts and
504-row total (360 scope pairs and 144 answer examples), with new dataset/review
identities for changed rows. This is a targeted data-design repair rather than
an expansion in dataset size. No new trained-model result is established by
preparing that packet; its review, profiling and finite training run remain
separate stages. The archived v2 data and evaluation evidence stay unchanged.

Temperature 1.5 was chosen from a finite six-value grid by validation binary
negative log-likelihood. Threshold 0.11 and ambiguity margin 0 were chosen from
101 thresholds and four margins by the declared cost:
`4 × false applications + missed positives + 0.25 × clarifications`.
The selected operating point has cost 12/120, zero clarifications and 120/120
non-clarification decisions. A zero selected margin does not demonstrate safe
handling of naturally ambiguous requests. The application's deterministic
conflict and incompatible-scope checks remain separate safeguards.

On these same 120 pairs, the calibrated Brier score is 0.098265 and ten-bin
equal-width expected calibration error is 0.083367. Calibration and threshold
selection use the validation population being reported, so these values are
descriptive and optimistic for new data. The graphs show all finite operating
candidates and their coverage rather than hiding clarification outcomes.
Classifier decision coverage is distinct from generated-answer coverage;
clarifications remain unsuccessful in fixed-denominator classification accuracy.

The embedding baseline uses cosine threshold 0.46. Its **raw** pair score has
TP 47, FP 48, TN 24 and FN 1: precision 47/95, recall 47/48. After compatibility
gates, the same policy has TP 47, FP 0, TN 72 and FN 1. Gates reject 60
incompatible pairs and make a substantial contribution to this result. Cosine
is a similarity, not a calibrated probability. Its separate raw and gated
metrics are in `reports/similarity-pilot-v2/calibration.json`.

![Recorded learned scope calibration and operating points](../reports/pilot-v2-figures/scope-calibration.png)

## Out-of-scope preservation and correctness

Both editors selected an edit on 0/72 out-of-scope questions. This is a measured
routing event, not proof that every generated answer is correct; correction
influence in RAG or prompt-only prose cannot be inferred from that event.
All configurations have 0/72 automatically credited complete assertions and
72 pending semantic reviews on these questions. The strict scorer found no
baseline-correct out-of-scope assertion, so **baseline-correct retention is
undefined (0/0), not 100% or 0%**.

The primary semantic output-preservation field is 0/72 with zero parsed answer
pairs. That field cannot establish changed answers when fragments remain
unparsed. As a separate literal-string diagnostic, both editors and updated
RAG preserve all 72/72 unchanged-base output strings; prompt-only preserves
70/72. Those exact-string counts are recomputed from raw predictions in the
figure manifest. Identical output does not establish factual correctness:
preserving an incorrect base answer remains an incorrect answer.

## Measured latency and resources

| Configuration | CLI inference p50 | CLI inference p95 | Samples |
|---|---:|---:|---:|
| Unchanged base | 159.992 ms | 282.157 ms | 120 |
| Updated RAG | 143.262 ms | 248.714 ms | 120 |
| Prompt-only | 150.658 ms | 275.170 ms | 120 |
| Similarity editor | 213.828 ms | 867.840 ms | 120 |
| Trained scope editor | 201.710 ms | 849.964 ms | 120 |
| Oracle answer diagnostic | 736.513 ms | 879.705 ms | 48 |

These measurements are serial CLI inference timers. Runner/model construction
and the base fingerprint pass precede those timers; HTTP/UI/network overhead
is excluded. There was no independent randomized configuration order or formal
latency warmup. Fixed per-question configuration order and shared embedding
caches can bias timing comparisons. The run's GPU peak after reset was
948.663 MiB allocated and 1008.0 MiB reserved; final sampled process RSS was
1510.188 MiB. These are a measured window and one process sample, not universal
resource requirements or full application end-to-end latency.

The training loops measured 89.448 s for scope and 245.856 s for the answer
model, including checkpoint I/O. Model loading, tokenization and final export
are outside these loop totals; complete command wall time was not recorded.
The selected scope checkpoint is step 120 of a 360-step run; the answer
checkpoint is step 240 of 240. Validation loss and resource details are in
[the training results](PILOT_TRAINING_RESULTS.md).

![Recorded training loss, loop components and resources](../reports/pilot-v2-figures/training-resources.png)

## Recorded failure examples

| Failure layer | Question and expected behavior | Recorded output or decision | Implication |
|---|---|---|---|
| Scope false negative | `How many devices does Aster 044 Basic version v2 support?`; approved value 1 | Calibrated focal probability 0.094326, below 0.11; frozen fallback emits `4` | Retrieving a correction did not ensure its application. This particular fragment is pending under the strict parser; the routing miss is established independently. |
| Answer polarity | `Can I connect 2 devices on Aster 044 Basic version v2?`; expected no, limit 1 | `Yes. Aster 044 Basic version v2 supports 1 device.` | Correct value mention plus an incorrect endorsement is an incorrect answer; oracle routing cannot repair it. |
| Strict scorer limitation | `What is the device allowance on A044 Basic version v2?` | Oracle answer: `Aster 044 Basic version v2 includes 1 device.`; semantic review pending | A different assertion verb falls outside the strict automatic grammar. Pending does not mean factually wrong. |
| Baseline output format | Updated RAG receives the approved one-device fact for the direct question | Emits `1`; semantic review pending | A short value does not satisfy the complete-assertion contract; fair factual comparison needs a separately stated bounded short-answer rule or semantic review. |

All 12 oracle comparison errors have expected `No` but generated `Yes`, despite
including the correct factual limit. They remain explicitly incorrect in the
separate short-answer diagnostic. This polarity failure is independent of the
scope-routing misses and must not be attributed to retrieval.

The raw files retain all failures and fragments. A supplementary short-answer
analysis examines value fragments in the context of an explicit question while
preserving this primary report and identifying its narrower scoring assumptions.
It never counts reconstructed template answers as generated output.

## Uncertainty and remaining work

The metrics file contains paired family-cluster percentile bootstrap intervals
(2,000 resamples, seed 42). There are only two validation families. Narrow or
degenerate intervals from those two clusters should not be presented as precise
population uncertainty, and pending baseline adjudication prevents a factual
superiority claim from the strict credited-count differences. More independent
families and a fixed final protocol are required before defensible numerical
resume claims.

This pilot establishes real training, artifact integrity and integrated
inference, with measured limitations. The final protocol still needs independent
label/output review where required, manually authored unfamiliar-wording
challenges, larger independent evaluation populations, frozen configurations
and scoring rules, and a single final held-out test evaluation. Conflict,
historical-scope and rollback consistency remain covered by separate engineering
tests and deployment evidence; they are not estimated by these 120 ordinary
validation questions. The scope misses and polarity failures are finite,
documented development targets, not a reason to expand the project indefinitely.

## Reproduce the figures

With the recorded artifacts and plotting environment installed:

```powershell
.venv-ml\Scripts\python.exe scripts/plot_pilot_evaluation.py --calibration reports/scope-pilot-v2 --scope reports/pilot-v2-training/scope --answer reports/pilot-v2-training/answer --output reports/pilot-v2-figures-reproduced
```

The command reads completed training logs/manifests, frozen calibration
probabilities/grid and recorded validation predictions/metrics. It loads no
model and reads no generated dataset, test or challenge file. A fresh output
directory is required. It checks hashes, full counts, selected operating counts,
Brier/ECE and reliability bins, emits PNG and SVG figures, records Matplotlib
version and all source/output hashes, and verifies that input bytes are unchanged.
The command uses exact archived metadata and logs checked into Git; reproducing
these plots does not require the ignored model weights or another training run.
The checked-in `reports/pilot-v2-figures/manifest.json` binds the displayed plots
to those exact inputs. First render drafts are retained only under ignored
`runtime/figure-drafts`; the checked-in files are the visually inspected revision.
