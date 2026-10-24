# EditGuard: measured visual editing results

The saved Tesla T4 development runs demonstrate actual model-weight updates,
export/reload and exact restoration. Adapted last-layer fine-tuning also produces
the requested full answer on the direct development question and a second
photograph, while failing the strict paraphrase target and changing an unrelated
answer to repetitive output. These observations support a demonstrable editing
and review workflow, with visible limitations.

**Final evaluation status:** the configured 20-edit, five-method evaluation has
not yet produced a completed measured report. No held-out accuracy, method
ranking or runtime guarantee is claimed here. The observations below are from
one fixed development edit drawn from the authors' training annotations.

## Evidence ledger

| Saved run | What it establishes | Primary receipt |
| --- | --- | --- |
| First MEND, raw questions/release alignment | A weight update and same-process restoration; complete persistence gate failed when the reload process was killed. | [First smoke](../reports/vlkeb/t4-smoke-1/smoke-1/smoke.json) |
| Context-aligned MEND, VQA scaffold | Eight mechanical checks passed, including fresh-process reload; generated targets still failed. | [Context/VQA smoke](../reports/vlkeb/t4-context-vqa-1/run/smoke.json) |
| Loaded-patch diagnosis | Explicit tokenization, teacher forcing and free generation under paired question formats; no new edit was fitted. | [Diagnosis](../reports/vlkeb/t4-context-vqa-diagnosis-1/diagnosis/diagnostic.json) |
| Adapted last-layer FT | Eight mechanical checks passed; exact direct/alternate-image targets, incomplete paraphrase and changed unrelated behavior. | [FT smoke](../reports/vlkeb/t4-ft-last-layer-1/run/smoke.json) |
| Updated-RAG threshold validation | A fixed gate selected on eight training rows; 24/24 applicable routes accepted, with 4/16 false applications outside scope. | [RAG validation](../reports/vlkeb/t4-rag-calibration-1/artifacts/vlkeb/rag/calibration.json) |
| Complete five-method pipeline profile | All 25 development calls executed, one MEND update and 15 FT steps completed, and exact rollback passed. | [Pipeline profile](../reports/vlkeb/t4-pipeline-profile-1/artifacts/vlkeb/pipeline-profile/profile.json) |

Each run directory also preserves its declared plan and input preflight. The
completed MEND and FT directories contain separate edit and reload reports with
original, edited and restored matrix hashes. A report is a measurement receipt;
its patch hash alone cannot reconstruct a missing patch or model checkpoint.

## The concrete development edit

The fixed target is `Albany, New York`. The direct question asks for the city in
a photograph; a paraphrase and alternate photograph probe transfer. A separate
county question targets `Hidalgo County`. These are controlled benchmark labels,
not verified real-world corrections. The image/question fixture came from
upstream TRAIN rows 0 and 1 and may have been seen during editor training.

The table copies whole generated answers from the saved context/VQA MEND and FT
receipts. It describes individual cases, without estimating general accuracy.

| Case | Desired answer | MEND after update | FT after update |
| --- | --- | --- | --- |
| Direct city question | `Albany, New York` | `depton, texas` — strict failure | `Albany, New York` — strict match |
| Paraphrase | `Albany, New York` | `New York, New York` — strict failure | `Albany` — strict failure |
| Alternate photograph | `Albany, New York` | `denton, texas` — strict failure | `Albany, New York` — strict match |
| Unrelated county question | `Hidalgo County` | Empty — strict failure | Repeated `Albany` — strict failure |

The FT [edit report](../reports/vlkeb/t4-ft-last-layer-1/run/edit.json) records
the direct baseline as `detroit, michigan` and the edited answer as
`Albany, New York`. Its baseline unrelated answer was already empty and wrong.
FT changes that behavior to a repeated city name; this is collateral behavioral
change, but does not demonstrate loss of an originally correct factual answer.
MEND preserves the empty unrelated answer. That preservation also does not make
the answer factually correct.

Both completed runs passed **8/8 mechanical checks**: an editable matrix changed,
original matrices and baseline answers were restored, a fresh process reproduced
the base state, loaded patch tensors and generated answers matched, and fresh
restoration reproduced the original tensors and answers. These checks establish
reversibility and persistence for this fixture. They do not certify locality or
usefulness on new corrections.

FT performs exactly 15 AdamW optimizer steps at learning rate `0.0002`, changing
only OPT layer 31's `fc1.weight` and `fc2.weight`. It uses FP32 optimizer master
copies while retaining the base model's released precision. Newline supervision
teaches answer termination. This is actual per-edit training, with adaptations
to the upstream recipe; the project has not meta-trained MEND.

## Why token correctness was insufficient

The loaded MEND patch's raw direct question reached **100% contextual
teacher-forced target-token correctness** in the
[diagnosis receipt](../reports/vlkeb/t4-context-vqa-diagnosis-1/diagnosis/diagnostic.json).
The same condition freely generated `Albany, New York, New York, …`, which fails
the fixed whole-answer scorer. With the VQA scaffold, the loaded patch's
contextual token score was 75%, and the generated direct answer was
`depton, texas`.

Teacher forcing includes the full correct target in the input. Free generation
uses only the image/question and its own previous tokens. These are different
measurements: a perfect teacher-forced token score is not a completed generated
answer. The diagnosis also records six standalone target tokens versus four
tokens after a separating space. The context-aligned adapter selects the actual
joined target suffix rather than using the standalone length.

## Measured development resources

The following values are copied from saved phase receipts, rounded to three
decimals. Phase duration includes model loading and diagnostic work, so it is
neither edit-only latency nor a per-query serving benchmark.

| Run and phase | Seconds | Sampled host RSS peak (GiB) | CUDA allocated peak (GiB) | CUDA reserved peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
| First MEND edit | 128.027 | 11.094 | 10.195 | 10.818 |
| Context/VQA MEND edit | 120.173 | 7.827 | 10.199 | 10.812 |
| Context/VQA MEND reload | 112.458 | 7.903 | 9.902 | 10.279 |
| FT edit | 132.053 | 7.874 | 11.150 | 11.609 |
| FT reload | 117.269 | 7.890 | 9.902 | 10.293 |

RSS was sampled every 0.05 seconds and can miss short peaks. CUDA allocator
figures exclude some driver allocations. One GiB is 2³⁰ bytes. The first reload
was killed with return code `-9`, without a phase report; memory pressure is a
plausible explanation, not proven by that code alone. Its missing resource
measurement is not filled with an estimate.

## Validation and full-pipeline readiness

The [RAG validation receipt](../reports/vlkeb/t4-rag-calibration-1/artifacts/vlkeb/rag/calibration.json)
uses upstream TRAIN rows 2–9: 24 applicable cases and 16 out-of-scope cases.
The fixed selection rule maximized validation F1, breaking ties by precision
then threshold. It selected `0.7559913590939377`, accepting **24/24 applicable
cases** and incorrectly applying the correction on **4/16 out-of-scope cases**.
Precision was 24/28 (85.7%); the false-application rate was 25%. These are routing
measurements on the subset used to select the threshold, not held-out answer
accuracy. The eight nonrandom training rows may repeat entity lineages, and the
cosine scope scores are not calibrated probabilities. No evaluation split was
opened and no retriever weights were trained.

The [complete pipeline profile](../reports/vlkeb/t4-pipeline-profile-1/artifacts/vlkeb/pipeline-profile/profile.json)
then ran TRAIN row 0 through all five methods and five cases, including the
text-only locality path. All **25 calls executed**, with one MEND update,
15 FT optimizer steps and exact restoration. Wrong answers and abstentions
remain observed outcomes; the execution gate does not require them to be correct.
The shared model load took **96.878 seconds**, the episode **41.428 seconds**,
and the worker **138.306 seconds** overall on the Tesla T4. This is development
evidence, with no evaluation split opened or editor meta-training performed.

The [preregistered run card](../configs/vlkeb-eval20-run-card.json) extrapolates
that single episode to about **925 seconds (15.5 minutes)** for one load plus
20 episodes, with an **1,800-second worker cap**. This is a planning estimate,
not a runtime guarantee or uncertainty interval; different answer lengths and
hardware load can change it. Verification and sealing add time outside the
worker. The card fixes the finite stopping rule before final inference.

## What the final comparison must answer

The [fixed evaluation protocol](VLKEB_EVALUATION_PROTOCOL.md) defines 20 distinct
source entities, five cases per edit, and five configurations: unchanged base,
prompt-only correction, updated image/question RAG, context-aligned MEND, and
adapted last-layer FT. This assigns 100 question instances per method, 500 in all.
These are protocol sizes, not obtained accuracy denominators in this report.

The comparison must separately show direct, paraphrase and alternate-image
strict accuracy, text/visual locality factual correctness, and retention among
baseline-correct unrelated answers. Errors, empty answers and unexecuted cases
remain failures in their assigned denominators. RAG routing success is separate
from answer correctness. Paired uncertainty resamples entire edit units; query,
edit and index-building latency have different meanings.

No numerical final-results table appears until its saved measured evidence
exists. Author-reported paper results remain in
[the upstream reference](VLKEB_UPSTREAM_RESULTS.md); their teacher-forced scoring
and locality definitions differ from this generated-answer experiment.

## Limits and a defensible demonstration

The development example is training-seen and cannot estimate generalization.
The planned final selection is small and nonrandom. Its source image directories
are excluded from the authors' training image slots, but target names, locality
entities, checkpoint training provenance and foundation-model pretraining can
still overlap. Labels have author provenance and automated structural checking;
our team has not independently human-reviewed their semantics.

Contextual alignment, generation formatting, precision handling and termination
supervision are disclosed adaptations. MEND and FT objectives differ, so this is
not an isolated comparison of editor architecture under identical objectives.
Strict whole-answer scoring can reject useful shorter answers such as `Albany`;
the fixed score is preserved rather than improved after inspecting outputs.
One-edit RAG memory does not establish large-memory scalability. Arbitrary
uploads, sequential edit composition, production authentication and guaranteed
locality are not established by these runs.

For a panel demonstration, show the FT direct before/after answer, its incomplete
paraphrase and repetitive unrelated output, the saved inactive candidate, the
operator's review decision, explicit activation, evidence export and rollback.
Explain why a maintainer may reject an update even when its direct target works.
The [workbench guide](VLKEB_WORKBENCH.md) documents this flow. Its 11 application
contract tests use a tiny fake backend to verify lifecycle behavior; GPU receipts
provide separate model evidence.

A supported claim is: **we integrated a published visual editor, implemented
bounded per-edit FT and retrieval comparisons, and built a reversible review
workbench with preserved measurements of successes and failures.** Broader
performance or superiority claims require the completed final comparison.
