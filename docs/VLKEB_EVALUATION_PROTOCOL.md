# Fixed small VLKEB evaluation protocol

Status: **prepared protocol; evaluation has not run**. Freeze the evaluator,
scorer, configuration, this protocol and prepared selection before reading model
outputs. The development smoke remains separate. This protocol supplies our own
small measured experiment; author-reported tables must retain their attribution
and cannot be presented as results from this experiment.

## Source and split audit

Use [the authors' dataset](https://huggingface.co/datasets/HymanH/VLKEB-data/tree/cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a)
at revision `cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a`, with the
[pinned VLKEB implementation](https://github.com/VLKEB/VLKEB/tree/10951b7b3788928f578b73f07eac9e1eaa0316f3).
The [paper](https://arxiv.org/abs/2403.07350) describes 5,000 training and
3,174 evaluation edits. The release provides:

| File | Exact bytes | SHA256 or purpose |
|---|---:|---|
| `train.json` | 2,616,367 | `6e40e0e9a28dc3e7523f8d48f032755fe050c5b5728344633f13113dbd6fbc01` |
| `eval.json` | 1,658,466 | `72c47efe00773bb16cd83258aff394a0894b8c29a6b02e6250717a30d1b18f13` |
| `eval_multihop.json` | 6,816,678 | Multi-hop data; excluded from this finite experiment |
| `eval_edit_onehop.json` | 1,033,271 | Different edit-one-hop experiment; excluded |

These are immutable-revision bytes inspected from the primary release; no model
outputs were inspected for selection. The ordinary eval file has 11 scalar
fields: `src`, `rephrase`, `pred`, `alt`, `image`, `image_rephrase`, `loc`,
`loc_ans`, `m_loc`, `m_loc_q`, `m_loc_a`. It does not contain portability labels.

Do not call the authors' raw split entity-held-out. A structural audit found
4,142 distinct training source image-directory IDs, 2,847 evaluation source IDs,
and **1,173 overlapping source IDs**. Including training edit, alternate and
locality image slots gives 6,440 training visual entity IDs. These counts concern
annotation paths, not the content of the pretrained foundation model.

The model card identifies released MEND weights but does not certify the exact
checkpoint's training example hashes or isolation from validation selection.
The paper and code establish intended training/evaluation roles; they do not
prove the entire released checkpoint's provenance. Foundation-model pretraining
may also contain these public entities and photographs.

## Frozen selection: exactly 20 independent edits

`scripts/prepare_vlkeb_eval.py` implements this fixed rule, without loading a
model:

1. Verify pinned train bytes and the already downloaded full image archive.
2. Fetch and verify only the 1.66 MB pinned `eval.json` annotation.
3. Exclude an eval edit when its source image-directory entity occurs in **any**
   training `image`, `image_rephrase` or `m_loc` directory.
4. Group remaining rows into edit lineages by `[source_entity, pred, alt]`.
   The stable edit ID is SHA256 of compact UTF-8 JSON for that list.
5. Sort edit IDs lexicographically, take the first lineage for each distinct
   source entity, and stop at exactly 20. Never select by answer success.
6. Select the lexicographically smallest canonical JSON row within each lineage
   before inference. Record every row index belonging to that lineage, so its
   paraphrases and image variants cannot be reused as development examples.
7. Extract only the three declared images per selected representative row,
   with the existing safe streaming copier; record source archive and image hashes.

Source image-directory entity disjointness is a bounded claim: target names and
locality entities can overlap the authors' training set. No unseen-target-name or
private-item recognition claim follows. The first sorted 20 are a small
nonrandom slice, with possible selection bias from the ordering rule and
canonical representative selection. Report that limitation.

The default command has no network or writes:

```powershell
python scripts/prepare_vlkeb_eval.py
```

Explicit preparation, after the existing train/archive assets are available:

```bash
python scripts/prepare_vlkeb_eval.py --root artifacts/vlkeb/assets --prepare
```

It refuses to overwrite an existing selection. It cannot execute evaluation or
training. Its default output is `artifacts/vlkeb/assets/eval20-selection.json`;
`--output` can choose a fresh destination. The fixture preserves original row
fields under `edits[].record`, with paths relative to `assets/images/`, lineage
IDs, source row indices, source hashes and image integrity records. Commit the
configuration and selection checksum before running the evaluator. Keep test
questions and expected answers out of logs shown during development.

## Five fixed cases per edit

| Case | Input | Expected answer | Purpose |
|---|---|---|---|
| Direct | `image` + `src` | `alt` | Requested correction |
| New wording | `image` + `rephrase` | `alt` | Author-provided paraphrase generalization |
| Alternate photograph | `image_rephrase` + `src` | `alt` | Generalization across photographs |
| Text locality | **No image** + `loc` | `loc_ans` | Unrelated language knowledge |
| Visual locality | `m_loc` + `m_loc_q` | `m_loc_a` | Similar but unrelated visual knowledge |

This is 100 scored question instances per method, **500 across five methods**. The paraphrase is the
authors' held-out wording, not independently authored by our team. Text locality
requires a target-free text-only OPT generation path; do not insert an arbitrary
image as a workaround. That path must pass a focused test before the experiment.

## Five configurations and matched settings

Use the same released base weights, image preprocessing, 24-token maximum,
greedy decoding, fixed VQA template `Question: {question} Answer:`, seeds and
whole-answer scorer in all configurations. Record all exact source/configuration
hashes. Each edit begins from the base, with exactly one correction active.

| Configuration ID | Behavior |
|---|---|
| `base` | Unchanged base model; diagnostic reference |
| `prompt_only` | Unchanged weights, correction provided as one explicit QA demonstration in the input prefix |
| `updated_rag` | Unchanged weights, current approved correction retrieved using text and reference-image features; validation-selected scope threshold |
| `mend_context` | Released learned MEND transform, one update using explicitly documented context-aligned target suffix labels |
| `ft_last_layer` | Bounded 15-step AdamW fine-tuning of OPT layer 31 `fc1.weight` and `fc2.weight`, learning rate 0.0002, context-aligned targets and newline termination supervision |

The FT schedule and parameters come from the
[published FT configuration](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/hparams/FT/blip2.yaml).
Our resource adaptation keeps released base precision and evaluation mode, using
FP32 AdamW master weights for only those two editable matrices. This is **not**
a faithful full-FP32 reproduction of the authors' FT experiment. It adds no
unmeasured KL or norm constraint. Record exact implementation and metadata in
the seal. FT performs per-edit optimization; the MEND editor remains pretrained.

MEND and FT each begin from exactly the same original base. Restore and
hash-check all declared editable matrices after MEND before starting FT, then
restore and check again after FT. Halt subsequent methods/edits on restoration
failure. Both receive the same edit image, question and target, and generate
without the target in their inference prompt.

FT explicitly supervises a newline after the approved target so the target-free
decoder can learn where its short answer should stop. The scored desired answer
remains the original `alt` value. This is a declared generation adaptation beyond
the authors' target-only objective, fixed before the FT development profile and
final evaluation. MEND retains its context-aligned target-only objective.
Both methods receive the same approved answer information, but their objectives
differ; disclose that difference when interpreting comparisons. Do not present
these generated-answer scores as reproduction of published teacher-forced MEND
accuracy or as an isolated test of editor architecture with identical objectives.

Freeze the exact prompt-only template before test inference. A simple declared
template is `Question: {edit_src} Answer: {alt}\nQuestion: {query} Answer:`.
Provide the correction to **all** query types, including locality queries; do
not use expected routing or test answers to suppress it. The actual queried
image is identical across methods. This simple single-image architecture does
not provide a second reference image in the prompt baseline, while MEND consumes
the correction image during the parameter update. Disclose that representational
asymmetry in the prompt-only comparison. The separately measured updated visual
RAG baseline receives the correction's reference image in its retrieval index.
No pretrained-model or MEND meta-training is claimed as our work.

Updated RAG stores exactly the current approved correction in each episode,
including its question, target and reference-image features. The correction is
updated before queries; it is never deliberately stale. Scope score is fixed as
`0.5 * text_cosine + 0.5 * visual_cosine`. Text embeddings use the official
[all-MiniLM-L6-v2 model card's mean-pooling recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41),
with attention-mask pooling, L2 normalization, maximum length 256, CPU evaluation
mode and no gradient. Visual features are the normalized mean of the first 32
Q-Former-to-OPT prefix vectors with the fixed `Image:` prefix; they contain no
query-answer target. The reference and current query image use identical
preprocessing. Text-only queries are outside this declared visual-edit memory
scope and deterministically use the base model.

Select the threshold only on fixed upstream **train rows 2 through 9**, with
three positives and two negatives per row: direct, paraphrase, alternate image,
visual locality and text locality. Maximize F1 over observed visual score cutoffs
plus a reject-all cutoff; ties prefer higher precision, then higher threshold.
These 40 cases are our threshold-validation set, not test results. They are
small, nonrandom, may contain repeated entities, and may have been seen in the
authors' meta-training. The fixed 0.5 weight is not tuned. Cosine scope scores
are **not calibrated probabilities**. Pin MiniLM to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and preserve its separate model manifest,
file hashes and calibration receipt; do not alter the base-asset manifest.

When scope accepts, use exactly the prompt-only QA demonstration template;
otherwise use the same VQA base prompt. Always generate with the actual query
image, or no image for text locality. Include retrieval and gating in per-query
latency; report reference-index construction separately. This is a one-edit
memory comparison, not evidence of large-memory retrieval scalability. Claims
about superiority must follow these measured comparisons.

Context alignment repairs the standalone/joined BPE-label mismatch diagnosed on
the fixed development example. It is an architectural adaptation, so these
results cannot be described as faithful reproduction of the paper's complete
benchmark. Keep teacher-forced token scores separate from independently
generated answers. The final scorer consumes only generated continuation text.

## Scoring, denominators and uncertainty

Primary scores are conservative normalized **whole-answer** exact matches:
Unicode NFKC, case folding, punctuation/underscore-to-space and whitespace
normalization. Do not count a target mentioned in rejected or incidental prose.
No test-derived alias additions, semantic judge tuning or post-hoc template
changes. Broader semantic review, if conducted, is a separate descriptive audit
that preserves the primary fixed score and discloses that it is automated.

Report direct, paraphrase and photograph target accuracy separately, each over
20 edits. Report text and visual locality factual accuracy separately over 20
each. Also report preservation among **baseline-correct** locality answers, with
that denominator displayed, and unconditional answer preservation separately.
Preserving a wrong baseline answer is not factual correctness.

Report updated-RAG applicability TP/FN over 60 positive cases, FP/TN over 40
out-of-scope cases, recall, precision and false-application rate. Display routing
errors/unexecuted counts. Base fallback is still an answer and stays in factual
accuracy denominators; a rejected correction is not automatically a factual
success. Report all 100 routing opportunities, including missing predictions.

Empty answers, abstentions, errors, timeouts and failed edits remain in the
denominators as primary failures. Report their counts explicitly. If a run is
interrupted, resume the same frozen selection/configuration and retain previous
rows; never drop difficult examples. If a mechanical problem requires code or
configuration changes after viewing test outputs, close this run as exploratory
and do not reuse it for a fresh confirmatory claim.

For paired method differences, bootstrap complete edit lineages (20 units), not
individual correlated questions, with a fixed seed and 10,000 resamples; report
95% intervals and raw counts. With n=20, intervals are wide and an isolated
successful example is not an accuracy estimate. Latency p50/p95 must disclose
their sample counts, warm-up rule, synchronization and whether loading/editing
time is included. Separate editing latency, per-query latency and load time.

## Finite run gate, resources and stopping rule

The experiment may start only after the fixed development example demonstrates
loading, one context-aligned edit, persistence, fresh-process reload and exact
rollback, and after the five-case evaluator and scorer pass their focused tests.
Prepare a concrete command/run card with measured runtime and memory estimates
from that development configuration. **Those estimates are currently unknown**;
the first release/raw attempt's 128-second worker profile does not establish the
new configuration's runtime. No long meta-training is required by this protocol.
The FT configuration needs its own finite development profile before releasing
the command/runtime estimate. Distinguish per-edit FT from training the editor.

Stop after **20 edit lineages and five cases for each of five configurations**:
at most 20 MEND updates, 20 fine-tuned edits and 300 FT optimizer steps per complete
attempt. Disclose interrupted incomplete-edit retries in attempt records.
Do not describe the FT comparison as no training: it performs bounded per-edit
optimization, while editor meta-training is not run.
Restore original editable tensors and confirm hashes between edits. Record
restore failures and halt further edits rather than contaminating subsequent
cases. A finite wall-clock cap must be fixed in the run card from profiling.
Do not expand the sample, tune additional models or change the project because
results disappoint. Report successes and failures honestly.

Required outputs: frozen selection/configuration/scorer hashes, per-edit
generated answers, expected labels, strict score, error/abstention status,
timings, update/rollback hashes, resource summary, aggregate counts and paired
intervals. Report schema must clearly distinguish `own_measured`,
`author_reported` and `not_executed` provenance. No current placeholder is an
obtained metric.

Labels use author review plus our automated structural audit. **No independent
human review by our team is claimed.** They are controlled counterfactual
associations, not newly established real-world facts.
