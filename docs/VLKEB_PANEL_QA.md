# EditGuard: panel rehearsal notes

Lead with the measured tradeoff: **we completed a controlled visual editing
experiment; fitting the requested answer did not guarantee preserving unrelated
answers.** The software executed all 500 assigned answer records with zero
execution errors. Incorrect answers and collateral changes are research outcomes.

Keep the [results and graphs](VLKEB_RESULTS.md),
[original evaluation JSON](../reports/vlkeb/t4-eval20-1/artifacts/vlkeb/eval20/evaluation.json)
and [verification receipt](../reports/vlkeb/t4-eval20-1/host-verification.json)
available during the presentation. The receipt verifies recorded identities,
arithmetic and restoration; it does not independently certify factual truth.

## Opening talk: approximately six minutes

| Time | What to explain |
| --- | --- |
| 0:00–0:45 | **Problem.** “A vision–language model can associate an image with the wrong answer. I investigated whether a targeted update can change that association while preserving unrelated answers. The difficult part is controlling side effects.” |
| 0:45–1:35 | **System and ownership.** Explain image encoder → Q-Former → OPT answer generator. “We reused BLIP-2/OPT-2.7B and the authors' pretrained MEND editor. Our work is the integration, documented adaptations, controlled comparisons, patch lifecycle and evidence.” |
| 1:35–2:20 | **Experiment.** Show the five methods: base, prompt only, image/text RAG, adapted MEND and adapted FT. Explain 20 independent edit episodes, five cases per method and 500 answers. Each episode starts from the original weights. |
| 2:20–3:40 | **Main result.** Show the accuracy graph. “FT matched all 60 edited targets, but scored 0/40 on unrelated questions and lost all five originally correct locality answers. MEND matched fewer targets. RAG matched 54/60 targets but falsely applied corrections to 13 unrelated cases.” Give the denominators before interpreting percentages. |
| 3:40–4:30 | **Interpretation and limits.** “Only five of the 40 unrelated baseline answers were correct. MEND retaining those five is encouraging within this slice, but is not broad locality evidence. These are counterfactual benchmark targets, a small nonrandom sample and automated scoring.” |
| 4:30–5:30 | **Application.** Explain generate → preview candidate → inspect side effects → explicit approval → activate → history/export → rollback. If the GPU UI is ready, demonstrate the fixed development example; otherwise show saved evidence and state that the temporary runtime is unavailable. Distinguish a completed evaluation from a live UI rehearsal. |
| 5:30–6:10 | **Contribution and next step.** “The contribution is an auditable maintenance workflow and a reproducible comparison that exposes failure modes. The next study should improve locality and use stronger baseline behavior and independent label review. We preserved the sealed results rather than changing them after seeing failures.” |

## Twelve likely questions

### 1. What is this project, and why is it more than a CNN classifier?

It is a workbench for proposing, applying, inspecting and reversing targeted
changes to a pretrained vision–language model. A fixed-label classifier predicts
one of its predefined classes. Here the model generates answers to different
questions about an image. We did not build a new vision architecture or train a
foundation model from scratch. The [architecture explanation](VISUAL_PROJECT_EXPLAINED.md)
describes the reused image encoder, Q-Former and language model.

### 2. Why edit weights when RAG or a prompt might solve the problem?

That is a comparison question, not an assumption that editing must win. Editing
puts the update in model parameters; prompt-only and RAG supply a correction at
inference time. RAG can be a practical choice for changing information. In our
experiment it matched 54/60 edited targets, equal to prompt-only, and outperformed
adapted MEND on those targets. It also falsely applied the correction in 13/40
unrelated cases. Our RAG holds one current correction per episode; we have not
established large-memory scalability or a general advantage for editing.

### 3. What did you contribute if MEND already exists?

Our contribution is the engineering and experimental platform: pinned assets,
audited loading, target-free bounded generation, explicit answer-token alignment,
matched baselines, candidate approval, patch persistence, exact rollback, history,
evidence export and a sealed comparison. We diagnosed concrete tokenization and
termination failures and documented our adaptations. We do not claim to have
invented MEND, created VLKEB or demonstrated a new state-of-the-art editing
algorithm. See [the workbench guide](VLKEB_WORKBENCH.md).

### 4. What did you actually train?

The base model and learned MEND editor came pretrained from the authors. We did
not meta-train MEND or train a retriever. MEND computes a gradient and uses its
released learned transform to update six OPT matrices. Our adapted FT baseline
does perform training: 15 AdamW steps per correction on two layer-31 matrices,
at learning rate 0.0002. The final run used 20 independent FT edits, totaling
300 optimizer steps, restoring the base between edits. This was not one
300-step model trained sequentially on all 20 corrections.

### 5. What are your actual final results?

Each target-category entry below has 20 cases. “Unrelated” combines 20 text and
20 visual locality questions. All values come from the
[completed report](../reports/vlkeb/t4-eval20-1/artifacts/vlkeb/eval20/evaluation.json).

| Method | Direct target | Paraphrase target | Other photograph target | Unrelated factual matches |
| --- | ---: | ---: | ---: | ---: |
| Base | 0/20 | 0/20 | 0/20 | 5/40 |
| Prompt only | 19/20 | 15/20 | 20/20 | 3/40 |
| Updated RAG | 19/20 | 15/20 | 20/20 | 4/40 |
| Adapted MEND | 7/20 | 4/20 | 4/20 | 6/40 |
| Adapted FT | 20/20 | 20/20 | 20/20 | 0/40 |

Thus MEND's target rates were 35%, 20% and 20%; RAG's were 95%, 75% and 100%.
The base's 0/60 target matches concern requested **counterfactual replacements**,
not its accuracy on the original facts. We report the categories separately
because one overall percentage would hide the target/locality tradeoff.

### 6. Does FT achieving 100% mean the project solved editing?

It achieved 60/60 target matches on this slice, while answering 0/40 unrelated
questions correctly. Crucially, all five previously correct locality answers
were lost. The other 35 were already wrong, so we cannot say FT destroyed
40 correct facts. This is strong target fitting with observed collateral damage,
not reliable general editing. Future work should study locality constraints and
stronger baseline behavior in a separately frozen experiment. The completed
software exposes this negative outcome; the result is not an unfinished metric.

### 7. Can you claim MEND preserves unrelated knowledge perfectly?

Only with the denominator attached: it retained **5/5 baseline-correct locality
answers**, comprising two text answers and three visual answers. Its total
locality accuracy was 6/40, including one newly correct visual answer. The weak
5/40 baseline leaves little evidence for broad factual preservation. RAG retained
4/5, prompt-only 2/5 and FT 0/5. Retaining an originally wrong answer would not
count as factual success. [The locality table](VLKEB_RESULTS.md#locality-correctness-and-preservation)
also separates normalized factual retention from verbatim preservation.

### 8. The paper reports about 98%. Why is your MEND result lower?

The authors' [Table 2](https://arxiv.org/html/2403.07350v3#S4.T2) reports 98.52%
BLIP2-OPT MEND reliability under their evaluation. That is their result, not ours.
The released implementation scores target-token predictions with the expected
answer included in the input. We generate from image/question alone and score
the whole continuation, capped at 24 tokens; repetition or an incomplete answer
fails. Our sample and adaptations also differ, so these are not directly
comparable benchmark scores and do not refute the paper.

We use contextual target labels to fix the standalone/joined token-boundary
mismatch diagnosed during development. FT additionally supervises a newline
to teach answer termination and uses FP32 optimizer master copies while retaining
released model precision and evaluation mode. These changes are disclosed; this
is not a faithful reproduction of the full upstream recipe. See
[published results versus our measurements](VLKEB_UPSTREAM_RESULTS.md).

### 9. How did you control leakage and cherry-picking?

Development used fixed training examples. RAG's fixed cosine threshold was
selected on training rows 2–9 before the final run. We selected 20 distinct source
entities absent from every upstream training image slot, using a deterministic
hash order without filtering by model success. Source, selection, configuration,
threshold and scorer were sealed before inference; no tuning followed the final
outputs. The [protocol](VLKEB_EVALUATION_PROTOCOL.md) records those rules.

This establishes bounded source-entity separation from the pinned annotations.
It does not prove absence from foundation-model pretraining, unseen target names,
or complete isolation of the released MEND checkpoint's original training data.
The slice is small and nonrandom. The retrieval weight was fixed, and its scope
score is not a calibrated probability.

### 10. Who verified the answers, and what do the 500 records prove?

Twenty edits × five cases × five methods gives 500 answer records. Cases test the
direct question, paraphrase, another photograph, unrelated text with no image,
and unrelated visual question. We compare the whole generated answer with author
labels using fixed Unicode/case/punctuation/whitespace normalization. There was
automated structural and arithmetic review, **no independent human label review**
and no LLM judge. Counterfactual targets are controlled labels, not certified
real-world corrections.

Abstentions and errors remain failures in their assigned denominators. Execution
errors were zero; quality failures were not. RAG's 60 true-positive routes and
zero false negatives measure applicability recall, not answer accuracy: it still
missed six edited targets and produced 13 false applications. Paired bootstrap
intervals use 20 edit units, not 500 independent observations; small-sample and
selection limits remain.

### 11. How do you know rollback really works?

We snapshot the original tensors and compare hashes of dtype, shape and exact
bytes. MEND may update six matrices in OPT layers 29–31; FT updates two, while
the backend still checks all six. We restore saved originals instead of
subtracting rounded floating-point updates. Every completed final episode and
both edit methods recorded exact restoration. Separate development runs also
tested saving a patch, fresh-process reload and rollback. This proves recorded
reversibility for the tested configuration, not semantic correctness or safety
of arbitrary edits. Only one patch can be active at a time.

### 12. Where is it deployed, and can the panel run it on your laptop?

The measured GPU path is a private Colab Tesla T4 runtime, with the browser UI
shown through Colab's signed-in port viewer. The final worker took about
815 seconds, including about 99 seconds loading, with roughly 11.15 GiB peak
CUDA allocation. These measurements do not establish operation on a 6 GB laptop
GPU. A Colab session can expire; it is not permanent public hosting.

The workbench implements preview, approval, activation, history, export and
rollback. A successful batch evaluation is separate from evidence of a live UI
rehearsal. Before presenting, confirm the current model-ready status and saved
interactive receipt. If the runtime expires, show the preserved report and
recorded evidence, and describe the interruption plainly. Container build/import
checks establish packaging; they do not substitute for a full GPU deployment.

## Defensible resume claim

> Built an auditable visual model editing workbench around BLIP-2/OPT-2.7B and
> published MEND, integrating bounded per-edit fine-tuning, image/text retrieval
> baselines, patch approval, persistence and exact rollback; completed a sealed
> five-method Tesla T4 comparison across 20 edits and 500 answer records to
> measure target-success and locality tradeoffs.

If asked for a number, pair **60/60 FT target matches** with **0/40 locality
matches**. Avoid “98% accurate,” “trained MEND,” “100% reliable editing,”
“human-verified facts,” or “permanently deployed” as descriptions of this work.

Primary references: [VLKEB paper](https://arxiv.org/abs/2403.07350),
[pinned upstream implementation](https://github.com/VLKEB/VLKEB/tree/10951b7b3788928f578b73f07eac9e1eaa0316f3)
and [pinned authors' dataset](https://huggingface.co/datasets/HymanH/VLKEB-data/tree/cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a).
