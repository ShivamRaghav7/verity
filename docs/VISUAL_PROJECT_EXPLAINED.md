# EditGuard: understand the visual knowledge-editing project

EditGuard is a research workbench: propose a visual association, update model
parameters, inspect consequences, and restore the original state. Reliable
arbitrary edits are not guaranteed. The completed 20-edit comparison exposes
target-success/locality tradeoffs; read its saved measurements and failures in
[VLKEB_RESULTS.md](VLKEB_RESULTS.md).

## 1. What problem does this address?

A vision–language model accepts an image and question and produces words, such
as a city's name. Its answer depends on learned parameters, also called weights.

An image classifier usually chooses among a fixed set of labels. A
vision–language model can answer many different questions about the same image.
Knowledge editing asks whether we can change a particular association while
keeping unrelated answers useful, without retraining the entire foundation model.
Correcting one answer is easy to demonstrate; controlling its side effects is
the difficult research question.

Retrieval may be better for frequently changing information, so we compare it
with editing. Our audience is model maintainers measuring and governing updates.
This academic benchmark does not establish private-object recognition, arbitrary
policy updates or production readiness.

## 2. What is borrowed, and what did we build?

| Component | Meaning and ownership |
| --- | --- |
| BLIP-2 with OPT-2.7B | The pretrained model that interprets images and generates answers; created upstream. |
| VLKEB | The authors' visual editing benchmark, model integration and released editor checkpoints. |
| MEND | A learned network that transforms gradients into parameter updates. Its pretrained editor was trained by the authors, not us. |
| Our reproduction bridge | Pinned assets, audited loading, explicit token alignment, bounded generation, parameter snapshots, export, reload and exact rollback. |
| Our FT baseline | Actual per-edit training of two model matrices for 15 fixed optimizer steps, with disclosed adaptations. |
| Our evaluation/application | Matched comparisons, conservative scoring, uncertainty, provenance, candidate review, history and evidence export. |

Per-edit FT trains on one correction. MEND meta-training trains an editor across
many corrections. We implemented a pretrained editor and per-edit FT updates;
we did not train the released MEND network.

## 3. The development timeline: what, why and how

The current implementation followed these stages on 22–24 October 2026:

1. **Audit research.** Select VLKEB/MEND for genuine visual parameter updates;
   pin code, assets and licenses (`0da0481`).
2. **Isolate dependencies.** Preserve Windows tools; use an isolated Colab T4
   environment. Import checks catch missing OpenCV before model execution
   (`44769db`, `c945561`).
3. **Test one fixed example.** Add target-free generation and saved patches
   (`917802b`). Parameters changed and rolled back, but answers failed; fresh
   reload was killed. Preserve that failure.
4. **Fix concrete problems.** Earlier GPU staging reduces host-memory overlap.
   Explicit contextual labels address the discovered tokenization mismatch
   (`a25a370`). Neither change establishes accuracy.
5. **Check persistence.** Context-aligned MEND passes eight mechanical checks,
   including fresh reload. Generation still misses targets. Diagnose teacher
   forcing separately from generation.
6. **Add bounded FT.** Implement the published last-layer schedule with FP32
   optimizer copies and newline supervision. This is per-edit optimization,
   not editor meta-training (`5699a9a`).
7. **Integrate and evaluate.** Build the workbench, candidate approval safeguards
   and image/question retrieval comparison. Keep the final source-disjoint
   experiment separate from demonstrations.
8. **Validate routing and profile the complete path.** Select the RAG threshold
   on eight fixed training rows, then execute one full five-method development
   episode: 25 queries, one MEND update, 15 FT steps and exact rollback. This
   gives a measured runtime estimate without opening the final split.
9. **Seal and run the final comparison.** Freeze source `460a91c`, selection,
   scorer and configuration before inference. Finish 20 units and 500 answers
   in one 815-second worker run on the T4. Verify recorded arithmetic, source
   identity and restoration, and preserve unfavorable results without tuning.

## 4. What happens inside the model?

BLIP-2's visual encoder converts pixels into numerical representations. Its
Q-Former selects information for the language model. A projection maps that
information into OPT's representation space. OPT then predicts the next token
repeatedly. A token is a piece of text, not necessarily a whole word.

Our fixed question scaffold is `Question: … Answer:`. It helps establish that a
short answer is requested. The correction itself is absent from MEND/FT inference
prompts: it supervises their update beforehand. The prompt-only baseline does
include the correction in the inference prompt, by design.

MEND obtains gradients: signals describing how an answer loss changes with the
weights. Its learned transform predicts an update to six MLP matrices in OPT
layers 29–31. The FT baseline instead optimizes the two layer-31 matrices directly.
All other base weights remain frozen during that fitting.
MEND has no separate applicability classifier here: every query uses the active
weights, making locality testing essential.

### Why token alignment mattered

OPT uses BPE tokenization, which breaks text into pieces. `Albany, New York`
encoded alone had six tokens; after a separating space it had four. Its beginning
could be split as `Al`, `b`, `any` alone but represented as one contextual token.
Using the wrong suffix length accidentally supervised positions inside the
question. That explained the first edited output repeating “any”.

BOS is a special beginning-of-sequence token. Counting question tokens with and
without that token can also shift the answer boundary. Our context variant
tokenizes the whole question-plus-answer input and selects its actual answer
suffix, requiring the question to remain an exact prefix. It refuses an
ambiguous boundary. The released convention remains recorded for comparison.

### Why teacher forcing can look excellent while generation fails

With **teacher forcing**, the complete expected answer is in the input. The model
predicts each next token while seeing the correct earlier tokens. During **free
generation**, it sees only its own previous output. One error can change every
later prediction.

Our MEND diagnosis reached 100% contextual target-token correctness on the raw
direct development question, yet freely generated `Albany, New York, New York…`.
That is not a correct completed short answer. The FT baseline therefore also
supervises a newline after the answer, teaching the generator's stopping token.
This is disclosed as a generative adaptation, not claimed as identical upstream FT.

FT performs exactly 15 AdamW updates at learning rate `0.0002`. FP32 master copies
accumulate optimizer updates for the two selected matrices; their values are
copied back to the model's original dtype. FP32 means 32-bit floating point;
FP16 uses 16 bits and less memory. The whole model is not converted to FP32.
Loss is recorded every step. Decreasing loss is training evidence, not independent
answer accuracy. No KL regularization is claimed.

## 5. What have the development runs actually shown?

The saved context/VQA MEND run passed **8/8 mechanical checks**, including fresh
reload and exact restoration, while its direct, paraphrase and alternate-image
answers did not match the requested target. Its unrelated visual answer stayed
empty, which preserves an already incorrect answer rather than factual success.
See the preserved [run summary](../reports/vlkeb/t4-context-vqa-1/run/smoke.json).

The saved Colab FT development run passed **8/8 mechanical checks**, including
fresh reload and exact restoration. It produced the exact target on the
direct question and another photograph. Its paraphrase answered `Albany` rather
than the full `Albany, New York`, which fails our frozen strict scorer. Its
neighboring answer changed from empty to repetitive city output; it was already
factually wrong before the edit. These are observations on one training-seen
example, not a held-out success rate. Its full [FT receipt](../reports/vlkeb/t4-ft-last-layer-1/run/smoke.json)
records the answers, checks and resource measurements. Neither method has earned
an arbitrary-edit guarantee.

These failures reveal side effects, incomplete answers, and the gap between token
metrics and usable answers. Successful rollback proves reversibility, not that
an update was useful or factually justified.

## 6. How to navigate the website

The new workbench's default local address is `http://127.0.0.1:8003`. It requires
the verified model assets and a suitable CUDA runtime. The preserved older
website at port 8002 is a different implementation. See
[the setup guide](VLKEB_SMOKE.md) for the isolated runtime.

1. **Check model status.** Wait for “Model ready”. A loading or unavailable model
   cannot perform an edit merely because the page itself opens.
2. **Choose visual context.** Select a development image or upload an image.
   Uploads are demonstrations, not automatically new benchmark cases.
3. **Generate the current answer.** Enter a question and inspect the original
   output. Keep the image and wording fixed for a meaningful before/after comparison.
4. **Choose a method and desired short answer.** MEND uses the pretrained learned
   transform; FT runs the fixed 15-step update. This operation proposes model
   behavior; it does not certify that your correction is true.
5. **Preview the candidate.** The candidate workflow generates before/after
   answers and a fixed unrelated probe, saves the patch, then restores the base.
   Read both the target result and the side-effect result. Preview is a separate
   completed UI stage; the candidate remains inactive while it awaits review.
6. **Approve and activate deliberately.** The operator workflow loads the saved
   candidate only after explicit approval. Approval records a research decision;
   it is not a factual-verification badge or a guarantee from one probe.
7. **Inspect history and roll back.** History stores question, target, method,
   source, results and patch identity. Rollback restores the saved original tensors.
8. **Open Evaluation.** It displays our saved measured report, with downloadable
   evidence. Published paper numbers are documented separately, never substituted
   for a missing local experiment.

The candidate, explicit approval, activation, history and rollback lifecycle is
implemented. Its 11 application contract tests use a tiny fake backend to check
state transitions and failure handling; they do not measure model quality.
The completed GPU evaluator is separate evidence from an interactive workbench
rehearsal; that rehearsal needs its own saved receipt.

Only one patch can be active; another activation requires rollback. This prevents
stacking, not semantic contradictions or lifelong interference. Hashes identify
editable matrices' exact bytes. Restore saved originals rather than subtracting
rounded FP16 updates. The decoding cache lasts one answer and must not carry
edited-state context into later queries.

## 7. How to read the completed final experiment

The frozen selection contains **20 distinct source entities**, chosen without
inspecting model success and excluded when the source image directory appears
in the authors' training-image slots. This is one small, nonrandom slice, with
possible target-name, locality-entity and foundation-pretraining overlap. Labels
are author-provided and automatically audited, without independent human review.
They are controlled counterfactual replacements, not newly verified facts.

Five cases accompany each edit: direct question, paraphrase, alternate photograph,
unrelated text question without an image, and similar unrelated visual question.
The five configurations are base, prompt-only, updated image/question RAG, MEND
and FT. **All 100 questions per method, 500 overall, executed.** No execution
errors or unexecuted cases occurred. Wrong generated answers still count as
quality failures.

Each table cell is a strict full-answer match count out of 20 assigned cases.
A count of 15/20 means 75%; one more match changes that category by five
percentage points. Empty, erroneous and abstained answers stay in denominators.

| Method | Direct | Paraphrase | Other photograph | Text locality | Visual locality |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0/20 | 0/20 | 0/20 | 2/20 | 3/20 |
| Prompt only | 19/20 | 15/20 | 20/20 | 1/20 | 2/20 |
| Updated RAG | 19/20 | 15/20 | 20/20 | 2/20 | 2/20 |
| MEND (adapted) | 7/20 | 4/20 | 4/20 | 2/20 | 4/20 |
| FT (adapted) | 20/20 | 20/20 | 20/20 | 0/20 | 0/20 |

![Measured final category accuracy](../output/vlkeb-final-2026-10-24/accuracy.png)

**Read FT's two outcomes together.** It matches all 60 target-edit questions but
none of the 40 unrelated factual targets. It also breaks all five locality
answers the base had originally answered correctly. Fitting the correction can
work while producing an unacceptable maintenance update. There is no single
best-method conclusion without deciding how much target success and collateral
harm matter.

**Read MEND's retention with its denominator.** MEND matches 7/20 direct targets,
4/20 paraphrases and 4/20 other photographs, while retaining 5/5 baseline-correct
locality answers. Those five are only two text and three visual answers. Perfect
retention on five answers is limited evidence, and base locality factual
accuracy was only 5/40. MEND's total locality accuracy is 6/40 because one further
visual answer becomes correct. Preserving an originally wrong answer is not
factual success.

**Read routing separately from answers.** RAG and prompt only both match 19/20
direct, 15/20 paraphrase and 20/20 other-image targets. RAG accepts all 60/60
applicable routes, but falsely applies the correction on 13/40 unrelated cases;
precision is 60/73 (82.2%). The route can be correct while generation still fails.
RAG rejects text-only use of its visual correction and preserves both correct
text-locality answers. Cosine scores are not probabilities. This one-correction
memory says nothing about retrieval at large memory sizes.

The base's 0/60 is measured against counterfactual replacement targets. It does
not state that the base knows none of the original associations. Some factual
reference strings are long relative to the fixed 24-token limit, and strict
matching may reject reasonable shorter answers. Keep those scoring limits
visible rather than changing the rules after observing the test.

Abstentions total 4/100 for base, 0/100 prompt only, 1/100 RAG, 2/100 MEND and
0/100 FT, all included as incorrect. Strict failures total 95, 43, 42, 79 and
40 respectively. Zero execution errors therefore does not mean zero failures.

### Differences, uncertainty and speed

A **paired difference** subtracts two methods' scores on the same edit units.
For example, RAG minus MEND direct accuracy is +60 percentage points, with a
saved 95% paired bootstrap interval of +35 to +85. FT minus prompt-only paraphrase
accuracy is +25 points, interval +10 to +45. MEND minus FT visual locality is
+20 points, interval +5 to +40. The bootstrap resamples the 20 complete edit
units 10,000 times with seed 42, retaining their linked cases. It communicates
sample uncertainty, not freedom from dataset bias or proof about unseen edits;
these descriptive comparisons have no multiple-comparison correction.

The full worker took **815.078 seconds (13.6 minutes)**, including the shared
**99.276-second model load**. Edit application p50/p95 was **3.553/3.645 seconds
for MEND** and **10.242/10.427 seconds for FT**, with 20 measured edits each.
RAG reference-index construction was 0.265/0.298 seconds, separately measured.
For direct queries, p50/p95 was 0.460/0.911 seconds for base, 0.387/0.479 for
prompt only, 0.404/0.501 for RAG, 0.428/0.564 for MEND and 0.486/0.543 for FT.
Every method/category has 20 query measurements; the
[results report](VLKEB_RESULTS.md) lists all categories without pooling them.

p50 is the median duration; p95 is the interpolated 95th percentile of recorded
calls, not a response-time promise. CUDA was synchronized around each call, with
zero excluded warmups; loading and editing are separate from query latency.
RAG query latency includes retrieval and gating. Host RSS peaked at 7.867 GiB,
CUDA allocated at 11.152 GiB and CUDA reserved at 11.711 GiB. RSS samples can
miss short peaks, allocator figures omit some driver memory, and 1 GiB is
1,073,741,824 bytes. Serial measurements do not establish concurrent serving.

### Why the comparison needs qualifications

MEND and FT receive the same correction image/question/answer, but their training
objectives differ: FT supervises a stopping newline and MEND does not. FT uses
FP32 master copies for two matrices while retaining the base precision. The
prompt-only architecture does not accept a second reference image; RAG receives
reference-image features in its index. These declared asymmetries limit claims
about the isolated editing algorithm. The released MEND editor was trained
upstream, while our FT performs 300 total per-edit optimizer steps across this
run. We did not meta-train an editor or tune on these final outputs.

The saved [final receipt](../reports/vlkeb/t4-eval20-1/artifacts/vlkeb/eval20/evaluation.json)
and [host verification](../reports/vlkeb/t4-eval20-1/host-verification.json) retain
all outcomes and recorded restoration checks. Byte/hash and scoring verification
do not certify image meaning or label truth. The
[results report](VLKEB_RESULTS.md) also shows actual failure examples, including
an unrelated `Jurassic Park` answer changing to the active target `Hudson Hawk`
under FT. These examples explain the completed test; they are not fresh
confirmatory evidence or a reason to tune the completed experiment.

## 8. What to demonstrate and say to the panel

Show the original answer, preview an update, inspect a paraphrase and an unrelated
probe, explain the approval decision, activate an acceptable candidate, then
roll it back and regenerate. Show a failed candidate too. Explain that governance
helps a maintainer reject harmful updates rather than making every update correct.

A defensible claim is: **“We built an auditable visual model-editing workbench,
integrated a published learned editor, implemented per-edit FT and retrieval
comparisons, and measured generated-answer quality and side effects under a fixed
protocol.”** Explain that FT achieved 60/60 edited-target matches alongside 0/40
locality matches, and that MEND's 5/5 correct-answer retention has a very small
denominator. Author-reported reference numbers remain attributed in
[VLKEB_UPSTREAM_RESULTS.md](VLKEB_UPSTREAM_RESULTS.md).

The final comparison and preserved measurements are complete. Report the
application's interactive GPU rehearsal separately when its evidence exists.
The contribution is controlled, recoverable model maintenance with measured
tradeoffs; every visual association is not guaranteed editable.
