# EditGuard: understand the visual knowledge-editing project

EditGuard is a research workbench: propose a visual association, update model
parameters, inspect consequences, and restore the original state. Reliable
arbitrary edits are not guaranteed. Read saved development measurements and the
final experiment's current status in [VLKEB_RESULTS.md](VLKEB_RESULTS.md).

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

Only one patch can be active; another activation requires rollback. This prevents
stacking, not semantic contradictions or lifelong interference. Hashes identify
editable matrices' exact bytes. Restore saved originals rather than subtracting
rounded FP16 updates. The decoding cache lasts one answer and must not carry
edited-state context into later queries.

## 7. How to read the final experiment

The planned comparison selects **20 distinct source entities** without inspecting
model success, excluding source directories in the authors' training-image slots.
Foundation-model pretraining and target-name overlap remain possible. Labels are
author-provided and automatically audited, without independent human review.

Five cases accompany each edit: direct question, paraphrase, alternate photograph,
unrelated text question without an image, and similar unrelated visual question.
The configured five methods are base, prompt-only, image/question retrieval memory,
MEND and FT. That would be **100 questions per method, 500 overall**. The retrieval
baseline is implemented and has no held-out measured result yet. A fair comparison
gives each updated method the same correction and available supporting information.
Retrieval combines image and MiniLM question embeddings to select correction
memory, then supplies that information to the unchanged model.

| Number | How to explain it |
| --- | --- |
| Strict accuracy | Correct complete generated answers divided by all assigned questions. `15/20 = 75%`. Missing, erroneous and abstained answers stay in the denominator. |
| Abstention | An empty answer or explicit nonanswer, handled by the fixed protocol rather than silently discarded. |
| Generality | Correctness on wording or photographs not used to fit that edit. It is different from direct edit success. |
| Locality accuracy | Whether unrelated questions receive their factual target answer. |
| Preservation/retention | Whether baseline-correct unrelated behavior remains correct or unchanged. Keeping an originally wrong answer is not factual success. |
| Paired difference | Method A minus method B on the same edit units. This controls which cases each method received. |
| 95% paired bootstrap interval | Repeatedly resample the 20 edit units, keeping their linked questions together. The interval communicates sampling uncertainty; it does not remove dataset bias. |
| p50 / p95 latency | The median duration and the duration below which approximately 95% of recorded calls fall. Query, edit and loading times are separate. |
| RSS | Host-process memory sampled during execution; short peaks can be missed. |
| CUDA allocated / reserved | Memory used by live tensors versus memory retained by PyTorch's allocator; neither is the entire machine's GPU usage. |
| GiB | `1,073,741,824` bytes. It differs from decimal GB. |

Scoring checks the endorsed complete answer, not a mention inside repetition,
examples or rejection. Strict matching can reject reasonable shorter answers;
disclose this instead of changing rules after viewing results. Configurations
and scoring are sealed before evaluation; demonstrations cannot replace it.

## 8. What to demonstrate and say to the panel

Show the original answer, preview an update, inspect a paraphrase and an unrelated
probe, explain the approval decision, activate an acceptable candidate, then
roll it back and regenerate. Show a failed candidate too. Explain that governance
helps a maintainer reject harmful updates rather than making every update correct.

A defensible claim is: **“We built an auditable visual model-editing workbench,
integrated a published learned editor, implemented per-edit FT and retrieval
comparisons, and measured generated-answer quality and side effects under a fixed
protocol.”** Add completed methods and numerical results only when their saved
evidence exists. Author-reported reference numbers remain attributed in
[VLKEB_UPSTREAM_RESULTS.md](VLKEB_UPSTREAM_RESULTS.md).

Completion requires finished comparisons, preserved measurements, a tested
operator workflow and an honest report. The contribution is controlled,
recoverable model maintenance; every visual association is not guaranteed editable.
