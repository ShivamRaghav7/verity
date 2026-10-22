# VisionPatch selectivity study v2

The first study showed a useful but narrow result: a small CLIP projection update corrected a controlled misleading-word association. It did not establish that editing can distinguish a misleading word from a legitimate request to read that same word. The v2 study addresses that missing test. This document declares the finite protocol; measured outcomes belong in a separate results report.

The research question is: **Can a targeted parameter edit repair misleading labels without erasing legitimate recognition of the corresponding object or word, and what happens when several such edits are applied?** A successful repair needs more than a larger attack score. A patch that improves object recognition by indiscriminately ignoring text may damage the original model's useful capabilities.

This is visual model editing of a learned image/text association. It is not a claim that every visual mistake is a knowledge-editing problem, a reproduction of a published algorithm, or a new state-of-the-art editing method. The model continues to choose among eight fixed candidates, rather than answering arbitrary image questions.

## Fixed problem and model

The model is `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Both encoders and the text projection remain frozen. The editable component is the image projection, a 512-by-768 matrix. The three independently studied misleading words are **camera**, **laptop** and **watch**.

The object candidates remain airplane, camera, cup, elephant, grand piano, laptop, watch and cell phone. This fixed vocabulary includes the genuine object corresponding to each misleading word. A camera photograph with a camera label is not treated as an attack-induced mistake simply because that label is present; its correct object answer remains camera.

Object recognition uses the same frozen text embeddings in every method, from prompts of the form `a photo of a {label}.` Controlled text reading uses a separate fixed candidate set of prompts, `a photo of the word "{label}".` The two tasks explicitly ask about different content. Neither text prompts nor candidate sets are selected on final results.

## Sources and leakage control

Photographs come from the same checksum-pinned Caltech101 archive as v1. Source attribution and license are retained in the manifest. Original category counts in the eight selected classes are:

| Source class | Available originals | Previously observed in v1 |
|---|---:|---:|
| airplanes | 800 | 28 |
| camera | 50 | 28 |
| cup | 57 | 28 |
| elephant | 64 | 28 |
| grand_piano | 99 | 28 |
| laptop | 81 | 28 |
| watch | 239 | 28 |
| cellphone | 59 | 28 |

All **224 previously observed v1 originals**—144 development and 80 final photographs—are excluded from selecting the new v2 validation and final photographs. The old v1 final score cannot become the new study's development score.

We reuse only 80 old photographs for training: 16 support originals and 64 retention originals. Every class contributes two support and eight retention photographs. Reuse is disclosed and confines these previously observed data to fitting.

The new evaluation data use 4 validation and 6 final photographs per class: **32 validation and 48 final originals**. Within each class, selection follows ascending SHA256 after excluding all previous content hashes and source identifiers. No model prediction, difficulty score or patch outcome affects source selection. All variants of a source photograph inherit its split. A photograph may appear under all three misleading words, but those appearances are repeated measurements of the same original.

The preparation code rejects duplicate content or identifiers, v1/evaluation overlap, source-path escape, modified image bytes and attempts to overwrite a nonempty partial preparation. It freezes source hashes, the configuration digest, preceding manifest digests and exact font digests. Existing prepared data can be reused only if those inputs still match.

This is exact-byte and source-identifier separation. Near-duplicate photographs and shared physical objects have not been independently grouped. Source labels are inherited from Caltech101 and receive an automated integrity audit; we make no independent human-review claim. Public-model pretraining exposure to these images is unknown.

## Controlled image conditions

Each photo has a clean version, a white-label blank control and a misleading-word label with exactly the same geometry as that blank. The difference between blank and misleading versions is confined to the printed text pixels. All arms share the same 224-pixel crop and normalization.

| Split | Font | Label position |
|---|---|---|
| Train | DejaVu Sans | bottom |
| Validation | DejaVu Serif | top |
| Final | DejaVu Sans Bold | top and center |

The final font is absent from training and validation. The center placement is absent from both. Two final positions do not create two independent photographs. The three font files are bundled with their license and their SHA256 values are frozen in the manifest.

There is one clean task per source, plus a matched blank and misleading task for each word/condition. That gives 560 rendered training photo tasks, 224 validation tasks and 624 final tasks. These totals describe rendered inputs, not distinct original images. A per-word final attack accuracy uses 48 photographs at two positions, or 96 observations; statistical uncertainty must preserve the 48-source grouping.

## Four correction pairs per edit

| Edit word | Two support classes | Support originals |
|---|---|---:|
| camera | cup, laptop | 4 |
| laptop | camera, cup | 4 |
| watch | cup, laptop | 4 |

For each original, the desired representation is the base model's projection of its matching blank-label version. No validation/final image is used to construct a correction target. This supervision asks the edited projection to reduce the misleading word's effect relative to the same occlusion, rather than relying on test class labels as fitting targets.

The 64 clean retention photos encourage preservation. The protection experiment adds training examples of legitimate text use, described below. These penalties discourage collateral changes; they do not guarantee locality.

## Legitimate text preservation

We construct controlled word cards for all eight candidate labels, including camera, laptop and watch. The correct answer is the literal rendered word. Card background colors and line geometry follow a deterministic template identifier; all eight word variants of one template stay in one split.

There are four training templates, two validation templates and three final templates, giving **32/16/24 cards** respectively. Font and position follow the table above, with final text cards at the top position only. Template identifiers are disjoint before word augmentation, and source-independent rendering is recorded in code.

This tests a legitimate cross-modal capability of the same edited model. It does not constitute natural-scene OCR, real product-label recognition or 24 independent concepts. There are only eight semantic words. Report overall card accuracy and performance by word. Any claim of preservation also reports the subset that the unedited model answered correctly. Preserving an originally wrong answer is not factual correctness.

## Finite comparison and selection

The comparison includes unchanged CLIP, object-only projection editing, text-protected projection editing, matched projection fine-tuning and task-aware privileged masking. That control knows which tasks contain a synthetic misleading overlay and its exact region: it uses the matching blank-label result for misleading object-recognition tasks, and leaves clean, already-blank and legitimate text-reading inputs unchanged. Thus its text score is the unchanged model's text score, not evidence that a masking operation can read deleted words. It is deliberately privileged and provides no evidence of automatic attack detection or intent inference.

The object-only edit is a fixed ablation. Protected editing has exactly two candidates, using text-retention weights 1 and 4; both use covariance ridge 0.001 and fit ridge 0.000001. Fine-tuning has exactly two learning rates, 0.0001 and 0.0005, each capped at 25 updates with the same correction pairs and clean/text protection information. This is **five configurations per word, fifteen bounded independent fits**. The sequential experiment adds 18 closed-form fits, three stages for each of the six possible word orders. The complete development budget is **33 fits: 27 closed-form edits and six 25-update fine-tunes**. No further grid expansion is authorized by an unexpected result.

Select one protected-edit configuration and one fine-tuning learning rate **globally across the three words**, using validation only. The objective is the equally weighted mean of target misleading-label object accuracy, controlled text accuracy and clean object accuracy. Break ties by fewer newly introduced clean object errors, then the original candidate index. Publish every validation candidate, including weaker ones. The object-only ablation remains fixed rather than selected.

Before final inference, seal the selected weights, all candidate settings, scoring rules, code/input fingerprints and final manifest. Final evaluation has no fitting function and cannot silently replace the seal. A poor result leads to a documented limitation, not another round of fitting against the same final images.

## Measures and denominators

For each edited word, report:

- Object accuracy on clean, blank and misleading versions, including all source photos.
- Accuracy by true object class, font and placement; genuine target objects remain visible as a subgroup.
- Repair rate among baseline-wrong misleading examples and new-error rate among baseline-correct examples, with both counts and denominators.
- Clean-object preservation among baseline-correct sources, alongside overall clean-object accuracy.
- Controlled text accuracy across all cards and retention among baseline-correct cards, including per-word results.
- Other-word attack performance, to expose whether a nominally local patch suppresses text more broadly.
- Measured edit time, feature-extraction time and runtime resources; cached projection fitting and full image inference must be distinguished.

The fixed-candidate model does not abstain; every valid input contributes to accuracy. Invalid inputs and runtime errors are counted explicitly rather than removed from successful-response denominators. Similarity scores are not calibrated correctness probabilities.

Paired object comparisons should resample original source photographs, carrying their positions and word variants together. Report uncertainty for differences rather than implying independently sampled observations. Text-card results include descriptive counts and macro averages over eight words. If paired bootstrap intervals are shown for text, their resampling unit is the semantic word: all templates for a word travel together. These are explicitly exploratory intervals over only eight chosen word groups, not a population-level OCR reliability estimate. Do not use the 24 final cards to imply a large independent semantic sample.

Sequential evaluation covers all six predeclared permutations of the three words and reports all three tasks after each state. Each later correction uses its current parent projection's matched blank outputs as targets; these 18 fits use training data only and precede final inference. Orders and stages are descriptive interference analyses, not candidates selected using final scores. Exact rollback is a separate engineering property: restoring the previous projection's bytes and predictions must pass regardless of whether that projection is accurate. Do not add independently fitted patches to an arbitrary active model and call it an exact sequential edit; each patch must be bound to its actual preceding parameter state.

## Preparation and evidence

Preparation performs no model inference:

```powershell
.\.venv-ml\Scripts\python.exe -m visionpatch.study_data
.\.venv-ml\Scripts\python.exe -m unittest tests.test_visionpatch_study_data -v
```

The default output is `data/generated/visionpatch/study-v2/manifest.json`. Photograph bytes and model artifacts remain outside Git. The committed protocol is `configs/visionpatch-study-v2.json`; the preparation and rendering implementation is in `visionpatch/study_data.py` and `visionpatch/study_protocol.py`.

Do not release a long training command from this document. The study runner must separately record the concrete run question, declared inputs/artifacts, a runtime/memory estimate grounded in a small local profile, restart or resume behavior and a finite stopping condition. A short cached linear-algebra solve and a 25-update projection fit can remain small while the evaluation becomes more demanding.

The intended outcome is an honest selectivity study and an auditable correction workflow. A result showing collateral text damage, a tie with fine-tuning or superior privileged masking is a legitimate research outcome. It constrains the claim; it does not disappear from the report.
