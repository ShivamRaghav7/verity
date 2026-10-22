# Choosing a reusable knowledge-editing foundation

Inspected on 22 October 2026. This is a source and asset audit, **not an executed
reproduction or a new performance report**. No model weights were downloaded.

## Decision

The first visual candidate to reproduce is **VLKEB's MEND editor for BLIP-2**.
Build **EditGuard: a multimodal knowledge-editing and regression-testing
workbench** around it if the small reproduction gate below succeeds.

VLKEB is a research benchmark and implementation, not an existing finished
application. It releases trained editing networks, so we can reuse a published
ML core instead of inventing another editing method. Its
[NeurIPS 2024 paper](https://arxiv.org/abs/2403.07350v3) tests changes to
image-associated knowledge, including transfer to related questions.
[The authors publish the editor checkpoints](https://huggingface.co/HymanH/VLKEB-models).

The fallback, if the actual Colab GPU cannot execute this small reproduction, is
**EasyEdit with ROME on GPT2-XL**. This is a predefined resource fallback, not a
promise that an arbitrary multimodal model can be made to fit through quantization.

The existing VisionPatch application and sealed results remain preserved at the
Git tag `visionpatch-preserved-2026-10-22`, commit
`56111d082889803df5de2571d7c51fb8dbcb7af4`. This scout changes no live inference,
datasets, scoring rules or historical results.

## What makes the proposed project meaningful

The intended users are people evaluating and maintaining open-weight visual
language models. They need to establish whether a correction changed the model,
whether it transfers to another photograph or question, and whether it damaged
other knowledge before releasing an edited checkpoint.

Our question is: **Can we release a particular model correction with evidence
about its intended effects, unintended effects and reversibility?**

A suitable application demonstration is maintaining an image-grounded collection
assistant. A curator corrects the attribution of a pictured object. The assistant
should answer questions about that attribution from another photograph, while
preserving the object's material and facts about other objects. These are example
requirements, not demonstrated capabilities of the inspected checkpoint.

A classifier recognizing an object category does not supply its changing factual
associations. However, visual retrieval plus an updated catalogue could be the
better solution. That must be a fair baseline. We are proposing a maintenance
and evaluation tool, not asserting that editing is always preferable to a database.

An interactive edit, saved weight differences, independently generated answers,
tests on neighboring entities, sequential corrections, approval, rollback and
exported experiment evidence form our application contribution. A new skin over
a notebook alone would not establish those contributions.

## Shortlist and selection evidence

| Foundation | What is available | Decision |
| --- | --- | --- |
| [VLKEB](https://github.com/VLKEB/VLKEB) | MIT implementation and published MEND/SERAC/KE checkpoints; multimodal editing and evaluation | First visual reproduction target: MEND + BLIP-2 |
| [MMKE-Bench](https://github.com/MMKE-Bench-ICLR/MMKE-Bench) | MIT research implementation; entity, visual-semantic and user-specific editing tasks | Useful secondary evaluation reference; not a drop-in VLKEB checkpoint replacement |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | MIT framework, supported model configurations and several genuine editing algorithms | Most practical language fallback: ROME + GPT2-XL |
| [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) | MIT implementation; sequential parameter editing | Its documented requirement is an A40 48 GB GPU; unsuitable as the default laptop/T4 route |
| [UCE](https://github.com/rohitgandikota/unified-concept-editing) | MIT diffusion concept-editing implementation | Different task, with closed-form editing; not the selected factual visual-answering project |

Published memory measurements are useful screening evidence, not measurements
of our implementation. MMKE-Bench reports roughly 21–23 GB for BLIP-2 MEND and
15–17 GB for BLIP-2 SERAC, depending on the task. EasyEdit reports 10 GB for its
default ROME/GPT2-XL configuration. These figures do not establish our runtime,
peak memory or accuracy.

The local inventory found an RTX 4050 Laptop GPU with 6,141 MiB total VRAM and
25,463,480,320 bytes of physical system RAM. Compute capability and memory
capacity are different constraints. Colab is possible, as agreed in the chat,
but its allocated GPU and free memory must be inspected. A 24 GB or larger GPU
is a more plausible first MEND target; even that is not a fit guarantee.

## What the selected ML core actually does

[MEND](https://arxiv.org/abs/2110.11309) learns auxiliary networks that transform
gradients into targeted changes to a model's parameters. It can apply a new edit
using an already trained editor. This is a parameter-changing method, rather than
inserting a correction into every question's prompt.

The inspected VLKEB
[BLIP-2 configuration](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/hparams/MEND/blip2.yaml)
targets `fc1` and `fc2` matrices in OPT decoder layers 29, 30 and 31. Its
[executor](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/easyeditor/models/mend/mend_main.py)
applies learned updates and can retain original matrices for restoration.
Exact restoration, persisted updates and fresh-process loading still need tests.

This changes the original project's architecture: the selected first method is
published MEND, not our original SERAC-inspired pair of trained auxiliary models.
Published weights must be credited as published weights. If we adapt the editor
on a new domain, we must record which parameters we train and report that run
separately from reuse of the authors' checkpoint. Our own ML training remains a
bounded later stage, after an actual profile; it is not completed by this scout.

## Important findings from source inspection

1. **Do not choose VLKEB SERAC for private-item scope without further work.** Its
   [cache and scope-classifier implementation](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/easyeditor/trainer/algs/SERAC.py)
   compares question/correction text. It does not use the support image for scope
   selection. Identical wording about two different items is a material risk.
2. **Benchmark scores are not automatically interactive answer accuracy.** The
   supplied [BLIP-2 module](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/easyeditor/trainer/blip2_models/blip2_opt.py)
   exposes a forward path for evaluating answer logits. A working image-conditioned
   generation adapter must be implemented and tested. We must keep token-level
   scores with supplied reference tokens separate from independently generated
   short-answer correctness.
3. **Do not combine repositories and claim a faithful reproduction.** Testing a
   VLKEB editor on MMKE-Bench or our own collection is a transfer experiment with
   declared adapters, data conventions and scoring rules.
4. **Precision is mixed.** The inspected loader explicitly loads OPT in FP16,
   while the visual encoder is requested in FP32. The `half: false` setting does
   not mean the entire base is FP32. Changing precision or offloading is an
   adaptation that needs validation.
5. **Dependencies need isolation.** Use a separate Linux/Colab environment pinned
   to inspected code. Do not replace the working application's Python environment
   with an old research environment. Inspect the upstream exported environment
   before installing; it is not a universal Colab requirements file.

## Assets and attribution

The exact editor revisions, file sizes and published LFS hashes are recorded in
[the machine-readable asset inventory](../reports/scouting/upstream-assets.json).
It is a partial inventory: visual-backbone, Qformer and tokenizer assets still
need a complete pinning and compatibility check before a download plan is issued.
The MEND checkpoint alone is about 2.36 GB; this is not the total download size
and not a VRAM estimate.

VLKEB code and editor weights declare MIT; its dataset declares BSD-3-Clause.
The [OPT base model](https://huggingface.co/facebook/opt-2.7b) has separate terms;
the [authors' model card](https://arxiv.org/html/2205.01068v4#A4.SS1) describes
noncommercial research restrictions. An MIT editor does not make every
underlying model MIT. Keep model/data notices and individual image-source rights
separate. The application must visibly credit VLKEB, MEND and the base models.
This is an academic reproduction/integration proposal, not a claim of unrestricted
commercial deployment rights.

## Finite next steps

1. **Inventory the actual Colab runtime.** Run
   [the hardware-only notebook](../notebooks/01_colab_hardware_check.ipynb).
   It installs nothing, downloads no weights and performs no training.
2. **Verify the complete dependency and asset set.** Pin the visual encoder,
   Qformer, language model and tokenizer alongside the editor; make a clean
   environment and verify imports. Resolve checkpoint/config mismatches here.
3. **Run one published development edit.** Use a declared development example,
   not final-test examples. Record the base answer, edited answer, a paraphrase,
   a neighboring image/question, real elapsed time and peak host/GPU memory.
   Verify image-conditioned generation, export the edited matrices, reload in a
   fresh process, and restore byte-identical original matrices and behavior.
   One smoke test is not an accuracy estimate. Stop at this finite test; do not
   launch a benchmark or meta-training loop by accident.
4. **Make the go/no-go decision.** If the visual test works, integrate one
   edit/evaluate/approve/restore workflow using the tested core. If the allocated
   GPU cannot support it, exercise the EasyEdit language fallback once. Do not
   begin another speculative editor rewrite to force a preferred result.
5. **Then conduct the ML study.** Define independent data splits, train/adapt a
   bounded editor configuration after profiling, compare updated visual RAG,
   prompt-only updates and matched fine-tuning, freeze configurations and evaluate
   direct answers, new wording/images, locality, conflicts and sequential edits.
   State abstention denominators and report failures. No final-test tuning.

Before any substantial training command, provide its scientific question,
inputs/artifacts, estimates from the measured profile, resume behavior and
finite stopping condition, as required in the original project brief.

Only after the reproduction gate should we spend effort replacing the website.
At this stage the deliverables are the inspected recommendation, pinned partial
asset inventory and executable hardware inventory notebook. No new editor,
application integration, model training or accuracy result is claimed.
