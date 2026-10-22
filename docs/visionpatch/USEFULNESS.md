# What VisionPatch currently justifies

## The honest status

The editing workbench and controlled V2 experiment are implemented. A convincing
operational need for this particular editor has **not** been established. Those
are different completion claims. The supported description is a reproducible
study of selective visual model repair, preservation and edit interference.

A personal airplane photograph that remains correctly classified despite a
camera caption is a successful baseline example. It does not need a correction.
We should retain such examples as preservation tests, rather than increase the
caption or choose another image until the demonstration fails. Susceptibility
varies across photographs, typography, candidate labels and models; we have not
diagnosed the user's particular photograph from a verbal description.

The current backbone is CLIP ViT-B/32, a pretrained image/text model with a vision
transformer. It is not an evaluation of the best available conventional object
classifier or current multimodal systems. A conventional classifier could be an
effective choice for a fixed object-category task. Our implementation cannot
establish that editing is necessary without measuring those alternatives.

## Is the underlying problem real?

Yes: typographic attacks on multimodal recognition are documented independently.
The original CLIP analysis demonstrates the phenomenon. The SCAM study evaluates
1,162 real-world attack images across many categories and finds that vulnerability
depends strongly on model architecture and training. The authors also report
substantially stronger resistance for some models. That supports investigating
the failure, and supports comparing stronger alternatives before claiming a
practical advantage for our editor.

Sources: [original CLIP analysis](https://openai.com/index/multimodal-neurons/),
[SCAM paper, revised March 2026](https://arxiv.org/abs/2504.04893),
[authors' evaluation code and results](https://github.com/Bliss-e-V/SCAM).
Their results use their own protocol and must not be presented as direct numerical
comparisons with our eight-category benchmark.

This project studies a recognition failure. It does not implement general visual
prompt-injection defense, reliably update arbitrary visual facts, or demonstrate
prevention of real operational incidents. Making those claims would require
different evidence.

## What our experiment establishes

The frozen V2 study uses 48 final source photographs, eight object categories,
three misleading words, and two held-out overlay positions. Protected editing
improved target camera accuracy from 55/96 to 91/96 and laptop from 31/96 to 77/96.
It preserved the tested clean and controlled text-reading tasks.

However, preservation is incomplete. The protected camera edit lost 17 of 99
previously correct answers under the other misleading words. Laptop editing was
weaker than ordinary matched projection fine-tuning. Sequential laptop accuracy
ranged from 51/96 to 74/96 across edit orders. These are documented limitations,
not hidden unfinished evaluation commands. Neither a repaired support image nor
many passing software tests establishes operational ML usefulness.

See [the results report](RESULTS_V2.md) for the full denominators, competing
methods, uncertainty and failures. The earlier results remain frozen.

## The remaining justification experiment

Before expanding the application or fitting more variants, define one concrete
workflow, such as object-based retrieval of product photographs containing
irrelevant packaging text. A deployed or collected incident is not assumed.
Establish whether failures recur in representative inputs and whether retaining
useful text information is actually required by that workflow.

One finite next experiment can answer this:

1. Freeze a small fresh development/final cohort before model predictions,
   including natural text-bearing images, already-correct examples, genuine
   target objects and tasks where reading text is useful. A public real-world
   benchmark such as SCAM can be considered after checking its task and license;
   it is not automatically the ground truth for our chosen application.
2. Compare the unchanged model, one hardware-feasible stronger pretrained model,
   a conventional object classifier where label coverage permits, matched
   fine-tuning, an implementable automatic text-handling control and our editor.
   Record class-mapping limitations and costs. The existing oracle mask is not
   that implementable baseline.
3. Specify acceptable repair, collateral-error and resource tradeoffs before
   looking at final results. Use development data for configurations. Existing
   published final data must not become a new selection set.
4. Stop after the declared comparisons. If a stronger unedited model or simple
   preprocessing already solves the workflow at acceptable cost, report that
   finding. Do not keep adding features to manufacture a need for editing.

This is a proposed next evidence gate, not an experiment that has already run.
Exact backbones, cohort size and resource estimates require profiling and a
registered protocol. It does not need a larger website or a collection of
unrelated features. A strong project can have a negative result if its question,
controls and conclusions are rigorous; it cannot claim a production benefit
that was never demonstrated.
