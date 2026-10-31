# Next milestone: model training and broader evaluation

Status: focused research roadmap, 31 October 2026. This supports slide 13 of the ten-minute mid-project presentation. The current application and released measurements remain the completed baseline platform. The adaptation study below is planned work.

## Research question

Does supervised adaptation followed by process-aware PPO improve the use of approved updates on unfamiliar wording, entities and connected questions under matched retrieval and inference conditions?

Success means completing and interpreting a defensible comparison. It does not require a predetermined gain, beating every baseline or preserving a perfect score on a larger challenge.

## Stage 1: data and pilot contract

- Construct source-grounded edit examples with original/update states, explicit scope, approved provenance, direct/paraphrased questions, connected targets and out-of-scope controls.
- Include difficult alternatives: similar names, wrong relations, old versions, neighbouring tiers and contradictory or missing support.
- Review both final targets and intermediate steps. Reserve independent human review for a meaningful challenge subset.
- Split edit lineages and entities before question augmentation. Create a fresh final test and keep it separate from training/development.
- Treat the already inspected released tests as regression evidence. They are not a fresh unseen final test for the new study.
- Profile a compact training and generation pilot before fixing the full backbone/adapter setup, GPU budget and run size.

Deliverables: data card, reviewed labels, split manifest, pilot profile, training specification and acceptance checks for checkpoint restoration. No runtime or memory estimate is issued before profiling.

## Stage 2: supervised adaptation

Use a verified common instruction-model starting checkpoint. Train it on applicable update/question/context inputs and the selected answer format. If intermediate response stages are included, ensure their labels are reviewed rather than merely plausible generated text.

Choose an adapter or other memory-efficient setup only after profiling. Pin source weights, tokenizer, data, optimiser settings and random seeds. Select checkpoints using development evidence. Save enough state to test exact resume of the declared training procedure.

Deliverables: SFT checkpoint/adapters, tokenizer/config identity, training log, validation results, resource measurements and model card.

## Stage 3: PPO adaptation

Start from the selected SFT stage. Define the process and final-answer reward contract before the main run. Candidate components include structural validity, relevant decomposition, intermediate subanswer correctness and endorsed final-answer quality. Validate the reward on malformed outputs, empty/extra steps, correct endpoints with wrong associations and unsupported but fluent explanations.

Use a fixed validation set for reward/design choices. Document the critic/reference-policy choices and reward scale relative to KL. A high reward must not be treated as a factual certificate. First run a bounded pilot with a checkpoint/resume check, then release a finite full-run command for the user to execute.

Deliverables: PPO-stage checkpoint/adapters, reward specification and tests, rollout/training metadata, validation findings and resource log.

## Stage 4: matched comparisons

| Arm | Starting identity | Context and evaluation |
| --- | --- | --- |
| Instruction baseline | Verified common instruction checkpoint | Same approved data access, retrieval and inference budget |
| SFT | Adaptation from that checkpoint | Same retrieval, prompts appropriate to the fixed response contract and scoring |
| SFT + PPO | Adaptation from the selected SFT stage | Same final evaluation contract and matched operating settings |

Keep the current reused ReasonKE_plus deployment as a separately identified reference. Its undocumented exact training ancestry does not make it a causal parent for this three-stage comparison. An outcome-only PPO ablation is optional only if it answers a specific question within the agreed budget.

## Stage 5: broader frozen evaluation

Vary entities, independently authored wording, number of distractor records and reasoning depth. Include source/version ambiguity and locality controls. Measure:

- Required-support recall and complete-path retrieval.
- Whole-endorsed-answer accuracy, answer coverage and abstention/error composition.
- Baseline-correct preservation and factual locality correctness separately.
- Intermediate-step support under a declared review/parser contract.
- Paired uncertainty by edit lineage.
- Warm p50/p95 latency, cold-start/resource requirements and output-token usage.
- Approval, conflict, history and rollback regression behaviour in the application.

Freeze all configurations, aliases and scoring rules before final inference. Preserve wrong and abstained outcomes. Analyse the observed tradeoffs rather than retuning against that test.

## Gate before every substantial training command

Provide the concrete question, required inputs, expected artifacts, profiled runtime/memory estimate with uncertainty, checkpoint/resume behaviour and finite stopping condition. The user executes long GPU training commands. Unexpected results lead to diagnosis within the agreed comparison, not an expanding sequence of unbounded experiments.
