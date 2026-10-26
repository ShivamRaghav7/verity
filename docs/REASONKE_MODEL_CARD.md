# Model card: current released-checkpoint deployment

EditGuard's current learned reasoner is the author's
`YukinoKaren/ReasonKE_plus`, revision
`ceaedc2ae7c8fed6257953587d991ce550db6106`. The comparison instruction model is
official `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`.

Both checkpoints contain 7,615,616,512 parameters. This project did not train
either checkpoint. Our earlier DistilBERT/FLAN-T5 auxiliary training is a distinct
experiment documented in [MODEL_CARD.md](MODEL_CARD.md). Publishing a new fact
in the current website updates external memory, not either model's weights.

## Intended use

Short factual questions, ordinary conversation, and bounded connected factual
questions with administrator-approved context. The deployed memory has explicit
source/version records and supports comparison, supersession, withdrawal and
restoration. These controls establish authority and history inside the project;
they do not independently verify generated prose.

## Configuration and measured deployment

Both models use pinned safetensors, disabled remote model code, NF4 double
quantization and FP16 computation. Generation is greedy with one beam, at most
4,096 input tokens and 512 new tokens. Transformers' 120-second generation setting
is soft; the benchmark parent adds a hard 240-second request watchdog and a
1,800-second load watchdog. Only one model is loaded on the T4 at a time.

The measured profiles use Python 3.13.16, PyTorch 2.11.0+cu130, Transformers
4.56.1, Accelerate 1.10.1 and bitsandbytes 0.48.1. On the inspected T4, load peak
reserved CUDA was 7,140,802,560 bytes and sampled generation peak reservation was
7,205,814,272 bytes. These are PyTorch CUDA measurements, not total-system VRAM.

The ReasonKE profile averaged 14.21 seconds per request; the Qwen profile averaged
1.12 seconds. Their sampled condition/question mixes and output lengths differ,
so these are resource observations rather than a matched speedup claim.
See [the raw profiles and failures](../reports/reasonke/operations-profiles-v1/README.md).

## Adaptations and evidence boundaries

ReasonKE uses the upstream updated-information/query prompt structure with
approved facts serialized as JSON. The instruction baseline uses its official
chat template with explicit short-answer/context instructions. The configurations
have compatible budgets and shared retrieved fact IDs; their difference is not
a causal ablation of ReasonKE training, because exact checkpoint ancestry is
not established.

The product supports bounded full memory and question-only two-stage lexical
retrieval. Full memory accepts up to 20 facts. Retrieved mode selects up to four
facts from at most 200 records. Its storage/retrieval bound is tested separately
from neural quality; the operations benchmark worlds contain nine facts.

The five earlier startup fixtures and the seven website workflow generations are
development acceptance evidence. They are not a general accuracy estimate.
ReasonKE's fuller-memory profile exposed two-hop association failures despite
correctly identifying the first-hop owner; the instruction model also abstained
on one sampled two-hop paraphrase. All such outcomes remain in the receipts.

The model's raw explanatory steps can contain unsupported or incorrect claims
despite a correct final answer. The interface extracts final answers but does
not call explanations verified. There is no calibrated answer-confidence or
scope-classifier probability in this backend, and no 100% reliability claim.

## Attribution and distribution

The [author's code](https://github.com/YukinoshitaKaren/Reason-KE) and
[Reason-KE++ paper](https://arxiv.org/abs/2511.12661) describe the research method
and upstream training. Their published numbers are attributed research results,
never inserted as EditGuard's measurements. The exact released revision's
association with the published score is undocumented in its model card.

The official Qwen model card identifies Apache-2.0 terms. The inspected ReasonKE
release supplies no explicit checkpoint-license metadata; the upstream code's
license alone does not establish weight terms. EditGuard versions configuration,
code and measured receipts, and downloads pinned weights at runtime rather than
redistributing them in the application container.
