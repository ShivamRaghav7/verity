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

The matched 120-request development comparison is complete: ReasonKE with
retrieved context answered 24/24, compared with 21/24 under full memory. Qwen with
the same retrieval also answered 24/24, so there is no superiority claim. The
retrieved policy was selected before final inference; see
[`configs/deployment-policy.json`](../configs/deployment-policy.json) and the
[actual development evidence](../reports/reasonke/operations-development-v1/README.md).
These are three independent development lineages, not held-out final accuracy.

The subsequent [frozen held-out study](../reports/reasonke/operations-final/README.md)
completed all 1,312 requests across 32 lineages. ReasonKE with updated retrieval
answered 256/256 main questions correctly; Qwen with identical retrieved fact IDs
matched 241/256 fixed targets and abstained on fifteen, with no wrong nonabstaining
answers. All fifteen abstentions were two-hop paraphrases (17/32 correct in that
category), despite the required facts being supplied. The observed
5.86-percentage-point difference has a 95% paired
lineage-bootstrap interval of [3.91, 8.20]. It reflects answer coverage in this
controlled workload; it does not isolate the causal effect of ReasonKE training.
Both retrieved configurations answered all 128 locality questions correctly.

Full-memory ReasonKE answered 231/256 correctly with 25 wrong answers;
full-memory Qwen matched 241/256 with thirteen strict mismatches and two abstentions.
The initial-fact ReasonKE diagnostic scored 136/256 against updated truth. Both
sixteen-question oracle diagnostics scored 16/16. No final request was unexecuted
or an execution error. These sample results do not guarantee arbitrary questions,
unfamiliar natural-language formulations or larger fact stores.
The [failure analysis](../reports/reasonke/operations-final/FAILURE_ANALYSIS.md) separates factual errors from
strict-format failures: two G-ALL mismatches include the correct requested person
alongside extra owner text. The fixed primary scores are not revised afterward.

Warm pipeline p50/p95 was 15.06/16.71 seconds for retrieved ReasonKE versus
0.92/1.20 seconds for retrieved Qwen (256 generated requests each), excluding
cold loading and browser networking. Qwen was faster on this measured workload.
See the [full aggregate](../reports/reasonke/operations-final/results.json) for
all conditions, paired uncertainty, resources and failures. The separate sixteen
website/rollback generations also completed: the
[HTTP receipt](../reports/reasonke/operations-final/http-acceptance.json) records
12/12 correct answers, and the
[history/rollback receipt](../reports/reasonke/operations-final/history-acceptance.json)
records 4/4 correct answers with all eleven consistency checks passing. Both have
zero execution, unexecuted or cleanup errors. The twelve-request HTTP p50/p95 was
15.42/18.78 seconds. An
[independent host replay](../reports/reasonke/operations-final/operational-host-audit.json)
checked all sixteen outcomes without new model calls. These reused development
worlds are a distinct operational population, not additional independent held-out
questions or a guarantee for future live requests.

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
