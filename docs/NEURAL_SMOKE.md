# Frozen neural inference smoke

The successful recorded run is `reports/neural-smoke-v2/`. Six curated Nimbus
engineering questions produced 30 matched configuration records: 18 frozen-model
generations and 12 explicit unavailable editor records. Three applicable oracle
answer records are also unavailable and are aggregated separately.

This is an engineering smoke, not a reviewed validation or frozen test evaluation.
It uses one correction and one product family; it cannot establish uncertainty
or unseen-edit/entity accuracy. No final test row was read. The full frozen-base
tensor fingerprints matched before and after inference.

The pinned base is FLAN-T5-small and retrieval is pinned MiniLM. All base and
retrieval parameters are frozen. Generation is greedy with 48 new tokens maximum.
The recorded peak Torch allocated memory was 395.183 MiB, reserved 434 MiB, on
the inspected RTX 4050. These inference counters exclude model-loading process
time and other software allocations. A first API request separately includes
lazy model loading plus before/after generation.

| Engineering question | Unchanged base | Updated RAG | Prompt-only |
|---|---|---|---|
| Direct Basic v2 device limit after 5→3 correction | `5` | `3` | `5` |
| Paraphrased device allowance | `9 USD per month` | `9 USD per month` | `9 USD per month` |
| Connect four devices on Basic v2 | `No` | `No` | `No` |
| Basic monthly price | `9 USD per month` | `9 USD per month` | `9 USD per month` |
| Pro device limit | `12` | `12` | `12` |
| Archived Basic v1 device limit | `5` | `5` | `5` |

These are exact raw outputs, never rewritten from ground truth. Updated RAG's
direct input contained three devices; the unchanged base contained five.
Prompt-only saw both the original fact and the approved correction, yet returned
five. Retrieval placed the device fact first on the paraphrase, so the wrong
price answer is a generation failure in this fixture. The unchanged base's bare
`No` is inconsistent with its five-device fact for a four-device request.

The conservative scorer cannot confirm an explicit subject/relation endorsement
from these fragments. They remain uncredited and queued for semantic review;
retrieved facts alone do not validate the text. Reporting the literal value `3`
as trained-generator accuracy would be incorrect.

There is no separate warmup; configurations execute in a fixed order and share
the embedding cache. Route timings are diagnostic only and must not be used to
claim a latency advantage between methods. A real latency evaluation must declare
warmup, ordering, cache policy, workload and sample size.

The first run, `reports/neural-smoke/`, generated its observations but failed
aggregation because unavailable records omitted request-context metadata. Its
manifest is marked failed and observations are preserved. The runner now records
identical requested context for every configuration; the second run completed.
This was a harness metadata error, not a reason to tune model prompts against
the test set.
