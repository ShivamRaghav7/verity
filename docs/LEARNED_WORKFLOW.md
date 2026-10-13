# Isolated workflow with actual trained models

The [recorded verification](../reports/pilot-v2-learned-workflow.json) exercised
the administrator API and actual trained scope/answer checkpoints in a new
SQLite fixture. It passed **31/31 governance and integration checks** across
33 in-process API requests. This result verifies lifecycle behavior and artifact
integration; it does not mean all generated answers were correct.

The fixture demonstrated the original five-device fact, an unauthorized-write
rejection, a nonauthoritative draft, approval of a three-device correction,
direct/paraphrased/capacity questions, related scope boundaries, an ambiguous
plan request, a conflicting draft, rejected conflicting approval, explicit
conflict blocking, conflict withdrawal and rollback. Reopening the SQLite file
confirmed both corrections remained withdrawn and the original five-device
fact was restored. Sources and audit history remained visible.

The API reported both loaded checkpoints as `trained`, with exact artifact
hashes, the scope's 360 completed/120 selected steps and the answer model's
240 completed/240 selected steps. It also disclosed automated development
review and `human_reviewed_data: false`. Raw responses include the actual
scope probabilities, selected edit, generation input and output, source cards,
before-answer trace and training provenance. Nothing was written to the live
application or its database.

## Generated behavior and failures

| Demonstration case | Observed learned-mode behavior |
|---|---|
| Original direct question | Frozen base answered `5` |
| Approved direct correction | Answer model generated `Nimbus Basic version v2 supports 3 devices.` |
| Approved paraphrase | Generated `Nimbus Basic version v2 includes 3 devices.`; pending under the locked assertion grammar |
| Connect three devices | Correct `Yes` with a complete three-device assertion |
| Connect four devices | Incorrect `Yes` with a three-device assertion |
| Basic monthly price | Incorrectly selected the device correction and generated `Nimbus Basic version v2 costs 3 devices.` |
| Pro, v1 and pre-effective-date questions | Compatibility guards prevented correction selection; base generated their original values |
| Nimbus Mini question | Compatibility guard prevented selection, but base generated `No`; pending and uncredited |
| Multiple plans | Explicit deterministic clarification |
| Unresolved correction conflict | Explicit deterministic conflict response; no correction-conditioned generation |
| Rollback | Correction removed from eligible memory; base answered `5` again |

The price example is a **false correction application**: scope compatibility
protects product/plan/version boundaries, while deciding the property relies on
the learned classifier. Its calibrated probability was approximately 0.151 at
the previously selected validation threshold of 0.11. The direct correction
probability was approximately 0.113. No threshold was changed after observing
this workflow.

Two separate oracle-routed probes guarantee execution of the actual answer
checkpoint while keeping routing success separate. The direct probe produced
the correct three-device assertion. The four-device probe again generated
incorrect `Yes`, locating this failure in answer generation even with correct
routing. Oracle outputs never count as end-to-end routing success.

Across the thirteen factual learned-mode demonstration calls, including repeated
questions, the primary scorer credited 2/13 and left 10/13 pending; the separate
exploratory short-QA diagnostic credited 8/13 and left 4/13 pending. Every case
remains in its denominator. The two nonfactual clarification/conflict calls and
two oracle probes are recorded separately. These handpicked, repeated
engineering examples are **not a representative accuracy estimate**. Pending
prose is not automatically a demonstrated factual error; the malformed price
answer and wrong capacity polarity are explicitly documented failures.

## Timing and resources

The recorded run took 40.282 seconds. Its first learned request took
19,867.912 ms, including validation/loading of four local backbones and first
inference. The immediate repeated warm request took 148.429 ms. Later corrected
requests ranged from 449.687 to 813.295 ms in this run. API timings include
before-answer generation and run through FastAPI's in-process TestClient;
they exclude browser/network transport and are not production latency quantiles.

The RTX 4050 Laptop GPU run used PyTorch 2.9.0+cu130 and peaked at 1,239.625 MiB
allocated / 1,326 MiB reserved. That measurement includes a second answer
checkpoint loaded for the separate oracle probes, so it is not the steady
application memory footprint. All watched source, configuration, calibration
and model artifact files had identical SHA-256 hashes before and after the run.

## Reproduction

Run from the repository root after the two trained artifacts and frozen
validation calibrations exist:

```powershell
.venv-ml\Scripts\python.exe scripts/verify_learned_workflow.py `
  --output reports/pilot-v2-learned-workflow-reproduction.json
```

The script requires local cached checkpoints and the configured CUDA device.
It runs no training, model download, threshold selection or final-test read.
It performs at most forty API requests and two oracle generations with the
configured finite 48-token output cap. It never retries failed workflow steps.
The default SQLite filename is unique under ignored `runtime/`; it is retained
for inspection. Existing fixture files, SQLite sidecars and report files are
refused so earlier evidence is preserved. The live `runtime/editguard.sqlite3`
path and database paths outside `runtime/` are explicitly rejected. There is no
HTTP connection to the application at port 8000.

Four regression tests verify the isolated-path boundary, preservation of earlier
fixtures/reports, request cap and refusal to mutate a fixture with missing or
ineligible trained checkpoints. Learned factual correctness is scored alongside
the workflow, and is never an all-correct prerequisite for governance checks.
