# API v0.3

Run the local server, then use `/docs` for the generated OpenAPI schema. Input
models reject unknown fields. Dates are ISO `YYYY-MM-DD`, with exclusive end dates.
Values are nonnegative decimal strings; device/storage/hour limits are integers.
Initial support hours are constrained to 0–24 hours/day.

| Method/path | Purpose |
|---|---|
| GET `/api/health` | Revision, artifact/training provenance, loading and mode readiness |
| GET `/api/facts` | Frozen initial catalogue and coherent revision |
| POST `/api/ask` | Scoped customer question with before/after answer |
| GET `/api/edits` | Corrections, immutable audit events and revision |
| POST `/api/edits` | Administrator creates a draft |
| POST `/api/edits/{edit_id}/approve` | Explicit approval; conflicting draft stays pending |
| POST `/api/edits/{edit_id}/withdraw` | Withdraw draft or approved leaf; preserve history |
| GET `/api/conflicts` | Overlapping contradictory live corrections |
| GET `/api/evaluation` | Disposable engineering regression, model readiness and optional recorded ML validation summary |
| GET `/api/export` | Coherent memory snapshot, model provenance and available evaluation evidence |

All three mutation endpoints require `X-Admin-Token` equal to server environment
`EDITGUARD_ADMIN_TOKEN`. Missing/wrong token is 401. Customer questions do not
accept authoritative correction fields. The first release has one shared
administrator identity; production accounts and session management are future work.

`POST /api/ask` example:

```json
{"question":"Can I connect four devices on Basic?","entity_id":"nimbus","plan":"Basic","version":"v2","as_of":"2026-10-11","mode":"structured"}
```

Response includes `status` (`answered`, `clarification`, `conflict`, `error`, `abstention`), `answer`,
`before_answer`, `answer_kind`, `route`, `fact`, `applicable_corrections`, `revision`,
`latency_ms` and `explanation`. Structured mode uses `deterministic_structured`.
Optional modes are `unchanged_base`, `updated_rag`, `prompt_only`,
`similarity_editor` and `learned`. Set `EDITGUARD_INFERENCE_CONFIG` and use the
isolated ML environment to load neural inference. Missing trained artifacts or
calibration return 503. Model-generated responses retain the original generated
text and use `model_generated`; clarification uses `deterministic_clarification`.
No checkpoint is implied by a route label.

Health distinguishes checkpoint evidence from inference loading. `models` and
`model_details` report `trained` for configured completed auxiliary artifacts that
pass file-integrity/input-format validation, with `loaded=false` until inference
loads. Other states are `not_configured`, `missing`, `invalid` and
`loaded_unvalidated` for an injected runner without artifact evidence. No global
claim that models are untrained follows from a clean structured-only deployment.
Health does not load neural model weights into a model or import Torch.

`neural_inference` is `disabled`, `configured_not_loaded`, `loaded`, `unavailable`
or `configuration_changed`. Each `modes` entry is `available`, `load_required`
or `unavailable`; `available=null` means loading remains necessary and hardware/
checkpoint loading has not yet succeeded. Missing/invalid calibration prevents
the affected editor; any invalid configured dependency can prevent the shared
neural runner loading. A loading error is preserved explicitly. Restart after
changing a loaded inference configuration or resolving a loading failure.
Loaded inference also binds the validated model-manifest and calibration-file
fingerprints. Changed or invalid configured dependencies set
`restart_required=true`, block neural modes and suppress matching the recorded
report to that changed snapshot. Current configured metadata is not overwritten
by stale loaded-model provenance. Dependencies changing during model loading
are refused before the candidate runner becomes active.

The first metadata check validates artifact hashes. Later health requests use a
cached result with inexpensive config/manifest/model-file/directory stat checks;
changed signatures invalidate it. This avoids hashing large checkpoints on every
health request. It is an integrity check for trusted local artifacts, not a
security attestation against deliberate changes that preserve file metadata.
`model_details` include the manifest identity, training dataset/model identity,
completed optimizer steps, selected checkpoint step, review policy and
`human_reviewed_data`. Automated development training remains explicitly distinct
from human-reviewed training. Training completion does not establish answer
accuracy or frozen-test generalization; `final_evaluation` remains `pending`.

Neural responses also include `before_answer_kind`, `before_trace`, exact
`input_text`, retrieved facts/scores/sources, correction candidates, scope scores,
generation token/truncation trace, and `api_latency_ms`. Route `latency_ms` excludes
the original-answer generation and model initialization; API processing time
includes them and excludes HTTP transport. The comparison uses one coherent
memory revision. Explicit question scope conflicting with a selector clarifies.
Pending conflicts conservatively block neural output throughout their scoped
product/plan/version/date; see [COMPARISONS.md](COMPARISONS.md).
`model_training` records configured auxiliary training provenance separately from
the selected answer route. A low-confidence editor can use the frozen base even
while trained auxiliaries are loaded. The interface preserves that distinction
and does not call generated prose human-verified or correct from provenance alone.
Without a configured report, the evaluation endpoint reports its isolated
deterministic checks with `ml_results=null`; `model_readiness` remains metadata,
not model accuracy.

Set `EDITGUARD_EVALUATION_REPORT` to the recorded
`reports/pilot-v2-evaluation.json` development summary to expose actual validation
results beside the engineering checks. This optional read-only path never trains
or evaluates a model. The summary must explicitly identify automated development
review, `human_reviewed_data=false`, `test_rows_read=false`, all five matched
configurations and full question denominators. Model artifacts must match the
configured completed checkpoint manifests. Raw validation manifest and prediction
file hashes must match the summary; changed evidence is rejected. `ml_report`
reports `not_configured`, `available`, `missing`, `invalid` or `incompatible`.
Missing or incompatible evidence leaves `ml_results=null` and preserves its reason.
An unavailable metric can retain a nonzero population only with an explicit
reason and `value=null`. For example, correction application inferred from
generated RAG prose is not measured by an editor's selected-edit identifier;
the dashboard shows that metric as unavailable, without converting it to zero.

`ml_report.configuration_matches=false` means the configured inference settings
have changed while model identities still match. The recorded results remain
visible with that mismatch disclosed; they describe the saved configuration,
not a newly measured run. Report checks use cached stat signatures like artifact
readiness, and do not claim protection against deliberate metadata-preserving
tampering.

The dashboard labels strict full-assertion accuracy as a conservative lower bound.
Generated fragments pending semantic review receive no accuracy credit; that
does not establish that every uncredited fragment is factually wrong. It shows
accuracy, coverage, out-of-scope factual accuracy and baseline-correct retention
with their distinct populations. False edit applications, pending reviews,
generation errors and unavailable routes remain explicit. Oracle-routed answer
results retain a separate applicable-question denominator. Model-generated
results never incorporate the deterministic slice checks. Both the complete
workspace export and the dashboard's validation-summary download include these
recorded measurements, model identities, provenance and limitations. Final
frozen-test evaluation remains `pending`.

`POST /api/edits` example:

```json
{"fact_id":"nimbus-basic-v2-devices","original_value":"5","corrected_value":"3","source":"Fictional release EG-001","reason":"Approved plan change","effective_from":"2026-10-01","effective_to":null,"supersedes":null}
```

201 returns the draft with copied immutable scope, edit ID and provenance. No
customer-supplied scope can overwrite the chosen fact's scope. `original_value`
must match the authoritative value at `effective_from`, and approval revalidates
it transactionally. 400 means semantic validation failure, 404 unknown fact/edit,
409 state/overlap/stale-authority failure, and 422 Pydantic schema validation.

Successors require an approved predecessor in the identical fact scope, a later
start, a contained interval and matching prior corrected value. The preceding
date interval is preserved. Rollback restores the predecessor or initial fact.
Active descendants must be withdrawn before an ancestor. Withdrawn corrections
cannot be reapproved; publish a new reviewed correction with explicit provenance.

No answer cache exists in this release. All mutations append a revisioned audit
event inside their transaction. Read snapshots are coherent across another SQLite
writer. HTTP-export evaluation is a separate fixture: its `revision` records live
memory at evaluation request time, not the exported snapshot's earlier revision.
