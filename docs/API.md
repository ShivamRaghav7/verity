# API v0.1

Run the local server, then use `/docs` for the generated OpenAPI schema. Input
models reject unknown fields. Dates are ISO `YYYY-MM-DD`, with exclusive end dates.
Values are nonnegative decimal strings; device/storage/hour limits are integers.
Initial support hours are constrained to 0–24 hours/day.

| Method/path | Purpose |
|---|---|
| GET `/api/health` | Revision, answer mode and checkpoint status |
| GET `/api/facts` | Frozen initial catalogue and coherent revision |
| POST `/api/ask` | Scoped customer question with before/after answer |
| GET `/api/edits` | Corrections, immutable audit events and revision |
| POST `/api/edits` | Administrator creates a draft |
| POST `/api/edits/{edit_id}/approve` | Explicit approval; conflicting draft stays pending |
| POST `/api/edits/{edit_id}/withdraw` | Withdraw draft or approved leaf; preserve history |
| GET `/api/conflicts` | Overlapping contradictory live corrections |
| GET `/api/evaluation` | Disposable engineering regression; no learned accuracy |
| GET `/api/export` | Coherent memory snapshot plus isolated engineering report |

All three mutation endpoints require `X-Admin-Token` equal to server environment
`EDITGUARD_ADMIN_TOKEN`. Missing/wrong token is 401. Customer questions do not
accept authoritative correction fields. The first release has one shared
administrator identity; production accounts and session management are future work.

`POST /api/ask` example:

```json
{"question":"Can I connect four devices on Basic?","entity_id":"nimbus","plan":"Basic","version":"v2","as_of":"2026-10-11","mode":"structured"}
```

Response includes `status` (`answered`, `clarification`, `conflict`), `answer`,
`before_answer`, `answer_kind`, `route`, `fact`, `applicable_corrections`, `revision`,
`latency_ms` and `explanation`. Current answer kind is `deterministic_structured`.
Latency covers the service call; it excludes HTTP transport. `learned` mode is
503 until model registration/calibration is implemented. No checkpoint is implied
by a route label. Explicit question scope conflicting with a selector clarifies.

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
