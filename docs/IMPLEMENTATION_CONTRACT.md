# First vertical slice contract

This slice is a controlled fictional support demonstration. All answers are
deterministic structured answers. Neither trained auxiliary model exists yet;
the application must display this explicitly and reject learned mode until
models are available. No result from this slice is an ML accuracy claim.

## Data and storage

`editguard/models.py` defines Pydantic `Fact`, `CorrectionCreate`, `AskRequest`.
Facts include fact_id, entity_id, entity_name, aliases (list), relation,
value (string), unit (string), plan, version, effective_from (ISO date),
effective_to (optional exclusive ISO date), source, lineage_id.
Relations initially: device_limit, price_monthly, storage_gb, support_hours.

CorrectionCreate includes fact_id, original_value, corrected_value, source,
reason, effective_from, effective_to (optional), supersedes (optional edit id).
Server copies the immutable entity/relation/plan/version/lineage from the fact.
Correction records add edit_id, status (draft/approved/withdrawn), created_at,
approved_at. Conflict approval is refused with HTTP 409 and persists its draft.
No implicit latest-write-wins. Scope keys are exact tuples. Dates are half-open.

`Store(path)` creates schema; `seed()` idempotently loads `data/catalogue.json`.
Public methods return JSON-serializable dict/list:
- facts(): list[dict]
- edits(): list[dict]
- events(): list[dict] chronological audit trail
- conflicts(): list[dict] with edit_ids, fact_id, reason
- revision(): monotonically increasing integer
- create_correction(CorrectionCreate, actor='administrator'): dict
- approve(edit_id, actor='administrator'): dict
- withdraw(edit_id, actor='administrator'): dict; refuses active descendants
- resolve_fact(fact_id, as_of ISO-date): {fact, value, edits, conflict, revision}

DomainError(message, code=400) is raised for validation/state errors.
Transactions protect approval, withdrawal and revision increments. Supersession
requires same fact scope, explicit approved predecessor, later start, and matching
original_value; date queries prior to the successor retain the earlier value.
Rollback withdrawal makes the predecessor applicable again without rewriting
audit history. Withdrawn descendants do not suppress ancestors. Unresolved
overlapping draft contradictions block answering for the affected date and scope.

## API

GET /api/health: {status, revision, models:{scope, answer}, answer_mode}
GET /api/facts: {facts, revision}
GET /api/edits: {edits, events, revision}
GET /api/conflicts: {conflicts, revision}
POST /api/edits: CorrectionCreate -> correction (201)
POST /api/edits/{id}/approve -> correction
POST /api/edits/{id}/withdraw -> correction
POST /api/ask: {question, entity_id?, plan?, version?, as_of?, mode:'structured'}
-> {question, status:'answered'|'clarification'|'conflict', answer,
before_answer, answer_kind:'deterministic_structured', route, fact?,
applicable_corrections:[], revision, latency_ms, explanation}
GET /api/evaluation: {status, disclaimer, metrics:[], cases:[], revision}
GET /api/export: JSON export of catalogue, edits, events and slice evaluation.
Errors: {detail: message}. Administrator mutation endpoints require
X-Admin-Token matching EDITGUARD_ADMIN_TOKEN (default local-demo-only).
UI asks for token; never hardcode a token in JS. Bind to localhost by default.

## UI

Static files served by FastAPI at /. Vanilla JS uses same-origin /api endpoints.
Pages/tabs: assistant comparison, correction editor, history, conflicts,
evaluation. Seed question chips; product/plan/version/date controls; correction
source; history withdrawal/approval actions; exports; responsive accessible UI.
Show empty experimental metrics, not fabricated performance charts.
