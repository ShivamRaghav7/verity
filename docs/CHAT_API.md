# Conversational application API

The local single-operator website binds to loopback. The optional Colab proxy
requires an exact configured HTTPS origin. All mutation requests require
`Content-Type: application/json` and `X-EditGuard-Token` from `GET /api/status`.
That session token is private and must not be committed or placed in URLs.
Customer chat does not publish corrections.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/api/status` | Model/backend identity, readiness, knowledge revision, policy and session token |
| GET / POST | `/api/chats` | List conversations / create one using `{}` |
| GET | `/api/chats/{id}` | Conversation and historical messages, evidence and revisions |
| POST | `/api/chat` | Generate from `chat_id`, `message`, `mode` (`edited` or `original`) |
| POST | `/api/compare` | Two fresh generations using `chat_id` and an existing user `message_id`; optional integer `baseline_revision` selects historical approved memory |
| GET | `/api/memory?revision=N` | Replay approved facts at revision N without changing live memory |
| GET / POST | `/api/corrections` | List records/audit events / explicitly approve a correction |
| POST | `/api/corrections/{id}/withdraw` | Deactivate while retaining history; idempotent |
| POST | `/api/corrections/{id}/restore` | Restore a version unless an active version conflicts |
| GET | `/api/workspace/export` | Consistent checksummed text snapshot, refused during a mutation/model operation |
| POST | `/api/workspace/restore` | Validate and atomically import `{ "snapshot": ... }` into an empty workspace |
| GET | `/api/experiments` | Typed current ReasonKE development or controlled-final evidence |
| GET | `/api/experiments/artifacts/{name}` | Allowlisted current PNG/SVG figures, report, scorecard and figure provenance |
| GET | `/api/examples/operations` | Fictional development catalogue and example questions, without approving anything |
| POST | `/api/examples/operations/approve` | Explicitly approve nine fictional initial facts only when knowledge history is empty; existing chats are preserved |
| GET | `/api/results` | Explicitly archived visual-study JSON, a separate model/protocol |

## Approve an identified fact

```json
{
  "question": "Which team owns Atlas in release 2?",
  "answer": "Amber team",
  "scope": "Atlas service; release 2; owner",
  "source": "Controlled fictional handbook, ownership revision 2",
  "identity": {
    "entity_id": "atlas-service",
    "relation_id": "owner",
    "scope_id": "release-2",
    "aliases": ["Atlas", "Atlas service"]
  }
}
```

Canonical IDs are optional for legacy free-form records. If used, all three IDs
are required. IDs accept letters, numbers, dots, hyphens, underscores and colons,
and normalize to lowercase. Aliases are recorded, not learned entity resolution.
The canonical key controls version/conflict identity even when wording differs.
To replace a conflicting active record, include its ID as `supersedes` in the
new approval. Conflicts return HTTP 409 with the competing records; an operator
must explicitly resolve them. This does not detect all semantic contradictions.

## Workspace restore

Snapshots contain exactly the versioned schema, creation timestamp, revision,
four data tables and a SHA-256 integrity digest. They contain no executable SQL,
model weights or session tokens. The server validates row fields, identifiers,
lineage references, positions, JSON, active-key uniqueness and revision events.
It refuses overwriting a nonempty target with HTTP 409. Invalid input changes
nothing. A snapshot checksum establishes integrity, not factual truth or authorship.
Event replay must agree with every final fact state, and each revision must occur
in order. Import cannot create competing active canonical keys or deactivate a
record that was already inactive.
The restore action is the local operator's explicit approval of that workspace.

Text snapshots are capped at 6 MiB and 10,000 rows per table. They refuse local
image paths; image conversations require a private SQLite plus images-directory
backup. Export before terminating the temporary Colab runtime.

## Failures and generation provenance

Invalid requests return 400/413/415; unauthorized/cross-origin mutations return
403; unavailable models return 503. Generation fails explicitly, without a
template answer standing in for the neural model. Messages retain model revision,
knowledge revision, selected fact IDs, raw model output and answer-format details.
Sources describe supplied context and operator approval; they are not a factual
verification label for arbitrary generated prose.

The CLI defaults to `--context-policy retrieved`, selected before final evaluation
using the matched development comparison recorded in `configs/deployment-policy.json`.
`--context-policy all` supplies at most 20 active records. `retrieved` supplies up
to four selected records from at most 200 active facts.
Both enforce the worker's 12,000-character supplied-context and 4,096-token input
budgets. Oversized context is rejected rather than silently truncated.

## Compare an approved historical state

```json
{"chat_id": "<existing chat ID>", "message_id": "<existing user message ID>", "baseline_revision": 9}
```

On the ReasonKE backend this generates once with approved memory at revision 9
and once with current approved memory. Both use the same pre-question history;
old correction-conditioned messages cannot leak the newer answer into the older
side. It changes no facts and saves no replacement answer into the conversation.
Omitting `baseline_revision` compares without correction context instead. Revision
zero is empty approved memory. Invalid/future revisions are rejected. This is an
external-memory comparison, not two different weight checkpoints.

The operations example starts with nine approved facts. After explicitly approving
the ownership update, compare against revision 9 to inspect old versus new owners
or their contacts. The helper never publishes the proposed ownership update
without the operator pressing **Approve & publish**.
