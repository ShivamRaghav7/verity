# Failure analysis: first engineering slice

No learned-model failure statistics exist yet. These findings came from source
review, automated regression and the local workflow. They are not test-set results.

| Finding | Consequence | Resolution/status |
|---|---|---|
| Naive numeric endorsement ignored “more than” and “fewer than” | Wrong Yes/No despite correct fact retrieval | Structured grammar now clarifies unsupported comparisons; regression covers both. Neural comparison quality remains unmeasured. |
| Unknown tier wording could be silently ignored when Basic was selected | Answer could target the wrong subscription | Conservative vocabulary/scope checks clarify unknown tiers and conflicting selectors. |
| Short product alias was nested in a similar product name | Nimbus could capture Nimbus Mini | Span-aware alias matching preserves longest contained name while retaining separately mentioned products; multi-product questions clarify. |
| Independent reads could combine a prior revision and later edits | Inconsistent audit/export provenance | SQLite read snapshots, thread lock and cross-instance regression now keep state coherent. |
| Windows sandbox blocks event-loop socketpair initialization | API test client hangs before app startup | Localhost tests run in the permitted host context; no model or app logic workaround. |
| Default sandbox temporary directory blocks SQLite | Fixtures could fail to create/clean files | Store/data tests use checkout-local temp paths; dashboard uses in-memory SQLite. |
| Unreviewed generator output can contain systematic shortcuts | Misleading apparent generalization | Review gate, family-first splits and negative taxonomy tests; independently authored challenge wording still required. |
| Shared question templates across splits | Entity-heldout accuracy does not prove novel-wording accuracy | Disclosed in dataset card; separate manual challenge set before final freeze. |
| Candidate checkpoints absent from local cache | Cached-only feasibility run cannot complete | Recorded as incomplete, never a throughput estimate; bounded download/profile follows. |

The structured parser intentionally has limited language coverage. Unsupported
questions clarify instead of being counted as correct model generation. Future
learned evaluation must count these abstentions in the question denominator.

For trained runs, append reviewed failure records with question ID, split,
expected fact, retrieved candidates, probabilities, routing decision, actual text,
scorer verdict and a component cause (retrieval, scope, generation, state or scorer).
Keep successful and failed examples together, and disclose semantic-review cases.
