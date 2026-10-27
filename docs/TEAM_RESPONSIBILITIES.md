# Five-person ownership of EditGuard

The project is one system: **an assistant whose approved knowledge can change,
whose connected answers can be measured, and whose changes can be traced and
reversed**. The five responsibilities below cover that system together. They
are a proposed ownership map, not a record of who historically wrote each file.
Replace the member numbers with names after the team confirms actual work.
Implementation, review, reproduction, testing and documentation are all useful
contributions; record which of those each person actually performed.

## The responsibilities

| Member | Main responsibility | Concrete work in the repository | Shared responsibility |
| --- | --- | --- | --- |
| **1 — Dataset and experiment design** | Define what a successful edit means and construct independent, testable examples. | Forty-eight fictional service/team graphs; original and updated views; direct, paraphrase, two-hop and locality questions; lineage-first development/final split; supporting fact IDs; deterministic graph audit; documented automated review. | With 2: inspect retrieval without leaking expected answers. With 3: fixed conditions, denominators and uncertainty. With 5: explain the demonstration handbook. |
| **2 — Models and context selection** | Make the learned reasoner run within measured hardware limits and receive the right context. | Pinned ReasonKE and Qwen integration; T4 4-bit loading; bounded prompts and outputs; authenticated worker; two-stage retrieval; handling of stale conversation history. | With 1: question/fact contracts and two-hop retrieval. With 4: the approved memory supplied to the model. With 3: resources, latency and fair model comparisons. |
| **3 — Evaluation, reproducibility and delivery** | Establish what the system actually achieves and make the evidence reproducible. | Finite sequential runner; frozen source/data/configuration protocol; exact-plan resume; endorsed-answer scoring; retrieval/oracle separation; locality versus preservation; paired lineage uncertainty; host receipt audit; graphs/PDF; CI, container and deployment checks. | With 2: model budgets and serving evidence. With 1: valid scoring populations. With 4 and 5: separate software checks from real model website checks. |
| **4 — Knowledge lifecycle and backend reliability** | Make publishing, superseding and reversing facts consistent and recoverable. | Canonical entity/property/scope identity; approval and conflicts; immutable audit events; historical fact views; atomic rollback; atomic conversation turns; validated text-workspace export and transactional restoration. | With 5: API contracts and user-visible conflict/rollback behavior. With 2: revision-consistent model context. With 3: regression and failure-injection evidence. |
| **5 — Application and interaction design** | Turn the core into a usable product with a clear evidence trail. | Normal chat; administrator editor; searchable knowledge library; approval/source display; conflict choices; history; before/after historical comparison; explicit example approval; raw-answer explorer; results dashboard and downloads. | With 4: complete editing and recovery flows. With 3: accurate phase labels, metric displays and exports. With 1: clear examples and demonstration script. |

No member needs to work alone. For example, one person can implement an API,
another integrate its screen, and a third reproduce its failure test. Record
those different contributions instead of giving all three the same authorship
claim.

## What each person should be able to demonstrate

**Member 1:** Reconstruct an owner-to-contact two-hop answer from its two facts.
Show why the old release, similarly named service and other property must stay
unchanged. Explain the split before paraphrase generation and the limitation of
automated labels. Start with the [dataset card](OPERATIONS_DATASET_CARD.md),
[data authoring](../editguard_benchmark/data.py) and
[dataset/benchmark tests](../tests/test_operations_benchmark.py).

**Member 2:** Trace one question through retrieval, prompt construction and an
actual saved model response. Explain that ReasonKE was trained upstream, why
we use NF4 on the T4, why only one model loads at a time, and why changing an
approved fact does not change the model weights. Start with the
[model card](REASONKE_MODEL_CARD.md), [runtime](../editguard_reasonke/runtime.py),
[retriever](../editguard_benchmark/retrieval.py) and
[measured profiles](../reports/reasonke/operations-profiles-v1/README.md).

**Member 3:** Recompute a saved answer's score, explain every reported denominator,
and reproduce receipt integrity checks and graphs. Explain why abstention/error
is not silently discarded and why a passing software test is not model
accuracy. Start with the [benchmark guide](OPERATIONS_BENCHMARK.md),
[scorer](../editguard_benchmark/scoring.py),
[metrics](../editguard_benchmark/metrics.py),
[host audit](../scripts/audit_operations_evidence.py),
[report generator](../scripts/build_operations_report.py) and
[CI](../.github/workflows/ci.yml).

**Member 4:** Approve an ownership replacement, demonstrate a conflicting active
version, roll back atomically, and inspect the corresponding audit events.
Explain why two rollback events share one revision and why a malformed restore
cannot partially overwrite a workspace. Start with the
[store](../editguard_chat/store.py), [snapshot validation](../editguard_chat/workspace.py),
[API](CHAT_API.md), [store tests](../tests/test_chat_store.py) and
[recovery tests](../tests/test_chat_workspace.py).

**Member 5:** Perform the complete [website demo](OPERATIONS_DEMO.md), including
an ordinary chat, approved edit, paraphrase, connected question, unaffected
question, historical comparison and rollback. Identify model output versus
deterministic controls and show a saved failure honestly. Start with the
[browser application](../editguard_chat/static),
[server](../editguard_chat/server.py) and
[browser evidence](../output/operations-2026-10-26/ui/README.md).

## Review and rehearsal after the primary evaluation

The frozen primary benchmark is complete: all 1,312 outputs were recorded and
independently host-audited. Review and rehearsal use that saved evidence without
changing the models, dataset, retrieval, generation settings or scoring rules.
The separate sixteen-generation website/rollback supplement also completed:
[12/12 HTTP answers](../reports/reasonke/operations-final/http-acceptance.json)
and [4/4 history/rollback answers](../reports/reasonke/operations-final/history-acceptance.json)
were correct, all eleven consistency checks passed, and neither workload had
execution, unexecuted or cleanup errors. Their
[independent host audit](../reports/reasonke/operations-final/operational-host-audit.json)
uses saved outputs without new model calls. Reusing development worlds does not
expand the held-out study's population.

1. **1 + 2:** Reconstruct a saved final example end to end, using its actual facts
   and output.
   Write a short explanation of one direct question, one two-hop question and
   one failure. Do not tune settings or aliases against final answers.
2. **4 + 5:** Rehearse approval, conflict, historical comparison, rollback and
   restoration in an isolated local workspace. Record the state/revision before
   and after each operation. CPU software checks need no competing GPU worker.
3. **1 + 3:** Review dataset limitations, each condition's information access,
   and the difference between preservation and factual correctness. Check that
   report wording matches those definitions.
4. **3 + 5:** Rehearse dashboard navigation and explain a raw response beside its
   score. Present the audited final counts, with development kept as the earlier
   policy-selection stage. Explain that R-RAG's 256/256 versus G-RAG's 241/256
   reflects fifteen Qwen two-hop-paraphrase abstentions with support present,
   not fifteen incorrect nonabstaining answers. Use the
   [failure analysis](../reports/reasonke/operations-final/FAILURE_ANALYSIS.md) to distinguish factual mistakes from
   strict-format mismatches while retaining the frozen scores.
5. **Everyone:** Read [the beginner guide](UNDERSTAND_THE_CURRENT_PROJECT.md),
   perform the full demo once, and answer questions outside the primary role.

Members 2 and 3 explain the actual model execution and resource evidence; 1 and 3
review the fixed populations and outcomes; 4 and 5 inspect the real website
editing/rollback supplement's own completion and consistency receipts; 3 and 5
check exports and rehearse the demo. All members review the resulting conclusion
and its limitations. Explain the difference between the 1,312 held-out requests,
the sixteen development-world website generations and the zero-generation
software checks instead of combining their scores.

## Record actual contributions before presenting

For each member, keep a small log:

| Person | Task and date | Contribution type | Reviewable evidence | Collaborators |
| --- | --- | --- | --- | --- |
| Fill in | Specific completed task | Implementation, review, test reproduction, analysis or documentation | Commit, reproduced command/output, issue or reviewed artifact | Names |

Suitable presentation wording is: “I owned the knowledge lifecycle, reviewed the
transaction rules with our interface teammate, and reproduced rollback and
restoration tests.” Say “implemented” or “trained” only when that describes the
person's actual work. The current ReasonKE checkpoint and Qwen baseline require
upstream attribution; our earlier auxiliary-model training is a separate study.

The shared primary-study conclusion follows the
[completed evidence](../reports/reasonke/operations-final/README.md): an implemented
governed knowledge-editing product, measured behavior on a bounded controlled
dataset, fair comparisons and documented limits. It does not depend on claiming
that every question succeeds or that ReasonKE beats every baseline. The separate
completed operational receipts establish measured website/rollback behavior;
continued live demo access depends on the serving runtime.
