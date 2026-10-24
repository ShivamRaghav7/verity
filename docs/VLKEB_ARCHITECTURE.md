# Architecture

```mermaid
flowchart TD
    Q[Image and short factual question] --> B[BLIP-2: EVA, Q-Former, OPT]
    B --> A[Bounded generated answer]
    C[Operator-proposed correction] --> M[MEND learned gradient transform]
    C --> F[15-step last-layer fine-tuning]
    M --> P[Candidate parameter patch]
    F --> P
    P --> T[Before / after generation and unrelated probe]
    T --> R[Restore original tensors and verify hashes]
    R --> H[Persist candidate evidence and source identities]
    H --> D{Explicit operator approval}
    D -->|Approve| L[Validate and activate own saved patch]
    L --> B
    L --> U[Rollback: copy saved originals]
    U --> B
    C --> V[Comparison: updated visual RAG memory]
    Q --> S[MiniLM text + visual-prefix scope score]
    V --> S
    S -->|Validation threshold accepts| X[Correction-conditioned prompt]
    S -->|Outside scope| Y[Base prompt]
    X --> B
    Y --> B
```

MEND and FT change model parameters; RAG and prompt-only comparisons leave them
unchanged. The RAG score is a validation-thresholded cosine combination, not a
calibrated probability. The evaluation always starts each parameter-editing
method from the same original weights and checks restoration between methods.

The stdlib loopback HTTP server serializes model operations. Uploaded images are
validated, file paths cannot be supplied by clients, POST requests require a
per-session token, and activation only accepts application-created patches.
Assets and history are external files; no additional database or queue is needed.

The experiment worker records complete generated continuations, fixed scores,
retrieval decisions, timing and resources. The read-only dashboard displays those
saved measurements. It cannot replace them with paper numbers or count a
structured lookup as a model-generated correct answer.

Preview checks are deliberately limited. A correct direct answer and unchanged
single probe cannot establish universal locality. Approval and exact rollback
are engineering controls around uncertain learned behavior.
