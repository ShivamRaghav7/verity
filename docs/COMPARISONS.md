# Matched comparison implementation

The frozen base is pinned FLAN-T5-small with greedy decoding and a fixed short
answer budget. It answers from retrieved fictional catalogue sentences; no
assumption is made that pretrained weights know the company. Every base
parameter is frozen. Engineering smoke records a full tensor fingerprint before
and after inference.

All routes receive the same question, explicit product/plan/version/date,
initial facts and approved memory. Scope selectors are inference inputs, not
scoring labels. The deterministic resolver rejects incompatible named scopes
and certain explicit unknown product/tier/version forms. It does not claim to
understand every natural-language reference.

| Configuration | Evidence supplied to generation | Router |
|---|---|---|
| unchanged_base | Initial facts | MiniLM fact retrieval |
| updated_rag | Approved corrections applied to a fresh fact snapshot | MiniLM retrieval over updated facts |
| prompt_only | Initial facts plus retrieved approved correction descriptors | MiniLM correction retrieval with scope gates |
| similarity_editor | Applicable correction and up to two supporting facts | Validation-selected similarity threshold |
| trained_scope_editor | Applicable correction and the same supporting-fact policy | Trained question/correction classifier plus validation calibration |

The two editors share the trained answer generator. When no correction is
selected they use the unchanged frozen base. Missing artifacts yield unavailable
records; no pretrained auxiliary model is substituted. Generation uses the same
48-token answer budget. The learned generator's input length follows its
profiled training artifact, while the frozen base has a declared 384-token cap.
Token truncation is logged and included in failure analysis.

Approved ancestor corrections are suppressed only while a compatible descendant
is effective. Date intervals are half open. Withdrawal or expiry restores the
applicable predecessor. Updated RAG applies current corrections before ranking
facts; it is not deliberately stale. Each answer records its exact prompt,
candidate scores, selected correction, sources, model revision and truncation.
Retrieved sources support inspection; they do not verify generated prose.

In the live app, unresolved drafts/approvals conservatively block neural answers
throughout the matching product/plan/version/date scope. This can also withhold
an unrelated property in that scope. The structured route can identify a unique
relation and block only that relation. This adaptation favors clarification
until the learned routing has evidence sufficient to refine it.

`scripts/run_comparison.py` implements two finite development modes:

- `--smoke`: six literal Nimbus engineering fixtures, no dataset partition read.
- Reviewed validation: manifest-checked scope validation rows and hash-bound
  human decisions; no test path option. A bounded selection alternates positive
  and negative examples and discloses its selection rule.

The initial validation scenario uses one focal correction plus at most eight
foreign-product distractors, excluding the question's explicit product.
Counterfactual variants are never concurrently approved in one lineage. This
keeps the existing single-edit ground truth valid. It is a modest initial
benchmark; it does not establish performance in dense memory containing several
simultaneous edits on the same product. A later separately specified challenge
must test that case before making such a claim.

Ground truth is attached only after inference. Retrieval recall for updated RAG
checks the corrected fact; for correction routes it checks the applicable edit
in raw candidates before gates. Scope pair quality is measured independently on
the focal pair, including hard negatives rejected by deterministic gates.
Baseline-correct retention and factual accuracy remain separate. Structured
answers never count as trained generator correctness.

`--oracle-answer` separately evaluates the trained generator with the known
applicable correction, question and the same supporting-fact policy. It receives
no target answer. Its population is applicable questions only and is aggregated
separately, so missing out-of-scope questions cannot distort its denominator.
Absent trained answer weights produce explicit unavailable oracle records.

Final test evaluation remains unreleased until data, configuration, checkpoints
and scoring rules are frozen. Validation results guide development; a successful
smoke is not a final evaluation.
