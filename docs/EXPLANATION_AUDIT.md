# A bounded audit of explicit explanation claims

The final protocol declared an explanation-support audit before held-out
inference: the same sixteen hash-preselected two-hop question IDs as the oracle
diagnostic, crossed with **R-RAG** and **R-ALL**, giving **32 assigned raw traces**.
`scripts/audit_operations_explanations.py` implements that postprocessing. It
does not generate an answer, change an outcome or tune scoring aliases.

```text
python -m scripts.audit_operations_explanations <final-reasonke.json> --protocol reports/reasonke/operations-final/protocol.json --output <new-audit.json>
```

Final mode requires the identical frozen protocol and source/config/data hashes.
It reconstructs the sixteen IDs using the predeclared `oracle-42:` hash ordering;
it does not choose successful or unsuccessful traces after seeing answers.
Every selected trace remains assigned, including execution errors, unexecuted
records, missing output and unparseable formats. The input context is checked
against the declared retrieval and request digest before the audit.

## What is checked

The parser recognizes exactly one `<decompose>` and one `<action>` block with
indexed `[Sub question N]` entries. Each action must have exactly one simple
`\boxed{answer}` and no recognized negation/rejection/example marker. It resolves
`[sub answer N]` only from an earlier extracted box. One explicit anaphora rule
resolves “Who is the on-call contact for that/this team?” at step 2 to the boxed
step-1 team. Other wording is not guessed.

Each extracted proposition is the pair **resolved subquestion → boxed answer**.
It receives two separate labels: support in the facts actually supplied to the
model, and correctness under the current world facts.

| Label | Meaning for the bounded proposition |
| --- | --- |
| supported | A normalized exact fact question and answer match in that view |
| contradicted | That exact fact question exists but endorses a different value |
| unsupported | A recognized controlled factual query has no fact in that view |
| unassessable | Unknown question wording, ambiguous/missing box, unresolved reference, competing facts or unsupported trace format |

For example, the development R-ALL Ember Relay trace correctly named **Willow
unit 75** as owner, then stated its contact was **Ilan Voss**. The authoritative
contact is **Lina Dane**. The audit labels the first indexed claim supported and
the second contradicted, even though Ilan Voss is a valid contact for the old
team. Removing a required contact fact from supplied context instead yields
unsupported in the supplied view and can remain supported in current truth.

## What is deliberately left unassessed

All raw output is retained. Surrounding action prose is stored as unassessable
material, as is acknowledgement, decomposition narrative, final-tag prose and
text outside the action block. Such prose may contain further factual claims
that an exact-format parser cannot safely certify. Missing or multiple blocks
are preserved without guessed claims. Therefore **fully verified traces is
always zero**; a supported subanswer is never a certificate for the whole trace.

Claim-label counts have an extracted-proposition denominator. They are not
whole-explanation accuracy. Trace-status counts use all 32 assigned final traces.
Errors and missing traces cannot vanish into a smaller success rate. The report
retains both denominators and unassessable segment counts.

Development mode is a separate parser diagnostic on all assigned two-hop
questions in the saved bounded development run: twelve traces across the same
two conditions. It is explicitly labelled development and does not replace the
preselected final audit. Unit tests cover real development correct and old-team
join-error outputs, missing support, negation, multiple boxes, unknown wording,
unresolved references, execution errors and absent trace format.

This is an **automated narrow-format support audit**, without independent human
semantic review. It does not prove internal reasoning faithfulness, complete
natural-language claim extraction or arbitrary factual verification. Primary
answer accuracy remains the unchanged frozen scorecard.
