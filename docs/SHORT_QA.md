# Exploratory request-bound short QA

The primary scorer requires a complete assertion of the product, plan, version,
relation, value and unit. The frozen assistant frequently produced short answers
such as `75 GB`, `10`, or `9 USD per month`, despite being prompted to state the
scope. Those outputs remain **pending and uncredited in the original primary
metrics**. A pending output is not evidence of a factually incorrect answer.

This separate diagnostic asks whether an output answers a narrowly specified
question. It was added after inspecting development validation outputs. It is
**post-hoc, automated development analysis**, not a frozen test result, human
semantic review, or a replacement for complete-fact endorsement scoring.

## Rules

`editguard/short_qa.py` supports exactly twelve controlled question forms: two
value requests and one comparison form for each of devices, monthly price,
storage and daily support hours. It independently parses the question's entity
surface, plan, version, property and comparison quantity. The parsed subject
must match a ground-truth entity name or exact alias; the plan and version must
agree with truth. The explicit request context must agree on entity ID, plan,
version and as-of date. A comparison label must also agree with independently
computed arithmetic, including the reversed inequality for “cost at most.”

An original **incorrect** verdict always prevents fallback credit, including a
wrong subject, property, scope, value or polarity, or an explicit contradiction.
A complete assertion already credited by the primary scorer keeps its stronger
evidence label after the question binding checks.

For a supported value request, a whole numeric output with an optional supported
unit is compared to the requested value. A bare number obtains its unit and
scope from the explicit question. A whole `$19` can answer a monthly USD price
question. An explicit incompatible unit fails. A substring, example, rejection,
extra prose, or unsupported output form does not receive automatic credit.

For a supported comparison, a whole `Yes` or `No` is checked against the requested
predicate. **Predicate correctness does not establish the exact limit.** A
quantity alone receives no comparison credit. Bare short answers never count as
complete scoped fact endorsements. Unsupported questions and prose remain
pending and uncredited. Errors, unavailable models, abstentions and structured
answers remain in denominators and receive no model-generated QA credit.

## Recorded development results

The [saved analysis](../reports/pilot-v2-short-qa/summary.json) includes all 600
end-to-end records, 120 per configuration, plus all 48 oracle-routed records.
Every configuration has the same 96 value requests and 24 comparison requests.
The 120 questions include 12 applicable direct requests, 12 applicable
paraphrases, 24 applicable comparisons and 72 out-of-scope requests.

| Configuration | Original complete-fact credit | Exploratory requested-QA credit | Value requests | Comparison predicates | Pending QA |
|---|---:|---:|---:|---:|---:|
| Unchanged base | 0/120 | 85/120 | 72/96 | 13/24 | 4/120 |
| Updated RAG | 0/120 | 110/120 | 88/96 | 22/24 | 4/120 |
| Prompt-only correction | 0/120 | 89/120 | 72/96 | 17/24 | 2/120 |
| Similarity editor | 30/120 | 102/120 | 90/96 | 12/24 | 5/120 |
| Trained scope editor | 19/120 | 91/120 | 79/96 | 12/24 | 5/120 |

Pending cases are included in every denominator above. The remainder comprises
incorrect or nonanswer outcomes. Complete-fact counts of zero for the three
frozen-baseline configurations comprise 120 pending primary outputs each; they
are not 120 demonstrated factual errors. Updated RAG leads this exploratory
requested-QA comparison. The learned editors produce more primary-credited
complete assertions, but that does not establish better requested-QA accuracy.

All five configurations receive 72/72 out-of-scope QA credit on this small slice.
Consequently retention of baseline-QA-correct out-of-scope answers is 72/72 here
as well. Preservation and overall factual correctness remain separately named
measurements; this observation is limited to these controlled questions.

Oracle routing receives 31/48 QA credit: 19/24 value requests and 12/24 comparison
predicates, with five pending answers and twelve incorrect comparison answers.
The generator's comparison-polarity failures remain visible even when routing
is supplied correctly. The diagnostic does not repair model outputs.

These are only two held-out entity families with shared question templates.
The labels were accepted through disclosed automated development review. They
have not received independent human review. The table therefore cannot support
a claim about unfamiliar wording, general support dialogue, production
reliability, statistical superiority, or performance on the final frozen test.
No question-template generalization claim or uncertainty interval is inferred
from this small correlated slice.

## Reproduction and integrity

```powershell
.venv\Scripts\python.exe scripts/analyze_short_qa.py `
  --input reports/pilot-v2-validation-initial `
  --output reports/pilot-v2-short-qa-reproduction
```

The output directory must be fresh. The finite command reads the original
manifest, metrics and two saved prediction files. It runs no model, training,
threshold selection or inference, and reads no test corpus. It validates matched
question identities, original row hashes, scorer identity, the recorded raw
predictions hash and complete oracle coverage. It refuses completed runs that
are not explicitly development validation runs.

The new [manifest](../reports/pilot-v2-short-qa/manifest.json) records the SHA-256
of all four original files, both analysis sources and all derived outputs. The
command checked that all four original files remained byte-identical afterward.
The original scorer version remains `editguard-short-facts-v2`; the diagnostic
has its own version, `editguard-request-bound-short-qa-v1`. Original primary
scores are preserved alongside separate `short_qa` records in the derived files.
The original raw predictions, oracle predictions, manifest and primary metrics
were not rewritten.

Eighteen independent regression tests cover exact alias boundaries, wrong tiers
and versions, mismatched request context, relation and unit errors, arithmetic
polarity checks, source preservation, malformed scores, duplicate/mismatched
records, refusal of test runs, no-overwrite behavior and unsupported prose.
