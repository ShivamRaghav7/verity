# Panel demonstration

Demonstrate one approved correction from beginning to rollback, then explain the
trained models using the recorded evidence. Name every answer mode. The first
sequence uses **deterministic structured answers** and tests governance; it is
not evidence that the answer model generated correct text.

The standard frozen short-QA comparison is complete: updated RAG 649/720, trained
editor 613/720. The original oracle run has 319 CUDA errors. After a zero-generation
preload interruption and GPU restoration, its separately declared recovery
answered all 319 selected cases without new runtime errors. Its combined view
retains 65 original answers and has 247/384 primary credits (90 pending, 47 wrong);
standard request-bound credit is 208/288. Present original and recovered evidence
separately, without substituting template answers into learned mode or claiming
that operational recovery improved model weights.

## Prepare a separate demonstration database

Keep the user's normal database and its edit history. From the repository root,
after installing the lightweight requirements, start an isolated demo on a free
loopback port, such as 8001:

```powershell
$env:EDITGUARD_ADMIN_TOKEN = 'replace-with-your-local-demo-token'
$env:EDITGUARD_DB = Join-Path 'runtime' ('panel-demo-' + [guid]::NewGuid().ToString('N') + '.sqlite3')
.\.venv\Scripts\python.exe -m uvicorn editguard.api:create_app --factory --host 127.0.0.1 --port 8001
```

Open [the isolated demonstration](http://127.0.0.1:8001). The API creates the new
database and seeds the fictional handbook. Verify no corrections exist, then
select structured mode, Nimbus, Basic, v2 and an as-of date of 2026-10-15.
Enter the configured token in Corrections. The token is a bounded demo authority
check, not production identity management. Retain the database after the demo;
rollback preserves history rather than erasing it.

## Six-part sequence

1. **Original answer.** Ask “How many devices does Basic support?” The initial
   Nimbus Basic v2 value is five. Show the selected product/plan/version and fact
   source. Explain that the initial fictional handbook is the base's information.
2. **Approved correction.** In Corrections choose its device-limit fact. Use
   original `5`, corrected `3`, effective date `2026-10-01` and source “Fictional
   release notice EG-001.” Save the draft. Show that a draft is not authoritative,
   inspect its scope, then explicitly approve it through the administrator action.
3. **Applicable wording.** Ask the direct question again, then “What is Basic's
   device limit?” and “Can I connect four devices on Basic?” Structured results
   show three and No. Inspect the edit ID, supporting fact, effective date,
   memory revision and before/after comparison. Clearly identify these as
   structured answers.
4. **Neighboring facts and history.** Ask Basic's monthly price: nine USD/month.
   Switch to Pro's device limit: twelve. Switch to Basic v1: five. Return to v2
   and a valid earlier date before the correction: five. These demonstrate exact
   property/plan/version/date boundaries, not unrestricted semantic routing.
5. **Conflict and ambiguity.** Create an overlapping competing draft changing
   original `3` to `4` in the same scope/date, without supersession. Approval is
   rejected and the affected question exposes the unresolved conflict. Inspect
   Conflicts, then withdraw that competing draft and confirm three is restored.
   An ambiguous product/plan request should explicitly clarify; do not supply an
   invented answer when scope cannot be resolved.
6. **Rollback.** Withdraw the approved three-device correction in History.
   Ask the direct question again: the value returns to five. Show the retained
   correction lineage and audit events. A withdrawn edit is immutable; restoring
   it later requires a new reviewed draft.

The previously recorded structured container workflow passed 23/23 lifecycle
and 3/3 restart checks. The isolated learned workflow passed 31/31 state/routing
engineering checks, which are distinct from its factual answer diagnostics.
See [deployment evidence](DEPLOYMENT.md) and the
[actual selected-model workflow](../reports/targeted-candidate-learned-workflow.json).

## Demonstrate the ML evidence

Open Evaluation lab. Identify the final cohort, automated-review policy, selected
model artifacts, calibration and scorer versions. Display all five matched
configurations and the complete denominators. The dashboard reads recorded
evidence; refreshing it does not train or re-run the held-out test.

Show the component distinction:

- Applicable-edit retrieval reaches 288/288 at rank 3 in the standard cohort.
- Raw scope classifies 701/720 correctly; runtime additionally uses deterministic
  compatibility guards and uncertainty handling.
- The editor gets 72/72 applicable direct short answers, 39/72 paraphrases and
  97/144 comparisons; overall updated RAG is stronger at 649/720 versus 613/720.
- The editor answers only 48/96 challenge questions. Their unsupported scoring
  forms and pending outputs do not establish general-language accuracy.
- Original oracle runtime errors remain in the original 384-case denominator.
  Show the completed 319-case operational supplement separately: its combined
  view retains 65 original answers and credits 247/384 complete assertions, with
  90 pending and 47 incorrect generated responses. Do not label original runtime
  errors as model wrongness or hide them to create a success rate.
- The separate learned HTTP fixture answers 30/30 measured requests with no
  runtime errors, warm p50/p95 230.42/372.04 ms and separate cold loading
  68,781.35 ms. Three warmups are excluded. Its four-device question still gets
  an incorrect Yes, so this is latency/availability evidence, not 30 correct
  answers or additional held-out accuracy.

Explain that the scope model and edit-conditioned answer model were trained
across corrections, while the base stays fixed. An approved new edit changes
memory without another fit. Show [the architecture diagram](ARCHITECTURE.md),
[model card](MODEL_CARD.md), [dataset card](DATASET_CARD.md),
[final results](FINAL_EVALUATION.md) and a representative
[failure analysis](FAILURE_ANALYSIS.md).

With GPU availability restored, live learned mode may be added to this sequence
using `configs/inference-final.json` and the saved local exports.
Display its actual generated answers and routing, including any failure or
clarification. Provenance is supporting evidence; it does not verify arbitrary
prose. The seven remaining failed development No comparisons and weaker final
comparison performance are appropriate discussion points, not facts to hide.

## Evidence to hand the panel

Use the recorded [summary](../reports/final/summary.json),
[CSV](../reports/final/results.csv),
[paired intervals](../reports/final/secondary-paired-uncertainty.json),
[scientific figures](../reports/final/figures-v2/manifest.json) and
[report source](REPORT.md) and [HTTP evidence](../reports/final/neural-http.json).
The 13-page report and current browser evidence include the completed supplements
and have been visually checked. Local release regressions pass 516 tests with
18 explicit optional skips; the separate 34-test CPU numeric/checkpoint group
also passes. Verify the three CI jobs on the pushed release commit for final
release sign-off.
Describe the lightweight structured container and optional local neural app as
separate deployment paths. No production, human-review or universal-accuracy claim
is part of this demonstration.
