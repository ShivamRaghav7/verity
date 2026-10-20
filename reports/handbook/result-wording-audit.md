# Result terminology and remaining-work audit

Audit date: 20 October 2026 (Asia/Calcutta).

## What the confusing terms mean

- `pending_review` and `pending_uncredited` are original machine-readable scoring categories. The fixed scorer did not safely credit those generated forms; they remain in the full accuracy denominator. They do not indicate a queued training/evaluation job. They can include incomplete facts, unsupported wording, and challenge forms that the frozen short-QA scorer does not support. They are not proof that every such output is factually wrong.
- Incorrect supported answers are real model failures and stay visible. Replacing or hiding those observations would invalidate the held-out evaluation. Further model revisions would require a new development/evaluation protocol and fresh independent evidence, not deleting failures.
- The original oracle diagnostic contains 319 CUDA execution errors. The separately declared recovery is completed: 319 selected cases processed, 65 original successes retained, 384 combined outcomes, zero combined execution errors. Original error records are preserved as history. The original process's lost GPU-memory state cannot be reconstructed by the later healthy process.
- Historical progress/failure records and raw limitation quotations are retained. A completed supplement can retain the prior failed session in its provenance without being unfinished work.
- A clarification or abstention deliberately withholds an answer; it is neither a runtime crash nor automatically a correct answer. It remains uncredited in accuracy.

## Presentation changes (no scoring changes)

Files: `editguard/static/app.js`, `editguard/static/index.html`, expected-terminology assertions in `tests/test_model_status.py`.

- Score table labels now say **Unscored by fixed rules** / **Unscored primary / QA**.
- Added a visible completed-score legend distinguishing correct, incorrect, unscored, clarification/abstention, and execution error, with fixed denominators and original export field-name explanation.
- Original CUDA error notice is explicitly historical and links conceptually to the separate completed recovery; when compatible recovery is loaded, it displays the combined recovered execution-error count without overwriting the original count.
- Initial loading and absent-evidence messages no longer assert that the project's final evaluation is still pending. Missing evidence in a particular running instance is described as unavailable/not loaded.
- A legacy structured-parser explanation saying auxiliary models are untrained is contextualized in the interface as **this structured mode does not run the auxiliary models**; current training/loading status remains separate. The original frozen service code and API payload remain unchanged.
- Customer-question maximum length is 1000, matching `AskRequest`. Original/corrected value limits are 40, matching `CorrectionCreate`, preventing avoidable form/API boundary mismatches.

## Verification

- `.venv/Scripts/python.exe -m unittest tests.test_model_status tests.test_final_dashboard tests.test_api`: **46 tests passed**, 20.838 seconds, no failures.
- Bundled Node.js `--check editguard/static/app.js`: passed.
- Parsed HTML maximum lengths and compared with Pydantic request JSON schemas: question 1000, original value 40, corrected value 40, all matched.
- `git diff --check`: passed.
- Frozen scorer/inference/training dependencies, API, raw JSON field names, recorded counts, scores, models, and sealed reports were not edited by this task.
- Browser visual review completed after the final UI updates. The completed
  evaluation, separate recovery, 14 workflow checks, original export fields,
  1000/40 input limits and lack of horizontal page overflow were verified.
  See `reports/ui/guide-outcome-legend.qa.json`.

## Actual unfinished work found in this audit

No unfinished model training, final-evaluation execution, or declared CUDA-recovery job was found. Remaining model errors and unsupported scoring forms are documented limitations of a completed controlled experiment. Optional independent human semantic review was not performed, per the chosen automated-audit policy; no human-review claim is made. Operational availability in a newly started server still depends on installing/configuring local model artifacts and starting the app.

## Loading-state follow-up

Visual QA found two pre-load evidence chips still said Not measured. Both now say Not loaded; a missing descriptor defaults to not_loaded. Explicit API descriptor states and all actual metric availability values remain unchanged.

The final follow-up passed 39 model-status/final-dashboard tests, JavaScript
syntax checking and diff checks. This group overlaps the earlier 46-test group.
The handbook's 31 pages were visually reviewed, and its command examples passed
PowerShell syntax parsing. The final integrity audit verified 21 frozen sources,
61 original summary-bound evidence files and both selected model exports.
