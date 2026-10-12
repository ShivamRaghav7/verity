# Validation selection for the similarity baseline

`scripts/calibrate_similarity.py` selects the cosine threshold for the
similarity-based correction editor on reviewed validation pairs. The frozen
embedding backbone is pinned in the inference configuration, loaded from the
project cache, and uses masked mean pooling with L2 normalization and 256-token
inputs. Cosine similarity is a geometric score, **not an applicability
probability**. This tool does not fit a temperature or report probability
calibration metrics such as ECE or Brier score.

The implemented similarity route selects the top compatible correction when its
cosine is at least the threshold, including equality. Otherwise it falls back to
the unchanged frozen base. It has no uncertainty band or tie-clarification rule.
Product/plan/version/date resolution and conflicts can separately clarify in the
application. This simpler policy is held fixed for threshold selection; the tool
does not quietly give it the trained scope classifier's policy.

## Finite objective and denominators

The finite grid has 202 thresholds: `-1..1` in increments of `0.01`, plus `1.01`
as an explicit never-apply operating point. Slight float32 endpoint overshoot of
at most `1e-6` is accepted and preserved. Missing, nonfinite or invalid scores
fail the run instead of disappearing from its denominator.

Selection minimizes compatibility-gated per-row cost: false correction
application costs 4, missed applicable correction costs 1 and correct decisions
cost 0. Clarification cost 0.25 is recorded for consistency with the scope
protocol but is inactive in this threshold-only pair experiment. Equal cost
prefers fewer false applications, then higher routing-decision coverage, then a
lower threshold. Routing-decision coverage is always 100% for complete binary
apply/reject pairs; it does **not** establish generated-answer coverage. This
development cost choice is not an accuracy guarantee.

Every accepted validation row remains in counts, cost, accuracy and recall.
Explicit reviewed rejections are separately counted as dataset exclusions.
The tool reports two distinct views at the selected threshold:

- Raw pair quality ignores compatibility gates and measures how the embedding
  score separates applicable corrections from all reviewed negatives.
- Compatibility-gated application quality also applies the deterministic
  entity/plan/version/effective-date and approval rules before thresholding.
  It preserves misses if a positive pair is rejected by a gate.

Neither view measures full-memory candidate retrieval recall, competition among
multiple corrections, generated-answer accuracy or end-to-end coverage. Only
one focal correction is scored per question. Those measurements belong to the
separate matched-comparison harness. Both editor routes require the same trained
conditional generator, input formatter, supporting-fact policy and fallback base;
their conditional generation settings are matched. Matching settings alone do not
establish the existence or correctness of a trained answer checkpoint.

## Reviewed inputs and artifacts

Only `scope_validation.jsonl` is opened. The tool verifies its manifest hash,
exact validation tags, duplicate IDs/JSON keys and complete hash-bound typed review
decisions before importing model packages. Question and correction descriptors
enter the embedding model; scoring labels and expected facts do not. The entire
accepted partition must fit the declared cap; silent subset selection is refused.
No test question, answer or challenge file is read, and there is no synthetic
production-output mode. Tests use clearly labeled temporary synthetic fixtures.
The default `human` gate refuses machine-typed decisions. The separately authorized
pilot-v2 registry reviewed 504/504 development rows (504 accepted, zero flagged)
and declares `human_reviewed_data=false`; source pending human flags remain
immutable. Select it only with `--review-policy automated_development`. This is
development data consistency evidence, not model accuracy or final-test approval.

Current automated-development preflight template (no model loading or output):

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_similarity.py --config configs/inference.json --dataset data/generated/pilot-v2 --reviews reports/pilot-v2-reviews.automated.json --review-policy automated_development --output reports/similarity-calibration-run-1 --max-rows 120 --preflight-only
```

This validates inputs without loading models or writing output. A human registry
uses the default policy instead. Restore the exact measured v2 source on a clean
checkout with the hash-checked offline unpack command in [DATA_REVIEW.md](DATA_REVIEW.md);
archived v1 bytes/pages stay unchanged. Current training/runtime formatting is v2.
Future embedding execution requires the isolated
ML environment and pinned cached checkpoint; its device defaults to the inference
configuration. Any explicit device override is recorded. No similarity threshold
selection run has been executed yet; numerical tests do not establish baseline quality.

Execution records `validation_cosine_scores.jsonl`, `selection_grid.json` and
an atomically published `calibration.json`. The frozen artifact binds embedding
model/revision, pooling and token limit, input format and routing policy,
retrieval settings, original row and review hashes, dataset/config/source hashes,
tokenizer identity, before/after embedding tensor fingerprints, truncation counts
and measured inference/selection time. Cosine records, grid, artifact and summary
state review policy and human-review status; automated development records `false`
and binds the separate audit hash. Inputs are checked again before publishing.
The output must be fresh and separate from dataset/review/configuration sources.

Runtime should accept only a frozen artifact with purpose
`similarity_validation_calibration`, split `validation`, the matching pinned
embedding identity and policy `top_compatible_cosine_gte_threshold_v1`. A raw
manually populated threshold alone is not evidence of validation selection.
Freeze this operating point before final testing. These validation selection
metrics remain optimistic development diagnostics; final test evaluation is a
separate milestone.
