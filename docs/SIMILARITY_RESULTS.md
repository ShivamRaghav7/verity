# Similarity baseline: validation operating points

These 2026-10-12 measurements describe a development baseline on 120 fixed
question–correction pairs from pilot-v2 validation: 48 applicable and 72
out-of-scope pairs across two entity families. Labels underwent explicitly
automated development review; `human_reviewed_data=false`. The embedding model
was pinned `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with masked mean pooling and L2
normalization. The recorded embedding tensor fingerprint stayed unchanged.

The finite selection grid has 202 thresholds: −1 through 1 in increments of
0.01, followed by a 1.01 never-apply sentinel. Selection minimizes the
compatibility-gated cost `4 × false applications + missed applicable pairs`,
then breaks ties using the recorded policy. It selected **0.46**. Cosine is a
similarity score, not a calibrated probability; these results establish no ECE
or probabilistic confidence claim.

![Validation precision, recall, false application rate and decision cost](../reports/similarity-pilot-v2/operating-points.png)

The [SVG export](../reports/similarity-pilot-v2/operating-points.svg) supports
scalable report figures. Rate panels share their scales; cost panels explicitly
use separate y scales. The plots show thresholds 0 through 1.01 because every
observed cosine score lies between 0.317 and 0.718; the complete source grid is
preserved. Precision becomes undefined when no correction is proposed, and
those points appear as gaps rather than zero precision.

## Selected-point results and denominators

The raw cosine panel is a diagnostic at the same selected threshold. Threshold
0.46 was selected for the gated policy; it was not separately optimized for
the raw panel.

| Measurement at threshold 0.46 | Raw cosine pair decision | Cosine with compatibility gates |
| --- | ---: | ---: |
| True applications | 47 | 47 |
| False applications | 48 | 0 |
| Correct rejections | 24 | 72 |
| Missed applicable pairs | 1 | 1 |
| Precision | 47/95 = 49.47% | 47/47 = 100% |
| Recall | 47/48 = 97.92% | 47/48 = 97.92% |
| False application rate | 48/72 = 66.67% | 0/72 = 0% |
| Correct binary pair decisions | 71/120 | 119/120 |
| Proposed correction rate | 95/120 | 47/120 |
| Weighted cost per pair | 193/120 = 1.6083 | 1/120 = 0.0083 |
| Clarifications / abstentions | 0/120 | 0/120 |
| Binary routing decision coverage | 120/120 | 120/120 |
| Generated-answer coverage or accuracy | Not measured | Not measured |

The compatibility checks rejected 60/120 pairs on approved-edit metadata and
explicit entity, plan, version and date selectors. All 60 were negative in this
controlled validation population. These deterministic checks explain the large
change in false applications: it is not a measured improvement from training
a scope classifier. The 12 compatible negative pairs still require property
discrimination; none crossed 0.46 in this small dataset. Zero observed false
applications does not establish zero future error.

No pair abstained: this baseline uses a binary threshold without an uncertainty
band. Rejecting an edit remains a scored routing decision; it does not mean a
correct answer was generated. Denominators include every validation pair.

## Recorded failure

One applicable price paraphrase fell below threshold in both views:

- ID: `pilot-v2:42:scope:edit:aster-044-core:Basic:v2:price_monthly:variant-0:p1`.
- Question: “How much would I pay each month for A044 Basic version v2?”
- Correction: Aster 044 Basic v2 monthly price changes from 44 to 7 USD/month,
  effective 2026-09-01; source
  `controlled://corrections/family-044/price_monthly/0`.
- Observed cosine: **0.4533250331878662**; compatible with the correction, but
  below 0.46. The policy misses the applicable edit. No fallback answer was
  generated or evaluated in this measurement.

## Limits on interpretation

Threshold selection and reported metrics use the **same validation population**.
The result is descriptive development evidence and is optimistic for unseen
data. No frozen test pair or manually authored test challenge was read. Two
families, repeated templates and related counterfactual variants provide few
independent observations; no external-generalization confidence interval is
claimed from these pair counts.

Each example contains one focal question–correction pair. This run does not
measure retrieval recall over full correction memory, competing edits, answer
generation, factual-answer retention, unrestricted questions or end-to-end
latency. Explicit request selectors simplify compatibility; reliable extraction
from arbitrary customer prose remains a separate concern. Neither auxiliary
model has yet completed reviewed-data training, and these results provide no
superiority claim over updated RAG, prompt-only correction or a learned editor.

## Reproduce the figure

The immutable inputs are
[`calibration.json`](../reports/similarity-pilot-v2/calibration.json),
[`selection_grid.json`](../reports/similarity-pilot-v2/selection_grid.json) and
[`validation_cosine_scores.jsonl`](../reports/similarity-pilot-v2/validation_cosine_scores.jsonl).
`scripts/plot_similarity.py` recomputes every grid point from the recorded scores
and checks exact agreement with saved counts and costs before plotting. PNG/SVG
metadata includes input and plot-script hashes. The source files are checked
again after export and remain unchanged.

```powershell
.\.venv-ml\Scripts\python.exe scripts/plot_similarity.py --input-dir reports/similarity-pilot-v2 --output reports/similarity-pilot-v2/operating-points-new
```

Existing plot outputs are refused. The export was visually inspected for legible
axes, denominators, selected-point markers and limitations. Matplotlib 3.10.7
was installed from the existing `requirements-plots.txt` in the isolated ML
environment; NumPy 2.4.2, Torch 2.9.0+cu130, Transformers 5.16.1, tokenizers
0.23.3 and safetensors 0.8.0 remained unchanged. No model, GPU or training
operation was performed for this figure, and the ML dependency lock was not
rewritten.
