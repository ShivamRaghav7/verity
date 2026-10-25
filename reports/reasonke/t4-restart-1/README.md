# Cold restart and complete website acceptance

On 25 October 2026, a fresh Colab Tesla T4 session loaded the pinned author
checkpoint and repeated the five predefined development questions. It then
ran seven actual generations through the website API. All assigned checks
completed, with no unexecuted cases or substituted answers.

These are **development acceptance checks**, not a held-out accuracy benchmark
or a reproduction of the paper's score. Repeating the same five fixtures on
another runtime does not create five additional independent test facts.

## Five model checks

| Question type | Generated final answer | Generation seconds |
| --- | --- | ---: |
| Direct fictional inventor | Tovan Vale | 14.62 |
| Independent paraphrase | Tovan Vale | 12.95 |
| Two-hop inventor → birthplace | Lyris | 13.35 |
| Unrelated capital control | Paris | 10.64 |
| Similar product name boundary | Serin Moss | 14.99 |

All five final answers matched their predefined targets. All supporting facts
were supplied explicitly; this is not a retrieval-quality measurement.
Download/loading took 399.24 seconds, followed by 66.62 seconds for these five
generations and receipt handling. Load peak allocated CUDA memory was
7,071,596,544 bytes and peak reserved memory was 7,140,802,560 bytes. These are
PyTorch measurements in this session, not total-system VRAM requirements or
runtime guarantees.

## Seven website generations

| Assigned check | Expected | Actual |
| --- | --- | --- |
| Original capital question | New Delhi | New Delhi |
| Approved counterfactual, direct question | Mumbai | Mumbai |
| Fresh-chat paraphrase | Mumbai | Mumbai |
| Matched comparison, without correction context | New Delhi | New Delhi |
| Matched comparison, with correction context | Mumbai | Mumbai |
| After withdrawal | New Delhi | New Delhi |
| Ordinary conversation about overfitting | Nonempty generation only | Relevant two-sentence explanation |

The six factual cases matched; the seventh is a nonempty-response check, **not
a seventh factual-accuracy success**. The full workflow took 49.47 seconds.
It published and withdrew its own demonstration record, leaving zero active
corrections at revision 2. Both comparison sides used the same trained model.
The synthetic Mumbai target does not describe India's actual capital.

The ordinary response was: “Overfitting occurs when a model learns the training
data too well, capturing noise along with the underlying pattern. This results
in poor generalization to new, unseen data.” Its content was inspected as a
basic qualitative check, not graded as an unrestricted dialogue benchmark.

## Reproducibility and provenance

- `acceptance.json` contains all five raw outputs, extracted final answers,
  token counts, timings, memory measurements and exact-match diagnostics.
- `live-acceptance.json` contains all seven raw website generations, operation
  responses, the two demonstration conversations, and publish/withdraw history.
  It confirms the owned correction is inactive at the end. CSRF tokens and
  authorization headers are omitted.
- `inventory.json` records the actual hardware/software and source revisions.
  The worker loaded from `7d3bfb34f9bd8d41f0f191ef43f18b1b4a9aec3f`.
  The website used `cb5fb1442e4b7c198ffc85309c772c6a5ad2f0cc`.
  Worker implementation and input manifests did not change between them.
- `model-manifest.json` and `predefined-cases.json` preserve the exact inputs.
  Their SHA-256 digests match the receipt and repository configuration files.
  The live verifier's recorded SHA matches `scripts/verify_reasonke_live.py`.
- The model is `YukinoKaren/ReasonKE_plus`, revision
  `ceaedc2ae7c8fed6257953587d991ce550db6106`, using NF4 weights with FP16
  computation. It was trained upstream. No training or weight updates ran here.

The new Colab region returned a `*.prod.r.colab.dev` website address. The first
website launch rejected it. A narrow allowlist update and regression tests
fixed access; the GPU worker stayed loaded. Exact-origin and CSRF checks remain
enabled. This deployment failure was not an answer-generation failure and is
not silently counted as a successful first launch.

## Limits retained in the evidence

The raw traces repeat unsupported explanatory claims from the
[first run](../t4-development-1/README.md), including an invented naming claim
and an incorrect intermediate scope statement. Correct final answers do not
verify those explanations. The earlier contradictory-source failure remains
documented; this repeat uses the revised, unambiguous demonstration source.

No universal edit success, arbitrary multi-hop reliability, human-reviewed
held-out score, training contribution, or paper-score reproduction is claimed.
A broader fixed-protocol evaluation is still required for those wider claims.

See [the website walkthrough](../../../docs/REASONKE_DEMO.md) and the current
[source-pinned launcher](https://colab.research.google.com/github/ShivamRaghav7/verity/blob/f6983b695ec2796de49026ac7e6361e51d9fbee9/notebooks/04_reasonke_chat.ipynb).
