# Own measured visual-RAG threshold validation

This is development/threshold-selection evidence, **not held-out test accuracy**.
It was measured on a Google Colab Tesla T4 on 24 October 2026, using wrapper
commit `da64e3e1de6e89408205b46c32ea499970d0aae3`. No encoder training or MEND
meta-training ran.

The fixed validation set is upstream training rows 2–9, five cases per row:
24 applicable cases and 16 out-of-scope cases. The cosine score is not a
probability. The frozen selection rule maximizes validation F1, with ties broken
by precision and then the higher threshold.

| Quantity | Measured result |
|---|---:|
| Selected threshold | 0.7559913590939377 |
| True positives / applicable cases | 24 / 24 |
| False positives / out-of-scope cases | 4 / 16 |
| Precision | 24 / 28 = 85.71% |
| Recall | 24 / 24 = 100% |
| Validation F1 | 92.31% |
| Model loading | 98.276 seconds |
| Sampled peak process RSS | 8,476,536,832 bytes |
| CUDA peak allocated / reserved | 10,633,114,112 / 10,942,939,136 bytes |

These eight nonrandom training rows may contain repeated entities and may have
been seen during the released editor's meta-training. This small validation
result cannot establish generalization, factual-answer accuracy or reliable
large-memory retrieval. In particular, half of the eight visual-locality cases
were incorrectly routed; the eight text-only negatives were outside the visual
memory scope by design.

The machine-readable receipt, individual scores, source/model/code hashes and
worker log are retained under `artifacts/vlkeb/`. The deterministic 20-edit test
selection is preserved here before test inference, without selecting on model
outputs. Labels have an automated structural audit; no independent human review
by our team is claimed.

The transferred evidence ZIP SHA256 was
`97d12ca9bf267d3150f223b4534dbad7ba764f91c71ad2a759c60cf6ef03166f`.
The ZIP contains JSON/log evidence only, without model weights or account data.
