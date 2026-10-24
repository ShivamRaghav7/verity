# Dataset card: fixed VLKEB evaluation subset

This is a reproducible subset of the authors' public benchmark, not a newly
authored independent dataset. It tests controlled counterfactual associations:
a requested target need not describe the real-world image. It is unsuitable as
a source of newly verified facts.

The source is `HymanH/VLKEB-data` at revision
`cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a`. Preparation verifies exact annotation
bytes, source archive checksums and extracted image hashes. Unsafe archive paths,
links and duplicate selected members are rejected.

## Roles and isolation

- The fixed first training example supplies the development smoke and pipeline
  profile. Its alternate photograph and unrelated probe are retained.
- Training rows 2–9 select the visual-RAG operating threshold: 24 applicable and
  16 out-of-scope cases. They are a small, nonrandom development subset.
- The evaluation contains exactly 20 correction lineages and five cases per
  lineage. Its source image-directory IDs are absent from every training image,
  alternate-image and visual-locality slot. Stable lineage hashes determine
  selection before model inference; selection is never based on observed success.

The original train/eval split has overlapping source entities. Our exclusion is
therefore stronger than simply trusting the source split, but does not prove
absence from foundation-model pretraining or certify the released editor's full
training provenance. Target names and locality entities can still overlap.

## Labels and scoring

Each selected record retains the authors' question, paraphrase, original answer,
counterfactual target, source/alternate image, text locality question/answer and
visual locality question/answer. The five categories are direct, paraphrase,
alternate photograph, text locality and visual locality. Five methods produce
500 scored question instances, clustered into 20 editing units.

Labels receive author provenance plus automated structural and checksum audits.
No independent human review by our team, independently authored challenge
questions, random population sample or calibrated confidence probabilities are
claimed. Cosine threshold selection is not probability calibration.

The primary scorer compares the entire generated continuation after fixed
Unicode/case/punctuation/whitespace normalization. It may reject useful partial
answers or longer valid variants. No test-derived aliases are added. Empty
answers, errors and unexecuted cases remain failures in the declared
denominators. Locality correctness and preservation of baseline-correct answers
are reported separately. Paired uncertainty resamples complete editing units.

The pre-inference seal binds the selection, model/data receipts, threshold,
configuration and scoring code. A resumed run preserves completed rows and
requires the same seal. Any further development after viewing these outputs
must disclose their use rather than treating them as unseen test data.

See [the frozen protocol](VLKEB_EVALUATION_PROTOCOL.md) for exact selection,
comparison, uncertainty and latency rules. Source annotations declare BSD-3-Clause;
individual source-image rights remain distinct. No private user uploads belong
in the benchmark or public report.
