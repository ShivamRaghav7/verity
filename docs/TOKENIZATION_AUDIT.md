# Pilot input tokenization audit

The 2026-10-12 v2 audit counted exact current training inputs with pinned cached
official tokenizers. It performed no model-weight loading, GPU work, backpropagation,
training, review decision or test/challenge-question access. Source rows remain
pending human review; the separate authorized automated development review
accepted 504/504 with zero flagged. The token audit did not read that registry
and makes no human-review claim. It is an advisory input-shape measurement, not
evidence of label quality, model accuracy, memory requirements or training throughput.

The reproducible command is `scripts/audit_tokenization.py`; its observed data,
manifest, tokenizer-file, trainer-source and audit-source hashes are recorded in
[`reports/pilot-v2-tokenization.json`](../reports/pilot-v2-tokenization.json). The
dataset is `editguard-controlled-v1-pilot-v2-42-7b7a9b1b-grammar-80ce36e5`.
The 2026-10-11 v1 audit, its report and its review pages remain archival and
unchanged. The v2 repair changed 28 correction descriptors, 4 questions, 8 targets
and 8 inputs for singular device/hour grammar, preserving structured labels and
creating new review-row IDs/hashes. Token counts were measured afresh on v2 bytes.

## Exact input formation

- Scope tokenizer: `distilbert/distilbert-base-uncased`, revision
  `12040accade4e8a0f71eabdb258fecc2e7e948be`. Count
  `tokenizer(question, correction_text, truncation=False)["input_ids"]` with the
  plain question first and stored correction second, matching the trainer.
- Answer tokenizer: `google/flan-t5-small`, revision
  `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`. Count the stored `input_text` and
  `target_text` separately with `truncation=False`, matching the trainer. The
  audit does not reformulate supporting context or correction text.

Both formats are `editguard-input-v2`. Counts include tokenizer special tokens
and exclude padding. The trainer pads to the declared input and target lengths.
The measured library versions were Transformers 5.16.1 and tokenizers 0.23.3;
the machine-readable report is authoritative if a future run differs.

## Observations

| Task / split | Rows | Input min | Input median | Input p95 | Input max | Target max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scope / train | 240 | 76 | 78 | 82 | 84 | — |
| Scope / validation | 120 | 76 | 77 | 82 | 82 | — |
| Answer / train | 96 | 121 | 125 | 130 | 132 | 20 |
| Answer / validation | 48 | 121 | 123 | 127 | 127 | 19 |

No scope or answer input would truncate at either checked input limit of 192
or 256 tokens: denominators are 360 scope rows and 144 answer rows. No answer
target would truncate at 48 tokens, with denominator 144. Input distributions,
exact-length histograms, p90/p99, target distributions and per-row token counts
are available in the JSON report. Reported percentile ranks use `ceil(p*n)`;
medians average the central values when the population size is even.

Among the two checked finite input limits, 192 contains every current pilot
input. The advisory candidate shape therefore remains batch 2, input 192, with
answer target 48. This matches the existing trainer interface and preserves
room beyond observed lengths. It is not a measured speed or memory prediction.
Reviewed edits or independently rewritten questions can change this result;
rerun the audit on the final reviewed input bytes with a new report filename.
A matching optimizer/resource profile is still required before releasing a
substantial training command; this tokenizer-only report supplies no throughput
or memory estimate. Current generator defaults are `controlled-v2` / generator 3.

## Integrity and refusal checks

The audit verifies selected file hashes against the pilot manifest, row split
and identifier consistency, manifest row counts, pinned tokenizer cache
revision, and unchanged data/config/manifest bytes after measurement. It reads
only explicitly selected `scope/answer` `train/validation` files. It does not
open a review registry or test file, and it never updates labels or approvals.

Verification also exercised existing-output refusal, test/unknown split
refusal, manifest checksum mismatch refusal and row split mismatch refusal.
The malformed examples were temporary copies of a training row; no actual
test row was read. Existing report bytes remained unchanged after these refusal
checks. Cached Hugging Face snapshot symlinks remain valid because revision
checking uses the lexical snapshot path before hashing tokenizer files.

## Reproduction

Use the isolated ML environment and existing pinned tokenizer cache. The command
refuses an existing output path; choose a fresh filename for another observation.

```powershell
$env:HF_HOME = (Resolve-Path runtime/hf-cache).Path
$env:HF_HUB_OFFLINE = "1"
.\.venv-ml\Scripts\python.exe scripts/audit_tokenization.py --data-dir data/generated/pilot-v2 --cache-dir runtime/hf-cache/hub --output reports/pilot-v2-tokenization-new.json
```

Optional `--task scope` or `--task answer` and `--split train` or
`--split validation` narrow the audited development population. Repeated options
must be unique. Unsupported splits fail before a dataset file or tokenizer is
loaded. No dependency was added for this audit.

For a clean checkout, first restore the exact measured packet:

```powershell
.\.venv\Scripts\python.exe scripts/pilot_bundle.py unpack --source data/pilots/pilot-v2-source.zip --manifest reports/pilot-v2-manifest.json --output data/generated/pilot-v2
```

The finite offline bundle verifies all allowed hashes and refuses existing output.
It excludes models, reviews and test/challenge files. New repair builds record
current source-code hashes and may produce a different manifest; use the bundled
packet for reproduction of these recorded observations. Archived v1 test/challenge
question files were neither read nor changed by this audit or v2 repair.
