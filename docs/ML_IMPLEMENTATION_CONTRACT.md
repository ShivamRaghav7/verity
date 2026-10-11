# Stage C/D/E implementation contract

Implement locally without changing structured-mode defaults or memory authority.
No final test evaluation and no substantial training automatically executed.

## Neural inference (`editguard/neural.py`, root)

- FrozenGenerator(model_id, revision, device='cpu', local_files_only=True,
  max_input_tokens=384, max_new_tokens=48): generate(input_text) -> str.
  All base parameters require_grad=False, eval/inference_mode, deterministic
  generation. Log token counts/truncation; never rewrite generated text from truth.
- EmbeddingRetriever compact MiniLM mean-pooling + L2 normalization, pinned
  revision. Scores correction descriptions/fact texts, exposes raw top-k as well
  as scope-filtered top-k. Model imports lazy: normal app/tests need no torch.
- ScopeCheckpoint and AnswerCheckpoint load only trained manifests/checkpoint
  files; absent artifacts -> unavailable, never substitute random/pretrained
  auxiliary as a trained model. Calibration separate, validation-only.

## Comparison harness (`editguard/comparison.py`, root)

Use explicit request_context to filter entity/plan/version/date; do not filter
relation from expected_fact or expected label. Question retrieval determines
relation. Scenario memory combines distractor approved corrections within split;
counterfactual variants of a lineage are mutually exclusive scenario states.
No configuration receives ground truth labels in its prompt/input. Each receives
the same approved edit memory and initial/support context availability.

Five configs: unchanged_base, updated_rag (apply corrections then reindex),
prompt_only (initial fact retrieval + eligible correction descriptors),
similarity_editor (embedding route + trained generator), trained_scope_editor
(pair checkpoint/calibration route + same trained generator). Missing checkpoints
yield explicit unavailable/error records counted in denominators if requested;
never silently label frozen generation as trained. Available-only smoke runs may
select first three but disclose missing two. Record exact inputs and route evidence.

## Trainer (`scripts/train_auxiliary.py`, training agent)

Use preflight-reviewed task train/validation, run-card pinned pretrained revision,
max_steps cap; deterministic sampler and shuffle RNG, batch2, float32 initially,
AdamW + declared LR schedule. Save atomic exact-state checkpoints with model,
optimizer, scheduler, RNG (Python/NumPy/Torch CPU+CUDA), global_step, sampler
cursor/order, run/data/review/model/environment hashes. Resume fails on differences;
finite cap exact. Do not import Torch at module import for core test compatibility.
No test paths. No full-run command released until reviewed inputs and trainer
profiling are real. Bounded developer smoke may use explicitly synthetic fixtures,
must mark checkpoint purpose=smoke_test and runtime refuses it as production.
Store model/tokenizer, selected/best checkpoint and selection metric (validation
loss initially), per-step/resource JSONL, run card, training manifest. Genuine
trained artifacts (reviewed rows + successful run) are distinct from smoke.

## Review tooling (`scripts/review_dataset.py`, data agent)

Human review CLI and HTML export per split/task/filter/family. Do not auto-accept
generated labels. Decisions match row hashes, reviewer, timezone-aware timestamp,
reject reason; accepted registry exported for existing loader. Preserve unknown
registry fields/decisions, reject duplicate/conflicting decisions and frozen test
changes. Deterministic pilot selects whole training/validation families, writes
same-schema task files/manifest; no test contents in pilot. Preserve origin hashes.
Inspect examples with independent rule checks; mark automated checks separately
from human review. Fixture smoke questions can use Nimbus handbook (curated).

## Scoring/metrics (`editguard/scoring.py`, `editguard/metrics.py`, eval agent)

Conservative accepted answer forms (subject aliases, requested relation, value,
unit, expected yes/no). Return {correct:bool|None, verdict, reason, endorsed_fact,
needs_review:bool}. Short model fragments without explicit endorsed subject or
relation remain review_required; no substring correctness. Examples mentioning
old/rejected target values must not pass. Wrong or contradictory answers fail;
unsupported prose -> review queue and counted uncredited until adjudicated.
Structured outputs scored separately and never ML generator evidence. Store raw
scoring results and allow hash-bound blinded manual decisions, no LLM-only judge.
Metrics: fixed denominators for errors/unavailable/abstentions, scope P/R/FPR and
Brier/ECE, retrieval recall@k, coverage, accuracy subsets, preservation vs factual
correctness, latency p50/p95, paired cluster bootstrap by family (only matched rows).
Ordinary tests use explicit synthetic outputs and stdlib; no model downloads.

Interface evaluator `score_answer(text, expected_fact, *, aliases=(),
expected_polarity=None, answer_kind='model_generated')` returns dict above.
`aggregate_predictions(records)` consumes rows {id,family_id,configuration,
applicable:bool,question_type,expected_fact,answer,status,answer_kind,score,
latency_ms,retrieval_hits:{k:bool},scope_probability?,scope_label?,selected_edit_id?}.
Return rates {numerator,denominator,value}, review/error/abstention counts and
subsets. `scope_metrics(labels, probabilities, threshold=.5)` standalone.
`paired_cluster_bootstrap(left,right,seed=42,resamples=2000)` paired id/family rows.

File ownership: root neural/comparison/API integration/config/docs; training agent
trainer/training support/tests; data agent review/pilot/tests/data-review docs;
evaluation agent scoring/metrics/CLI evaluation aggregation/tests only. Communicate
schema changes before editing another owner's files. No agent commits.
