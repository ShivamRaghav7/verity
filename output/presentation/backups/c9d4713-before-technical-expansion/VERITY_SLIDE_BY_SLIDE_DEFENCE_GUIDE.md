# Verity: slide-by-slide study and defence guide

Companion to **Verity_Mid_Project_Presentation.pptx**, 31 October 2026. This guide follows the thirteen-slide, ten-minute mid-project deck. The preceding deck and guide are preserved in `output/presentation/backups/72ff983-before-midproject/`.

Read this for understanding; rehearse from [the timed five-speaker script](VERITY_SPEAKER_SCRIPT.md). Presentation roles are independent of contribution ownership.

## Understand the project first

An AI assistant can answer from outdated or inappropriate reference information. Verity lets an operator publish approved facts, retrieve relevant records for questions, generate answers and inspect the information supplied. The system keeps versions, sources, scopes and audit events. A correction can be superseded or rolled back without rewriting old chat messages.

The current implementation is **a governed retrieval-augmented assistant evaluated with two frozen generators**. One is the author-published `YukinoKaren/ReasonKE_plus`; the other is `Qwen/Qwen2.5-7B-Instruct`. An approval changes external memory, not model weights. Our contribution is the integrated application, maintenance rules, retrieval adaptation, datasets, evaluation and diagnosis. Our own SFT/PPO adaptation is the next milestone. These fixed-system results are not a causal evaluation of training performed by our team.

A panel member may call the mechanism RAG. Answer directly, then explain the concrete governance and connected-question mechanisms. External-memory editing and parametric weight editing solve related maintenance problems through different mechanisms. The current system does not secretly rewrite parameters.

A website remains a good way to publish facts. The project is useful where an assistant already answers natural-language questions spanning changing records. It does not assume every company needs a chatbot or that a database lookup requires a language model.

## The ten-minute delivery

| Member | Slides | Clock | Focus |
|---|---|---|---|
| 1 | 1–2 | 00:00–02:00 | Objective and sourced Air India example |
| 2 | 3–4 | 02:00–04:00 | Architecture and connected retrieval |
| 3 | 5–8 | 04:00–06:00 | Lifecycle and recorded website walkthrough |
| 4 | 9–11 | 06:00–08:00 | Experimental design, findings and latency |
| 5 | 12–13 | 08:00–10:00 | Contribution, conclusion and training roadmap |

The walkthrough uses genuine recorded website states. It does not wait for GPU inference inside the ten-minute clock. Opening a stored conversation is replay; sending a new question causes fresh generation. An optional live demonstration can follow the timed talk.

## Slide 1 — Verity

**What:** controlled factual updates with source, history and reversal. An approved change should transfer to direct and connected questions while preserving other properties and historical scopes.

**Why:** retraining a large model after every fact changes is costly. Supplying approved reference records is practical, but introduces retrieval, association and state-management problems. Verity studies and exposes these components.

**How:** keep approved records outside the model; select bounded context; generate an answer; persist the response together with the supplied facts and revision.

There are two acceptance layers. Application checks test explicit approval, conflict rejection, supersession, transaction consistency, historical replay and rollback. Model evaluation measures correct final answers on assigned direct, paraphrased, connected and out-of-scope questions. Passing a database test is not model accuracy. A correct generated answer does not prove a transaction is atomic.

**Questions to expect**

- **Did you train the current model?** The deployed checkpoints are reused and frozen. SFT/PPO adaptation is the next milestone. Earlier separate experiments are not training evidence for this deployment.
- **Is this a new foundation model?** No. The completed contribution is the maintenance and experimental platform, including its retrieval adaptation.
- **What is the bounded task?** Short factual and connected questions over approved records, with declared context and memory limits. Arbitrary conversation is outside the accuracy guarantee.
- **What does success mean now?** A working maintenance workflow and reproducible evidence on the declared task; no universal perfect score is claimed.

## Slide 2 — Air India: one dated update, connected questions

### Verified real sources

The [CCI/PIB release of 20 December 2021](https://www.pib.gov.in/pressreleasepage.aspx?lang=2&prid=1783550&reg=48) describes Air India as government-owned. The [Ministry of Finance release of 27 January 2022](https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1792950&lang=2&reg=48) says the completed disinvestment transferred 100% of its shares to Talace Private Limited, a wholly owned subsidiary of Tata Sons Private Limited. [Tata's handover announcement](https://www.tata.com/newsroom/business/air-india-handover) corroborates it. The slide photograph is the genuine government announcement image.

The [official Tata contact page](https://www.tata.com/contact-us), checked on 31 October 2026, lists Tata Sons at Bombay House, Mumbai. That is a separately dated contact fact, not proof of its historical 2022 office address. [Tata's later merger release](https://www.tata.com/newsroom/business/air-india-vistara-merger-completion) records subsequent ownership arrangements. Therefore always say **the 2022 handover snapshot**, not present-day ownership.

### Trace the connected facts

The selected local reference initially describes December 2021. Its approved replacement describes the January 2022 handover. A separate historical government-ownership record stays available.

- Direct: “Who owns Air India in the selected reference snapshot?” → Talace after the update.
- Two edges: Air India → selected owner Talace → parent Tata Sons.
- Three edges: add Tata Sons → official contact city Mumbai.
- Historical control: “Who owned Air India before the January 2022 handover?” → Government of India.

A hop is one represented relation lookup. This chain is not evidence of unrestricted reasoning. The familiar example may be in model pretraining; it demonstrates product operation rather than unseen-entity generalisation.

**Why choose it?** It is familiar to an Indian panel, publicly sourced and clearly dated. It illustrates how updating one selected reference should affect connected questions while a different historical scope remains intact.

**Questions to expect**

- **Was the old government answer false?** It was correct for the old snapshot. The selected reference changes; the historical fact remains.
- **Does rollback undo the real handover?** It restores an earlier local memory state, not real-world history.
- **Does the system infer dates automatically?** The operator supplies scope and canonical identity. This is not a general temporal reasoning engine.
- **What if the model knows the answer already?** The product test still shows which authoritative reference was supplied and how it was maintained. It is not a claim that this familiar event necessarily produces a stale unassisted answer.

## Slide 3 — Implemented architecture

### One request, end to end

1. The browser submits a bounded question to the Python API. Mutations use a session token and origin checks.
2. The application captures the current memory revision under its operation lock and reads active approved records.
3. The selector constructs context. Compatible complete user/assistant history pairs are retained; older-revision conditioned messages are excluded from a new request.
4. The backend assembles the prompt and calls a separate authenticated GPU worker.
5. The worker generates output and extracts a final answer. The application saves the question/response pair, supplied fact identifiers, sources, model identity and revision transactionally.
6. The interface shows the answer and its evidence. Evidence records the supplied inputs; it is not a certificate for every generated sentence.

The operation lock prevents a question or edit from capturing a partially completed application operation. It spans generation; SQLite transactions remain short. These are local-process consistency mechanisms, not distributed consensus.

### Deterministic versus neural work

Validation, identity keys, version state, audit replay, bounds, lexical ranking and persistence are deterministic application operations. The frozen transformer generates tokens. Greedy decoding chooses the highest-scoring next token, but does not guarantee factual truth. The website refuses to count a deterministic fallback as a successful neural answer when its worker is unavailable.

### Actual configuration

| Setting | Recorded configuration |
|---|---|
| Reused generator | `YukinoKaren/ReasonKE_plus` |
| Revision | `ceaedc2ae7c8fed6257953587d991ce550db6106` |
| Comparison | `Qwen/Qwen2.5-7B-Instruct` |
| Comparison revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Parameter count, each | 7,615,616,512 |
| Inference | NF4 double quantisation; FP16 compute; SDPA |
| Limits | 4,096 input tokens; up to 512 new tokens |
| Decoding/hardware | Greedy; Colab T4; one model at a time |

Four-bit quantisation compresses stored weights. FP16 is the computation precision. Total runtime memory also includes activations, cache and overhead. A model fitting for inference does not establish its training requirements. Use actual resource receipts for numerical memory claims.

The `_plus` identity is exact. Its correspondence to a particular paper-headline score is undocumented; our deck uses our measured scores. The two models are alternative generators, not jointly trained modules. This is a research workspace with token/origin controls, not a hosted multi-user role-management service.

## Slide 4 — Retrieval across connected facts

### Why expand the query?

A question mentioning Air India may omit Talace or Tata Sons. Ranking only the original question can miss those bridge records. Supplying all memory includes support but adds distractors. The experiment separates those alternatives.

The iterative selector ranks approved records using the existing BM25-style lexical scorer and selects two seeds. It joins their stored answer text as a frontier, ranks remaining records against that frontier, adds the best unseen fact and continues from its answer. It stops at the cap, an empty remaining set or the round bound. If slots remain after the rounds, it fills them from the last frontier ranking. Ties follow the existing fact-ID order.

```text
selected = top two records ranked by question
frontier = answer text of selected records
for at most three expansion rounds:
    stop if four records selected or none remain
    next = best unseen record ranked by frontier
    add next; frontier = next.answer
fill unused slots from the last frontier ranking
return at most four original fact records
```

Normal retrieval receives no hidden target answer, gold support IDs or question-depth metadata. Its scores are relevance scores, not probabilities. It retrieves records; it does not deterministically compute the final answer.

The original operations study retains its two-stage selector. The depth study/application uses the iterative policy. These must not be relabelled as the same algorithm. On depth development questions, complete paths improved from **96/128 to 128/128** before final inference. This is retrieval development evidence, separate from final neural accuracy. Final depth support was **448/448 required occurrences** and **256/256 complete question paths**; original support was **320/320 occurrences** and **64/64 complete two-hop paths**.

A wrong seed can lead to an irrelevant chain. A four-fact budget may omit support in a larger memory. The current software guard allows at most 200 active records, but the benchmark measures 9- or 12-fact worlds, not accuracy at 200 records. More depth, larger memories and adversarial distractors belong in broader evaluation.

## Slide 5 — A correction has an explicit lifecycle

A record includes question, answer, readable scope, source and canonical identity `(entity_id, relation_id, scope_id)`. Versions retain lineage, supersession links and timestamps. The source enables inspection; the explicit operator approval gives the record authority in this workspace. Source text is not automatically verified by SQLite.

A partial unique index permits one active version per represented key. A replacement must name the active predecessor in the same key. A competing publication is rejected. Separate historical scopes can coexist. Two semantically conflicting statements assigned different keys may escape same-key detection, so identity/scope review matters.

Rollback groups withdrawal of the replacement, restoration of its predecessor, one revision increment and both audit events in `BEGIN IMMEDIATE`. All commit or all roll back. Injected late-event failure tests check this transaction boundary. Repeated/obsolete operations have explicit preconditions rather than blindly overwriting newer state.

**Questions to expect**

- **Can a customer make a fact authoritative?** Customer messages remain conversation input. Explicit publication is required.
- **Does rollback erase messages or events?** Earlier versions and messages remain inspectable with their original revisions.
- **Why not two separate updates?** An interruption between withdrawal and restoration could leave inconsistent state. One transaction prevents that partial outcome.
- **Does an export checksum prove truth?** It detects corruption. It does not establish truth or authorship, and a malicious actor could recompute it.

## Slide 6 — Website: approve the sourced update

The screenshot is the genuine Verity editor with the Air India question, Talace answer, dated PIB source and explicit approval control. It shows a prepared draft. The recorded workflow publishes through the same application API; a screenshot itself does not execute a change.

Navigate through **Knowledge**, locate the selected-reference owner and choose **Publish a new version**. Inspect answer, scope, source and canonical identity. **Approve & replace version** names the predecessor and activates the replacement. Merely typing draft text changes no authoritative fact.

The isolated sourced workspace starts with five records: selected owner, historical owner, Talace parent, Tata Sons contact city and handover date. `scripts/prepare_sourced_demo.py` creates it only in a fresh directory. `scripts/run_sourced_demo.py` records seven actual requests, saves every outcome and exports initial, updated, rollback and active snapshots. Do not load the separate fictional starter handbook into this sourced demo.

This run completed seven target-matching answers, conflict rejection and restored-predecessor rollback. It is acceptance evidence on selected examples, not a new held-out real-world accuracy estimate. The complete receipt preserves raw outputs and context.

## Slide 7 — Website: ask a connected question

The actual recorded question asks for the parent of Air India's owner in the selected snapshot. The supplied records link Air India to Talace and Talace to Tata Sons. The displayed answer is genuinely generated; it has not been replaced with a painted target string.

Use **Chat** to open the recorded conversation. Inspect the response and its supplied-source section. A saved message's revision can differ from the current header after later edits. That difference describes history. A fresh Send uses the current reference state and compatible history.

The receipt also includes the three-hop official-contact-city question and historical control. The talk uses the parent question for readability. Familiar sourced examples remain separate from the controlled benchmark totals.

If the GPU worker is unavailable during the panel, recorded views still show what actually ran. State that fresh generation needs the worker. Do not describe replay as fresh inference or source retrieval as independent factual verification.

## Slide 8 — Website: history and reversible state

Open **Knowledge → Approval and rollback history**. The sourced workflow approves a replacement, attempts a competing publication, rolls back and reapplies the handover for the active demonstration. The competing attempt returns HTTP 409 without changing the revision. Rollback restores the owned predecessor and records the withdrawal/restoration pair.

The active demo can therefore show the handover state while the audit contains an earlier rollback. Explain the chronological sequence. The receipt contains each state snapshot. Rollback restores the local selected reference, not the real historical event.

Historical comparison can generate answers against an earlier approved revision and the current revision using the same checkpoint and matched compatible history. It is not two model weight snapshots. Stored responses remain stored; comparison requests are fresh inference.

The engineering depth is consistency: lineage validation, audit replay, import validation, rejection of overwrite into a nonempty workspace and all-or-nothing event/state writes. These are separate from generated-answer correctness.

## Slide 9 — Experimental design

### Populations and splits

Original operations uses fictional 9-fact worlds, up to two hops. The separate depth challenge uses 12-fact worlds, up to three hops. Each study has 48 worlds: 16 development and 32 final lineages. Eight variants per lineage yield 128 development and 256 final questions.

A lineage groups an underlying world/edit and related question variants. Splitting before augmentation prevents paraphrases of one fact appearing across development and final partitions. Labels and graph structure were automatically audited; independent human semantic review is not claimed. Small memories, templated wording and controlled ground truth make reproducibility easier and limit external validity.

The existing final tests have now been inspected. Future training must use a fresh frozen test, with these released tests retained as regression evidence.

### Actual matched conditions

| Code | Generator | Context |
|---|---|---|
| R0 | ReasonKE_plus | No supplied corrections; diagnostic reference |
| R-RAG | ReasonKE_plus | Retrieved approved current facts |
| R-ALL | ReasonKE_plus | All current facts in the bounded original world |
| G-RAG | Qwen2.5-7B | Same retrieval/current information |
| G-ALL | Qwen2.5-7B | All current facts |
| Oracle | Each generator | Gold support on a declared 16-question diagnostic subset |

Original workload: `5 × 256 + 2 × 16 = 1,312` generations. Depth: `2 × 256 + 2 × 16 = 544`. Main denominators remain 256 per condition. Oracle subsets are diagnostic, not extra main successes. Updated full-memory baselines receive current facts; they are not deliberately stale.

### Scoring and uncertainty

Accuracy = correct endorsed final answers / **all assigned questions**. UNKNOWN abstentions count as unsuccessful for accuracy. The frozen scorer checks extracted final spans against reviewed/declared aliases; merely mentioning the right value as a rejected possibility is insufficient. Strict mismatches remain visible.

Preservation asks whether reference-correct locality answers stay correct. Factual locality accuracy asks whether all out-of-scope answers are true. Preserving an already wrong answer cannot establish correctness.

Paired bootstrap intervals resample whole lineages while keeping each system's outcomes paired. Variants from one world are related; treating them as independent would overstate precision. Such intervals describe sampling uncertainty within this design, not every uncertainty in real deployment.

## Slide 10 — Measured answers on controlled fact worlds

| Retrieved configuration | Original | Depth |
|---|---:|---:|
| ReasonKE_plus | 256/256 | 256/256 |
| Qwen2.5-7B | 241/256 | 248/256 |
| Qwen abstentions | 15 | 8 |

The perfect observed scores mean all assigned final answers passed the scorer on these controlled worlds. They do not imply universal accuracy, production reliability or guaranteed success on unfamiliar unrestricted language. Do not silently pool the studies into a claim about a different population.

The paired retrieved difference is **+5.86 percentage points**, 95% lineage interval **[3.91, 8.20]**, in original operations; depth is **+3.13 points [1.56, 5.08]**. These compare fixed systems, not the causal effect of an upstream training stage.

### Diagnose the failures

Original full-memory ReasonKE_plus scored **231/256**. Its 25 wrong final values all matched another supplied fact value. Fifteen matched Pro-owner values; others were neighbouring owner, release, office, contact or hosting values. In two cases an expected value appeared in an intermediate box but the final endorsed answer was wrong. This demonstrates why a target appearing somewhere in a response is not enough.

Original retrieved Qwen's 15 nonsuccesses were UNKNOWN. Depth Qwen's eight were UNKNOWN: seven in the direct two-hop template category and one in a three-hop paraphrase category. Hop count alone does not determine difficulty; wording and response behaviour matter. None of those eight is in the oracle subset, so that small subset cannot individually explain them.

Original full-memory Qwen has 13 strict mismatches and two abstentions. The released updated-method audit covers **63 nonsuccesses**: `25 + 15 + 13 + 2 + 8`. All have complete required support and no recorded budget-hit flag. Missing supplied support is therefore not the immediate explanation for these cases. The audit identifies observable wrong associations and abstention, not an unmeasured internal neural cause.

**Questions to expect**

- **Why can more context hurt?** Full memory adds distractors. Availability of the right value and correct association to the requested relation are different.
- **Is UNKNOWN a wrong fact?** It is an abstention, which fails the answer-accuracy task but avoids asserting a false value. Report both categories.
- **Does 100% look suspicious?** Small controlled worlds can be easy. State the population, preserve every receipt and expand a fresh challenge; do not manufacture failures.
- **Where are training losses?** These are frozen-checkpoint inference comparisons. Training curves belong to the planned adaptation study.

## Slide 11 — The accuracy–latency tradeoff

Depth warm timing includes all 256 assigned outcomes per arm, including abstentions:

| Generator + iterative retrieval | p50 | p95 |
|---|---:|---:|
| ReasonKE_plus | 15.94 s | 19.28 s |
| Qwen2.5-7B | 1.12 s | 1.38 s |

p50 is the median. p95 describes slower requests: 95% of measured latencies lie at or below that quantile. The timing covers selection, local IPC and generation. It excludes downloads, cold loading and browser-network delay. It is not a hosted-service latency promise.

ReasonKE_plus answers eight additional depth requests correctly but takes longer. Median output lengths are 212 tokens versus six, consistent with a cost from longer responses. This is not a causal decomposition of all latency sources. A ratio of these medians is not a measured paired per-request speedup.

**Which is best?** It depends on answered coverage versus response-time priorities. The measured conclusion is a tradeoff on this workload, not a universal winner. Training is not promised to improve both.

## Slide 12 — Our engineering contribution

Completed strands:

1. Governed memory: canonical identity, source/scope, approval, version links, conflicts, audit replay and atomic rollback.
2. Context handling: bounded retrieval, answer-text expansion and revision-compatible history.
3. Evidence: controlled data, lineage splits, fixed protocols, per-request ledgers, scoring, paired uncertainty and failure diagnosis.
4. Delivery: chat, correction editor, source/history inspection, results explorer, export/restore, regression tests, CI, containers and notebooks.

Original retrieved ReasonKE_plus retained **124/124 reference-correct locality answers**. Its factual locality accuracy was **128/128**. Four locality questions were incorrect in the reference condition, so those denominators differ. Preservation and factual correctness must remain separate.

The current conclusion is definite: the implemented platform completes the maintenance workflow and performs accurately on its measured bounded task, with an observed accuracy–latency tradeoff. The contribution is inspectable code and evidence, not ownership of upstream weights.

The script ends with an evidence-based five-person responsibility allocation. Each member should inspect/reproduce the assigned components and record actual work before attaching their name. Presentation assignments do not prove historical implementation ownership. Do not invent a retrospective research journey or labour history.

## Slide 13 — Next Milestone: Model Training and Broader Evaluation

The new question is whether **our own adaptation** improves transfer and robustness under matched conditions. That requires a common verified parent checkpoint, reviewed data and fresh evaluation, rather than relabelling two unrelated frozen checkpoints as training stages.

1. Construct source-grounded edits, connected questions, difficult negatives and reviewed intermediate steps. Include similar names, wrong relations, older scopes, ambiguity and missing support.
2. Split entities/edit lineages before wording augmentation. Keep fresh development and final partitions; existing tests become regression evidence.
3. Profile a compact pilot before choosing adapter settings, total run size or GPU budget. Verify checkpoint/resume behaviour.
4. Perform SFT from the common instruction checkpoint. Supervised fine-tuning learns from input/target pairs; use development evidence for selection.
5. Begin PPO from the selected SFT stage, using a declared process/final-answer reward contract. PPO is a constrained policy update from generated-response feedback. Validate rewards against formatting shortcuts and plausible but wrong intermediate steps; reward is not a truth certificate.
6. Compare instruction baseline, SFT and SFT+PPO with matched data access, retrieval, inference budgets and scoring. Keep reused ReasonKE_plus as a separate reference.
7. Evaluate new wording/entities, larger distractor memories, greater depth, locality, abstention, latency and resources. Preserve negative outcomes.

The goal is an interpretable comparison, not a guaranteed gain. Before a substantial user-run training command, provide the concrete question, inputs, expected artifacts, measured profile-based time/memory estimate with uncertainty, resume behaviour and finite stopping rule. No speculative runtime is promised.

See [the focused research roadmap](VERITY_MIDPROJECT_RESEARCH_ROADMAP.md). The slide deliberately avoids the upstream training tutorial and does not imply future stages have run.

## Quick panel answers

**“So it is RAG?”** Yes, the current answering mechanism is retrieval augmented generation. Verity adds governed memory, connected retrieval, revision-aware evidence and reversible maintenance. The next milestone studies training separately.

**“Why use a model over structured facts?”** Deterministic structured answers are useful for direct lookups. The measured model task is natural-language factual/connected answering. A database-template answer must not count as neural accuracy.

**“Can sources verify any answer?”** They identify what was supplied. A model can still associate the wrong property or generate unsupported prose.

**“Why not report the upstream score?”** It comes from another dataset and protocol. We identify the reused checkpoint and report our actual released receipts.

**“What would you improve first?”** Independently reviewed challenge wording, larger/adversarial memory and a matched adaptation pilot. These address the evidence's real limits while preserving the working baseline.

**“Can every fact be edited successfully?”** A valid represented record can be approved/versioned under tested rules. Retrieval and generation remain empirical components; arbitrary natural-language success is not guaranteed.

## Evidence map

- Architecture/transactions: [current architecture](../CURRENT_ARCHITECTURE.md), `editguard_chat/server.py`, `store.py`, `reasonke_backend.py`.
- Selectors: `editguard_benchmark/retrieval.py`, `editguard_depth/retrieval.py`.
- Checkpoints: `configs/reasonke-model.json`, `configs/instruction-model.json`.
- Sourced demo: `data/demos/air-india-handover.json`; prepare/run scripts; `output/presentation/midproject/air-india-demo-receipt.json`.
- Frozen protocols/results: `reports/reasonke/operations-final/`, `reports/reasonke/depth-final/`.
- Retrieval development: `reports/reasonke/depth-development/`.
- Failure audit: `output/presentation/analysis/failure-audit.json`.
- Inventory: [contributions and evidence](../CONTRIBUTIONS_AND_EVIDENCE.md).
- Exact order/speech: [timed script](VERITY_SPEAKER_SCRIPT.md); canonical `verity-midproject-content.json`.

When answering, state the mechanism, the population and denominator, then what the evidence establishes. That is more defensible than a larger-sounding label.
