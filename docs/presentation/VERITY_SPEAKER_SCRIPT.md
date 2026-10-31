# Verity: timed five-speaker technical script

**Ten minutes: 16 main slides, five two-minute slots. Four hidden Q&A backups are outside the clock.**

Read the quoted speech. Notes and questions are rehearsal material. Rehearse at approximately 125 words/minute, including pauses and handovers. Do not add a separate greeting, live model download or long generation inside the timed slot. Website views are genuine recorded states; optional fresh generation follows the talk.

## Speaking order

| Member | Clock | Main slides | Focus |
|---|---|---|---|
| 1 | 00:00–02:00 | 1–3 | Objective and request architecture |
| 2 | 02:00–04:00 | 4–6 | Transactions, retrieval and development finding |
| 3 | 04:00–06:00 | 7–9 | Data, matched conditions and original outcomes |
| 4 | 06:00–08:00 | 10–12 | Depth, failure diagnosis and preservation |
| 5 | 08:00–10:00 | 13–16 | Latency, product, conclusion and roadmap |

## Member 1

### Slide 1: Verity

**00:00–00:20 · 20 seconds · 37 spoken words**

> Verity studies how an assistant can use approved factual updates across connected questions while keeping its knowledge state inspectable and reversible. We will explain the implementation, the experiments and six findings, then define the next training milestone.

Pacing: approximately 2.2 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Current system: governed external memory, bounded deterministic retrieval and alternative frozen generators. Corrections alter approved memory; they do not update the deployed weights. The reused ReasonKE_plus checkpoint is an upstream model, while our contribution is the maintenance, retrieval, application and evaluation system. Technical depth comes from implemented mechanisms and evidence, not a claim of a new trained checkpoint.

Why this slide: A maintenance system must coordinate authority, retrieval, generation and history. Measuring only a direct edited answer would miss connected failures and state inconsistencies.

Evidence:

- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)
- [docs/CONTRIBUTIONS_AND_EVIDENCE.md](../../docs/CONTRIBUTIONS_AND_EVIDENCE.md)

### Slide 2: A factual change can affect a chain

**00:20–01:05 · 45 seconds · 79 spoken words**

> Air India gives a familiar motivation. The completed handover on twenty-seven January 2022 transferred its shares to Talace, a Tata Sons subsidiary. An older government-ownership record and that handover record answer different dated questions. Connected questions must follow the applicable owner to its parent, while unrelated facts stay intact. Our bounded objective is consistent reference updating: explicit approval, correct direct and connected answers, locality preservation and atomic rollback. The example motivates the problem; our controlled experiments measure the system.

Pacing: approximately 7.1 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Government/CCI reporting on 20 December 2021 describes government ownership. The Ministry of Finance/PIB release of 27 January 2022 documents completion and Talace’s Tata Sons parent. This is a historical handover snapshot, not present-day shareholding; subsequent arrangements changed. The previously recorded sourced demo also links the parent to its official contact city, checked separately on 31 October 2026. The current introduction does not use a current contact page to infer a historical address. The official photograph is from the PIB handover release. Application acceptance and neural accuracy are separate criteria: transaction tests establish state consistency, while assigned-question evaluation establishes observed answer quality.

Why this slide: One dated change makes scope and connectedness understandable before introducing the general architecture. It is an introduction, not the experimental dataset or an unseen-entity claim.

Evidence:

- [https://www.pib.gov.in/pressreleasepage.aspx?lang=2&prid=1783550&reg=48](https://www.pib.gov.in/pressreleasepage.aspx?lang=2&prid=1783550&reg=48)
- [https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1792950&lang=2&reg=48](https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1792950&lang=2&reg=48)
- [https://www.tata.com/newsroom/business/air-india-handover](https://www.tata.com/newsroom/business/air-india-handover)
- [data/demos/air-india-handover.json](../../data/demos/air-india-handover.json)

### Slide 3: Revision-consistent request architecture

**01:05–02:00 · 55 seconds · 86 spoken words**

> The API captures a memory revision under an application operation lock. It retrieves active approved records and keeps only compatible conversation pairs. A private GPU worker generates the answer. The application stores raw output, extracted answer, supplied fact identifiers, model identity and revision together. This separates deterministic state management and retrieval from neural generation. We compare two frozen seven-billion-class checkpoints with pinned identities, four-bit weights, FP16 computation and matched input and output ceilings. Source evidence records the model inputs; it does not automatically certify generated prose.

Pacing: approximately 13.7 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: The browser, loopback application API, SQLite store and authenticated GPU worker are separate components. The operation lock spans generation, preventing a concurrent approval/export from capturing a half-completed application operation; short SQLite transactions persist state and message pairs. Current-revision compatible complete history pairs are retained; old correction-conditioned context is excluded. Historical comparisons use matched pre-question history and exclude correction-conditioned pairs from both sides. Worker failure is reported explicitly rather than replaced with a template counted as neural success. ReasonKE_plus and Qwen2.5-7B-Instruct each contain 7,615,616,512 parameters. Both use NF4 double quantisation, FP16 compute, SDPA, greedy decoding, 4,096 input tokens and up to 512 new tokens on T4, one model at a time.

Why this slide: Capturing revision and evidence lets us diagnose whether a bad answer used wrong state, missing support or complete support but wrong generation. The process boundary keeps the lightweight application independent of GPU loading.

Evidence:

- [editguard_chat/server.py](../../editguard_chat/server.py)
- [editguard_chat/store.py](../../editguard_chat/store.py)
- [editguard_chat/reasonke_backend.py](../../editguard_chat/reasonke_backend.py)
- [editguard_reasonke/runtime.py](../../editguard_reasonke/runtime.py)
- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [configs/instruction-model.json](../../configs/instruction-model.json)

## Member 2

### Slide 4: Fact identity and atomic rollback

**02:00–02:45 · 45 seconds · 77 spoken words**

> A represented fact is keyed by entity, relation and scope. A partial unique index permits one active version per key. Replacement names the active predecessor. A competing publication returns conflict instead of silently choosing a value. Rollback withdraws the replacement, restores its predecessor, increments the revision once and writes both audit events inside one immediate transaction. Fault-injection tests check that a late event-write failure restores the entire previous state. Historical versions and messages remain available for inspection.

Pacing: approximately 8.0 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Canonical identity is the tuple (entity_id, relation_id, scope_id), normalised by the store. Alias metadata is not a learned resolver. Legacy records use normalised question/scope identity. Approval stores answer, provenance, scope, lineage, version and links. BEGIN IMMEDIATE groups updates, revision and events. An active replacement must own the direct predecessor in the same key; obsolete rollback cannot overwrite a newer unrelated state. Restore refuses a competing active version. Event replay reconstructs the approved set at historical revision r. User and assistant message insertion is separately atomic. Same-key conflict rejection does not solve arbitrary semantic contradiction under distinct operator-assigned keys.

Why this slide: Without a single transaction, interruption between withdrawal and restore could produce missing knowledge or an audit/state disagreement. A visible version chain makes recovery reviewable.

Evidence:

- [editguard_chat/store.py](../../editguard_chat/store.py)
- [editguard_chat/workspace.py](../../editguard_chat/workspace.py)
- [tests/test_chat_store.py](../../tests/test_chat_store.py)
- [tests/test_chat_workspace.py](../../tests/test_chat_workspace.py)

### Slide 5: Bounded answer-frontier retrieval

**02:45–03:30 · 45 seconds · 80 spoken words**

> Our ranker uses a BM25-style lexical score over question, scope and answer text. We select two seeds. Their answer text forms the search frontier; each expansion adds the highest-ranked unseen record and advances that frontier. The iterative policy stops at four records, with at most three rounds. Neither gold support identifiers nor expected answers enter normal retrieval. This makes connected selection transparent and independently measurable. The original study keeps its two-stage selector; the separate depth challenge evaluates the iterative adaptation.

Pacing: approximately 6.6 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Tokenisation lowercases alphanumeric words and removes the fixed stop list. The score uses k1=1.2 and b=0.75 with log(1+(N-df+0.5)/(df+0.5)); query tokens are a set. Ties use fact ID. Original retrieval chooses two question-ranked seeds, re-ranks using original question plus seed answers and fills to four. Iterative retrieval chooses two seeds, searches from their joined answers, adds the best remaining record each round, then optionally fills unused slots from the last frontier. Scores are relevance values, not confidence probabilities. The active-memory guard is 200 records, but measured worlds are 9 or 12 records. With this unoptimised small-memory scorer, each ranking recomputes document frequencies; the cap makes the chosen workload finite rather than establishing large-scale performance.

Why this slide: A connected question may not mention bridge entity names. Answer-text expansion introduces those names without using evaluation targets. Separating retrieval from generation prevents crediting a generator for support that a normal retriever could not find.

Evidence:

- [editguard_benchmark/retrieval.py](../../editguard_benchmark/retrieval.py)
- [editguard_depth/retrieval.py](../../editguard_depth/retrieval.py)
- [reports/reasonke/depth-development/retrieval-selection.json](../../reports/reasonke/depth-development/retrieval-selection.json)

### Slide 6: Retrieval development recovered three-hop support

**03:30–04:00 · 30 seconds · 50 spoken words**

> On 128 development questions, the original selector completed 96 paths and missed every three-hop path. Iterative expansion completed all 128, including all thirty-two three-hop paths. Selection used the declared candidate family before neural profiling and final freeze. This is a retrieval improvement, not an accuracy gain inferred from model outputs.

Pacing: approximately 6.0 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: The baseline found 192/224 required fact occurrences, completing 96/128 questions. Each depth group has 32 questions; the baseline completes [32,32,32,0] and the selected iterative policy [32,32,32,32]. All six declared seed/cap candidates completed the packet. Before neural profiling, two separate display-development states revealed a one-seed distractor; the final two-seed/four-record choice completed packet and display supports. The displayed chart covers the 128-question packet only. Final depth retrieval later supplied 448/448 required occurrences with 256/256 complete question paths, including 192/192 occurrences for 64 three-hop paths. These are shared selector outcomes, not independent duplicate samples for each model.

Why this slide: The graph isolates a mechanism change with an explicit support metric. It explains what the retrieval adaptation fixed before looking at generated answers.

Evidence:

- [reports/reasonke/depth-development/retrieval-selection.json](../../reports/reasonke/depth-development/retrieval-selection.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)

## Member 3

### Slide 7: Controlled worlds and leakage controls

**04:00–04:40 · 40 seconds · 74 spoken words**

> We constructed two controlled fictional handbook studies with explicit ground truth. Each has forty-eight edit lineages, split into sixteen development and thirty-two final worlds before question variants. The original uses nine facts and reaches two hops; the depth challenge uses twelve facts and reaches three. Each final condition has 256 assigned questions. Labels were automatically audited. The split protects against sharing an edit’s variants across partitions, while controlled wording and small memories remain limits.

Pacing: approximately 4.5 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Each study has 384 authored question records: 128 development and 256 final, eight variants per lineage. Original worlds separate current/previous ownership, related properties, other tier, other entity and version controls. The depth study adds a contact-office-city relation for a third edge. Template and relation vocabulary can repeat; the depth challenge uses separate development/final wording banks, while the original shares template families. Automated structural/graph review is not independent human semantic review. Final configurations and scorer identities were frozen before final inference. Because the released tests are now inspected, the next training study requires fresh unseen held-out material. The sourced introduction/demo is a separate population, not the dataset behind these charts.

Why this slide: Controlled worlds provide known answers for direct, paraphrased, connected and locality questions. Splitting the underlying edit rather than individual rows prevents paraphrase leakage and supports lineage-based uncertainty.

Evidence:

- [docs/OPERATIONS_DATASET_CARD.md](../../docs/OPERATIONS_DATASET_CARD.md)
- [editguard_benchmark/data.py](../../editguard_benchmark/data.py)
- [editguard_depth/data.py](../../editguard_depth/data.py)
- [reports/reasonke/operations-final/protocol.json](../../reports/reasonke/operations-final/protocol.json)
- [reports/reasonke/depth-final/protocol.json](../../reports/reasonke/depth-final/protocol.json)

### Slide 8: Matched conditions and strict scoring

**04:40–05:20 · 40 seconds · 76 spoken words**

> The initial ReasonKE reference receives the old full handbook. Updated conditions receive either retrieved current facts or the full current handbook; Qwen uses the same approved information. Every assigned question stays in the denominator, including UNKNOWN. The scorer checks the endorsed final span against declared aliases, rather than accepting any target mention. Original and depth studies contain 1,312 and 544 generations respectively, including smaller oracle diagnostics. These compare fixed systems, not stages of our own training.

Pacing: approximately 3.5 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: R0 is the initial-full-memory stale reference, confirmed by request_for in editguard_benchmark/run.py. It is not the application’s empty-memory comparison mode. R-RAG/R-ALL use ReasonKE_plus with updated retrieved/full context; G-RAG/G-ALL use Qwen with matching retrieval or updated full context. The original has five main arms ×256 plus two preselected oracle arms ×16 =1,312. Depth has two ×256 plus two ×16=544. Fact IDs match across generators in retrieved conditions. Model-appropriate response-format prompts differ, so describe matched information and budgets rather than identical prompts. Accuracy=correct/all assigned. UNKNOWN, execution error and unexecuted outcomes get zero primary credit. Oracle support uses gold IDs only on the diagnostic subset; it is not arbitrary-chat retrieval. Whole-final-span scoring is stricter than accepting a correct token anywhere.

Why this slide: Fairly updated baselines distinguish the benefit of current information, selection policy and checkpoint behaviour. Fixed denominators prevent abstention from making accuracy look artificially high.

Evidence:

- [editguard_benchmark/run.py](../../editguard_benchmark/run.py)
- [editguard_benchmark/models.py](../../editguard_benchmark/models.py)
- [editguard_benchmark/scoring.py](../../editguard_benchmark/scoring.py)
- [editguard_benchmark/metrics.py](../../editguard_benchmark/metrics.py)
- [editguard_depth/metrics.py](../../editguard_depth/metrics.py)

### Slide 9: Original study: context changes the outcome

**05:20–06:00 · 40 seconds · 64 spoken words**

> With updated retrieved facts, ReasonKE_plus scored 256 of 256, compared with 231 using full updated memory. Qwen scored 241 in both modes, but retrieval changed its failures from mostly wrong final spans to fifteen UNKNOWN responses. The initial handbook reference scored 136. Current information matters, yet more context is not automatically better. This chart preserves every assigned outcome and distinguishes wrong answers from abstention.

Pacing: approximately 9.3 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Original main outcomes: R0 136 correct/120 wrong; R-RAG 256/0; R-ALL 231 correct/25 wrong; G-RAG 241 correct/15 UNKNOWN; G-ALL 241 correct/13 strict wrong/2 UNKNOWN. All have 256 assigned, zero execution errors and zero unexecuted. R0 is a staleness diagnostic, not a deliberately stale updated RAG baseline. The paired R-RAG minus G-RAG difference is +5.859375 percentage points, 95% lineage-bootstrap interval [3.90625,8.203125]. R-ALL minus R-RAG is −9.765625 points, interval [−13.671875,−6.25]. The former compares fixed generators/configurations and the latter a same-checkpoint context policy. Neither isolates the causal effect of upstream training. Perfect observed scores are bounded by 32 small final worlds and controlled wording.

Why this slide: Outcome composition answers more than an aggregate success percentage. Equal accuracy can conceal different wrong-versus-abstain behaviour, while full memory reveals distractor sensitivity.

Evidence:

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)

## Member 4

### Slide 10: Depth challenge: wording still matters

**06:00–06:35 · 35 seconds · 69 spoken words**

> The separate depth challenge has sixty-four questions per group. ReasonKE_plus answered every group correctly. Qwen answered 64, 64, 57 and 63 across locality, one, two and three hops. Its eight nonsuccesses were UNKNOWN. Seven cluster in one direct two-hop wording family; only one occurs in a three-hop paraphrase. Greater hop count alone therefore does not explain difficulty. The three-hop difference is one question, with an interval that includes zero.

Pacing: approximately 1.9 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Depth totals are 256/256 versus 248/256. Paired advantage +3.125 points has 95% lineage interval [1.5625,5.078125]. On three-hop questions alone, +1.5625 points has interval [0,4.6875]. The seven two-hop abstentions are in a direct-template subgroup with 9 assigned and 2 correct; other direct two-hop templates have 23/23 correct. Three-hop paraphrases have 31/32 correct. These are frozen, post-hoc wording subgroups, not a randomised causal experiment. None of the eight failures belongs to the 16-question oracle subset; its 16/16 results cannot explain these cases individually. Final iterative retrieval supplied complete support to every main depth question.

Why this slide: A depth breakdown prevents the misleading inference that three-hop performance proves all simpler wording is solved. It also scopes the perfect observed score to the controlled challenge.

Evidence:

- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)

### Slide 11: Failure audit: the right facts, wrong association

**06:35–07:25 · 50 seconds · 82 spoken words**

> We traced all twenty-five full-memory ReasonKE failures. Each had complete support, but its wrong final value matched another supplied record. Fifteen selected a Pro-service owner; other errors crossed version, property or contact boundaries. For Garnet Relay, the requested owner was Silver unit 77, but the model endorsed the Pro-service value Saffron unit 137. Across the released updated conditions, all sixty-three nonsuccesses had complete support and no recorded budget-hit flag. Retrieval success is necessary here, but not sufficient for a correct generated association.

Pacing: approximately 10.6 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: R-ALL off-target roles are Pro-service owner 15, release-2 owner 3, unrelated office city 3, previous-owner contact 2, hosting region 1 and current-owner contact 1, summing to25. The concrete Garnet case is R-ALL:ops-006-direct; requested f1 is Silver unit 77, selected wrong value matches f5 Saffron unit 137. This is observed output/value association, not proof of an internal attention mechanism. Two R-ALL cases mention an expected intermediate boxed value but endorse a wrong final. The 63 updated-method nonsuccesses comprise 25 R-ALL wrong +15 G-RAG UNKNOWN +13 G-ALL strict mismatches +2 G-ALL UNKNOWN +8 depth G-RAG UNKNOWN. G-ALL includes10 off-target values,2 correct owner/contact chains with extra final text and1 contaminated chain. Their frozen scores stay unchanged. R0 is excluded because it is the stale-reference diagnostic.

Why this slide: A complete support path does not ensure the model selects the requested subject/relation. Auditing raw values and exact record IDs turns a small failure count into a concrete engineering finding.

Evidence:

- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)
- [output/presentation/analysis/failure-ledger.csv](../../output/presentation/analysis/failure-ledger.csv)
- [reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md](../../reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md)

### Slide 12: Preservation and factual correctness differ

**07:25–08:00 · 35 seconds · 68 spoken words**

> The initial reference got 124 of 128 locality questions right. Retrieved ReasonKE_plus retained all 124 and answered all 128 correctly. Full memory retained 117 of the 124, with 119 of 128 correct overall. The denominators distinguish retaining previously correct answers from factual correctness on all out-of-scope questions. Preserving an old mistake is not factual correctness. The depth study has no separate initial reference for a new preservation estimate.

Pacing: approximately 2.4 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Original R-RAG baseline-correct retention124/124=100%, locality factual accuracy128/128=100%. R-ALL retention117/124=94.3548%, factual locality119/128=92.96875%. R0 locality124/128=96.875%. Four reference locality mistakes are excluded from the preservation denominator but included in factual correctness. Qwen original locality is128/128 retrieved and126/128 full (one wrong and one abstain). Depth locality is64/64 for both generators; there is no depth R0 arm, so do not infer a separately measured depth retention rate. These are dataset-defined locality controls, not proof that arbitrary unrelated knowledge is unchanged.

Why this slide: The distinction prevents a common editing-evaluation error: rewarding a system merely for retaining an answer that was already incorrect.

Evidence:

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [editguard_benchmark/metrics.py](../../editguard_benchmark/metrics.py)

## Member 5

### Slide 13: Higher answer coverage has a latency cost

**08:00–08:35 · 35 seconds · 66 spoken words**

> Depth median latency is fifteen point nine four seconds for ReasonKE_plus and one point one two for Qwen; the ninety-fifth percentiles are nineteen point two eight and one point three eight. ReasonKE_plus answered eight more questions, with median output length 212 tokens versus six. This exposes the accuracy–latency tradeoff. Timing includes selection, local communication and generation, including abstentions, but excludes cold loading and browser network delay.

Pacing: approximately 3.3 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: The chart uses depth final generated_latency for 256 assigned generated responses per condition. p50/p95: Reason15.93582438600015/19.282786326999826 seconds; Qwen1.1189718869995886/1.3847337005004192. Original retrieved timings are15.056786640499883/16.71352412099983 and0.9161561650001246/1.1980873600000268. Median output tokens are204/5.5 original and212/6 depth. Longer output is consistent with increased decoding time, but does not causally decompose every latency source. A ratio of separate medians is not a paired per-question speedup. Resource receipts distinguish PyTorch allocation from total device usage; inference fit does not imply training fit.

Why this slide: Accuracy alone is insufficient for a user-facing assistant. The measured workload exposes a practical response-time cost for the higher-coverage configuration.

Evidence:

- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)

### Slide 14: Product: evidence remains inspectable

**08:35–09:00 · 25 seconds · 44 spoken words**

> Chat retains model and revision evidence. Knowledge manages versions and rollback. The results explorer exposes wrong answers and their supplied facts. This genuine recorded view shows the full-memory failure cohort. Stored responses are distinct from fresh generation. An optional walkthrough can follow the talk.

Pacing: approximately 3.9 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: The main screenshot is the existing genuine Verity-branded results explorer, captured30 October2026 from released evidence. It filters25 original R-ALL wrong records and expands a wrong-release case. No pixel editing or new model call is needed. Stored responses retain the original generation/revision; a header may show a newer current state. Historical comparison invokes fresh requests with the same checkpoint and matched pre-question history against an earlier approved revision and current context. It does not compare trained weight snapshots. Hidden backup 20 contains the recorded correction editor and rollback-history interface. Existing sourced Air India acceptance is7 selected requests, separate from the controlled benchmark.

Why this slide: A filterable receipt view makes an aggregate graph auditable. The product supports investigation as well as a successful demonstration.

Evidence:

- [output/presentation/screenshots/technical-v2/provenance.json](../../output/presentation/screenshots/technical-v2/provenance.json)
- [editguard_chat/results.py](../../editguard_chat/results.py)
- [editguard_chat/static/app.js](../../editguard_chat/static/app.js)
- [output/presentation/midproject/air-india-demo-receipt.json](../../output/presentation/midproject/air-india-demo-receipt.json)

### Slide 15: Engineering contribution and current conclusion

**09:00–09:25 · 25 seconds · 45 spoken words**

> Our contribution combines transactional memory, connected retrieval, revision-aware chat, recoverable inference and failure diagnosis. The measured conclusion is clear: selected current context supports accurate bounded answers; full memory exposes association failures; higher answer coverage costs latency. This establishes a working baseline for the next study.

Pacing: approximately 3.4 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: Completed work spans memory/store invariants, retrieval, model transport, context/history policy, controlled datasets, fixed protocols, checkpointed per-request execution, source/config/data identities, strict scoring, paired uncertainty, raw-output inspection, website, export/restore, containers, notebooks and CI. The method does not require a new claim of ownership of upstream weights to be technically substantial. Original and depth evidence remain separate and unchanged. Application acceptance and transactional regression checks establish different properties from neural factual evaluation. Small templated worlds, automatic labels and bounded memories constrain the conclusion; larger real-world accuracy is not established.

Why this slide: The conclusion should answer the objective using observed mechanisms and tradeoffs, rather than equating a perfect controlled score with universal reliability.

Evidence:

- [docs/CONTRIBUTIONS_AND_EVIDENCE.md](../../docs/CONTRIBUTIONS_AND_EVIDENCE.md)
- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)
- [Dockerfile.chat](../../Dockerfile.chat)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)

### Slide 16: Next milestone: model training and broader evaluation

**09:25–10:00 · 35 seconds · 70 spoken words**

> Next, we will compare a common instruction baseline, supervised adaptation and SFT plus PPO under matched retrieval and inference budgets. We will start with reviewed source-grounded edit data, fresh lineage-separated tests and a profiled pilot. Broader evaluation will vary wording, entities, memory size and depth while measuring accuracy, locality, abstention, latency and resources. The research question is whether adaptation improves transfer and robustness; gains and runtime remain to be measured.

Pacing: approximately 1.4 seconds remain for reading the graphic, pointing and handover. Advance on the clock mark.

Technical rehearsal: This is planned work. Establish a verified common instruction parent; adapt SFT, then PPO from the selected SFT stage. Do not describe the reused ReasonKE_plus as the verified causal parent of this study. Review final and intermediate targets, validate process/final rewards against plausible wrong chains and formatting shortcuts, and preserve a fresh frozen test because previous tests have been inspected. Before a substantial user-run GPU command, supply the concrete question, inputs/artifacts, profiled time/memory uncertainty, exact resume behaviour and finite stopping cap. Completion is an interpretable matched study, not a predetermined gain. The detailed upstream training tutorial remains outside this presentation.

Why this slide: The current platform already supplies the stable governance, retrieval and scoring controls needed for a focused adaptation experiment.

Evidence:

- [docs/presentation/VERITY_MIDPROJECT_RESEARCH_ROADMAP.md](../../docs/presentation/VERITY_MIDPROJECT_RESEARCH_ROADMAP.md)
- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [configs/instruction-model.json](../../configs/instruction-model.json)

## Hidden technical Q&A backups

Do not include these in the timed talk. Select a backup only when it answers a panel question.

### Slide 17: Backup: schema and transaction contract

**Untimed.** Optional Q&A: use this slide to show the exact represented key and transaction boundary.

Fact key=(entity_id,relation_id,scope_id). Approved versions retain immutable identity, question, answer, scope, source, lineage/version, predecessor and timestamps. Partial unique index: at most one active record per key. Rollback pseudocode: BEGIN IMMEDIATE; validate active replacement and same-key predecessor; withdraw replacement; restore predecessor; increment revision once; append withdrawn/restored audit events; COMMIT. Any exception rolls back state, revision and events together. The operation lock is a separate process-level guard spanning generation; it is not a long-held SQLite transaction. Store tests inject a failure late in the event write and assert the entire prior state survives. Snapshot validation checks digest, references, bounds, lineage monotonicity, event order and agreement between replayed and current state. A digest is corruption detection, not authenticity.

Evidence:

- [editguard_chat/store.py](../../editguard_chat/store.py)
- [editguard_chat/workspace.py](../../editguard_chat/workspace.py)
- [tests/test_chat_store.py](../../tests/test_chat_store.py)
- [tests/test_chat_workspace.py](../../tests/test_chat_workspace.py)

### Slide 18: Backup: scoring and paired uncertainty

**Untimed.** Optional Q&A: explain exact denominators and the lineage bootstrap.

Accuracy=correct/all assigned. Coverage=answered/all assigned. Answered-only precision is conditional on answering and must not replace primary accuracy. UNKNOWN, errors and unexecuted assignments remain zero-credit outcomes. Final extraction retains raw output/format flags and scores a whole endorsed span against the requested subject/relation target and declared aliases. Two G-ALL examples have correct chains but extra final text and remain strict mismatches. Paired bootstrap: draw 32 lineage IDs with replacement; include all related variants per selected lineage; compute the paired accuracy difference; repeat 10,000 draws at seed 42; use2.5th/97.5th percentiles. Original retrieved difference+5.86pp[3.91,8.20]; depth+3.125pp[1.56,5.08]; three-hop+1.5625pp[0,4.6875]. These are within-design uncertainty estimates, not a cure for template dependence or external-validity limits.

Evidence:

- [editguard_benchmark/scoring.py](../../editguard_benchmark/scoring.py)
- [editguard_benchmark/metrics.py](../../editguard_benchmark/metrics.py)
- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)

### Slide 19: Backup: pinned worker and resource boundary

**Untimed.** Optional Q&A: identify the reused checkpoint, exact settings and measured hardware.

ReasonKE model: YukinoKaren/ReasonKE_plus at ceaedc2ae7c8fed6257953587d991ce550db6106. Instruction baseline: Qwen/Qwen2.5-7B-Instruct at a09a35458c702b33eeacc393d103063234e8bc28. Each has 7,615,616,512 parameters. NF4 double quantisation, FP16 compute, SDPA, greedy decoding,4096 input and512 new-token ceilings. One loaded model at a time on T4. The saved sourced demo inventory records total device15,637,086,208 bytes, load peak allocated7,071,596,544 bytes=6.59 GiB and reserved7,140,802,560 bytes=6.65 GiB. Allocation excludes other device usage and is a specific measured load, not a training estimate. Source weight tensor bytes15,231,233,024=15.23 decimalGB are larger than the quantised loaded weights; cold transfer/loading is outside warm latency. Exact mapping of the plus artifact to a headline paper score is undocumented, so we report our own receipts.

Evidence:

- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [configs/instruction-model.json](../../configs/instruction-model.json)
- [editguard_reasonke/runtime.py](../../editguard_reasonke/runtime.py)
- [output/presentation/midproject/air-india-demo-receipt.json](../../output/presentation/midproject/air-india-demo-receipt.json)
- [output/presentation/midproject/screenshots/provenance.json](../../output/presentation/midproject/screenshots/provenance.json)

### Slide 20: Backup: correction editor and audit evidence

**Untimed.** Optional Q&A: show the recorded editor and the rollback event pair.

Two genuine Verity captures show the sourced correction draft and actual rollback/reapproval history. The editor draft advances a selected historical reference from December2021 to the January2022 Air India handover, with dated official source and explicit approval. The finite API workflow makes7 actual neural requests: initial direct, updated direct, connected parent, three-hop contact city, historical control, restored direct and final active connected. All7matched their declared targets; competing approval returned409 without revision change; rollback restored the predecessor. Reapproval leaves active demo r8 while history retains r7 rollback. These are selected acceptance cases, not a held-out accuracy population. Screenshots/receipts preserve their actual outputs and original pixels. Rollback reverses local memory, not the historical event; source evidence is not an arbitrary-prose verifier.

Evidence:

- [output/presentation/midproject/screenshots/provenance.json](../../output/presentation/midproject/screenshots/provenance.json)
- [output/presentation/midproject/air-india-demo-receipt.json](../../output/presentation/midproject/air-india-demo-receipt.json)
- [scripts/run_sourced_demo.py](../../scripts/run_sourced_demo.py)

## Clock discipline

Prepare handovers at 01:55, 03:55, 05:55 and 07:55. End slide 16 at 10:00. Hidden slides 17–20 have no automatic timings. If the panel interrupts, ask whether to pause the talk’s clock. The script’s 1067 words leave time for figures; extra unscripted explanations use that margin.

## Cross-cutting Q&A

**Is the current system RAG?** Yes. Explain the governed memory, bounded selector, revision consistency and measured failure behaviour directly.

**Did we train ReasonKE_plus?** It is a reused author-published checkpoint. Our own adaptation is planned. These are fixed-system comparisons, not our training-stage ablation.

**Does 256/256 establish universal accuracy?** It establishes the observed final-span result on the stated controlled population. The original full-memory failures, Qwen abstentions and broader-evaluation roadmap show its boundary.

**Are sources verification?** They identify model inputs. Wrong associations remain possible with complete support.

**Why no training-loss chart?** This completed experiment measures frozen-model inference. Training curves will come from the planned adaptation study.

## Evidence-based contribution allocation

These are proposed lead areas for implementation, reproduction, review and analysis. Attach names only to work actually performed and documented. Members can collaborate across areas; speaking order does not establish historical authorship.

| Lead | Work area | Concrete evidence |
|---|---|---|
| Member 1 | Data and label/split audit | Fact-world modules, dataset cards, split identities and source fixture |
| Member 2 | Retrieval and model integration | Lexical ranker, iterative selector, worker, pinned configs and context/history policy |
| Member 3 | Backend reliability | Canonical keys, constraints, event replay, atomic transitions and regression tests |
| Member 4 | Evaluation and failure analysis | Frozen protocols, per-request ledgers, strict scoring, paired bootstrap and 63-record audit |
| Member 5 | Product and reproducibility | Chat/editor/explorer, export/restore, containers, notebooks, CI and presentation |
