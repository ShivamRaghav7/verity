# Verity team handbook

This is the technical onboarding and presentation reference for all five team members. It describes the completed system and evidence behind the twelve-slide **Verity** presentation. You can start without knowing machine learning, retrieval or databases. Read Part A for the mechanism, Part B for the evidence, then Part C for operating and defending the project. The separate [speaker script](VERITY_SPEAKER_SCRIPT.md) gives the exact words, timings and handovers.

**The conclusion:** an approved ownership correction can change a frozen model system's direct and connected answers through versioned external memory. Our controlled experiments measure that behaviour and its costs. The final model was trained upstream. We built the governed application, context-selection adaptations, independent fictional data, matched evaluation and evidence tooling. Selected-context ReasonKE scored every main question correctly in both small controlled tests, while its full-memory condition made errors and its responses were much slower than Qwen's. These are measured boundaries, not unfinished failures to delete.

Verity is the new presentation and documentation name. Existing application screens, code modules, archived reports and frozen evidence can still say EditGuard. We preserve those historical identities and protocol hashes. This documentation change does not claim a newly trained checkpoint or a new experiment, and it does not silently rename the running application's historical UI.

## Reading map

- [Presentation map](#presentation-map)
- [Part A The system](#part-a-the-system)
- [Part B Data experiments and findings](#part-b-data-experiments-and-findings)
- [Part C Development history and practical operation](#part-c-development-history-and-practical-operation)
- [Additional technical questions](#additional-technical-questions)
- [Glossary for first time readers](#glossary-for-first-time-readers)
- [Evidence and attribution map](#evidence-and-attribution-map)

## Presentation map

| Slide | What to understand before speaking | Main reading |
| --- | --- | --- |
| 1 Verity | The problem, final method and bounded goal | Part A sections 1 and 2 |
| 2 Edit contract | Entity, relation, scope, locality and the three-hop example | Part A sections 6 and 7 |
| 3 Architecture | How a question reaches the model and how evidence is stored | Part A section 4 |
| 4 Model configuration | Frozen weights, tokens, quantization, compute precision and hardware | Part A section 3 and Part B resources |
| 5 Context selection | BM25-style ranking, seeds, answer frontier and development selection | Part A section 5 |
| 6 Transactions | Active-version invariant, conflict, events, revision and atomic rollback | Part A section 6 |
| 7 Dataset | Lineages, variants, split controls and automated labels | Part B experimental data |
| 8 Protocol | Matched baselines, request counts, oracle and scoring | Part B fair comparisons and scoring |
| 9 Original results | Full-memory reversal, preservation versus correctness | Part B original results |
| 10 Depth results | Per-depth scores, uncertainty and latency | Part B depth results and bootstrap |
| 11 Failure analysis | Why perfect observed scores are limited and explanations remain unchecked | Part B limits and evidence questions |
| 12 Product and conclusion | The actual website workflow, acceptance population and our contribution | Part A example, Part B acceptance and Part C |

## Part A The system

This material explains the final text system that the team is presenting. The project now uses the display name **Verity**. Earlier files, Python packages, saved screenshots and frozen experiment schemas still use **EditGuard**. Those names identify the same development history. Renaming a presentation does not change model identities, experimental outcomes or the files bound by a frozen protocol.

### 1 What problem are we actually solving

Suppose an internal assistant answers questions about services, their owning teams, on-call contacts and office locations. A service changes owners. Its old owner and new owner each retain their own contact. Updating one ownership record should change answers that follow the new ownership path. It should preserve the service's hosting location, the owner of an older release and the owner of a similarly named Pro service.

The task therefore has two parts. First, maintain authoritative information safely. Second, make a learned language model use the right information when answering a relevant question. A database transaction can establish the first part. A correct database transaction alone cannot establish the second part, because a model can ignore, confuse or misinterpret supplied facts.

For a handful of fixed policies, a normal web page or database lookup is often enough. We do not claim that a chatbot is necessary for every fact. The intended benefit here is allowing a user to ask natural-language questions across related records and compare answers across approved historical states. A structured graph query could also answer the same controlled example if a reliable interface translated the user's request into that query. Our experiment measures a language model route to this task; it does not prove that an LLM beats every deterministic alternative or that a real organisation has purchased this product.

The final scope is a controlled research demonstration. Ordinary text conversation is available in the website, but the reported factual evaluation concerns short explicit answers. It does not score every conversational reply, arbitrary language, every possible paraphrase or all operational situations.

### 2 Knowledge editing and the final method

**Knowledge editing** means changing a model system's response to selected information while preserving behaviour outside the intended scope. A useful edit must be effective on its target, generalise to relevant wording, and avoid changing unrelated answers. We also care about connected consequences: changing a service's owner can change the answer to a question about that owner's contact.

There are different ways to implement editing:

- **Parametric editing** changes numerical model weights. ROME is a research example of directly changing factual associations in model parameters.
- **External-memory editing** stores approved information outside the model and supplies applicable information during inference. Model weights can remain unchanged while the system's answer changes.
- **SERAC-style editing** combines explicit edit memory with learned applicability and counterfactual response components. Our initial support-editor study explored that separation. The final ReasonKE application does not contain those separately trained auxiliary components and is not a faithful SERAC reproduction.

Verity's final method is an **upstream-trained reasoner conditioned on versioned approved external memory**. The administrator changes memory. A deterministic context selector chooses a bounded set of active records. The frozen learned generator receives those records and generates an answer. There is no new optimiser run for each edit, no online gradient update, no current standalone trained binary scope classifier and no calibrated probability that an arbitrary correction applies.

This route overlaps with retrieval-augmented generation, usually abbreviated **RAG**. RAG supplies retrieved material to a generator. Verity adds an explicit correction lifecycle, canonical scope identity, historical states, rollback and a replayable evaluation. Those operational features are meaningful engineering work, but they do not by themselves make the underlying retrieval algorithm a new learned editing method. The comparison uses both ReasonKE and an instruction-model baseline with properly updated retrieved information.

#### The honest project history

The repository retains an earlier bounded support-editor study with DistilBERT/FLAN-T5 auxiliary training, followed by visual experiments, then the final text ReasonKE integration. These are distinct approaches and evidence populations. They should not be combined into a fictional single training run. The final presentation concentrates on the current product and its two text evaluations. The upstream authors own the ReasonKE method and training. The project owns its integration, controlled data, context-selection adaptations, governance, serving, evaluation and analysis. We do not claim that the exact released ReasonKE checkpoint reproduces a particular published paper percentage.

Implementation references: [current architecture](../CURRENT_ARCHITECTURE.md), [model card](../REASONKE_MODEL_CARD.md), [upstream ReasonKE repository](https://github.com/YukinoshitaKaren/Reason-KE), [archived auxiliary model card](../MODEL_CARD.md).

### 3 The learned model explained from first principles

A language model turns input text into tokens, processes those tokens using a neural network and predicts output tokens. A **token** is a unit from the model's vocabulary, often a word fragment rather than an entire word. The **tokenizer** converts between text and token IDs. It also supplies the checkpoint's expected chat-template format.

A **parameter** is a learned numerical value in the network. Both compared checkpoints contain **7,615,616,512 parameters**, approximately 7.6 billion. These parameters encode general language behaviour and learned associations. A new approved ownership fact does not rewrite those values in Verity. It becomes part of the next input context.

The models use a causal Transformer language-model interface. **Causal** means each generated token depends on the prompt and preceding output tokens. The attention mechanism lets the model combine information from positions in that context. A multi-hop answer therefore emerges from the generator using several supplied records; the website does not replace its output with a database-computed target.

The exact checkpoints are:

| Role | Model identifier | Pinned revision |
| --- | --- | --- |
| Final learned reasoner | `YukinoKaren/ReasonKE_plus` | `ceaedc2ae7c8fed6257953587d991ce550db6106` |
| Comparison instruction model | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |

A revision pins a particular repository snapshot rather than downloading whatever happens to be the latest model later. The model ID alone is insufficient for exact reproduction. Exact ancestry between the two released checkpoints is not established, so their comparison does not isolate the causal effect of ReasonKE training.

#### Why four-bit loading matters

**Quantization** stores selected model values at lower precision to reduce memory requirements. The fixed loader uses bitsandbytes **NF4** four-bit quantization, with **double quantization** of associated quantization constants. It configures **FP16**, or sixteen-bit floating-point computation, for the quantized layers' compute dtype. It does not mean every byte of the complete runtime occupies exactly four bits. Unquantized components, activations, attention state and framework allocations also need memory.

The checkpoint still has the same number of parameters. Quantization is a deployment adaptation, not evidence that we trained a smaller new model. It can affect behaviour, so the configuration remains fixed and recorded. Only one comparison model loads on the T4 at a time. The measured loader requires supported NVIDIA CUDA hardware; this fixed deployment is not a CPU inference claim.

The final inference configuration includes:

| Setting | Meaning and value |
| --- | --- |
| Weight format | Pinned `safetensors`, with remote model code disabled |
| Weight adaptation | NF4 double quantization through bitsandbytes |
| Compute dtype | Configured `float16` |
| Attention implementation | PyTorch scaled dot-product attention, `sdpa` |
| Generation mode | Greedy, `do_sample=False`, `num_beams=1` |
| Input ceiling | 4,096 tokens for the assembled prompt |
| Output ceiling | At most 512 newly generated tokens |
| Evaluation mode | `model.eval()` and `torch.inference_mode()` |
| Seed | The runtime sets PyTorch seed 42 |

**Greedy** means selecting the highest-scoring next token rather than randomly sampling a token. It removes sampling randomness, but hardware/software differences can still affect exact numerical execution. A fixed seed is not a proof of universal bit-identical output. An output ceiling is a limit, not a requirement to emit that many tokens.

`model.eval()` selects inference behaviour in the neural module. `torch.inference_mode()` avoids training-gradient tracking. There is no optimiser, learning rate, epoch, training loss or new final-checkpoint training curve in these experiments. Earlier auxiliary training has its own separate evidence.

Implementation references: [runtime loader and generation](../../editguard_reasonke/runtime.py), [instruction baseline adapter](../../editguard_benchmark/models.py), [pinned manifest](../../configs/reasonke-model.json).

### 4 The complete online request path

The browser and GPU inference worker are separate processes. This lets the application display knowledge, history and saved research results even when the model is offline. It also keeps the large ML dependencies out of the lightweight website process.

1. The user types a question in **Chat**. The browser submits it to the Python application API. An API is the structured interface through which one component requests an operation from another.
2. The application validates the request and serializes operations through a lock. It reads the active approved facts and current knowledge revision from SQLite. SQLite is the local transactional database, stored in a file.
3. The configured context policy inspects those active records and selects the supplied context. The India launcher requests the iterative policy. The normal CLI default retains the original two-stage policy, so the launch configuration matters.
4. The server selects eligible conversation history. It keeps complete user/assistant pairs from the current revision within the history budget. Old answers remain visible, but earlier-revision pairs do not enter a fresh prompt.
5. The backend client sends the question, selected facts and permitted history to the private model worker. The worker has already loaded the pinned checkpoint.
6. The worker serializes facts as JSON in an updated-information/query prompt and applies the checkpoint's chat template. JSON clearly delimits records but does not guarantee resistance to malicious prompt instructions inside text.
7. The tokenizer measures the assembled prompt. An input above 4,096 tokens is rejected rather than silently discarding important context. The model generates within the output budget.
8. The worker retains raw generated text. If exactly one nonempty `<answer>...</answer>` block exists, it extracts that block as the final answer. Otherwise it returns the raw fallback and marks the format state. It does not silently manufacture a correct response.
9. The controller stores the complete conversation turn, selected fact IDs, provenance, raw output, model revision, memory revision and generation statistics, then returns the answer.

The comparison model uses its own official Qwen chat template, with explicit context, short-answer and UNKNOWN instructions. ReasonKE uses the upstream updated-information/query prompt and answer-tag extraction. We match approved facts, question IDs and generation budgets, not byte-identical model prompts. Each checkpoint needs its declared input format. This remains a system comparison rather than a controlled causal ablation of one training change.

A failed worker connection or failed neural generation remains an explicit error. A deterministic template is never substituted and counted as a successful model answer. The displayed source means “this record was supplied”; it does not certify every generated explanatory sentence.

Important bounds are **200 active candidate records** in retrieved mode, at most **four supplied facts** in the selected depth policy, **12,000 supplied fact characters**, and the separate 4,096-token prompt ceiling. Full-memory mode accepts at most 20 active records. The candidate collection also has a one-million-character bound. These are software limits, not measured accuracy at those sizes; final benchmark worlds contain nine or twelve facts.

Implementation references: [controller](../../editguard_chat/server.py), [backend and context policies](../../editguard_chat/reasonke_backend.py), [worker runtime](../../editguard_reasonke/runtime.py).

### 5 Exactly how retrieval works

**Retrieval** selects records to place in the model's prompt. It does not generate the final answer. A retrieval failure omits needed support. A generation failure can happen even when every required fact was supplied. We measure them separately.

Both selectors use the same dependency-free **lexical** ranker. Lexical means matching word tokens, rather than learned semantic embeddings. It lowercases text, keeps alphanumeric tokens, removes a fixed stop-word set and ranks each fact's question, scope and answer together. The score follows a BM25-style term-frequency and inverse-document-frequency calculation. Frequent appearances in one fact help a match, common words across many facts carry less information, and document length adjusts the score. The code uses the term-frequency factor `2.2`, saturation coefficient `1.2` and length mix `0.25 + 0.75 * document_length / average_length`. Ties resolve deterministically by fact ID.

The current score is neither a learned applicability probability nor calibrated confidence. A higher lexical score does not prove that a correction applies. Similar names and relation wording can still confuse the ranker.

#### Original two-stage policy

```text
seeds = top two facts ranked against the original question
expanded_query = original question + seed answer text
reranked = rank all facts against expanded_query
selected = seeds followed by unseen reranked facts
return first four selected facts
```

The first pass can find an ownership record. Adding its answer to the query can retrieve the owning team's contact. The bounded second pass does not automatically traverse an arbitrarily long graph.

#### Separate iterative policy

```text
selected = top two facts ranked against the original question
frontier = answer text of those seeds
repeat at most three expansion rounds:
    stop if four facts already selected or no unseen fact remains
    next = highest-ranked unseen fact against frontier
    append next to selected
    frontier = next.answer
fill any unused slots from the last frontier ranking
return at most four original records
```

At the selected four-fact cap, two seeds leave room for at most two additional records. “Three rounds” is a maximum loop count; it does not promise that each question performs three extra expansions. Nor is it a general symbolic graph traversal: ranking still depends on answer text and lexical matching. It does not inspect hidden relation labels, requested depth, expected answers or gold support IDs.

Development considered a finite family of one or two seeds with caps four, five and six, keeping three rounds. The selected setting was two seeds and four facts. The development record discloses that a separate India display fixture exposed a one-seed hosting-city distractor before neural profiling and final freezing. The same declared family was checked on the development packet and both display states. That is a documented development choice, not test-set tuning.

**Gold support** means the author-defined records that are actually required to derive an answer. Only the evaluator reads those IDs to check retrieval recall or to construct explicitly named oracle diagnostics. Normal retrieval never receives those answer labels. Otherwise the experiment would be leaking its solution into the selector.

Implementation references: [shared ranker and original selector](../../editguard_benchmark/retrieval.py), [iterative selector and development selection](../../editguard_depth/retrieval.py), [depth run plan](../DEPTH_RUN_PLAN.md).

### 6 Fact identity and safe state changes

A correction needs more than a sentence containing a new value. The application identifies a canonical fact with `(entity_id, relation_id, scope_id)`. An entity is the subject, a relation is the property, and scope distinguishes a release or plan. For example, the service's release-2 owner and hosting city share an entity but have different relations. Its release-1 owner has a different scope. The three canonical IDs must be provided together and are normalized. Optional aliases help record identity but are not a trained general-purpose entity linker.

Each approved version retains its question, answer, source, scope, canonical identity, lineage, version number, predecessor/successor links and timestamps. A **lineage** groups versions of the same fact. A **knowledge revision** identifies the approved memory state after an operation. Neither is the model's checkpoint revision.

SQLite stores corrections and immutable audit events. Correction status can be `active`, `superseded` or `withdrawn`. A partial unique index permits at most one active version for the represented key. An event records the action, actor, time and revision. This makes state changes inspectable rather than overwriting the old answer and losing its history.

#### Approval, replacement and conflicts

Only the administrator workflow approves information. A customer message cannot silently publish a correction. A source field records what the operator supplied; it does not independently fact-check that source.

Publishing the first version requires valid fields and no active competing key. Replacing an active version requires explicitly naming it through `supersedes`, in the same scope. One transaction creates the new version, supersedes the old one, writes both events and increments the revision. A **transaction** commits all its database changes together or rolls them all back if something fails. A competing approval without a valid replacement relationship receives HTTP **409 Conflict** rather than silently selecting an answer.

The guarantee applies to represented identity. Two semantically contradictory records under unrelated keys can evade that check. Free natural-language overlap and conflicting calendar-effective date intervals are not solved by this schema. The application does not implement a complete semantic contradiction detector, calendar policy engine or calibrated ambiguity detector.

#### Atomic rollback and historical reads

Atomic rollback starts `BEGIN IMMEDIATE`, acquires SQLite's write transaction, checks that the replacement is currently active and restores its direct predecessor in the same key. It withdraws the replacement and restores the predecessor, writes two events at one new knowledge revision, then commits. It rejects an obsolete rollback that would undo a newer state. Repeating an already completed rollback is idempotent only in the valid restored state; idempotent means the repeat does not keep changing the result.

A historical read replays ordered events through an existing revision and reconstructs which facts were active then. It does not mutate today's state. A before/after comparison runs two fresh generations with the same checkpoint, question and matched permitted history, using different approved memory snapshots. It compares memory states, not saved model-weight snapshots.

Old chat answers remain visible after an edit. New requests exclude earlier-revision conversation pairs so the old answer does not contaminate current context. Comparison requests exclude correction-conditioned pairs from both sides. Memory changes call the context interface's invalidation hook. The current ReasonKE selector has no persistent generated-answer or embedding cache, so it reads approved state again rather than serving a previously cached correction.

Implementation references: [store schema and transactions](../../editguard_chat/store.py), [API contract](../CHAT_API.md), [architecture invariants](../CURRENT_ARCHITECTURE.md).

### 7 A complete India three-hop example

The fictional initial handbook contains nine records:

| Relation | Initial approved answer |
| --- | --- |
| Kaveri Payment Gateway release-2 owner | Bengaluru Platform Team |
| Bengaluru Platform Team on-call contact | Meera Iyer |
| Pune Reliability Team on-call contact | Arjun Deshmukh |
| Kaveri Payment Gateway release-2 hosting city | Bengaluru |
| Kaveri Payment Gateway Pro release-2 owner | Hyderabad Integration Team |
| Kaveri Payment Gateway release-1 owner | Chennai Legacy Team |
| Konkan Records Office city | New Delhi |
| Meera Iyer office city | Chennai |
| Arjun Deshmukh office city | Hyderabad |

The user asks: “In which Indian city is the office of the on-call contact for the team that owns Kaveri Payment Gateway in release 2?” Initially, the required path is service ownership, owning-team contact, then that person's office. The target is Chennai. It is deliberately different from Bengaluru in the team's name, so guessing a city from a team label is insufficient.

Approving the starter facts one by one creates knowledge revision 9 in a fresh workspace. An explicit ownership replacement changes only the first answer to Pune Reliability Team and creates revision 10. The new path uses the already existing Pune contact record and Arjun's office record. The target becomes Hyderabad. The old team's contact remains Meera; that fact itself was not edited. Hosting remains Bengaluru, the Pro owner's value stays separate, and release 1 keeps Chennai Legacy Team.

Rollback creates revision 11 and restores the original ownership record. A fresh three-hop generation should therefore return Chennai again. These are controlled ground-truth targets. Only the actual raw response establishes whether a particular model generation matched them. A table or deterministic target calculation is not a substitute for generated-answer evidence.

Source: [India catalogue implementation](../../editguard_chat/examples.py), [operator demonstration guide](../INDIA_DEMO.md).

### 8 Deployment, export and security boundaries

The lightweight application uses HTML, CSS and JavaScript with a Python standard-library HTTP server and SQLite. The private ML process adds pinned PyTorch, Transformers, Accelerate, bitsandbytes and safetensors. The container excludes downloaded model weights, private chat databases and secrets. A running website container alone is not a loaded GPU model. Colab can host the T4 worker and proxy the website; if that temporary runtime ends, the durable published evidence remains but live inference needs the worker loaded again.

The website uses loopback binding, mutation session tokens, JSON content-type checks and origin checks. Colab access uses one exact configured HTTPS proxy origin. The private worker uses a bearer token and rejects direct browser-origin requests. A bearer token authorises whoever possesses it; it must not be included in public receipts. These controls do not provide production user accounts, separate administrator roles, tenant isolation or enterprise access management. The local operator is trusted.

Portable text-workspace export includes records, events, revision metadata and a SHA-256 digest. A digest is a content fingerprint. It helps detect accidental corruption relative to recorded bytes, but anyone who changes content can also recompute a digest. It does not prove factual truth, human authorship or source authenticity.

Restore validates the exact schema, finite JSON, identifiers, timestamps, references, lineages, unique active keys, ordered events and agreement between event replay and final state. It accepts at most 6 MiB and 10,000 rows per table. It inserts data transactionally into an empty workspace instead of executing supplied SQL. Image histories need a separate private backup; portable text export does not expose local image paths.

Scientific result exports are public fictional research evidence. A user's workspace export is private conversation and knowledge data. They are separate artifacts with different purposes. The same distinction applies to a source displayed beside an answer: provenance records the input supplied, whereas factual verification requires checking the actual endorsed answer and any explanation claims separately.

Implementation references: [snapshot validation](../../editguard_chat/workspace.py), [application API](../../editguard_chat/server.py), [worker API](../../editguard_reasonke/server.py), [container](../../Dockerfile.chat), [deployment guide](../CURRENT_ARCHITECTURE.md).

## Part B Data experiments and findings

### Experimental data and ground truth

Verity's final evidence comes from two separately authored controlled studies. A controlled dataset means that we define a fictional world and know its facts exactly. We do not ask a model to guess whether an invented company really has a particular owner. Each inference request supplies the relevant configured fact memory explicitly.

A **fact** is a relationship such as service A in release 2 has owner team B. A **graph** is a collection of those relationships: service to team, team to contact, contact to office. A **lineage** groups the original world, its correction and all question variants derived from that edit. A **question** is one dataset record. A **request** is one actual generation under a model and information configuration. Running the same question through five configurations produces five requests, not five independent questions.

| Quantity | Original operations study | Separate depth study |
| --- | ---: | ---: |
| Fictional graph lineages | 48 | 48 |
| Facts per world | 9 | 12 |
| Questions per lineage | 8 | 8 |
| Authored questions | 384 | 384 |
| Development lineages and questions | 16 and 128 | 16 and 128 |
| Final lineages and questions | 32 and 256 | 32 and 256 |
| Split seed | 42 | 43 |

Development data guides choices. Final data estimates behavior after those choices are fixed. Splitting lineages before wording augmentation keeps related paraphrases and original/corrected variants in one partition. Otherwise the model development process could see a near-copy of a final question, producing misleading evidence of generalisation.

The original study reuses wording constructions across the development/final split. It therefore tests new entity assignments more strongly than genuinely unfamiliar language. The depth study improves that separation with disjoint authored wording banks, three constructions per category per split, and disjoint service/team/person aliases. Templates still repeat within each split. City labels, relation vocabulary and canonical fact syntax can repeat. These are important remaining limits.

Only one release-2 ownership fact changes in a graph. Old and new contacts retain their own valid assignments. The depth graph also contains the Pro service and release-1 branch as distractors. A distractor is valid information that answers a different question. Each question stores its requested subject, relation, scope, expected whole-answer aliases, original answer, edit applicability and ordered supporting fact IDs.

The data audit mechanically checks current and original graph paths, joins, aliases, scope, category counts, split membership and file identities. For the depth packet, 384 questions have both current and initial paths, giving 768 path checks. This is **automated review**, without independent human semantic review. The audit can catch a broken join or inconsistent expected answer. It cannot establish that every sentence sounds natural or that this fictional population represents real customers.

Sources: [operations dataset card](../OPERATIONS_DATASET_CARD.md), [depth dataset card](../DEPTH_DATASET_CARD.md), [depth data authoring](../../editguard_depth/data.py).

### Fair comparisons and finite execution

A configuration pairs a generator with a policy for supplying information. `R` means ReasonKE and `G` means the Qwen instruction generator. These abbreviations are labels, not different datasets.

| Condition | Model | Supplied facts |
| --- | --- | --- |
| R0 | ReasonKE | Initial facts, scored against corrected truth as a staleness diagnostic |
| R-RAG | ReasonKE | Updated facts selected by retrieval |
| R-ALL | ReasonKE | Full updated memory |
| G-RAG | Qwen | Same updated selection policy and fact IDs as R-RAG |
| G-ALL | Qwen | Same full updated memory as R-ALL |
| R-ORACLE and G-ORACLE | Respective model | Gold supporting facts on a preselected diagnostic subset |

An **oracle** diagnostic bypasses ordinary retrieval by supplying the known supporting facts. It helps separate context selection from generation, but it is not ordinary deployed accuracy. A properly updated RAG baseline receives the approved correction. R0 intentionally receives stale facts and must never be presented as the updated RAG competitor.

The original final workload is `5 × 256 + 2 × 16 = 1,312` generations. The depth final workload is `2 × (256 + 16) = 544`. Its two main conditions use iterative retrieval, and each model has sixteen separate preselected three-hop gold-support questions. The original oracle subset contains two-hop questions. These scores are kept separate from the main denominator.

Each study has its own protocol that binds source files, data, configuration, scoring and prerequisite receipts by SHA-256 hashes. A hash identifies exact file bytes. It detects an unexpected change relative to the saved identity; it does not prove that the contents are true. The depth protocol was published in commit `08d274e204e0dc8c7fcff94cc3aabcefda0debef` before final inference.

Finite runners save every completed outcome atomically and resume only the unfinished plan. A completed wrong answer is not retried for a better result. All final assigned requests completed without execution errors or unexecuted rows. Browser disconnection did not erase already saved outputs. One earlier temporary development runtime expired before export; its unavailable outcomes remain disclosed and excluded, rather than reconstructed as successes.

Sources: [original protocol](../../reports/reasonke/operations-final/protocol.json), [depth protocol](../../reports/reasonke/depth-final/protocol.json), [depth preservation audit](../../reports/reasonke/depth-final/resume-audit.json).

### Scoring and what each metric means

The scorer uses the model's **endorsed final answer**, the response it actually gives as its answer, against fixed aliases for the requested entity, relation and scope. Mentioning the target somewhere in a rejected example does not earn a match. Normalisation handles case, Unicode normalisation, outer bold markers, whitespace and trailing full stops. It does not freely reinterpret arbitrary prose with an LLM judge.

Every assigned record has one outcome: correct, wrong, abstain, execution error or unexecuted. `UNKNOWN` is an abstention: a model-generated refusal or inability to answer. It is different from a crash. It still receives zero primary accuracy credit.

```text
N = all assigned requests in the stated population
C = correct answers
W = wrong endorsed answers
A = abstentions
E = execution errors
U = unexecuted requests

N = C + W + A + E + U
Primary accuracy = C / N
Answer coverage = (C + W) / N
Answered-only precision = C / (C + W)
```

For Qwen in the depth main comparison, `C=248`, `W=0`, `A=8` and `N=256`. Accuracy and coverage are `248/256 = 96.875%`. Answered-only precision is `248/248 = 100%`. Quoting only the last percentage would conceal eight unanswered questions. For original Qwen full memory, the same 241 correct answers accompany thirteen wrong answers and two abstentions. Coverage is `254/256 = 99.21875%`, and answered-only precision is `241/254 ≈ 94.88%`.

Whole-answer matching also has a limit. Two original Qwen full-memory outputs contain a correct contact together with its compatible owner but violate the request to give only the person's name. A third combines a Pro owner with the requested contact. They remain strict mismatches under the frozen contract. We disclose that some score failures are format failures and do not call all thirteen false factual endorsements. Changing aliases after inspecting final outputs would change the test retrospectively.

**Retention** and **locality accuracy** answer different questions. Retention asks whether an answer that was already correct before the edit stays correct outside the edit's scope. Locality accuracy asks whether every tested out-of-scope answer is factually correct. In the original study, R0 answered 124 of 128 locality controls correctly. R-RAG retained `124/124` and answered `128/128` locality questions correctly. R-ALL retained `117/124` and answered `119/128` correctly. Preserving the initial model's four wrong answers would not count as factual correctness. The depth study has no unchanged-base condition, so its `64/64` locality scores do not supply a new baseline-retention estimate.

The bounded false-correction metric flags an out-of-scope answer that matches the inapplicable edit's value. R-RAG had `0/128` and R-ALL `3/128` original locality cases. This narrower check is not a count of all mistakes or a general semantic contradiction detector.

Sources: [fixed scorer](../../editguard_benchmark/scoring.py), [original metric implementation](../../editguard_benchmark/metrics.py), [original failure analysis](../../reports/reasonke/operations-final/FAILURE_ANALYSIS.md).

### Results of the original study

| Condition | Correct / assigned | Wrong | Abstain | Primary accuracy |
| --- | ---: | ---: | ---: | ---: |
| R0 | 136/256 | 120 | 0 | 53.125% |
| R-RAG | 256/256 | 0 | 0 | 100% |
| R-ALL | 231/256 | 25 | 0 | 90.234375% |
| G-RAG | 241/256 | 0 | 15 | 94.140625% |
| G-ALL | 241/256 | 13 | 2 | 94.140625% |
| R-ORACLE | 16/16 | 0 | 0 | 100% diagnostic |
| G-ORACLE | 16/16 | 0 | 0 | 100% diagnostic |

There are zero execution errors and unexecuted requests in this scorecard. R0's 136 correct answers comprise twelve applicable answers and 124 locality answers correct against updated truth. Its score is a staleness diagnostic, not separately measured accuracy against the initial handbook.

The original retriever supplied `320/320` required fact occurrences and complete paths for `64/64` two-hop questions. A fact occurrence counts a supporting fact once for each question that needs it. It does not mean there are 320 unique facts. All fifteen G-RAG abstentions had the needed support and occurred in the 32 two-hop paraphrases: seventeen correct and fifteen UNKNOWN. The other seven categories were 32/32 each. The retrieval-model difference is therefore a coverage difference in this population, not fifteen incorrect nonabstaining answers.

Every R-ALL strict wrong answer endorsed a value belonging to another supplied fact. This supports the observation of incorrect associations under full memory. It does not expose the model's internal attention or prove a causal explanation. With full memory, Qwen scored higher than ReasonKE. The presentation must retain this reversal rather than claim that ReasonKE wins all conditions.

Both oracle subsets scored 16/16. Only one of the fifteen original Qwen retrieval-abstention question IDs overlaps the oracle subset. Consequently the oracle result is not evidence of repairing all fifteen failures.

Sources: [original aggregate](../../reports/reasonke/operations-final/results.json), [all original failures](../../reports/reasonke/operations-final/FAILURE_ANALYSIS.md), [ReasonKE full-memory failures](../../reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md).

### Results of the separate depth study

| Main question group | ReasonKE correct / assigned | Qwen correct / assigned | Qwen abstentions |
| --- | ---: | ---: | ---: |
| Locality controls | 64/64 | 64/64 | 0 |
| One hop | 64/64 | 64/64 | 0 |
| Two hops | 64/64 | 57/64 | 7 |
| Three hops | 64/64 | 63/64 | 1 |
| All main questions | 256/256 | 248/256 | 8 |

Each depth group has 32 direct and 32 paraphrased questions except locality, which combines 32 historical-release and 32 other-property controls. Depth zero is a control label; those questions still require an independent fact.

Iterative retrieval supplied `448/448` required fact occurrences and `256/256` complete paths. The sum is `64×1 + 64×1 + 64×2 + 64×3 = 448`. Three-hop support alone was `192/192` required occurrences over `64/64` complete paths. Both generators use the same selector, so duplicating it with two models does not create two independent retrieval studies.

All eight non-correct Qwen outputs were literal UNKNOWN. None exhausted its token or generation-time budget. Seven direct two-hop failures concentrate in one final wording construction, with seven abstentions among nine instantiations. The other direct two-hop constructions were 23/23 correct, and all 32 two-hop paraphrases were correct. Three-hop direct questions were 32/32 and paraphrases 31/32. Thus a deeper group scored higher than the two-hop group. These data do not support a monotonic error-versus-depth claim. The concentration is a descriptive finding, not proof of an internal cause or permission to retune final prompts.

Both sixteen-question depth oracle diagnostics scored 16/16, but none of the eight main failure question IDs belongs to that subset. Do not say oracle context repaired those eight cases. The sixteen questions span thirteen lineages, not thirty-two.

Sources: [depth aggregate](../../reports/reasonke/depth-final/results.json), [all eight saved failures](../../reports/reasonke/depth-final/FAILURE_ANALYSIS.md).

### Paired uncertainty and scientific graph interpretation

A **percentage-point difference** subtracts percentages directly. `100% − 96.875% = 3.125 percentage points`, rather than a 3.125% relative improvement. A paired comparison evaluates both models on the same question IDs, allowing successes and failures to be compared directly.

The bootstrap estimates uncertainty by repeatedly resampling complete lineages with replacement. For each of 10,000 draws, seed 42, it samples 32 lineage groups and computes the mean question-level correctness difference within those sampled groups. The 2.5th and 97.5th percentiles form the reported 95% interval. Keeping all variants of a lineage together respects their shared construction better than pretending every question is independent.

| Comparison | Observed difference | 95% paired lineage interval |
| --- | ---: | ---: |
| Original ReasonKE retrieved minus Qwen retrieved | +5.859375 pp | +3.90625 to +8.203125 pp |
| Original ReasonKE full minus Qwen full | −3.90625 pp | −7.8125 to −0.390625 pp |
| Depth main ReasonKE minus Qwen | +3.125 pp | +1.5625 to +5.078125 pp |
| Depth three-hop subset ReasonKE minus Qwen | +1.5625 pp | 0 to +4.6875 pp |

The three-hop interval includes zero, so its comparison supports a weaker conclusion than the overall main difference. A zero-width interval when both methods succeed on every sampled group describes identical differences inside that sample. It is not certainty about future user traffic. This analysis does not establish that a particular upstream training step caused the difference, because Qwen's exact status as a causal ablation of ReasonKE training is unverified.

Read each graph by its actual question. The original outcome-composition figure distinguishes correct answers, wrong answers and abstentions. Its paired-difference plot shows direction and uncertainty, including the full-memory reversal. Depth accuracy shows the actual non-monotonic pattern. Retrieval-and-answer quality shows why complete support does not guarantee an answer. Latency shows the cost accompanying accuracy. Never infer task quality from the number of figures or treat counts pooled across both studies as one larger independent test.

### Runtime and resource measurements

| Configuration | Timed generated requests | Warm p50 / p95 seconds |
| --- | ---: | ---: |
| Original ReasonKE retrieved | 256 | 15.0568 / 16.7135 |
| Original Qwen retrieved | 256 | 0.9162 / 1.1981 |
| Original ReasonKE full memory | 256 | 15.3450 / 18.7162 |
| Original Qwen full memory | 256 | 1.4398 / 1.6594 |
| Depth ReasonKE iterative | 256 | 15.9358 / 19.2828 |
| Depth Qwen iterative | 256 | 1.1190 / 1.3847 |

**p50** is the median: half the measured responses are at or below it. **p95** is the 95th percentile: about 95% are at or below it in the specified sample. The implementation sorts values and linearly interpolates at index `(n−1)×q`. Warm pipeline time includes support selection, local interprocess communication and generation. It excludes model loading, downloads and browser network. Abstentions remain in the timed population.

The T4 depth runs measured loading at about 79.15 seconds for ReasonKE and 77.24 seconds for Qwen. Both had peak loading allocated CUDA memory of `7,071,596,544 bytes ≈ 6.59 GiB` and reserved memory of `7,140,802,560 bytes ≈ 6.65 GiB`. A GiB is `1,073,741,824` bytes. Allocated memory tracks active PyTorch tensor allocations; reserved memory includes the allocator's retained pool. Neither equals total operating-system or GPU-driver memory. Per-request generation resource samples remain in the receipts. A 6 GiB laptop configuration does not automatically fit a loader whose measured allocation exceeds that amount.

These final checkpoints were trained upstream, so there are no project-created training or test-loss curves for them. Loading, inference, scoring and database tests are real project work, but none should be relabelled as local training.

Sources: [model card](../REASONKE_MODEL_CARD.md), [depth measured report](../../reports/reasonke/depth-final/derived/REPORT.md), [depth per-request latency and resources](../../reports/reasonke/depth-final/derived/latency.csv).

### Why 100% accuracy remains a bounded finding

ReasonKE's 256/256 main answers in each study are actual saved neural outputs, rather than a database template or a copied upstream paper score. We retain the result and do not invent failures to make it look more realistic. However, each world has nine or twelve explicit facts; relation families and wording constructions repeat; supporting facts were complete; answers are short controlled values; and labels lack independent human semantic review. The 256 variants arise from only 32 held-out lineages. The test therefore has limited power to distinguish strong models on its easier groups.

The result does not establish arbitrary paraphrase robustness, safe handling of unknown ambiguity, multilingual quality, large-memory retrieval, unrestricted conversations or correctness of every explanation. Full-memory failures demonstrate that model behavior depends on the actual context policy. The final system also has no calibrated scope probability, so the report cannot invent a classifier precision/recall or confidence-versus-coverage curve.

The original explanation audit selected 32 traces before inspection: sixteen R-RAG and sixteen R-ALL. It extracted 32 indexed subanswers per condition. Retrieved context had 32 supported claims. Full memory had 26 supported, three contradicted and three unsupported. Each condition also retained 64 unassessed surrounding prose segments. **Fully verified traces is zero in both conditions.** This is a narrow automated proposition audit, not a certificate of faithful internal reasoning or every sentence's truth.

Sources: [explanation audit design](../EXPLANATION_AUDIT.md), [saved claim audit](../../reports/reasonke/operations-final/explanation-audit.json).

### Website acceptance and software checks

The separate India display workload produced fourteen actual model answers correctly and passed twelve transaction/provenance checks. Thirteen HTTP generation requests produced fourteen answers because one historical comparison invokes the model twice. Loopback HTTP p50/p95 was 17.11/33.39 seconds for those thirteen mixed requests, including the two-generation comparison. That is not per-generation latency. It covers one fictional display lineage, not fourteen extra independent held-out facts.

The recorded workflow checks initial, edited and rolled-back answers, paraphrases, locality, historical scope, rejected conflicting approval, revision identity and restored memory. Revision 9 is the initial nine-fact handbook, revision 10 the approved replacement and revision 11 its rollback. A separate manual screenshot demonstrates a real saved conversation but adds no benchmark observation. The older operations website supplement recorded twelve workflow plus four history/rollback generations, separately.

Deterministic regression tests establish properties such as atomic rollback or rejecting malformed restore data. They do not count as correct neural answers. Conversely, a correct generated contact does not prove database transaction isolation. The saved engineering deployment receipt has 24 software checks, zero model calls and null model accuracy. The release regression workflow passed six CI jobs, which also do not increase the neural-accuracy denominator.

Sources: [India receipt](../../reports/reasonke/depth-final/india-acceptance.json), [India CPU replay](../../reports/reasonke/depth-final/india-host-audit.json), [India operator guide](../INDIA_DEMO.md).

### Questions the panel may ask about the evidence

1. **Did you train ReasonKE?** No. Its authors trained the released checkpoint. We integrated and evaluated it. Earlier auxiliary-model training remains a separate archived study.
2. **Is Verity better than every baseline?** No. Selected ReasonKE context had the strongest observed main coverage here, while Qwen was faster and scored higher under full memory.
3. **What makes the comparison fair?** Updated models receive the same approved facts and selection policy, compatible generation settings and the same questions and fixed scoring contract.
4. **Is R0 the RAG baseline?** No. It deliberately receives stale initial facts. G-RAG and R-RAG are properly updated retrieval configurations.
5. **How large is the test?** Each study has 256 main questions per condition from 32 final lineages. The request count also includes repeated configurations and separate oracle diagnostics.
6. **Why count UNKNOWN as failure?** The user requested an answer. Safe abstention can be preferable to a false statement, but it still fails to supply the target. Accuracy, coverage and wrong-answer counts show the distinction.
7. **Does complete retrieval prove the model reasoned correctly?** No. Required support can be present while the model abstains or joins the wrong records. We measure support and final answers separately.
8. **Why did two-hop Qwen score below three-hop Qwen?** Seven failures concentrate in one direct wording template. Depth is not the only variable. We report the actual pattern without claiming monotonic difficulty.
9. **Did oracle support fix the eight depth failures?** That was not tested. Their IDs do not overlap the preselected sixteen-question oracle subset.
10. **Does the confidence interval prove future accuracy?** No. It describes paired differences for a small constructed lineage population. The dataset's representativeness and label validity remain separate limitations.
11. **Why use lineage bootstrap?** Related paraphrases share an underlying edited graph. Sampling them independently would ignore that dependence.
12. **Does the source label verify the answer?** It identifies the supplied approved evidence. Generated prose can still be wrong. The narrow explanation audit leaves surrounding prose unassessed.
13. **Why no loss curve?** The final study performs inference with upstream-trained checkpoints. Adding a training curve would misrepresent the work performed.
14. **Can the demo run without Colab?** The website can display saved evidence without weights. Live neural generation needs an available worker with sufficient measured memory. Stored results must be described as stored results.
15. **What would strengthen the result next?** Independently reviewed operational questions, larger memories, missing facts, ambiguous scopes and broader wording would test limits. That would be a new study, not a retrospective change to this frozen test.


## Part C Development history and practical operation

### The actual development sequence

We did not start with a single method that solved every issue. The repository preserves several approaches. The final presentation focuses on the last text platform. Knowing this distinction prevents mixing old training losses with new inference results.

| Phase | What happened | Why it matters to the current presentation |
| --- | --- | --- |
| First support editor | Built explicit corrections, approval, conflicts and transactional rollback, then trained DistilBERT and FLAN-T5 auxiliary models for a bounded support task. | Establishes earlier work, but those auxiliary models do not run in the final ReasonKE backend. |
| Visual exploration | Implemented separate visual prototypes and comparisons. | Archived evidence, excluded from all current text scorecards. |
| Conversational text integration | Added normal chat, durable history, provenance and before/after comparison around a pinned upstream ReasonKE checkpoint. | The final product supports conversation while its scientific evaluation keeps a narrower short-answer task. |
| Original operations study | Authored controlled service/team graphs, profiled both checkpoints, ran matched development comparisons and froze the original evaluation. | The five main information configurations explain the full-memory reversal and retrieval result. |
| Reliability and recovery | Added canonical keys, historical reads, atomic rollback, workspace restore, raw-output inspection and resumable finite runners. | Makes state changes and experimental evidence recoverable instead of relying on one browser session. |
| Separate depth extension | Authored new graphs and wording banks, inspected development retrieval, added iterative answer-frontier expansion, froze and completed 544 requests. | A separately declared study tests three-hop behaviour without modifying the original study. |
| Application acceptance | Recorded the India workflow through initial, edited, historical and rolled-back states, then replay-audited its saved output on the laptop. | Establishes measured end-to-end application behaviour on one display lineage, separate from held-out accuracy. |
| Report and presentation | Produced scientific exports, the complete report, a five-speaker presentation and this technical handbook. The presentation now uses Verity. | Communication changes reuse saved evidence and launch no new model experiment. |

The dated history is in [Understanding the current project](../UNDERSTAND_THE_CURRENT_PROJECT.md) and [the project journey](../PROJECT_JOURNEY_AND_FINISH_PLAN.md). Earlier support or visual percentages should appear only if explicitly identified as those earlier studies. We do not invent a historical author for an implementation file or pretend every iteration was planned in advance.

### Opening the actual neural demonstration

1. Open the repository's [India chat notebook](../../notebooks/07_india_chat.ipynb) in Colab. Select an available T4 runtime. The notebook is a launcher, not a training exercise. Its source commit and checkpoints are pinned inside it.
2. Run its setup and worker cells in order. Keep only one GPU model process in that runtime. Downloads and cold loading are different from the warm response times in the scorecard. Do not expect a chat response while the worker is still loading.
3. Wait for the readiness status and open the served website/proxy link from the notebook. A website shell can open while the worker is unavailable, so visible HTML alone is insufficient evidence of readiness.
4. Use the India launcher's explicit iterative context policy. The CLI's default retrieved policy belongs to the original two-stage design. `scripts/run-chat.ps1` is an older Ollama-oriented launcher and is not the route for reproducing the final ReasonKE/T4 demonstration.
5. Use a fresh demonstration storage directory. In Knowledge explicitly approve the fictional India handbook. Nine approvals produce revision 9. Withdrawing every old fact does not erase history or make a previously used workspace fresh.
6. Keep the GPU runtime active during live generation. If you close or lose it, the committed results survive, but live chat needs a worker again. Export the private text workspace before ending a temporary runtime if you want its current conversations and fact history later.

These instructions identify the existing pinned launcher. They do not claim to have rerun GPU readiness today. The notebook and older UI can retain their historical EditGuard branding. Check [the detailed India demo guide](../INDIA_DEMO.md) for its exact controls.

### Where each website tab fits

**Chat** submits a fresh question, displays model output and lets you inspect supplied facts. Reopening a stored conversation displays history. It does not regenerate those old answers. A stored answer's revision indicates the memory used when it was generated.

**Knowledge** contains approval, source/scope entry, canonical identity, active and historical versions, conflict choices and rollback. Use the same canonical IDs for alternate descriptions of the same property and scope. Changing wording while assigning unrelated IDs can bypass the intended shared lineage.

**Research results** displays saved experimental receipts, scorecards, raw responses and downloadable figures. It does not evaluate every new conversation automatically. Filters help inspect a subgroup but must not replace the full assigned denominator with a favourable subset.

**Historical comparison** generates both sides anew. Choose approved memory at revision 9 for the real before state. The option without correction context removes context and is a different condition. A comparison consumes two model generations, which explains why fourteen website answers arose from thirteen generation requests.

### A rehearsal that exercises the actual contract

Start with the worker ready and initial memory at revision 9. Ask these as separate fresh questions and inspect the answer evidence:

| Stage | Question or action | Controlled target |
| --- | --- | --- |
| Before | Which team owns Kaveri Payment Gateway in release 2? | Bengaluru Platform Team |
| Before | Who is the on-call contact for the team that owns Kaveri Payment Gateway in release 2? | Meera Iyer |
| Before | In which Indian city is the office of the on-call contact for the team that owns Kaveri Payment Gateway in release 2? | Chennai |
| Edit | Draft the example ownership update, inspect it, then approve replacement. | Revision 10 and Pune Reliability Team active |
| After | Repeat the direct and connected questions. | Pune Reliability Team, Arjun Deshmukh, Hyderabad |
| Locality | Which Indian city hosts Kaveri Payment Gateway in release 2? | Bengaluru |
| Historical scope | Which team owns Kaveri Payment Gateway in release 1? | Chennai Legacy Team |
| Compare | Compare the office question at approved revision 9 against current memory. | Before target Chennai, current target Hyderabad |
| Rollback | Roll back the active ownership replacement. | Revision 11, original owner restored |
| Restored | Ask a fresh office question. | Chennai |

The target column is the truth inside the fictional handbook. The actual generation can fail. If it fails, inspect supplied fact IDs and raw output, then explain whether support was absent or the generator failed despite support. Do not paste the target into the chat as a supposed model response. A customer statement is never an authoritative update without approval.

The short presentation demo can use only the initial office question, ownership approval, edited office answer, one locality check and rollback. Historical comparison adds two responses and extra waiting. Rehearse the full path before selecting the shorter panel version.

### Replaying evidence without a GPU

Recomputation reads already saved model outputs and checks their identity and score. It is not a new model evaluation or a way to improve an old result. From the repository root, use an isolated Python environment containing the project's lightweight runtime requirements. In the existing laptop checkout, `.venv-chat` is the lightweight environment.

```powershell
.\.venv-chat\Scripts\python.exe -m scripts.audit_india_http reports/reasonke/depth-final/india-acceptance.json --output artifacts/team-india-replay.json
.\.venv-chat\Scripts\python.exe -m editguard_depth.metrics reports/reasonke/depth-final/final-reasonke.json reports/reasonke/depth-final/final-instruction.json --protocol reports/reasonke/depth-final/protocol.json --output artifacts/team-depth-replay.json
```

Choose fresh output filenames if those already exist. The tools intentionally preserve prior files. These commands audit the saved receipts and launch zero neural calls. The first checks fourteen saved website answers and transaction snapshots. The second reconstructs the 544-request depth aggregate and protocol identities. Read the emitted status and output JSON instead of treating process completion alone as model correctness.

Useful software regression commands are below. Their fixtures verify code behaviour and do not add neural accuracy observations:

```powershell
.\.venv-chat\Scripts\python.exe -m unittest tests.test_chat_store tests.test_chat_workspace tests.test_depth_data -q
```

Do not run training or the full GPU benchmark simply to show the website. A scientific rerun needs the exact pinned source, data, protocol, environment and finite plan. A modern checkout used for presentation documentation is not automatically identical to the source hashed by an earlier protocol.

### Common demonstration problems

| Symptom | What it means and what to check |
| --- | --- |
| The page loads but no neural answer appears | Inspect worker readiness and its logs. The page and worker are separate processes. |
| Colab disconnects | Reopen the connection and check the existing process before launching another worker. A browser disconnect and a runtime termination are different events. |
| GPU memory fails during loading | Check concurrent workers and available memory. The measured fixed loader allocates more than 6 GiB. Do not promise the same model fits a 6 GiB GPU. |
| The new fact does not appear in an answer | Confirm actual approval, active version and revision. Then inspect which facts the selector supplied. A draft alone changes nothing. |
| The stored old answer still shows the old owner | That is expected history. Ask a fresh question at the new revision. |
| Approval returns 409 | Resolve the represented active-key conflict by naming the intended predecessor. This is a controlled refusal, not a neural crash. |
| Rollback is rejected | Check whether it is still the active replacement with the permitted direct predecessor. An obsolete edit cannot undo a newer state. |
| An audit refuses an output path | The file already exists. Select a new path instead of overwriting evidence. |
| The live GPU becomes unavailable during the panel | Say the worker is unavailable and show recorded outputs explicitly as recorded evidence. Do not claim a saved screen is a new inference. |

## Additional technical questions

The evidence-specific questions appear in Part B. These cover implementation and project defence.

1. **Is this simply updated RAG?** It uses retrieval and external-memory conditioning, so there is real overlap. Its maintenance contribution is the approval, canonical scope, historical states, transactional rollback, serving and evaluation around a learned reasoner. We do not call the lexical selector a new learned method or claim an advantage over every RAG system.
2. **Why call it knowledge editing if weights never change?** Some editing approaches modify parameters; memory-based approaches change the context that controls system behaviour. We state our route explicitly so the panel can distinguish them.
3. **Where is machine learning central?** Both final generators are large learned neural models. The project adapts and evaluates their inference, not their training. The answer cannot be credited to the model unless it appears in actual neural output. Earlier auxiliary training is a separate experiment.
4. **Why not just use SQL or a graph query?** A correctly specified structured query can solve these fictional relations. The evaluated task begins with a natural-language question. We have not shown that this approach beats a carefully engineered semantic parser plus database on every objective.
5. **What does ReasonKE add to the comparison?** Its released checkpoint embodies an upstream learned approach to using edits. We measure its behaviour beside an instruction checkpoint with matched approved context. Exact checkpoint ancestry is unverified, so we cannot attribute every difference to one training step.
6. **Did we reproduce the ReasonKE++ paper exactly?** No. We integrated a released checkpoint, adapted loading/retrieval/serving and evaluated independent tasks. Published paper scores belong to their own models and datasets. The exact released-checkpoint-to-paper-score association is undocumented in our inspected evidence.
7. **Why does lexical retrieval include stored answers?** The stored answer is an approved fact field, such as a team name. Following it can retrieve the team's contact. This is different from reading the hidden expected answer for the evaluation question. Gold labels enter only the evaluator or explicitly named oracle condition.
8. **Is the selector a graph search engine?** It approximates connected retrieval through lexical answer text. It does not consume the hidden structured graph labels or guarantee a graph join under arbitrary wording.
9. **How do similar names or wrong releases stay distinct?** Canonical keys separate stored facts. Scope text and retrieval influence what reaches the model. Those mechanisms reduce risk inside the controlled task, but they do not provide perfect natural-language scope classification.
10. **Can administrators enter completely new facts?** The workflow can store and approve new values within its schema and bounds. That does not guarantee that the selector retrieves them or that the model answers every related paraphrase. Test a new edit as a new workload.
11. **Can any prompt injection become approved knowledge?** Chat messages never auto-approve facts. An approved source can still contain problematic instructions, so authority control alone is not a complete prompt-injection defence.
12. **What does BEGIN IMMEDIATE achieve?** It starts a SQLite write transaction before changing the replacement and predecessor. Combined validation, updates and event writes commit together. It does not make generated text deterministic or support arbitrary distributed writers.
13. **Why two events but one revision?** Withdrawing the new version and restoring the predecessor are two auditable changes within one committed memory state. Splitting them across revisions would expose an unintended intermediate state.
14. **Does rollback retrain or reload the model?** No. It restores approved fact state. A fresh request uses that state with the same checkpoint.
15. **What is the difference between checkpoint and revision?** The checkpoint identifies neural weights. The knowledge revision identifies approved external memory. A single checkpoint can answer against many memory revisions.
16. **Why remove old conversation pairs from fresh prompts?** An old generated owner can contradict the newly approved owner. Excluding stale pairs reduces this contamination while preserving visible conversation history.
17. **Why a partial unique index?** The database can retain many historical versions while allowing only one with active status for a represented key. Historical storage and current authority therefore coexist.
18. **Does a checksum prove the experiment is honest?** It binds byte identity relative to a record. It does not prove independent authorship, semantic validity or that someone did not design an easy task. Raw outputs, declared protocols and clear limits provide additional evidence.
19. **Can the demo container run the complete model?** The lightweight website image serves the product and saved evidence. The separate GPU worker requires its ML dependencies, checkpoint and sufficient hardware.
20. **Is this ready for a company deployment?** It is a locally deployable research application. Production deployment additionally needs authenticated roles, access policies, better semantic conflicts, source validation, broader evaluation and service-level measurements.
21. **Do numerical losses prove model accuracy?** No. A loss is an optimisation objective, not an accuracy percentage. The final experiment has no new training loss to plot. We show accuracy, coverage, retrieval, resources and latency instead.
22. **What is actually complete?** The implemented workflow, original and depth final ledgers, reported audits, acceptance checks and reports are complete. Broader real-world validation is a stated next study, not a pending row hidden from the completed denominator.
23. **Can we say each member trained a model or invented the method?** Only if that person actually performed the stated work. Use the contribution allocation in the speaker script to assign implementation, review, reproduction and documentation accurately. Speaking a section does not establish authorship.

## Glossary for first time readers

| Term | Meaning in this project |
| --- | --- |
| Inference | Using fixed neural weights to generate an answer. |
| Fine-tuning | Updating learned parameters with training data and an optimiser. The final deployment does not do this for an edit. |
| Token | A model vocabulary unit, often part of a word. Token count differs from word count. |
| Checkpoint | A saved set of model weights and associated configuration. |
| Quantization | Lower-precision storage of selected neural values to reduce memory. |
| NF4 | The chosen four-bit quantization representation in the loader. |
| FP16 | Sixteen-bit floating-point computation. This precision is unrelated to the evaluation metric called precision. |
| SDPA | Scaled dot-product attention, the selected PyTorch attention implementation. |
| Greedy decoding | Selecting the highest-scoring next token without random sampling. |
| Context | The facts, instructions and permitted history supplied with the question. |
| Retrieval | Selecting records from approved memory for that context. |
| Seed fact | An initial question-ranked record that starts expansion. |
| Frontier | Answer text from selected facts used as the next lexical query. |
| Gold support | The known facts required by a labelled question, used for evaluation or a separate oracle diagnostic. |
| Ground truth | The controlled expected answer and supporting relations in the authored fictional world. |
| Lineage | One underlying fact world/edit and its related variants. |
| Locality | Behaviour outside the intended edit scope. |
| Abstention | A model response such as UNKNOWN instead of an endorsed factual answer. |
| Coverage | The fraction of all assigned cases with an endorsed answer. |
| Answered-only precision | Correct divided by answered cases, which can hide abstentions if reported alone. |
| Calibration | Agreement between stated confidence and observed correctness. No calibrated scope confidence is reported here. |
| Percentage point | Direct subtraction of percentage scores. |
| Bootstrap | Repeated resampling to estimate uncertainty under the chosen sample structure. |
| p50 and p95 | The median and 95th percentile of the specified latency population. |
| Transaction | Database changes that commit together or roll back together. |
| Idempotent | Repeating a permitted operation leaves the same result instead of changing it repeatedly. |
| Revision | A committed approved-memory state, distinct from a neural checkpoint. |
| Provenance | Recorded source and origin information. It does not certify arbitrary generated prose. |
| CI | Continuous integration when discussing software jobs, confidence interval when discussing statistics. Context matters. |

## Evidence and attribution map

| Claim or component | Where to inspect it |
| --- | --- |
| Original complete model outputs and aggregate | [Original study folder](../../reports/reasonke/operations-final/README.md) |
| Separate complete depth outputs and aggregate | [Depth study folder](../../reports/reasonke/depth-final/README.md) |
| All eight depth abstentions | [Depth failure analysis](../../reports/reasonke/depth-final/FAILURE_ANALYSIS.md) |
| Exact model checkpoint revisions and limitations | [Current model card](../REASONKE_MODEL_CARD.md) |
| Database invariants and online interfaces | [Current architecture](../CURRENT_ARCHITECTURE.md), [API](../CHAT_API.md) |
| Original retrieval | [BM25-style rank and two-stage selector](../../editguard_benchmark/retrieval.py) |
| Iterative retrieval and development candidate selection | [Depth retriever](../../editguard_depth/retrieval.py) |
| Final answer contract | [Scoring code](../../editguard_benchmark/scoring.py) |
| Fourteen real website outputs and rollback snapshots | [India acceptance receipt](../../reports/reasonke/depth-final/india-acceptance.json) |
| Human-readable demonstration | [India operator guide](../INDIA_DEMO.md) |
| Bounded explanation audit | [Audit design](../EXPLANATION_AUDIT.md) |
| Our implementation packages and supporting evidence | [Contribution map](../CONTRIBUTIONS_AND_EVIDENCE.md) |
| Current presentation speech and team ownership allocation | [Verity speaker script](VERITY_SPEAKER_SCRIPT.md) |
| Complete project report under its historical title | [Word report](../../output/docx/EditGuard_Project_Report.docx) |

Primary research attribution: [ReasonKE authors' repository](https://github.com/YukinoshitaKaren/Reason-KE), [ReasonKE++ paper](https://arxiv.org/abs/2511.12661), [SERAC paper](https://arxiv.org/abs/2206.06520), [RAG paper](https://arxiv.org/abs/2005.11401), [ROME paper](https://arxiv.org/abs/2202.05262). Model identities: [ReasonKE_plus](https://huggingface.co/YukinoKaren/ReasonKE_plus), [official Qwen model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Code licence and checkpoint redistribution permissions are different questions; the released ReasonKE weight-licence metadata was not established by the inspected evidence, so the project does not claim a blanket redistribution right.
