# Verity: slide-by-slide technical study and defence guide

**Matches:** the 32-slide technical presentation released in commit `1fa7c2d` on 30 October 2026. Slides 1–29 are the core presentation. Slides 30–32 are optional technical backups.

**Audience:** someone joining the project now, including a reader who has never trained a language model, implemented retrieval or read a knowledge-editing paper.

**Purpose:** understand each slide, the mechanisms underneath it, why we made the documented choices, how to interpret every important number, and how to answer likely panel questions. No document can anticipate every possible question. When evidence is absent, the correct answer is “we did not measure that,” followed by the experiment that would be needed. This guide supplies those boundaries as well as the positive findings.

Read this alongside the [presentation](../../output/presentation/Verity_Technical_Presentation.pptx). The [speaker script](VERITY_SPEAKER_SCRIPT.md) is what to say aloud. This guide is the deeper explanation behind that speech. The [team handbook](VERITY_TEAM_HANDBOOK.md) adds launch and operation instructions.

## Reading route

1. Read the foundations below before memorising any numbers.
2. Study slides 1–16 to understand the problem, method and complete implementation.
3. Work through slides 17–23 with a calculator. These are the main scientific findings.
4. Study slides 24–29 to connect the evidence to the website and conclusion.
5. Use slides 30–32 and the final appendices for difficult technical questions.

| Slide | Subject | Jump |
| --- | --- | --- |
| 1 | Objective and system identity | [Slide 1](#slide-01) |
| 2 | India ownership-change example | [Slide 2](#slide-02) |
| 3 | Bounded task and acceptance | [Slide 3](#slide-03) |
| 4 | Method choice and recorded decisions | [Slide 4](#slide-04) |
| 5 | Published SFT/PPO learning pipeline | [Slide 5](#slide-05) |
| 6 | Structured stages and reward | [Slide 6](#slide-06) |
| 7 | Inference architecture | [Slide 7](#slide-07) |
| 8 | Request sequence and persistence | [Slide 8](#slide-08) |
| 9 | Identity, authority and lineage | [Slide 9](#slide-09) |
| 10 | Transactions and atomic rollback | [Slide 10](#slide-10) |
| 11 | Iterative lexical retrieval | [Slide 11](#slide-11) |
| 12 | Retrieval development selection | [Slide 12](#slide-12) |
| 13 | Prompt and history policy | [Slide 13](#slide-13) |
| 14 | Model configuration and extraction | [Slide 14](#slide-14) |
| 15 | Datasets and splits | [Slide 15](#slide-15) |
| 16 | Baselines and diagnostics | [Slide 16](#slide-16) |
| 17 | Scoring and uncertainty | [Slide 17](#slide-17) |
| 18 | Original operations findings | [Slide 18](#slide-18) |
| 19 | Separate depth findings | [Slide 19](#slide-19) |
| 20 | Reason-KE++ full-memory failures | [Slide 20](#slide-20) |
| 21 | Qwen abstentions and mismatches | [Slide 21](#slide-21) |
| 22 | Retrieval support and preservation | [Slide 22](#slide-22) |
| 23 | Accuracy, coverage and latency | [Slide 23](#slide-23) |
| 24 | Correction-editor screenshot | [Slide 24](#slide-24) |
| 25 | History and application acceptance | [Slide 25](#slide-25) |
| 26 | Stored chat versus fresh inference | [Slide 26](#slide-26) |
| 27 | Results explorer and evidence | [Slide 27](#slide-27) |
| 28 | Engineering contribution | [Slide 28](#slide-28) |
| 29 | Definite conclusion and limits | [Slide 29](#slide-29) |
| 30 | PPO mathematics and settings | [Slide 30](#slide-30) |
| 31 | Individual answer-contract failures | [Slide 31](#slide-31) |
| 32 | Evidence and configuration map | [Slide 32](#slide-32) |

## Foundations: understand the project before the slides

### The project in one paragraph

Verity is a locally deployable application for maintaining the factual behaviour of a conversational assistant. An operator approves a replacement fact in versioned external memory. A deterministic retriever selects a small context from that memory. A frozen, previously trained Reason-KE++ checkpoint generates an answer using the selected information and the question. The application records the answer, exact supplied facts, model identity and memory revision. It supports supersession, conflicts, historical comparison and rollback. We evaluated controlled fictional operations handbooks with direct questions, paraphrases, connected questions and unrelated controls.

### Three different things that must stay separate

| Layer | What changes or runs | Who supplied it | What evidence establishes it |
| --- | --- | --- | --- |
| Published learning method | SFT followed by process-aware PPO produces learned weights | Reason-KE++ researchers | Primary paper and official code |
| Verity implementation | Approval/state/history/retrieval/serving/evaluation/application | This project | Source, tests, protocols and operating receipts |
| A runtime knowledge edit | An approved external-memory record changes | The operator through the workflow | New revision and audit events, followed by actual generated answers |

We use **Reason-KE++** as the method name and **`YukinoKaren/ReasonKE_plus`** as the exact model identifier. We did not execute this checkpoint's SFT/PPO training. We did train auxiliary models in an earlier, separately preserved support-editor experiment. Those earlier weights are not the source of the current Reason-KE++ results.

### Why model weights, memory and answers are different

A language model contains learned numerical parameters, usually called weights. During inference, it predicts successive tokens from its input and previously generated tokens. Training changes weights. Supplying a new fact in context changes the input and can change the answer while keeping the weights fixed.

Our runtime contract can be written as:

```text
approved memory M_r, question q, compatible history h
    selected facts C = retrieve(q, M_r)
    generated response y = model_theta(prompt(q, C, h))
    persist(q, y, C, memory_revision=r, model_identity=theta)
```

`r` identifies memory state. `theta` identifies neural weights. The model does not permanently learn each approved fact. The next request needs the applicable memory again. Rollback changes `M_r`, not `theta`.

### Three meanings of “correct”

1. **Authoritative state:** the database contains the approved version and rejects an invalid competing mutation.
2. **Retrieved support:** the selected context contains the facts needed by the labelled question.
3. **Generated answer:** the model actually endorses the expected final value.

A database can be correct while retrieval fails. Retrieval can succeed while generation abstains or chooses a distractor. A correct final answer can coexist with an unsupported explanation. Our tests and metrics distinguish these cases.

### The main result, without overclaiming

The retrieved Reason-KE++ condition answered 256/256 main questions in each of two separate controlled studies. Retrieved Qwen answered 241/256 in the original study and 248/256 in the depth challenge, with its remaining main responses abstaining. Reason-KE++ was substantially slower. Full-memory Reason-KE++ made 25 strict wrong answers in the original study. The correct interpretation is a successful bounded configuration with visible limitations, rather than a universally reliable or universally superior model.

<a id="slide-01"></a>
## Slide 1: Verity

### What the slide establishes

The goal is maintainable conversational behaviour: authoritative corrections should transfer to appropriate questions, including questions that connect several facts. We also want unaffected information to remain correct and every change to be reversible and inspectable.

“Knowledge editing” can describe parameter-changing methods or external-memory methods. Our presentation states the latter route explicitly. A useful formal objective is:

```text
For questions affected by edit e: use the revised fact and relevant consequences.
For other questions: use their applicable facts.
For every change: retain its source, scope, history and recovery path.
```

These are system objectives. A universal mathematical guarantee for arbitrary natural language is not part of the completed project.

### Why this is a problem even if a company can edit its website

Editing the source establishes the desired fact. An assistant still has to select that source, prefer the applicable revision, interpret the user's wording and use it correctly. A service's new owner may change which on-call person's office answers a question, even though the office record itself did not change. The project studies that transfer and the operational control around it.

An organisation with five static facts might reasonably use a page or form. Our defensible use case is an existing conversational knowledge interface over connected, changing operational information. We have not established that every organisation needs this interface.

### Likely questions

**What exactly is the input and output?** A natural-language question, approved fact context and permitted conversation history enter the system. The neural model returns generated text. The application extracts the final span and stores its evidence.

**Where is ML central?** A learned generative model interprets and answers the question. The final measurements evaluate its actual output. The governance and lexical retrieval components are deterministic software.

**Did we invent Reason-KE++?** No. Our contribution is the surrounding platform, adaptations, independent controlled data, evaluation and analysis. Attribute the learning method when introducing slide 5.

**Can we claim an unrestricted AI assistant?** The UI supports ordinary text conversation, but the reported factual-accuracy claim concerns short answers in the controlled task.

**Evidence:** [architecture](../CURRENT_ARCHITECTURE.md), [contribution map](../CONTRIBUTIONS_AND_EVIDENCE.md).

<a id="slide-02"></a>
## Slide 2: One ownership edit changes a connected answer

### Read the chain from left to right

The controlled India example uses real Indian city names with fictional services, people and organisational assignments. It comes from the actual implemented demonstration, not a claim about a real payment company.

```text
Initial memory:
Kaveri Payment Gateway, release 2
  owner -> Bengaluru Platform Team
  on-call contact of that team -> Meera Iyer
  office city of that person -> Chennai

After one approved ownership replacement:
Kaveri Payment Gateway, release 2
  owner -> Pune Reliability Team
  on-call contact of that team -> Arjun Deshmukh
  office city of that person -> Hyderabad
```

The edited record changes only the service's owner. Both teams' contact records and both people's office records already exist. We do not insert the three-hop answer “Hyderabad” as the answer to the long question during the edit. The model must use the newly selected chain.

### What a hop means

A hop is one relation traversal. “Who owns the service?” needs ownership: one hop. “Who is on call for its owner?” needs ownership and contact: two hops. “In which city is that person's office?” adds office city: three hops. Counting words or clauses does not determine hop count.

The hosting city remains Bengaluru because hosting is a different relation. Release 1 retains Chennai Legacy Team. The similarly named Pro service retains Hyderabad Integration Team. Those questions test locality, version scope and entity scope.

### Why use controlled fictional facts?

We can state exact before/after truth, build difficult adjacent cases and audit the expected chain. Public current-affairs facts can be ambiguous, change during evaluation or already be memorised by a checkpoint. Fictional data makes this first evaluation easier to interpret. Its artificiality limits external validity.

### Likely questions

**Did we edit three facts to get the new city?** No. Only ownership changes. Existing downstream relations produce the new expected answer.

**Does rollback need to reverse the contact and office records?** No. Restoring the original owner restores the original path. The other team's valid contact and office facts remain available for questions about that team.

**Is this real-world verified information?** It is a real software workflow over a clearly fictional handbook. The city labels are familiar locations, not evidence about an actual organisation.

**Could SQL solve this chain?** A correctly specified structured join could. Our input is natural language, and our experiment measures a generative assistant's use of the resulting context. We did not prove superiority to a semantic parser plus SQL.

**Evidence:** [India guide](../INDIA_DEMO.md), [fixture construction](../../editguard_chat/examples.py), [actual acceptance receipt](../../reports/reasonke/depth-final/india-acceptance.json).

<a id="slide-03"></a>
## Slide 3: The task and acceptance contract

### What “bounded” means

We constrain the scientific task to factual questions about explicit operations facts, short endorsed answers, represented entity/release scopes and paths of at most three relations. We evaluate controlled paraphrases and nearby out-of-scope questions. We do not measure arbitrary dialogue, document synthesis, mathematical reasoning, ambiguous policy interpretation or unrestricted long-chain inference.

### The acceptance criteria, one by one

| Requirement | How we check it | What a pass does not establish |
| --- | --- | --- |
| Direct correction | Final span matches updated target | Every future wording will work |
| Paraphrase transfer | Separate wording variants match | Universal semantic equivalence |
| Connected transfer | Correct endpoint on a labelled path | Faithful explanation on every trace |
| Locality | Outside-scope answers remain factually correct | Perfect semantic scope detection |
| Governed authority | Only explicit publication changes approved memory | Source truth is independently certified |
| Conflict handling | Competing active key is rejected | All contradictions across different keys are detected |
| Rollback | State/events restore atomically, then fresh answers are checked | Every model request is deterministic across hardware |

### Why define acceptance before looking at the test?

Without a fixed contract, a result can be redefined as successful after seeing an output. For example, counting any target mention would reward “Hyderabad is not the answer.” Removing abstentions would make a cautious model look more accurate than it was over assigned questions. Fixed labels, aliases and denominators prevent these changes.

### Likely questions

**What does the database guarantee?** Represented-key invariants and transactional state changes. Natural-language applicability remains partly dependent on retrieval and generation.

**Does approval mean an external fact checker certified the source?** No. It means the designated operator made that version authoritative in this workspace. A mistaken operator can approve a mistaken fact.

**Why not guarantee 100%?** Language coverage, retrieval quality and learned generation are empirical properties. Transactions can provide strong local invariants; model correctness on unbounded questions cannot be inferred from them.

**What counts as complete if there are failures?** Complete implementation and a finished, reproducible evaluation with all outcomes disclosed. Completion does not require erasing measured failures or attaining an arbitrary universal score.

**Evidence:** [architecture guarantees](../CURRENT_ARCHITECTURE.md), [scoring contract](../../editguard_benchmark/scoring.py), [store tests](../../tests/test_chat_store.py).

<a id="slide-04"></a>
## Slide 4: Method choice and the recorded decisions

### Distinguish the three approaches

**Conventional updated RAG:** update the knowledge source, retrieve relevant records and ask a general instruction model to answer from them. RAG describes how context reaches the generator.

**Reasoning-aware use of corrections:** supply approved updates to a model trained to use them in connected questions. Reason-KE++ describes a learned behaviour that can operate on retrieved context. These approaches can coexist in the same pipeline.

**Parametric editing:** update selected neural parameters, or train an editor that produces weight changes. The correction is then represented through changed weights. That introduces different preservation, edit-interference and recovery questions.

Our final runtime uses the first two together. New corrections do not change weights. Do not describe the database transaction as a neural parameter edit.

### Why this choice was practical

The released checkpoint allowed us to deploy a learned editing reasoner on available T4 hardware without reproducing an eight-GPU training pipeline. External memory gave us explicit authority, inspection and reversal. Those properties matched the actual operational objective. This is a documented integration choice, not proof that parametric approaches are inferior.

### What the development comparison actually selected

The recorded operations development subset had three lineages and 24 questions per condition. Retrieved Reason-KE++ answered 24/24; full-memory Reason-KE++ answered 21/24. Retrieved Qwen also answered 24/24; full-memory Qwen answered 22/24. The criterion selected the context policy for the existing reasoner. It did not select a model by final-test performance or establish a Reason-KE++ advantage on development.

### Likely questions

**Is Reason-KE++ “better than RAG”?** That wording is too broad because our Reason-KE++ configuration uses retrieval. Say which generator, context policy, workload and metric are being compared.

**Why not use WISE, ROME or another editor?** The final studies did not compare those methods. Our chosen route delivered reversible context updates with feasible measured inference. A superiority claim would require implementing and evaluating the alternatives fairly.

**Was this the plan from day one?** No. Earlier support and visual experiments remain documented. The final roadmap refined the task after those experiences. The historical appendix below gives the supported timeline.

**Why not choose Qwen since it tied in development?** The final system already integrated the released editing reasoner. That development test chose retrieval, not a winner between checkpoints. We subsequently measured their accuracy–latency tradeoff under frozen conditions.

**Evidence:** [policy record](../../configs/deployment-policy.json), [development results](../../reports/reasonke/operations-development-v1/results.json), [integration decisions](../REASONKE_INTEGRATION.md).

<a id="slide-05"></a>
## Slide 5: Reason-KE++ learning pipeline

### Which training are we discussing?

This slide explains the published method behind the reused checkpoint. It does not show training performed during our final deployment. The paper describes cold-start supervised fine-tuning followed by reinforcement learning with intermediate-process and final-answer feedback. Its Appendix A reports 718 curated cold-start instances and an 8,296/922 PPO partition. These are reference-study counts, not our evaluation datasets. [Primary paper](https://aclanthology.org/2026.findings-acl.1137/)

### How data and supervised learning fit together

The inspected official generator constructs a rewritten fact and pairs it with a portability question, requesting a structured teacher response through `gpt-4o-mini`. “Portability” here concerns answering a related question under the edit. Our operations worlds were independently authored; they were not used by us to reproduce this teacher-data stage. [Official generator](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/data_preprocessing/generate_sft_data.py)

Supervised fine-tuning teaches the model to predict an example response token by token. In a simplified assistant-response objective:

```text
L_SFT = - sum_t log P_theta(target_token_t | prompt, earlier_target_tokens)
```

`theta` is the set of trainable weights. Minimising this loss increases the probability of the teacher's response. The official trainer attempts assistant-completion-only loss, with a general language-model collator fallback if the specialised TRL collator is unavailable. That software branch is a reproduction caveat, not proof of which branch created the released weights. [Official SFT trainer](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py)

### Why add PPO after SFT?

Imitating a format does not necessarily teach correct factual associations. A model can write plausible stages while ignoring an update. In reinforcement learning, the current policy generates responses, an evaluation function scores them, and optimisation shifts the policy toward higher-reward behaviour. PPO constrains the update rather than treating one high-reward sample as unlimited evidence for a huge change. Slide 30 explains the clipped objective.

### Likely questions

**Where are our final training-loss curves?** There are none for this reused checkpoint because we did not train it. Our final project measures inference, state management and evaluation. Earlier auxiliary-model training is a separate experiment.

**What is the teacher?** A model used to construct target examples. Its outputs are supervision, not automatically true facts.

**Is PPO happening when I press Approve?** No. Approval writes external memory. Runtime generation does not run an optimiser, a critic or reward training.

**Does `_plus` prove the exact paper experiment?** It identifies the released checkpoint we selected. Our inspected release does not certify that this precise weight revision produced a particular paper score. Keep the model ID and revision pinned and avoid claiming a faithful training reproduction.

**Evidence:** [method verification note](REASONKE_PLUS_PLUS_TECHNICAL_NOTES.md), [pinned deployment manifest](../../configs/reasonke-model.json).

<a id="slide-06"></a>
## Slide 6: Structured reasoning and stage-aware rewards

### What the tags mean

The structured response names the update, decomposes the task, gives indexed intermediate answers and emits a final answer. The literal delimiters make the text parseable during training. They are not separate software agents. One autoregressive model can generate the entire sequence.

```text
<think>
  <acknowledge>recognise the supplied update</acknowledge>
  <decompose>describe the needed subquestions</decompose>
  <action>
    [Sub question 1] owner question ... \boxed{team}
    [Sub question 2] contact question ... \boxed{person}
    [Sub question 3] office question ... \boxed{city}
  </action>
</think>
<answer>city</answer>
```

This is an explanatory template, not a newly measured model output.

### What each reward component measures

The pinned reward code rejects malformed structure with −1. For valid positive-hop cases it adds `0.2H + 0.4D + 0.4A + F`: correct hop count `H`, aligned subquestion similarity `D`, correct boxed-subanswer fraction `A`, and final token F1 `F`. The similarity uses MiniLM on action-section subquestions when hop count agrees. The acknowledgement's presence/order is checked; this is not independent semantic certification of every acknowledgement. [Official reward code](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py)

For an illustrative calculation, let `H=1`, `D=0.8`, two of three boxed subanswers match, and `F=1`. Then reward is `0.2 + 0.32 + 0.4*(2/3) + 1 ≈ 1.787`. This is teaching arithmetic, not a logged training observation.

### Why process feedback matters

An endpoint can be right through an invalid association, accidental overlap or a memorised shortcut. Checking intermediate components can penalise some such traces during training. However, correct hop count alone is weak: three wrong subquestions still have the expected count. Semantic similarity is also imperfect. A nonzero reward therefore is not a proof of reasoning validity.

### Likely questions

**Are these rewards used by our website?** No. The website runs inference and records output/evidence. It does not compute this upstream training reward to certify every response.

**Is the reward exactly the paper's equation?** The paper describes a normalised equal blend at its main `alpha=0.5`; the pinned implementation sums process and outcome. The relative component weighting agrees, but the overall scale differs. Slide 30 covers why the KL relationship can matter.

**Does final token F1 equal our final accuracy?** No. Upstream reward F1 measures token overlap. Our evaluator uses frozen whole-final-span aliases. A person plus extra team text can receive token overlap while failing our answer contract.

**Does dense supervision mean every token is independently verified?** No. There are multiple component signals about a generated response. We do not claim a separate trusted label for every token or faithful access to the model's internal computation.

**Evidence:** [verified distinctions](REASONKE_PLUS_PLUS_TECHNICAL_NOTES.md), [our extraction](../../editguard_reasonke/runtime.py), [our scorer](../../editguard_benchmark/scoring.py).

<a id="slide-07"></a>
## Slide 7: The implemented inference architecture

### Follow the responsibilities

| Component | Concrete responsibility | Deterministic or learned? |
| --- | --- | --- |
| Browser | Gather input, show state and evidence | Application code |
| Python API | Validate, coordinate operations and persist outcomes | Application code |
| SQLite | Approved versions, events, chats and revisions | Database operations |
| Context selector | Rank records and expand answer-text frontier | Deterministic lexical algorithm |
| History filter | Select permitted complete conversation pairs | Deterministic rules |
| Private worker | Tokenise prompt and generate text with fixed weights | Learned neural inference |
| Evidence/display layer | Preserve and expose actual inputs/output metadata | Application code |

The diagram separates authority from generation. A customer message cannot update approved memory simply because it appears in a prompt. The API obtains facts from the governed store.

### Why separate the website and model worker?

The lightweight application can run without loading billions of parameters. The GPU process owns the expensive checkpoint, ML dependencies and inference lock. The application talks to it through an explicit contract. Worker unavailability becomes a visible error rather than a silently substituted generator or template answer.

The worker uses an authentication token and loopback binding in the documented deployment. These controls reduce accidental access in a research setup. They do not create enterprise user roles, distributed access control or a public service's full security model.

### Likely questions

**Why Python and SQLite rather than a large microservice stack?** They support a complete local workflow with few dependencies and clear transaction semantics. We had no measured requirement justifying a distributed database or message queue.

**Where is the graph database?** The evaluation ground truth is graph-shaped, but runtime approved memory is relational storage and retrieval uses fact text. We did not deploy a graph database or a symbolic graph-query planner.

**Can the website work with the worker offline?** Existing chats, approved state and saved results can be inspected. Fresh model answers require a ready worker. A recorded answer on screen does not prove readiness.

**Is the 200-record limit tested neural capacity?** No. It is an input guard for the retrieved policy. Neural evaluation used nine- and twelve-fact worlds.

**Evidence:** [server](../../editguard_chat/server.py), [fact-context adapters](../../editguard_chat/reasonke_backend.py), [runtime](../../editguard_reasonke/runtime.py).

<a id="slide-08"></a>
## Slide 8: A question through the actual request path

### Step-by-step trace after rollback

1. The browser sends the current question and chat identifier to `POST /api/chat` in edited mode.
2. `mutate()` acquires the application operation lock without waiting indefinitely. A competing operation receives HTTP 409.
3. `ask()` validates identifiers, message and mode, checks model readiness, obtains history and captures knowledge revision 11.
4. The selector reads the active approved snapshot. It chooses four fact records under the iterative policy.
5. The API filters conversation history and sends question, selected facts and permitted history to the worker.
6. The worker validates again, builds/tokenises the prompt, checks token length, generates, extracts the final text and returns metadata.
7. Only after successful generation does `add_turn()` save the user message and assistant message together with evidence.
8. The API returns the stored turn. The UI displays the answer and the exact sources associated with it.

### Why the lock and transaction are different

The application lock spans the long operation to keep memory revision consistent while generation runs. A SQLite write transaction stays short and stores the completed turn afterward. Holding a database write transaction open for a roughly sixteen-second model call would unnecessarily occupy database writer state. Dropping the application lock before generation could allow an edit to race with the captured revision unless another snapshot-consistency design replaced it.

### What failure means here

If the worker is unavailable, input is too large or inference fails, the operation returns an explicit error. It does not store a newly dangling user turn and does not substitute the expected database answer. A failure before persistence therefore differs from a generated-but-wrong answer in the scientific ledger.

### Likely questions

**Can two operators edit concurrently during one generation?** The current local application serialises these operations with its process lock. We do not claim distributed multi-server consistency.

**Does the lock make the model deterministic?** No. It fixes the application state seen by the request. Greedy decoding is a separate generation choice and still does not establish bit-identical computation across hardware.

**What if storing the turn fails after generation succeeds?** The message transaction must fail atomically rather than save half a turn. The model's computation already happened; database rollback cannot undo GPU work. Do not confuse state atomicity with exactly-once inference across arbitrary crashes.

**Is this timing what slide 23 reports?** Slide 23 measures benchmark context selection, local IPC and generation. It excludes the website/browser network path. HTTP acceptance has a different timing boundary.

**Evidence:** [`mutate`, `ask`, `add_turn`](../../editguard_chat/server.py), [message transaction](../../editguard_chat/store.py), [request tests](../../tests/test_chat_server.py).

<a id="slide-09"></a>
## Slide 9: Fact identity, authority and version lineage

### Why a question string is insufficient identity

“Who owns Kaveri release 2?” and “Name the owning team for Kaveri's second release” can refer to the same fact. Conversely, replacing “release 2” with “release 1” changes the scope. A safe store should not let wording alone decide whether two approved records compete.

For the bounded catalogue we use three explicit identifiers:

```json
{
  "entity_id": "india-demo-001-service",
  "relation_id": "owner",
  "scope_id": "release-2"
}
```

The store encodes entity/relation into `question_key` and scope into `scope_key`. It requires all three canonical IDs together. Optional aliases are retained as metadata. Without canonical IDs, legacy entries use normalised question and scope text. This fallback has weaker protection against paraphrased duplicate identities.

### What else is stored and why

| Field family | Purpose |
| --- | --- |
| ID, lineage ID, version | Distinguish one record from its edit family and its version number |
| Question, answer, scope | Human-readable fact and its applicability description |
| Canonical identity and aliases | Stable represented subject/property/scope |
| Source | Record where the operator says the fact came from |
| Status | Active, superseded or withdrawn |
| `supersedes`, `superseded_by` | Link replacement and predecessor |
| Approval actor/timestamps | Record the explicit approval operation |
| Knowledge revision | Link the state change to a committed memory state |

A browser draft is not a fourth approved database status in this workflow. It remains unapproved interface input until publication. Approved content stays inspectable through historical version records; status/link fields can change as the lifecycle advances.

### Why the partial unique index matters

The database index enforces at most one active row for a represented `(question_key, scope_key)`. Older versions remain stored. An unconditional unique index would prevent retaining multiple versions; no unique constraint would permit conflicting active authority.

### Likely questions

**Do aliases automatically resolve arbitrary entity names?** No. They are metadata, not a trained entity-linking model. Runtime retrieval chiefly uses question, scope and answer text.

**What if an operator supplies the wrong canonical ID?** The system cannot infer every intended real-world identity. It can faithfully enforce the wrong represented key. Operator review and broader identity validation remain necessary.

**Can contradictory facts under different keys slip through?** Yes. Same-key conflict rejection is deterministic; general semantic contradiction detection is not implemented.

**Is `approved_by` a logged-in enterprise user?** The current implementation records a local research operator. Do not claim full role-based administrator authentication.

**Do approval timestamps implement effective-date reasoning?** They record approval/state-change time. Historical snapshots reconstruct committed memory. They do not implement a general engine for valid-from/valid-to intervals or arbitrary natural-language dates. Explicit release scope and selected revisions are the supported temporal controls.

**Evidence:** [`correction_identity` and schema](../../editguard_chat/store.py), [workspace validation](../../editguard_chat/workspace.py), [API](../CHAT_API.md).

<a id="slide-10"></a>
## Slide 10: Atomic supersession and rollback

### Understand the three operations

**Publish:** create a first active version for a free identity. **Supersede:** explicitly replace the currently active version for the same identity. **Rollback:** withdraw the active replacement and restore its direct predecessor in one transaction. Withdrawal alone removes an active record; it does not necessarily restore a predecessor. These actions have different semantics.

An ownership replacement conceptually performs:

```text
BEGIN IMMEDIATE
  validate the currently active predecessor
  allocate next version and one new knowledge revision
  mark predecessor superseded
  insert active replacement with same lineage
  append predecessor-superseded and replacement-published events
COMMIT
```

Rollback conceptually performs:

```text
BEGIN IMMEDIATE
  find active replacement and direct predecessor
  validate the shared represented identity and permitted state
  allocate one new knowledge revision
  mark replacement withdrawn
  restore predecessor to active
  append withdrawn and restored events at that same revision
COMMIT
```

An exception triggers `ROLLBACK`, so none of those writes commits. `BEGIN IMMEDIATE` reserves the SQLite writer transaction early. It is not the same as a universal exclusive lock blocking every reader. The application also enables foreign keys, WAL and a busy timeout.

### Why two events share one revision

The two events describe one externally visible memory transition. Exposing a committed intermediate state with neither valid owner, or with both owners active, would violate the intended contract. Revision 11 therefore represents the restored state as a whole.

Historical reads replay events up to a chosen revision without changing current memory. Repeating a completed rollback can return the already-restored predecessor without another state change. An obsolete replacement cannot roll back over a newer successor.

### How we tested the important boundary

Tests inject a failure during a late event write. The expected outcome is no partial status change, no partial revision increment and no incomplete event history. Ordinary success-path tests alone would not expose that failure mode. These are database reliability tests, separate from neural accuracy.

### Likely questions

**Why not delete the bad correction?** Deletion would lose history and its connection to saved answers. Withdrawal plus events preserves an explanation of what happened.

**Does rollback restore the old model checkpoint?** No. It restores approved fact state; the checkpoint stays fixed.

**What about caches?** Mutation calls the retriever's cache-clear interface. The current lexical implementation has no stored embedding/index scores or answer cache to invalidate. Do not claim an implemented vector-index rebuild.

**Does this survive an application restart?** Committed SQLite state is persistent. Unsaved Colab VM files still require export before the VM disappears. Local transaction durability and remote-runtime retention are different questions.

**Evidence:** [store transactions/lifecycle](../../editguard_chat/store.py), [late-failure regression tests](../../tests/test_chat_store.py), [state import checks](../../tests/test_chat_workspace.py).

<a id="slide-11"></a>
## Slide 11: Bounded iterative retrieval algorithm

### First understand the ranker

Each approved record becomes a small document containing its question, scope and answer. The ranker lowercases text, extracts alphanumeric tokens, removes a declared stop-word set, and scores matches to the query. It uses a BM25-style term-frequency score with `k1=1.2` and `b=0.75`. Rare matching terms get more weight; repeated terms have diminishing benefit; document length is normalised. Ties use fact ID order.

The implemented per-term contribution is:

```text
idf(t) = log(1 + (N - df(t) + 0.5)/(df(t) + 0.5))
score(q,d) = sum over unique query terms t of
  idf(t) * tf(t,d)*(1.2+1)
  / [tf(t,d) + 1.2*(1-0.75 + 0.75*len(d)/average_len)]
```

`N` is the number of candidate documents, `df` counts documents containing a term and `tf` counts that term in a document. This is a ranking score, not an applicability probability. It is not a learned cross-encoder or embedding model.

### Then understand frontier expansion

```text
selected = top 2 facts ranked by the user's question
frontier = concatenated answers of selected seed facts

for at most 3 rounds:
    stop if 4 facts selected or no unseen facts remain
    next = highest-ranked unseen fact for frontier
    append next
    frontier = next.answer

fill unused slots, if any, using the last frontier ranking
return selected
```

With two seeds and cap four, the deployed choice normally needs at most two additions. The configurable family permits caps four, five and six; the final application selects four. Each new ranking uses its current candidate set. There is no learned relevance threshold or stop-on-low-confidence rule; even zero-score ties can be ordered by ID. These are important limits under unfamiliar language.

For the India office question, hosting and ownership can be the seeds. Ownership provides the team name, which retrieves its contact; that person's name retrieves the office. A hosting distractor remains in the four-fact context. The algorithm does not magically know the gold three-hop path.

### Why expand approved answer text?

The team name needed for hop two may not appear in the user's original question. Once an ownership fact provides it, that approved value becomes a useful retrieval query. This is different from reading the hidden expected answer: the selector sees only memory fields, never target labels, depth labels or required support IDs.

### Likely questions

**Is this symbolic graph traversal?** It is lexical traversal through answer text. It approximates connected retrieval but has no runtime typed-edge planner or graph-join guarantee.

**Why only four facts?** Development evidence supported the smallest successful declared budget. Fewer distractors and bounded context are practical benefits, not a guarantee that four is universally optimal.

**Why not embeddings?** We implemented and evaluated a simple transparent baseline/adaptation. A semantic retriever could be a future controlled comparison; we did not measure its benefit in these final studies.

**Can it handle cycles or long paths?** Selected IDs prevent duplicate records. The finite round/context caps stop expansion. Success beyond the evaluated three relations is not established.

**Does an India-themed demonstration establish Hindi support?** No. The benchmark wording is English and the lexical tokenizer extracts `[a-z0-9]+` terms. Hindi-script retrieval can therefore fail even if the neural generator has multilingual capabilities. We did not measure multilingual editing accuracy.

**Where is the calibrated low-confidence threshold?** The final lexical selector has none. Its score is a ranking score; UNKNOWN is a generated abstention, not calibrated confidence. Same-key conflict clarification is a deterministic control. The earlier learned scope classifier belongs to a different preserved experiment.

**Evidence:** [ranker and original selector](../../editguard_benchmark/retrieval.py), [iterative selector](../../editguard_depth/retrieval.py).

<a id="slide-12"></a>
## Slide 12: Retrieval selection before final evaluation

### What the chart measures

This chart concerns **development support retrieval**, not generated answers. A path is complete if every required supporting fact for the labelled question is in the selected context. Fact recall instead counts individual required fact occurrences. Neither measure asks whether a model ultimately answered correctly.

The 128-question depth development packet has 32 questions at each depth category: locality, one hop, two hops and three hops. The original two-stage selector completed 96/128 paths: all of the first three categories and 0/32 three-hop paths. It found 192/224 required fact occurrences. The missing third-hop facts explain why a separate iterative design was justified.

### What changed and how selection remained finite

We declared six candidates: one or two seeds, a fixed three-round limit, and cap four, five or six. All six completed the packet's 128/128 paths and 224/224 required occurrences. Packet-only selection initially favoured one seed and four facts.

Before neural profiling and protocol freeze, an explicit India display-development question exposed a hosting seed distractor. The same six candidates were checked on both initial and updated display memory states. The selected policy became two seeds and four facts: complete packet support plus both display paths, 230/230 required occurrences over those selection inputs.

The two extra memory states are development controls for the display scenario, not two additional independent held-out test questions. This adjustment is documented as selection revision two. The old operations retriever and original results remain unchanged.

### Why not just increase every context budget?

A larger context can contain the needed chain but also more distractors. Retrieval quality and generation quality are different. Full-memory Reason-KE++ failures later show why including every available fact need not improve generation. We selected the smallest eligible declared cap rather than treating more context as automatically better.

### Likely questions

**Did you tune retrieval on the final test?** The selection receipt records development questions and display-development states. Final question labels and model outputs were not used to choose the policy.

**Is 100% retrieval here suspicious?** It describes a small controlled development packet with explicit lexical links. It supports feasibility for that design, not arbitrary large-corpus retrieval.

**Can you add the 230 facts to the final 448 facts?** No. They are different populations and count repeated occurrences, not unique entities or a growing training set.

**Did the support improvement prove neural improvement?** No. It removed a measured retrieval deficiency before final generation evaluation. Final neural accuracy is reported separately on slide 19.

**Evidence:** [candidate-selection receipt](../../reports/reasonke/depth-development/retrieval-selection.json), [selection implementation](../../editguard_depth/retrieval.py).

<a id="slide-13"></a>
## Slide 13: Prompt assembly and revision-safe history

### How approved facts enter the model

The worker serialises selected fact fields as JSON inside an updated-information block, followed by the query in the latest user message. The tokenizer then applies the checkpoint's chat template. We do not prepend a secret symbolic answer from the evaluator.

The conceptual message is:

```text
[Updated Information]:
JSON list of selected fact IDs, questions, answers, scope and source

[Query]:
the user's current question
```

The actual implementation also includes its update instruction. JSON provides explicit boundaries between fields; it is not a complete prompt-injection defence. Approved text can still contain problematic content.

### Why old history is dangerous after an edit

Suppose a stored assistant message says Bengaluru Platform Team owns the service. After approval, the source says Pune Reliability Team. Passing the old message into the next prompt creates contradictory context. We preserve it visibly as history but exclude its complete user–assistant pair when its knowledge revision is stale.

The filter also checks model compatibility and model-generation evidence. Normal edited chat may retain complete correction-conditioned pairs from the current revision. History is capped at eight messages and 6,000 total characters, dropping full pairs rather than leaving old unanswered user messages.

### How comparison differs from history display

A historical comparison calls the model twice: once with an approved historical snapshot and once with current approved facts. Both use the same pre-question history after removing correction-conditioned pairs. This prevents a newer answer from leaking the correction into the historical side.

“Without correction context” is another comparison mode. It gives the same trained checkpoint no approved fact context; it is not the original pretraining model. “Approved memory at revision 9” is the meaningful controlled before-handbook state in the India example. Comparison does not rewrite stored conversation messages.

### Likely questions

**Does a historical comparison retrieve an old stored answer?** No. It generates fresh answers using selected memory states. Opening a stored chat is the operation that displays saved output.

**Can a follow-up rely on an older revision's answer?** That pair is excluded from the prompt. The visible chat can contain context that the model no longer receives. Explicit subjects are safer during the factual demo.

**Why remove whole pairs?** Removing only the assistant response leaves an old question dangling, potentially distracting the model from the current question.

**Does a fresh answer have to reproduce an old recorded answer byte for byte?** No. The target and state can match while generated wording or execution details differ. The recorded output remains the evidence for its original observation.

**Evidence:** [history and comparison code](../../editguard_chat/server.py), [prompt assembly](../../editguard_reasonke/runtime.py), [history tests](../../tests/test_chat_reasonke.py).

<a id="slide-14"></a>
## Slide 14: Frozen model worker and answer evidence

### Understand every configuration number

| Setting | Meaning | Why it matters |
| --- | --- | --- |
| 7,615,616,512 parameters | Number of learned values in each selected checkpoint | Identifies model scale, not number of stored catalogue facts |
| Pinned model revision | Exact weight/config version | Avoids silently downloading changed weights |
| NF4 double quantization | Four-bit weight-storage adaptation with quantised scaling metadata | Reduces memory compared with storing every weight at full source precision |
| FP16 compute | Selected arithmetic dtype for the quantised deployment | Distinct from four-bit storage |
| SDPA | Selected scaled-dot-product attention implementation | Part of the reproducible runtime |
| Greedy, one beam | Highest-scoring next-token choice, no sampling | Controls decoding variability and budget |
| 4,096 input tokens | Maximum tokenised request length | Oversize input is rejected rather than silently changing the contract |
| 512 new tokens | Maximum generated continuation | Includes the explanation and answer, not just the final short name |
| Approximately 6.59 GiB | Measured loading peak allocated CUDA memory | Not total device usage or a universal minimum-VRAM guarantee |

The checkpoint is `YukinoKaren/ReasonKE_plus` at `ceaedc2ae7c8fed6257953587d991ce550db6106`. Qwen is `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28`. Each was loaded separately on the T4. Quantising one tensor class does not remove KV-cache, activation, allocator or runtime overhead. Four-bit storage therefore does not imply that a 6 GiB laptop can safely run the complete fixed workload.

### Extraction is parsing, not answer correction

Exactly one nonempty `<answer>...</answer>` block yields its inner text. Otherwise the runtime returns the raw text with fallback/format flags. Empty model output raises an error. The parser does not consult expected answers, replace a wrong city or repair multiple malformed final blocks into a scored success.

The response preserves raw output, supplied fact IDs, checkpoint revision, token counts, generation time, CUDA measurements and budget flags. The explicit `verified` flag remains false for generated text.

### Likely questions

**Is NF4 fine-tuning?** Our use is an inference loading adaptation. It is not a new per-edit optimisation run.

**Why can a short answer require 200 output tokens?** The checkpoint often emits structured reasoning before the final name. The total continuation budget covers all generated text.

**What happens after 120 seconds?** The Transformers generation setting is a soft time limit. The benchmark separately uses a hard 240-second parent request watchdog and 1,800-second loading watchdog. Do not describe the soft setting as an exact interrupt guarantee.

**Can we call the extracted answer verified because its source is displayed?** No. Extraction identifies the endorsed span; provenance shows inputs. Neither establishes factual correctness without evaluation.

**Does the final checkpoint process images?** This Reason-KE++ worker is text-only and reports image support as false. Earlier visual experiments are preserved separately; their capabilities and results are not part of the final text study.

**Evidence:** [Reason-KE++ configuration](../../configs/reasonke-model.json), [Qwen configuration](../../configs/instruction-model.json), [worker code](../../editguard_reasonke/runtime.py), [measured profiles](../../reports/reasonke/operations-profiles-v1/README.md).

<a id="slide-15"></a>
## Slide 15: Fact worlds, lineages and split control

### The counting units

A **world** is one authored fictional operations handbook with interconnected facts. An **edit lineage** groups the original state, replacement state and related question variants. A **question record** contains one wording, target, scope/category and required supporting facts. A **generation record** is one model condition's attempt at that question.

Each study has 48 authored worlds and 384 question records. Sixteen lineages produce 128 development questions; 32 lineages produce 256 final questions. Eight variants per lineage account for those totals. These are not 384 independent real companies, and 256 variants do not become 256 independent edit lineages.

The original operations worlds have nine current facts and questions up to two hops. The depth worlds have twelve facts and questions up to three hops. The latter also separates development and final wording banks. Service, team and person identities separate across partitions; simple values such as cities and relation vocabulary can repeat.

### How leakage control works

1. Assign underlying lineages to development or final.
2. Generate related wording and counterfactual variants within that assigned partition.
3. Keep required-support labels and expected aliases available to auditors/evaluators, not ordinary retrieval.
4. Select configurations on development evidence.
5. Freeze source/data/configuration/scoring identities before final inference.
6. Preserve the complete final ledger; do not retune prompts or aliases after seeing failures.

Randomly splitting paraphrases after augmentation would let nearly identical versions of the same fact enter both partitions. Our lineage-first split addresses that particular leakage route. It cannot eliminate all shared structure or establish independence from upstream public pretraining.

### Why automated review is disclosed

The graph audit reconstructs expected relations and checks IDs, targets, partitions and support paths. It is reproducible but shares assumptions with the authored controlled schema. Independent humans might notice unnatural wording or ambiguous meaning that mechanical consistency misses. The final labels were automatically audited; we do not claim independent human semantic review.

### Likely questions

**Where is the training split for these datasets?** They are independent development/final evaluation datasets for frozen checkpoints. We did not fine-tune the final generators on them.

**Why such a modest dataset?** It makes complete per-request recording, replay and failure inspection manageable. The final packet tests more than a showcase, but it remains a small controlled benchmark.

**Is the test set completely unlike development?** Underlying identities separate, and depth wording banks separate. Relation types, construction rules and simple labels still share structure. We disclose that limitation.

**Did an LLM judge create the final correctness score?** No. Explicit graph targets and fixed whole-span aliases drive scoring. Automated audits are not synonymous with an unrestricted LLM judge.

**Evidence:** [operations dataset card](../OPERATIONS_DATASET_CARD.md), [original authoring](../../editguard_benchmark/data.py), [depth authoring](../../editguard_depth/data.py), [depth dataset manifest](../../data/reasonke/depth-v1/manifest.json).

<a id="slide-16"></a>
## Slide 16: Matched conditions and diagnostic populations

### Decode the condition names

| Code | Generator | Supplied knowledge | Main or diagnostic? |
| --- | --- | --- | --- |
| R0 | Reason-KE++ | Initial handbook facts | Original main staleness reference |
| R-RAG | Reason-KE++ | Retrieved current approved facts | Main |
| R-ALL | Reason-KE++ | All current facts in the original world | Original main |
| G-RAG | Qwen instruction model | Same retrieved current fact IDs | Main |
| G-ALL | Qwen instruction model | All current facts in the original world | Original main |
| R-ORACLE | Reason-KE++ | Gold supporting facts for a fixed subset | Diagnostic |
| G-ORACLE | Qwen | Gold supporting facts for a fixed subset | Diagnostic |

`G` names the general instruction-model comparison; it does not mean a different ground-truth source. In the depth study `R-RAG`/`G-RAG` use the iterative selector, not the original two-stage selector. Always state the study when using these codes.

### Reconstruct the total workload

```text
Original operations:
5 main conditions * 256 questions + 2 oracle conditions * 16 questions
= 1,280 + 32 = 1,312 generations

Separate depth challenge:
2 main conditions * 256 questions + 2 oracle conditions * 16 questions
= 512 + 32 = 544 generations
```

These are actual evaluation requests, not dataset sizes or training examples. The studies remain separate. Repeated attempts at the same lineage in different conditions are paired comparisons, not new independent entities.

### What makes the comparison fair, and what remains unequal

The updated baselines get current approved truth. Retrieved fact IDs, question populations, compatible decoding budgets and scoring rules match. Qwen uses its official instruction template with short-answer/UNKNOWN instructions; Reason-KE++ uses its update-conditioned format. Appropriate templates differ because the models have different learned interfaces.

The precise pretrained ancestry of the released ReasonKE_plus revision is not established as this exact Qwen checkpoint. Thus we compare two fixed systems rather than isolating the causal effect of SFT or PPO. No final parametric-editing baseline was measured.

### Likely questions

**Why include R0?** It demonstrates staleness against updated truth. It does not make the updated RAG baseline stale.

**Why give an oracle?** Gold support provides a diagnostic view of answer generation when support selection is controlled. Its smaller population cannot replace main accuracy.

**Is full memory an oracle?** No. It contains all nine current facts, including distractors. Oracle context supplies the known required support for selected questions.

**Why not evaluate the depth full-memory condition too?** It was outside the frozen depth design. We do not add experiments merely to make the project appear larger or retrofit final conclusions.

**Evidence:** [operations protocol](../../reports/reasonke/operations-final/protocol.json), [depth protocol](../../reports/reasonke/depth-final/protocol.json), [configuration policy](../../configs/deployment-policy.json).

<a id="slide-17"></a>
## Slide 17: Scoring and paired uncertainty

### Accuracy, coverage and answered-only precision

Let `N` be all assigned questions, `C` correct responses and `A` nonabstaining answered responses. Our primary metrics are:

```text
accuracy = C/N
coverage = A/N
answered-only precision = C/A, when A > 0
```

Original retrieved Qwen has `N=256`, `C=241` and `A=241`. Accuracy and coverage are therefore 94.140625%, while answered-only precision is 100%. Saying only “Qwen was 100% correct when it answered” hides fifteen unfulfilled requests. We keep those requests in the primary denominator.

The outcome taxonomy is correct, strict wrong, abstain, execution error or unexecuted. Abstentions, execution errors and unexecuted cases receive zero primary accuracy credit. The completed final studies happened to contain no execution errors or unexecuted cases; that is a measured fact, not a scoring rule that removes such outcomes.

### What the scorer actually checks

It normalises Unicode with NFKC, case, whitespace, wrapping bold markers and limited trailing full stops. It matches the **entire endorsed final span** against aliases fixed for that question before final inference. It does not give substring credit to a target mentioned as a rejected possibility or in an unrelated explanation.

The target belongs to the requested subject/relation/scope in the authored ground truth. This is stronger than arbitrary token overlap, but it is not a general semantic proof that the model understood the subject or reasoned correctly. Two subjects can sometimes share a value; a matching value alone does not certify the explanation. No independent semantic review of every trace was performed.

### Why uncertainty is paired by lineage

Each model sees matched questions. A hard world can affect both, and variants from one world are correlated. For each of 10,000 bootstrap draws, we sample whole lineages with replacement, keep all their variants together, and recompute the difference between conditions. Seed 42 fixes the replay. The 2.5th and 97.5th percentiles give the reported 95% interval.

Resampling 256 variants as though each were an independent fact would overstate the effective evidence. The main studies have 32 final lineage clusters each. A narrow interval does not repair an unrepresentative dataset.

### Likely questions

**Is a confidence interval a 95% guarantee for future questions?** No. It describes uncertainty under the resampling design and observed controlled population, not universal generalisation.

**Why not use an LLM judge for everything?** Explicit targets make scoring inspectable and reproducible. A judge could help review complex outputs, but adds its own errors and cannot replace fixed denominators or ground truth.

**Why not accept “team plus person” after discovering a correct person in it?** That changes the answer contract after test inspection. We diagnose such cases separately while preserving the frozen primary score.

**Does zero bootstrap variance mean universal certainty?** No. If every sampled cluster has the same observed outcome, resampling may yield a zero-width interval. It cannot invent failures or capture unseen domains.

**Evidence:** [whole-span scorer](../../editguard_benchmark/scoring.py), [operations metrics](../../editguard_benchmark/metrics.py), [depth metrics](../../editguard_depth/metrics.py).

<a id="slide-18"></a>
## Slide 18: Original operations results

### Read every bar as 256 assigned questions

| Main condition | Correct | Strict wrong | Abstain | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| R0: initial facts | 136 | 120 | 0 | 53.125% |
| R-RAG: retrieved Reason-KE++ | 256 | 0 | 0 | 100% |
| R-ALL: full-memory Reason-KE++ | 231 | 25 | 0 | 90.234375% |
| G-RAG: retrieved Qwen | 241 | 0 | 15 | 94.140625% |
| G-ALL: full-memory Qwen | 241 | 13 | 2 | 94.140625% |

There are zero execution errors and zero unexecuted cases in these released main conditions. The chart's small abstention segments retain their true values; its caption explicitly gives 15 for retrieved Qwen and two for full-memory Qwen.

### What the comparisons mean

Retrieved Reason-KE++ exceeds retrieved Qwen by `15/256*100 = 5.859375` percentage points. Its paired lineage interval is `[3.90625, 8.203125]` points. The difference comes from fifteen Qwen abstentions, not fifteen confidently wrong answers.

Full-memory Reason-KE++ is **worse** than full-memory Qwen by `10/256*100 = 3.90625` points; the Reason-minus-Qwen interval is `[-7.8125, -0.390625]`. Reason-KE++ therefore does not win every context condition. Within Reason-KE++, full memory loses 25 correct outcomes relative to retrieval, a 9.765625-point difference.

R0 is evaluated against updated truth. Its 136 correct answers comprise 12/128 edit-applicable questions and 124/128 locality questions. The other 120 outcomes include more than simple staleness because R0 also has four locality errors. Do not describe its overall number as accuracy on the unchanged original handbook.

### What can be concluded

The selected retrieved configuration succeeds on this frozen population. Supplying all current facts does not ensure selective generation. The comparison motivates investigating context interference and generated associations; it does not identify a model-internal mechanism by itself.

### Likely questions

**How can full-memory Qwen and retrieved Qwen have the same accuracy but behave differently?** Their error composition differs. Retrieved Qwen abstains fifteen times. Full memory answers thirteen incorrectly and abstains twice. Same `C/N` does not mean same risk or coverage.

**Can we claim Reason-KE++ has 100% accuracy?** Say “256/256 assigned main answers in the original controlled retrieved condition.” Add the separate depth result and limits when relevant. Avoid dropping the population from the claim.

**Were wrong answers retried until correct?** No. Completed outcomes remain in the ledger. Resume handles unexecuted tasks under the fixed plan.

**Did updated RAG receive the correction?** Yes. Both retrieved generators receive the same selected current facts. It is a fair updated baseline.

**Evidence:** [original results](../../reports/reasonke/operations-final/results.json), [released study explanation](../../reports/reasonke/operations-final/README.md).

<a id="slide-19"></a>
## Slide 19: Separate depth challenge results

### Why this is a separate study

The original completed study tested paths up to two hops. The depth challenge added twelve-fact worlds, three-hop questions, different wording banks and a development-selected iterative retriever. It has its own protocol and final ledger. This is an extension, not a silent replacement of the original test or a pooled larger sample.

Each main condition has 64 questions in each of four groups:

| Group | Reason-KE++ correct | Qwen correct | Qwen abstentions |
| --- | ---: | ---: | ---: |
| Locality, labelled depth 0 | 64/64 | 64/64 | 0 |
| One hop | 64/64 | 64/64 | 0 |
| Two hops | 64/64 | 57/64 | 7 |
| Three hops | 64/64 | 63/64 | 1 |
| Total main | 256/256 | 248/256 | 8 |

Depth 0 is an out-of-edit-scope control label. It does not mean an answer needs no supporting fact. A hosting fact or older-release owner can still be required to answer it.

### Interpret the uncertainty correctly

The main difference is `8/256*100 = 3.125` points, with interval `[1.5625, 5.078125]`. The three-hop difference is `1/64*100 = 1.5625` points, with interval `[0, 4.6875]`. The latter includes zero, so do not claim a decisive general three-hop advantage based on one discordant question.

Both oracle groups answered 16/16. None of the eight main Qwen failure IDs is in that oracle subset. Those diagnostics cannot be described as successful targeted retries of the failures.

### Why deeper questions can score better here

The groups vary in wording and assignments as well as depth. Seven two-hop failures concentrate in one direct template; all 32 two-hop paraphrases are correct. The three-hop group has one paraphrase abstention. These results do not establish a monotonic law relating hops to errors. A planned factorial experiment controlling wording, entities and distractors would be needed to isolate depth causally.

### Likely questions

**Are the 64 three-hop cases 64 independent edits?** No. They include direct and paraphrased variants across 32 final lineages.

**Can we now handle any three-hop question?** The evidence covers the authored relation patterns and wording banks with small explicit worlds. It does not establish arbitrary relational composition or commonsense inference.

**Was Qwen wrong on eight facts?** It abstained on eight questions. Several questions share construction patterns. Describe request outcomes, not eight unique factual misunderstandings.

**Why not make the dataset harder until Reason-KE++ fails?** A new challenge can be planned, but retuning a completed evaluation until a desired-looking score appears would undermine its meaning. We report the actual result and its limited scope.

**Evidence:** [depth results](../../reports/reasonke/depth-final/results.json), [depth failure analysis](../../reports/reasonke/depth-final/FAILURE_ANALYSIS.md).

<a id="slide-20"></a>
## Slide 20: Reason-KE++ full-memory failure diagnosis

### What we examined

We analysed all 25 strict wrong answers from original R-ALL, using their saved endorsed answer, raw output, supplied facts and requested target. Every case had its required support. None had a recorded token/time-budget-reached flag. Missing required facts and the recorded generation limits therefore do not explain these particular failures.

Every wrong final exactly matches another supplied fact's answer value. The off-target roles are:

| Role of matched supplied value | Cases |
| --- | ---: |
| Pro-service owner | 15 |
| Release-2 owner in an inappropriate request | 3 |
| Unrelated office city | 3 |
| Previous-owner contact | 2 |
| Hosting region | 1 |
| Current contact when an owner was requested | 1 |
| Total | 25 |

“Release-2 owner” is wrong in a question requesting another release; a value's truth elsewhere does not make it the answer to this question. Sixteen failures are edit-applicable and nine are locality questions.

### A concrete case

In `R-ALL:ops-006-direct`, Garnet Relay release 2 should have owner Silver unit 77. The model endorses Saffron unit 137, which belongs to the Pro-service record f5. Its explanation names f5 as though it were the requested service. That supports a diagnosis of wrong fact association in the visible output.

Another case, `R-ALL:ops-045-direct`, asks for a team. Cedar unit 116 appears in an intermediate boxed answer, but the final endorsed answer is contact Nira Wren. This is an endpoint/relation error despite a correct intermediate value appearing in text. The conservative parser finds two wrong-final records containing the expected value in boxed action text.

### What the diagnosis proves and what it does not

It locates these failures after support delivery and describes the observed wrong association. It does not prove attention saturation, quantify a particular distractor's causal effect or reveal the model's internal computation. A controlled remove-one-distractor experiment could investigate that hypothesis, but was not part of this frozen comparison.

### Likely questions

**Is this hallucination?** Some wrong finals are true supplied values attached to the wrong subject or relation. “Fact-association error” is more precise than implying the model invented every value.

**Does the correct boxed answer deserve success credit?** The user receives an endorsed final answer. The frozen scorer evaluates that final span. Intermediate mentions do not fix a wrong final.

**Why not automatically output the boxed team instead?** That would introduce a new extraction algorithm and require a new fixed evaluation. It might help some cases and break others. We did not post-hoc repair the test.

**Is this only cherry-picked evidence?** The audit includes the complete 25-case R-ALL failure cohort, and the individual example illustrates its recorded pattern.

**Evidence:** [complete failure audit](../../output/presentation/analysis/failure-audit.json), [compact ledger](../../output/presentation/analysis/failure-ledger.csv), [diagnosis narrative](../../output/presentation/analysis/README.md).

<a id="slide-21"></a>
## Slide 21: Qwen failures and strict-format mismatches

### Three Qwen cohorts, three descriptions

**Original retrieved Qwen:** all fifteen nonsuccesses are literal `UNKNOWN` on two-hop paraphrases. It answers 17/32 in that category and all other original main questions correctly. All required support is present.

**Depth retrieved Qwen:** seven direct two-hop questions and one three-hop paraphrase return `UNKNOWN`. The direct template “Identify [service]'s release-2 owning team and give only its on-call contact's name” accounts for 7/9 abstentions. Other direct two-hop templates are 23/23 correct, and two-hop paraphrases are 32/32 correct. This is a post-test concentration, not proven wording causation.

**Original full-memory Qwen:** fifteen nonsuccesses consist of thirteen strict mismatches and two abstentions. Of the thirteen mismatches, ten exactly match off-target supplied fact values. Two other answers join a compatible correct owner/contact chain but violate the requested person-only span. One joins a Pro owner with the correct requested contact.

### Why abstentions must remain visible

Abstaining can be preferable to confidently supplying a false fact. In a service workflow, however, an answerable request still goes unresolved. Accuracy gives it no success credit; coverage makes the unresolved fraction visible. This avoids treating cautious nonanswers as if they disappeared from the workload.

The recorded outputs do not tell us whether Qwen's abstention was caused by wording, a learned preference, distractors or another interaction. All eight depth failures are after successful support selection. No alternative prompts were tested on those final failures.

### How the 63-case audit adds up

```text
25 Reason-KE++ full-memory wrong finals
+ 15 original retrieved Qwen UNKNOWN
+ 13 full-memory Qwen strict mismatches
+  2 full-memory Qwen UNKNOWN
+  8 depth retrieved Qwen UNKNOWN
= 63 updated-method nonsuccesses

38 strict mismatches + 25 abstentions = 63
```

R0's 120 wrong answers are excluded from this particular cohort because it is a deliberately initial-state staleness diagnostic. They remain visible in the original results.

The 63 count refers to condition-specific generated records, not 63 independent facts. A question can fail in more than one condition. This diagnostic tally does not change the lineage-based uncertainty unit.

### Likely questions

**Does “all failures have support” mean retrieval is perfect in general?** No. It describes these saved cases and the controlled final packets. It is not a general large-corpus guarantee.

**Can we fix Qwen with a better prompt?** Possibly, but this experiment did not establish that. A new prompt must be selected on separate development evidence and evaluated under a new protocol.

**Are all thirteen full-memory mismatches factually false?** No. Two are compatible chains with excess text. We identify this stricter answer-contract issue while retaining their original wrong scores. Slide 31 shows the cases.

**Does UNKNOWN make Qwen safer?** It avoids some wrong assertions here. We did not run a broad safety or calibrated selective-prediction study. State the measured coverage/error composition rather than a general safety ranking.

**Evidence:** [new reproducible audit](../../scripts/analyze_presentation_failures.py), [all 63 cases](../../output/presentation/analysis/failure-audit.json), [depth case descriptions](../../reports/reasonke/depth-final/FAILURE_ANALYSIS.md).

<a id="slide-22"></a>
## Slide 22: Retrieval support and preserved locality

### First separate two denominators

**Baseline-correct retention:** among locality questions R0 already answered correctly, how many remain correct? There are 124 such original-study questions.

**All-locality factual accuracy:** among every locality question, how many are correct now? There are 128 such questions.

| Original Reason-KE++ condition | Retention | All-locality correctness |
| --- | ---: | ---: |
| Retrieved current context | 124/124 = 100% | 128/128 = 100% |
| Full current memory | 117/124 ≈ 94.35% | 119/128 ≈ 92.97% |

Full memory loses seven of the 124 previously correct cases. It also fixes two of the four originally incorrect locality cases. Hence `117 retained + 2 repaired = 119` correct overall. The two columns answer different questions.

Preserving a baseline's wrong answer is not a factual success. Conversely, changing a wrong baseline answer to a correct one is not an undesirable locality violation merely because the text changed.

### Then interpret retrieval support counts

Original retrieval supplies 320/320 required fact occurrences and completes 64/64 two-hop paths. The occurrence denominator comes from 128 locality questions needing one fact, 64 direct applicable questions needing one, and 64 two-hop questions needing two: `128 + 64 + 128 = 320`.

Depth retrieval supplies 448/448 required occurrences: `64*1 + 64*1 + 64*2 + 64*3 = 448`. It completes 256/256 main paths, including 64/64 three-hop paths and their 192 required occurrences. The depth-0 locality label still requires a factual record.

Both checkpoints use the same selector per study. Running two generators on those fact IDs does not create two independent retrieval experiments.

### The narrow false-correction metric

The original scorer flags a false correction application when an out-of-scope answer exactly endorses a declared inapplicable edited-value alias. For R-ALL this is 3/128, while all locality errors number nine. Other wrong values are outside that narrow flag. Do not call 3/128 the total scope-error rate. Retrieved Reason-KE++ has 0/128 for this flag and all locality answers correct.

### Likely questions

**Why is there no separate depth retention number?** The depth study did not include an unchanged-base condition. Its 64/64 locality correctness per model is factual accuracy, not an independently measured preservation comparison.

**Does required-support recall count unique facts?** It counts fact occurrences required across questions. The same world record can be needed by multiple variants.

**Can 100% support coexist with wrong answers?** Yes. The generator can ignore, misassociate or abstain despite complete support. That is exactly why we separate stages.

**Evidence:** [original locality/retention metrics](../../reports/reasonke/operations-final/results.json), [depth support metrics](../../reports/reasonke/depth-final/results.json), [narrow flag definition](../../editguard_benchmark/scoring.py).

<a id="slide-23"></a>
## Slide 23: The measured accuracy–latency tradeoff

### What p50 and p95 mean

Order the measured request times. The p50 is the median: approximately half the observations are no greater than that value. The p95 estimates the upper-tail 95th percentile for this population. It is not the maximum, a timeout or a promised service-level guarantee.

| Retrieved condition | Original p50 / p95 | Depth p50 / p95 |
| --- | ---: | ---: |
| Reason-KE++ | 15.06 / 16.71 seconds | 15.94 / 19.28 seconds |
| Qwen | 0.92 / 1.20 seconds | 1.12 / 1.38 seconds |

Each displayed condition contains 256 generated main requests. Qwen's generated abstentions stay in the latency population. The measurement covers selection, local IPC and generation. It excludes checkpoint download/loading and browser/network time. Cold-start time would be a different metric.

### Why the difference is plausible but not a causal decomposition

Median output lengths are 204 versus 5.5 tokens originally, and 212 versus six in depth. Reason-KE++ often emits an explicit reasoning sequence; Qwen often emits a short answer or UNKNOWN. Generating more autoregressive tokens costs time. However, prompt behaviour, model implementation and other factors can also differ. We did not isolate each contribution.

Ratios of the recorded p50 values are about 16.43 originally and 14.24 in depth. These are **ratios of quantiles**, not an average of paired request speedups. Reporting rounded 15.94/1.12 gives a slightly different ratio because the original measurements have more precision.

### What a product owner can infer

On these workloads, Reason-KE++ provides higher observed answered coverage/final accuracy with substantially slower response. Qwen provides a fast baseline with strong answered-only precision and some unanswered requests. Which tradeoff is acceptable depends on task importance, latency tolerance and the cost of abstaining. The study does not identify a universal winner.

### Likely questions

**Could we shorten Reason-KE++ reasoning to make it faster?** That could change answer quality and requires profiling/evaluation. It is not a proven improvement from these receipts.

**Why not compare tokens per second?** It can add useful runtime information, but simple division of medians is not a substitute for per-request throughput accounting. The released primary chart reports request latency with a clear boundary.

**Is the website's 33-second p95 inconsistent?** The India HTTP workload has a different boundary and mixes twelve single-generation requests with one two-generation comparison. Its 17.11/33.39-second p50/p95 is not this benchmark's per-request population.

**Does slower mean more accurate reasoning?** No. The association here does not establish that extra tokens cause correctness or that a long explanation is faithful.

**Evidence:** [original timing](../../reports/reasonke/operations-final/results.json), [depth timing](../../reports/reasonke/depth-final/results.json), [output-token audit](../../output/presentation/analysis/failure-audit.json), [HTTP timing](../../reports/reasonke/depth-final/india-acceptance.json).

<a id="slide-24"></a>
## Slide 24: Correction editor and explicit authority

### What the genuine screenshot shows

The Knowledge view presents an active approved fact beside an operator's replacement draft. The screen is Verity-branded, and the model-availability state remains truthful. A filled form is not an approved edit. The screenshot's replacement draft is browser-local against the restored revision-11 workspace; it does not establish that another publication occurred during capture.

### Why each field exists

| Field/control | Why we need it |
| --- | --- |
| Question/fact description | Lets the operator identify the intended property |
| Corrected answer | Supplies the replacement value |
| Scope | States applicability such as release or plan |
| Source/reason | Makes approval inspectable rather than an unexplained overwrite |
| Canonical identity | Connects alternative descriptions to one represented fact key |
| Explicit predecessor/replacement choice | Prevents accidentally replacing a competing active fact |
| Approve and publish | Marks the authority boundary between draft and active memory |

The store validates these inputs and enforces same-key lifecycle rules. It cannot prove the source's truth or that every human description matches the supplied canonical identity.

### How to demonstrate it

Open Knowledge, inspect the active owner's source/scope, draft the intended replacement and explain which existing version it supersedes. Only after reviewing the scope should you approve. The revision and active state change on successful publication. Returning to Chat and asking a fresh question tests model behaviour against the new state.

Do not pretend that typing “Pune Reliability Team” permanently changes the neural weights. It changes an approved external-memory record after publication.

### Likely questions

**Can a customer update this through ordinary chat?** No. A message is not silently promoted to authoritative memory. Publication is an explicit operator action through the correction API/workflow.

**Do drafts persist as approved database rows?** No. The draft shown here is interface input. Approved rows have active/superseded/withdrawn lifecycle states.

**Is this an enterprise administrator portal?** It is a local research-operator workflow. Authentication roles, organisation-level permissions and independent source certification would be production additions.

**Why not just give the model the whole edited handbook?** That is a measured full-memory condition, and it produced association errors. The deployed selector instead bounds the context.

**Evidence:** [frontend](../../editguard_chat/static/), [publication contract](../CHAT_API.md), [capture provenance](../../output/presentation/screenshots/technical-v2/provenance.json).

<a id="slide-25"></a>
## Slide 25: History, rollback and application acceptance

### What the history demonstrates

The initial example handbook is revision 9. Approving the ownership replacement produces revision 10. Rolling it back produces revision 11. Version numbers describe records in one lineage; revision numbers describe whole committed memory states. Revision 11 restoring version 1 is therefore consistent.

The timeline retains revision-10 publication/supersession and revision-11 withdrawal/restoration. The earlier approved value is active again, while the withdrawn replacement and past messages remain inspectable.

### What 14/14 and twelve checks mean

The real India HTTP acceptance workload generated fourteen neural answers through thirteen requests. One comparison request generated both historical and current responses. All fourteen matched their controlled final targets. This is one display lineage, explicitly separate from held-out study generations.

Its twelve consistency checks were:

1. Ready pinned model identity.
2. Iterative serving policy selected.
3. A fresh separate workspace.
4. Exactly nine initial fictional facts.
5. The update replaces the intended predecessor.
6. A competing approval is rejected.
7. Rejection preserves revision and record identities.
8. The replacement is revision 10.
9. Historical comparison leaves the live revision unchanged.
10. Rollback restores the predecessor in one revision.
11. Initial facts return while history is retained.
12. Rollback audit events share the revision.

The twelve checks establish operational consistency. The fourteen generated answers establish correctness on those demonstration requests. Neither count should be added to the held-out 256-question denominator.

### Why the captured site can show this without a running GPU

The capture workspace reconstructs the actual audited acceptance receipt's approved state, events and stored messages. It preserves source text, timestamps, revisions and generation evidence. Capture made zero model calls and created zero new accuracy observations. A replay is useful for presentation resilience, but must be called recorded evidence.

### Likely questions

**Did you fake the screenshots because the worker was offline?** No. They show the real frontend with restored state and actual saved responses from the acceptance receipt. The unavailable-worker indicator and recorded labels remain visible.

**Can these results be called a test set?** They are application acceptance on one known display lineage. They are not independent held-out generalisation evidence.

**Why does rollback advance the revision instead of returning to 9?** Revision numbers track new committed events. The content can return to its earlier state while history advances to 11.

**Does 14/14 prove the workflow cannot fail?** It establishes one measured acceptance workload. Transaction/error tests cover additional software boundaries; broader neural performance needs broader data.

**Evidence:** [actual acceptance](../../reports/reasonke/depth-final/india-acceptance.json), [host audit script](../../scripts/audit_india_http.py), [replay utility](../../scripts/prepare_verity_screenshot_workspace.py).

<a id="slide-26"></a>
## Slide 26: Chat evidence and stored-response semantics

### Why Hyderabad can remain visible after rollback

The screenshot opens a stored three-hop response generated at revision 10, when the approved owner was Pune Reliability Team. Its final city is Hyderabad. Current approved memory has since reached revision 11 and restored the Chennai path. Both statements are consistent:

```text
Stored response: generated using memory r10 -> Hyderabad.
Current memory: r11 -> original owner path, whose expected city is Chennai.
```

We do not rewrite old answers to match the present. Their stored evidence explains the historical output. A fresh request at revision 11 must use current facts and revision-compatible history. The real acceptance workload measured the restored behaviour; merely viewing an old chat does not generate it again.

### What source cards mean

They show the selected records supplied with that answer. A visible hosting fact can be an irrelevant selected distractor rather than the basis of the final city. Source display establishes context provenance; it does not prove the model used every shown record correctly or that all generated explanation sentences are supported.

### Ordinary conversation versus the factual benchmark

The text UI can generate ordinary responses. The controlled evaluation scores short factual values under explicit memory. We have no equivalent numerical accuracy claim for casual dialogue, advice, arbitrary follow-ups or unstated subjects. Revision filtering also means some visible old conversation is deliberately absent from the next model prompt.

### Likely questions

**Is the visible answer cached?** It is a persisted historical message. That differs from an answer cache used to satisfy a new question without generation. The current selector has no such generated-answer cache.

**Why show current memory revision as well as response revision?** Users need to know whether they are reading historical output or a response created under current authority. One number cannot represent both facts.

**If the worker is offline, can I ask a new question?** The application should report model unavailability. It must not turn stored expected values into apparently fresh neural answers.

**Does rollback delete evidence of the changed owner?** No. It changes active state while preserving the historical replacement and response evidence.

**Evidence:** [stored messages and revisions](../../editguard_chat/store.py), [history rules](../../editguard_chat/server.py), [capture mapping](../../output/presentation/screenshots/technical-v2/provenance.json).

<a id="slide-27"></a>
## Slide 27: Results explorer and failure inspection

### Why this screen is more useful than an accuracy badge

The Research results view connects aggregates to individual saved experiment records. You can filter by condition, outcome and category, inspect expected and endorsed answers, expand the supplied facts/raw text and export evidence. Filters change the visible subset; they do not redefine the experiment's assigned denominator.

The shown case is original R-ALL asking for Beryl Relay's release-1 owner. The expected answer is Willow unit 192. The model returns Amber unit 72, the release-2 owner. The correct value for the wrong release is still a wrong answer. This illustrates why a source record's truth and its applicability to the question must be assessed separately.

### How to diagnose a case in the explorer

1. Identify the exact study, condition and question ID.
2. Read the subject, relation and scope requested by the question.
3. Check the frozen target and aliases.
4. Inspect the actual supplied facts, not only the global handbook.
5. Inspect the endorsed final span and its raw-output origin.
6. Separate missing-support, wrong-association, abstention, format-contract and runtime failures.
7. Preserve the original outcome. Add diagnosis in a separate derived artifact.

The narrow explanation audit also shown on the page checks extracted explicit subanswer claims. It is not a semantic verifier for all prose or access to the model's internal reasoning.

### Why saved results are distinct from live chat

A chat request is not automatically a new frozen benchmark observation. The results explorer reads the selected released experiment. This prevents user experimentation from silently changing published denominators or selecting only successful new attempts.

### Likely questions

**Does changing the filter change accuracy?** It changes the inspected subgroup. Published condition totals remain tied to the full assigned population. Any subgroup accuracy must name its own denominator.

**Can someone recompute the score without a GPU?** Yes. Saved raw responses, fixed labels and scoring code support host replay with zero new neural calls.

**Do exports include fabricated reasoning graphs?** They expose stored evidence and derived analysis. We do not claim that an evidence graph is a faithful representation of the model's internal computation.

**Why keep wrong outputs in the project?** They support diagnosis, honest comparison and reproducibility. Removing them would make the reported results misleading.

**Evidence:** [results backend](../../editguard_chat/results.py), [original raw receipt](../../reports/reasonke/operations-final/final-reasonke.json), [reproducible diagnosis](../../output/presentation/analysis/README.md).

<a id="slide-28"></a>
## Slide 28: Engineering contribution and reproducibility

### What we contributed around the learned core

| Work area | Concrete implemented contribution | Why it matters |
| --- | --- | --- |
| Governed memory | Identity, source, approval, supersession, events and rollback | Defines current authority and makes changes reversible |
| Context adaptation | Bounded lexical selectors, iterative answer-text expansion and history filtering | Supplies relevant connected facts without blindly passing every record |
| Evaluation data | Independent fictional worlds, variants, splits and support labels | Makes new measurements possible beyond an upstream showcase |
| Experimental method | Matched conditions, frozen protocols, explicit abstentions and paired uncertainty | Makes numerical claims interpretable |
| Evidence engineering | Atomic per-request ledgers, resume checks, host audits and failure diagnosis | Protects observations during interruptions and supports inspection |
| Product | Chat, editor, source/history, comparison and results explorer | Makes the maintenance workflow usable |
| Delivery | Separate environments, container/notebooks, tests, CI and documentation | Makes the system runnable and reviewable |

These are meaningful contributions even when a foundation model is reused. Their value depends on working mechanisms and evidence, not on pretending the upstream algorithm is ours.

### How resume protects a finite experiment

The runner records a fixed plan and source/data/configuration identities. Completed request records are saved atomically. Resume checks compatibility and proceeds with unexecuted tasks; it does not replace a completed wrong answer with a better retry.

The depth recovery audit compares earlier saved ledger states containing 174, 197, 235 and 247 records with the completed ledger. Those rows agree. This is evidence of preserving previously recorded observations across the run. It does not recover arbitrary unsaved VM state or guarantee survival after a lost runtime with no export.

### What a hash establishes

A SHA-256 digest binds file bytes to a recorded identity. Changing prompts, labels, outputs or configuration can break that binding. A hash alone does not prove semantic truth, independent review, unbiased data construction or honest authorship. It supports an evidence chain; it is not the whole evidence chain.

### Likely questions

**Is this just a wrapper?** It integrates a reused model, but also implements and evaluates a complete governed maintenance workflow, context adaptations and evidence system. Do not answer by inventing new neural training.

**Did we build a new SOTA algorithm?** No such claim follows from these studies. We built a substantive evaluated system using an attributed learned method.

**Can five people own meaningful work?** Yes: data, integration/retrieval, evaluation/analysis, transactional backend, and product/deployment overlap substantially. Assign names only to work each member actually implemented, reproduced, reviewed or documented.

**Are all automated tests model tests?** No. Most reliability tests use fixtures and establish software behaviour. Neural answer accuracy comes from actual generated receipts. The distinction is a strength of the evaluation design.

**Evidence:** [contribution map](../CONTRIBUTIONS_AND_EVIDENCE.md), [depth resume audit](../../reports/reasonke/depth-final/resume-audit.json), [CI](../../.github/workflows/ci.yml), [container](../../Dockerfile.chat).

<a id="slide-29"></a>
## Slide 29: A reversible editing platform, evaluated

### The definite conclusion

We set out to make approved factual changes usable in connected conversational questions, preserve outside-scope information, and make changes inspectable and reversible. We implemented that workflow and evaluated fixed retrieved configurations on independent controlled operations data.

The selected Reason-KE++ configuration achieved 256/256 final main answers in the original study and 256/256 in the separate depth challenge. Retrieved Qwen achieved 241/256 and 248/256. Original retrieved Reason-KE++ preserved all 124 baseline-correct locality answers and answered all 128 locality questions correctly. The application acceptance measured fourteen correct neural responses plus twelve passing consistency checks on one display lineage.

The accuracy/coverage gain has a substantial latency cost: depth p50 is approximately 15.94 seconds versus 1.12 seconds. Full-memory Reason-KE++ also fails 25 original main questions. The evidence therefore establishes the selected bounded configuration and implemented workflow, not an unconditional model ranking.

### Why 100% is bounded evidence

1. There are only 32 final lineage clusters per study.
2. Worlds contain nine or twelve explicit facts, not millions of production documents.
3. Relation types and construction patterns are narrow.
4. Wording variants share structure, even with split controls.
5. Simple answer values and vocabulary can repeat.
6. Labels were automatically audited without independent human semantic review.
7. Scoring concerns the endorsed final span, not every explanation sentence.
8. These tests do not establish open-domain dialogue, arbitrary ambiguity or unbounded multi-hop reliability.

Zero observed failures in a condition is a legitimate observation. It is neither evidence of fabricated data by itself nor a universal guarantee. We retain failures in other conditions and keep the limitations visible instead of changing the test to obtain a more aesthetically believable percentage.

### Likely questions

**Did we achieve our goal?** Yes, for the stated implemented workflow and controlled evaluated task. Say the domain, data size and latency tradeoff as part of the answer.

**What remains unestablished?** Larger memories, independently authored language, ambiguous scope, production security/concurrency and fully faithful explanations.

**What would be the next study?** A predeclared larger-memory and independent-wording evaluation, with human review and the same complete-denominator discipline. That is a future validation study, not an unfinished hidden portion of these released results.

**Can we guarantee the panel's arbitrary new question works?** No. We can demonstrate the measured workflow and explain unexpected outcomes. Live generation on a new question is a new observation, not part of the released score.

**Did we test thousands of sequential edits?** No. The controlled worlds and display scenario establish much smaller update workloads. Transaction tests exercise lifecycle rules, but do not establish neural accuracy after a long stream of interacting edits.

**Evidence:** [original findings](../../reports/reasonke/operations-final/results.json), [depth findings](../../reports/reasonke/depth-final/results.json), [application acceptance](../../reports/reasonke/depth-final/india-acceptance.json).

<a id="slide-30"></a>
## Slide 30: PPO objective and reward implementation

### Build the equation from familiar ideas

The **policy** is the generator's distribution over the next token. A **rollout** is a response sampled/generated during training. A **reward** scores that response. A **critic** estimates expected return, helping form an **advantage**: how much better an observed action/continuation was than expected.

For token/action `o_t`, define the probability ratio:

```text
r_t(theta) = pi_theta(o_t | question, earlier output)
             / pi_old(o_t | question, earlier output)
```

`r=1` means unchanged probability. A positive advantage encourages raising probability; a negative advantage encourages lowering it. PPO's clipped surrogate is:

```text
J_clip(theta) = E[min(r_t*A_t,
                     clip(r_t, 1-epsilon, 1+epsilon)*A_t)]
```

The clipping discourages excessively large beneficial changes within an update. It does not mean weights are clipped to an interval or factual answers are guaranteed. The expression is the policy surrogate, not a complete listing of critic, entropy and KL terms in a training implementation.

### Why KL control exists

A reference-policy penalty limits departure from an anchor model distribution. It helps prevent reward optimisation from drifting too far from useful language behaviour. KL control and ratio clipping play related but distinct roles; neither is a database conflict rule.

### Reference settings and reward scaling

The official example launchers specify SFT learning rate `1e-5` for ten epochs, and PPO actor `1e-6`, critic `1e-5`, KL coefficient `0.001`, fifteen epochs and eight GPUs. The PPO launcher uses vLLM rollouts and a custom reward function. These are upstream reference settings, not measured training resources for our T4 deployment. [SFT launcher](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh), [PPO launcher](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh)

The paper's main equal blend is `0.5*process + 0.5*outcome`; pinned reward code returns `process + outcome`. Scaling rewards can change their relationship to KL or other terms, so equal relative component weights do not certify identical optimisation. We disclose this reference-code difference rather than claiming faithful reproduction.

### Likely questions

**Why not train PPO ourselves?** The released checkpoint gave us a feasible inference route; reference training is a different, larger compute task. Reproducing it was outside the final system scope.

**What is an epoch?** One pass through a training dataset under the training pipeline's batching conventions. It is not one approved correction.

**Does a learning rate of `1e-6` mean one-millionth accuracy?** No. It scales optimiser updates. It is not a probability or evaluation result.

**What if process reward is high but final answer is wrong?** The combined objective can still give partial reward. That is why the final outcome term exists and why a reward is not a binary factual certificate.

**Evidence:** [primary method note](REASONKE_PLUS_PLUS_TECHNICAL_NOTES.md), [primary paper](https://aclanthology.org/2026.findings-acl.1137/).

<a id="slide-31"></a>
## Slide 31: Answer-contract failures in detail

### The three full-memory Qwen cases

| Record suffix | Requested final person | Actual endorsed final span | Interpretation |
| --- | --- | --- | --- |
| `ops-045-two_hop_paraphrase` | Nira Wren | Cedar unit 116 Nira Wren | Compatible owner/contact chain, but extra owner text |
| `ops-047-two_hop_paraphrase` | Serin Wren | Indigo unit 118 Serin Wren | Compatible owner/contact chain, but extra owner text |
| `ops-005-two_hop_paraphrase` | Nira Dane | Indigo unit 136 Nira Dane | Pro-owner contamination joined to the correct contact |

All are `G-ALL` records. The frozen request asks for the person's name and the scorer requires the whole final span to match fixed aliases. We keep all three wrong in the released score. The first two are different from inventing a false person; the third includes an incompatible association. This distinction makes the analysis more informative without changing evaluation rules after the fact.

### Why a final-answer metric needs complementary diagnosis

One strict outcome can have several causes: a false value, a true value for the wrong entity, a wrong relation endpoint, or extra content violating an explicit response contract. A single accuracy score cannot distinguish them. Conversely, a more forgiving semantic score should not be introduced only after we know which outputs it would rescue.

Two wrong Reason-KE++ finals also contain the expected target in a boxed action subanswer. That parser observation says the target text appeared in a particular intermediate format. It does not certify the entire trace or prove what internal representation the model used.

### Likely questions

**Would a human accept the first two?** A human may recover the intended person from the text. The frozen machine-readable contract is stricter. We disclose that difference and do not label every strict mismatch an equally severe factual error.

**Can you report a secondary semantic-review score?** A separately specified review protocol could do that. We have not conducted independent human semantic scoring for all outputs. This post-hoc diagnosis remains descriptive.

**Are the three cases found by a general semantic classifier?** No. The audit documents inspected fixed IDs for these concatenations. It must not be presented as a trained general-purpose failure detector.

**Why not improve the parser now?** A new parser is a new configuration requiring development and a new evaluation. It must not overwrite the already frozen scorecard.

**Evidence:** [unchanged case ledger and raw text](../../output/presentation/analysis/failure-audit.json), [audit implementation](../../scripts/analyze_presentation_failures.py).

<a id="slide-32"></a>
## Slide 32: Evidence and configuration reference

### How to answer “where is the proof?”

Choose the evidence type matching the question. Source code proves what an implementation attempts to do; a test or receipt shows an observed execution; an aggregate summarises that execution. A proposed plan or upstream result cannot substitute for our measured output.

| Panel question | Start here | What to inspect |
| --- | --- | --- |
| Which weights and budgets? | [Reason-KE++ manifest](../../configs/reasonke-model.json), [Qwen manifest](../../configs/instruction-model.json) | ID, revision, quantization and limits |
| What happened on the original test? | [Original results](../../reports/reasonke/operations-final/results.json) | Counts, assigned cases and paired differences |
| What happened on three-hop questions? | [Depth results](../../reports/reasonke/depth-final/results.json) | Per-depth groups and separate protocol |
| Are failed outputs inspectable? | [Failure JSON](../../output/presentation/analysis/failure-audit.json), [CSV](../../output/presentation/analysis/failure-ledger.csv) | IDs, actual spans, supplied facts and diagnosis |
| Did the real application edit/rollback? | [India receipt](../../reports/reasonke/depth-final/india-acceptance.json) | Fourteen actual responses, operations and twelve checks |
| Are screenshots genuine? | [Capture provenance](../../output/presentation/screenshots/technical-v2/provenance.json) | Capture mapping, JPEG hashes and recorded-state disclosure |
| Can we inspect the database rules? | [Store](../../editguard_chat/store.py), [tests](../../tests/test_chat_store.py) | Partial uniqueness, events and transactions |
| Was the test protocol frozen? | [Original protocol](../../reports/reasonke/operations-final/protocol.json), [depth protocol](../../reports/reasonke/depth-final/protocol.json) | Source/data/config identities and declared plan |
| Is the presentation backed by a preserved release? | [Presentation manifest](../../output/presentation/verity-technical-manifest.json) | Slide/artifact identities and evidence mapping |

### How to read a receipt

Identify its schema and study, phase, model revision, request condition and question ID. Read the actual returned output, not only a `passed` field. Check its support, outcome, timing and any error/budget flags. If it is an operational receipt, identify which counts are model answers and which are deterministic checks.

### Likely questions

**Can we reproduce the metrics without downloading the model?** Yes, by scoring/auditing saved outputs. Fresh neural reproduction additionally needs the pinned checkpoint and compatible GPU environment.

**Does matching a hash prove the label is right?** No. It proves identity relative to the recorded bytes. Dataset construction, review and assumptions still matter.

**Does Git commit time prove every historical motivation?** No. Git and dated records support a development chronology, not an invented account of what each person privately intended.

**What if someone asks for an unsupported claim?** State exactly which part was not measured. Explain the additional test needed. Do not substitute an upstream headline number or a successful demo for absent evidence.

## Appendix A: What we actually did, and why each phase matters

This is the supported chronology, not a claim that the final method was our original grand plan. The “why” column explains the technical value of a step. It does not invent private motivations or assign work to named people.

| Documented phase | What and how | Why it matters | Evidence boundary |
| --- | --- | --- | --- |
| 11 October: first correction workflow | Defined explicit facts, approval, conflicts, transactions and rollback; built a small end-to-end slice | A model answer cannot establish current authority or reversible state by itself | Earlier support release, not the final Reason-KE++ benchmark |
| 13–20 October: auxiliary learning | Trained a DistilBERT applicability model and FLAN-T5 conditional answer model; evaluated controlled support questions | Established actual project ML training and a testable editing pipeline | Those weights/results remain separate from the final checkpoint |
| 21–22 October: visual prototypes | Built visual correction workflows and investigated recognition changes | A visible prediction change alone is insufficient evidence of selective knowledge editing | Preserved visual experiments |
| 24 October: comparative visual study | Evaluated multiple methods and neighbouring question types with saved generated outputs | Exposed why edit success and locality must both be measured | Different visual dataset and models |
| 25 October: conversational workflow | Added ordinary text chat, correction/source controls, comparison and version history; addressed stale conversation context | Made the approved-memory state observable in continuing conversations | Application integration evidence |
| 25 October: released reasoner integration | Pinned the author checkpoint, adapted inference to T4, separated/authenticated the worker and ran finite development checks | Made a learned connected-fact reasoner available without attempting multi-GPU training | Feasibility and small development checks, not universal accuracy |
| Development before original freeze | Compared retrieved versus full current context on 24 questions per condition | Selected a bounded context policy from development evidence | Three development lineages; Qwen tied in retrieval |
| 26 October: original protocol freeze | Bound source, data, configs, budgets and scoring before final inference | Prevented silent changes during the final comparison | Protocol timestamp 26 October; completed results are separately released |
| Original final study and product checks | Recorded 1,312 generations; replayed metrics; tested real HTTP/history operations | Measured main quality, locality, latency and operational behaviour | Do not merge operational/demo answers into the held-out population |
| Depth development before its freeze | Authored separate twelve-fact worlds; tested six iterative selectors and explicit display states | Addressed a measured missing-third-hop support problem | Development-only selection, including its documented revision |
| 28 October: depth protocol freeze | Froze the separate iterative-retrieval experiment | Kept the extension independent of the earlier completed study | Separate protocol timestamp and identity |
| Depth evaluation, interruption/recovery and India acceptance | Completed 544 requests; checked preserved ledger prefixes; ran fourteen real website answers and twelve consistency checks | Established bounded three-hop evidence and the actual reversible display workflow | Saved generations/operational checks remain distinct |
| 30 October: Verity technical presentation | Rebranded genuine interface views, built editable diagrams/charts and matched notes; inspected saved failures | Made mechanisms and evidence defensible to a panel | Zero new training or neural evaluation for presentation authoring |
| Current study guide | Explain all 32 slides, equations, decisions, failures and limits; inspect source and pinned model metadata | Enable every member to understand and defend the system | Documentation work, not extra accuracy observations |

The [dated journey record](../PROJECT_JOURNEY_AND_FINISH_PLAN.md) contains the early phases and commit references. It was written as a plan on 25 October, before later evaluation completed. Its historical “not run” language describes that date; use the released final receipts for present completion status. The [auxiliary model card](../MODEL_CARD.md), [visual study](../VLKEB_RESULTS.md), [reasoner integration](../REASONKE_INTEGRATION.md) and final protocols preserve the distinctions.

### Why the earlier work is useful without becoming fake final training

Earlier learning and visual experiments informed the engineering discipline: test locality, separate deterministic answers from generation, keep raw failures and use fair updated baselines. They are legitimate parts of the development record. They do not make us authors of Reason-KE++ or mean that the selected final reasoner was trained on our support data.

If asked why the project changed, a sound answer is: “We refined the task and implementation after building and measuring earlier versions. The final objective became governed editing over connected operations facts, and the final evidence is attached to that configuration.” Avoid claiming every pivot was anticipated from the start.

## Appendix B: Neural-model primer for detailed technical questions

### B1. What kind of network is the final model?

The two pinned checkpoint configurations both declare `Qwen2ForCausalLM`, a decoder-only causal language-model architecture. Their inspected metadata agrees on:

| Configuration field | Pinned value | Meaning |
| --- | ---: | --- |
| Transformer layers | 28 | Repeated attention/feed-forward processing blocks |
| Hidden size | 3,584 | Width of a token's internal representation |
| Query attention heads | 28 | Parallel attention subspaces |
| Key/value heads | 4 | Shared key/value groups for grouped-query attention |
| Intermediate size | 18,944 | Feed-forward intermediate width |
| Vocabulary size | 152,064 | Number of token IDs in the declared vocabulary |
| Maximum position metadata | 32,768 | Configured model position capacity, not our permitted input budget |

These values were checked from the exact pinned [ReasonKE_plus config](https://huggingface.co/YukinoKaren/ReasonKE_plus/resolve/ceaedc2ae7c8fed6257953587d991ce550db6106/config.json) and [Qwen config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/config.json). Matching architecture metadata does not prove identical starting weights or establish a causal training ablation. Our deployment still rejects inputs above 4,096 tokens.

### B2. What does “decoder-only causal” mean?

The model predicts a continuation of a token sequence. “Causal” means a generated position cannot attend to future output tokens that have not yet been generated. It can use the prompt and preceding output. It does not require a separate encoder network like a classic encoder–decoder sequence-to-sequence model. The earlier FLAN-T5 experiment did use that different family; do not mix their architectures.

A simplified inference loop is:

```text
encode prompt into token IDs
repeat until an end token, budget or stopping condition:
    compute next-token logits from prompt plus earlier output
    choose the highest-scoring next token in greedy mode
    append that token
decode newly generated tokens into text
```

“Greedy” finds the locally highest-scoring token at each step. It does not guarantee the globally most probable sentence, factual correctness or cross-hardware byte identity.

### B3. How attention works, at a useful level

For token representations, the network computes query, key and value projections. A simplified attention expression is:

```text
Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k) + causal_mask) * V
```

Queries score compatibility with keys; normalised weights combine values. Heads allow several representation subspaces. The 28 query heads and four key/value heads indicate sharing of key/value information across groups. This is part of neural computation, distinct from our external BM25-style record retrieval.

Attention weights are not automatically a faithful explanation of why an answer was generated. Our failure diagnosis uses visible text and supplied facts, not a claim that we measured an internal attention mechanism.

### B4. What are tokens, logits and the chat template?

A tokenizer maps text to vocabulary IDs. A token can represent a word, part of a word, punctuation or other text. The model returns logits, unnormalised scores for possible next tokens. Softmax would turn those into a distribution. Greedy selection chooses the maximum-scoring next token.

A chat template serialises roles and boundaries in the format expected by the checkpoint. Supplying raw strings with invented role markers could change behaviour. We use the pinned tokenizer's template and retain the selected update instruction.

Do not translate 4,096 tokens into a guaranteed number of words or characters. Tokenisation varies with language and text. The application also has character guards because preventing oversized JSON input and enforcing a token budget solve different problems.

### B5. Why quantization needs more than “7B times four bits”

Four-bit storage reduces selected weight memory. Total inference also includes scales/metadata, tensors retained at other precision, activations, attention KV cache, temporary buffers, CUDA context and allocator reservation. Source checkpoints can still download at their original storage size before quantised loading.

Allocated memory is memory occupied by tracked tensors; reserved memory includes allocator-held blocks that may not currently contain active tensors. Neither equals every byte reported by the GPU driver. That is why the recorded approximately 6.59 GiB allocated loading peak is not a promise that every 6 GiB GPU can run the workload.

### B6. What loss and learning rate mean

Cross-entropy loss penalises assigning low probability to target tokens/classes. For an illustrative one-token target, probability 0.1 gives `-ln(0.1) ≈ 2.303`; probability 0.9 gives about 0.105. These are explanatory values, not Verity training measurements.

A loss of 0.48 is not 48% error or 52% accuracy. Its meaning depends on the objective, label distribution, data split and calibration. The earlier scope-training console and final neural answer accuracy belong to different experiments. A learning rate similarly controls update scale; it is not a success probability.

### B7. Why a learned reasoner can still ignore an explicit fact

Context supplies information, but the neural network still chooses tokens. Conflicting associations, wording, learned priors, distractors and response-format behaviour can affect output. Our experiments show wrong associations with complete support, but do not causally separate those possibilities. Writing a stronger instruction is a hypothesis to evaluate, not proof that a failure is solved.

### B8. Why the project has several different input limits

| Limit | Layer | What it protects |
| --- | --- | --- |
| 200 records and one million lexical characters | Iterative selector's candidate memory | Bounded selection input |
| Four selected records | Deployed retrieval policy | Context size and the tested retrieval configuration |
| Twenty fact objects | Worker request validator | Maximum accepted fact list for supported modes, including full-memory use |
| 12,000 total checked fact-value characters | Worker request validator | Oversized fact payloads |
| Eight history messages, 6,000 total content characters | Application history filter | Bounded complete prior turns |
| 4,096 tokenised input tokens | Loaded deployment | Actual prompt length after facts, query and history combine |
| 512 newly generated tokens | Decoder | Finite generated continuation |

These limits are compatible because they apply at different stages. A worker capable of accepting twenty facts need not receive twenty from a four-fact selector. A request under character guards can still exceed the tokenizer budget and receive an explicit rejection. The 32,768-position model metadata does not override the smaller evaluated deployment ceiling. None of these software bounds is an accuracy guarantee at the maximum allowed memory size.

## Appendix C: One complete worked example across all layers

Use this if the panel asks, “Explain the entire project through one question.” This is a mechanism walkthrough tied to the measured India scenario; the compact fact descriptions below are explanatory, not a new raw model trace.

### 1. Define the controlled source of truth

Initial approved facts say service release 2 is owned by Bengaluru Platform Team, that team's contact is Meera Iyer, and her office is Chennai. Hosting remains Bengaluru. The alternate team's contact/office records already define Pune Reliability Team, Arjun Deshmukh and Hyderabad.

### 2. Ask before the edit

The user asks which city contains the office of the on-call contact for the service's owning team. The API captures revision 9. Retrieval selects approved support and possibly a distractor. The unchanged checkpoint generates a response. Only the actual endorsed response is scored; the expected Chennai target is not injected by the evaluator into the request.

### 3. Draft and publish one correction

The operator identifies the same entity/relation/release key, sets owner to Pune Reliability Team, supplies provenance and explicitly names the active predecessor. The store supersedes the old version, inserts the new one and writes events at revision 10 in one transaction.

### 4. Ask the connected question again

The selector now starts from the current owner record and expands through the approved answer text to Arjun Deshmukh's office record. The model receives that selected context. The expected final answer is Hyderabad. Stale revision-9 pairs remain visible but do not contaminate the new prompt.

### 5. Check what should not change

Ask the hosting question: target Bengaluru. Ask the older-release owner: target Chennai Legacy Team. Ask the Pro-service owner: target Hyderabad Integration Team. These controls distinguish the ownership edit from a blanket string substitution.

### 6. Compare historical and current memory

Request approved revision 9 versus current revision 10. The API makes two new generations under matched permitted history, with the respective memory snapshots. It does not load two different model weight files and does not overwrite prior stored responses.

### 7. Roll back

The rollback transaction withdraws the new owner and restores its predecessor at revision 11, with both audit events sharing that revision. Fresh connected questions use the Chennai path again. Stored revision-10 Hyderabad output remains historical evidence of the former approved state.

### 8. Inspect the evidence

The acceptance receipt contains fourteen actual generated responses and twelve consistency checks. The UI can display saved messages without a GPU. A fresh request requires the worker. This separation prevents a deterministic database reconstruction from masquerading as new neural success.

## Appendix D: Terms that are easy to confuse

| Term | Meaning here | Common mistake to avoid |
| --- | --- | --- |
| Knowledge editing | Changing system factual behaviour under a correction | Assuming all editing changes neural weights |
| Parametric editing | Changing model parameters | Calling an SQL update a weight edit |
| External-memory editing | Changing authoritative context used with fixed weights | Claiming the model permanently memorised each edit |
| RAG | Retrieve context and condition generation on it | Treating it as mutually exclusive with the trained reasoner |
| SFT | Supervised optimisation on target responses | Calling runtime prompting a training run |
| PPO | Policy optimisation using rollout feedback with constrained updates | Saying it runs on every Approve action |
| Reward | Numerical training feedback | Treating it as a deployment correctness certificate |
| Inference | Generating with fixed weights | Confusing it with fine-tuning |
| Checkpoint | Saved model weights/config identity | Confusing it with a memory revision |
| Memory revision | One committed approved-memory state | Assuming rollback decreases the revision number |
| Version | A record's position in its lineage | Confusing version 1 with global revision 1 |
| Canonical ID | Explicit stable represented identity | Assuming it provides general semantic entity resolution |
| Scope | Release/plan/applicability boundary | Treating any similarly named fact as applicable |
| Provenance | Recorded source/origin | Assuming retrieved prose is independently verified |
| Lineage | Underlying world/edit and related variants | Counting paraphrases as independent edits |
| Locality | Questions outside the intended edit scope | Equating unchanged text with factual correctness |
| Preservation/retention | Keeping answers correct on the baseline-correct locality subset | Preserving baseline errors as successes |
| Ground truth | Authored expected facts/paths/answers | Calling model output ground truth |
| Alias | Predeclared acceptable final-answer variant | Adding a favourable alias after seeing the test |
| Oracle context | Gold support for a diagnostic subset | Calling all-memory context an oracle |
| Coverage | Answered divided by assigned | Dropping UNKNOWN from primary accuracy |
| Precision | Correct divided by answered in this report | Confusing it with FP16 numerical precision |
| Calibration | Confidence matching empirical correctness | Treating BM25 score as a probability |
| Percentage point | Difference between percentage values | Confusing +5.86 points with a 5.86% relative gain |
| Bootstrap | Resampling under a stated unit/design | Sampling correlated variants as independent facts |
| p50/p95 | Median/95th percentile for a named timing population | Calling p95 a maximum or a guaranteed SLA |
| Transaction | Writes commit or revert together | Assuming it makes a wrong neural answer correct |
| Idempotence | A repeated permitted operation does not repeat its state change | Confusing it with a retry that regenerates a different answer |
| Replay | Reconstruct/score saved observations | Counting it as new neural evaluation |
| Regression test | A software check against a known behaviour | Counting fixture outputs as fresh model accuracy |
| CI | Confidence interval in statistics; continuous integration in engineering | Using the acronym without context |

## Appendix E: Rehearsal questions with answer keys

Try answering these without looking, then check the explanation.

1. **A model answers 90/100 and abstains ten times. What are accuracy, coverage and answered-only precision?** Accuracy 90%, coverage 90%, answered-only precision 100%. All 100 stay assigned.
2. **A baseline had 80 correct locality answers out of 100. A new system retains 75 of those and fixes ten old errors. What are retention and factual locality accuracy?** `75/80 = 93.75%` retention and `85/100 = 85%` correctness. They use different denominators.
3. **All gold support is present but the model says UNKNOWN. Which stage visibly failed?** Answer production under the tested prompt/context/decoding configuration. This does not identify its internal cause.
4. **The expected city appears in a sentence saying it is wrong. Does that earn credit?** No. Whole endorsed final-span scoring avoids simple substring credit.
5. **Why does original full-memory Reason-KE++ have nine locality errors but only three false-correction applications?** The latter flags a narrow inapplicable edited-value endorsement, not every possible outside-scope wrong value.
6. **Why do 256 questions provide 32 bootstrap clusters?** Eight related variants belong to each final lineage. Lineages are the resampling units.
7. **Does 16/16 oracle performance repair the eight depth failures?** No. None of those eight question IDs is in the oracle subset.
8. **Can the current memory be revision 11 while a displayed answer says revision 10?** Yes. Stored messages preserve their generation state; fresh requests use current authority.
9. **Does a successful database rollback prove a fresh answer is correct?** No. Verify restored state deterministically and score an actual fresh generation separately.
10. **Why is full-memory Qwen higher than full-memory Reason-KE++ but lower under retrieval?** The measured context conditions produce different error profiles. The evidence supports configuration-specific comparisons, not an absolute model ordering.
11. **If the transformer config says 32,768 positions, can we send that much text?** Our deployment imposes a 4,096-token input ceiling. Model metadata and evaluated operating bounds differ.
12. **Where are our SFT/PPO training logs?** Not part of this final deployment; we reused a pinned upstream checkpoint. Earlier auxiliary training has its own artifacts and cannot substitute for those missing upstream logs.
13. **What should we do if a new panel paraphrase fails?** Acknowledge it, inspect support/output/revision and identify the stage. Do not relabel the demonstration as a released benchmark success or substitute a template answer.
14. **Why not present the paper's headline score as ours?** It belongs to a different dataset, evaluator and training study, and the precise checkpoint-to-score correspondence is undocumented in our inspected release.
15. **What can we responsibly say each member contributed?** Implementation, reproduction, review, analysis or documentation they actually performed, tied to artifacts. Presentation assignments alone do not establish ownership.

### A reliable answer structure when surprised

Use four sentences:

1. “The implemented mechanism is …”
2. “The observation we measured was …, over … assigned cases.”
3. “That establishes …, within this scope.”
4. “It does not establish …; testing that would require …”

For example: “The selector expands approved answer text with two seeds and a four-fact cap. It supplied all 448 required fact occurrences in the depth main population. That establishes complete support on those controlled questions. It does not establish arbitrary large-corpus retrieval; testing that needs a separate larger-memory and independently worded evaluation.”

## Appendix F: Evidence-first preparation checklist

Before the presentation, every speaker should be able to:

- Explain the three-hop India chain and the unchanged controls without reading.
- Distinguish fixed neural weights, active external memory and stored messages.
- Explain why approval is explicit and why the canonical key includes scope.
- Trace retrieval through seeds, frontier and the stopping budget.
- Explain the application lock versus the short database transaction.
- Reconstruct the 1,312 and 544 generation totals.
- Calculate accuracy, coverage and the two locality denominators.
- Explain all 63 updated-method nonsuccesses without changing their scores.
- State the latency boundary and the measured accuracy–latency tradeoff.
- Attribute published training and identify our concrete contribution.
- Tell recorded output from fresh generation and use the offline fallback honestly.
- State at least three remaining limits of the 100% result.

Use the [speaker script](VERITY_SPEAKER_SCRIPT.md) for speaking assignments and its final contribution allocation. Use the [team handbook](VERITY_TEAM_HANDBOOK.md) and [India guide](../INDIA_DEMO.md) for launch/demo operation. Use the [method note](REASONKE_PLUS_PLUS_TECHNICAL_NOTES.md) for primary-source details. The presentation and its scientific results remain unchanged by this study guide.
