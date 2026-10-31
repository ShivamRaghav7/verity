# Verity technical presentation: five-speaker script

Current version: 29 core slides and three optional technical backup slides. Allow approximately 24–26 minutes including transitions. Read the quoted paragraphs aloud. Technical detail and evidence are for rehearsal and Q&A, not additional required speech. Speaking roles need not match implementation ownership. Replace Member 1–5 with names only after agreeing the speaking order.

The deployed checkpoint is `YukinoKaren/ReasonKE_plus`, used for the Reason-KE++ method. Runtime corrections change approved external memory while its weights stay frozen. We do not claim new SFT/PPO training or reproduction of the paper’s headline score. The single concise oral attribution is on slide 5. References stay in notes and this script rather than crowding slide faces.

## Speaking order

| Speaker | Core slides | Spoken words | Speech at 125 words/minute |
| --- | --- | ---: | ---: |
| Member 1 | 1–6 | 540 | 4.3 minutes |
| Member 2 | 7–12 | 596 | 4.8 minutes |
| Member 3 | 13–18 | 583 | 4.7 minutes |
| Member 4 | 19–23 | 491 | 3.9 minutes |
| Member 5 | 24–29 | 578 | 4.6 minutes |

## Member 1

### Slide 1: Verity

**Visual cue:** State the maintenance problem and the evaluated contract.

> Verity addresses a specific maintenance problem: an assistant must use an approved correction when answering connected factual questions, preserve information outside its scope, and support reversal. Updating a webpage or database establishes the new fact. The additional question is whether a conversational model uses that state correctly. We built a versioned memory platform around Reason-KE++, then evaluated the complete retrieval and generation path. Our evidence covers short factual answers, controlled paraphrases and reasoning paths of up to three relations.

**Technical detail for rehearsal:** The final system uses a frozen released checkpoint and external approved memory. It is a parameter-preserving editing deployment. It supports ordinary text conversation, but the scientific accuracy claim concerns the bounded factual task. The presentation separates the published learning method, our implementation and our measured outcomes.

**Evidence:**

- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)
- [docs/CONTRIBUTIONS_AND_EVIDENCE.md](../../docs/CONTRIBUTIONS_AND_EVIDENCE.md)

### Slide 2: One ownership edit changes a connected answer

**Visual cue:** Trace the initial and replacement paths, then point to the unchanged controls.

> This example comes from the implemented India demonstration. Kaveri Payment Gateway in release two initially belongs to Bengaluru Platform Team. That team’s contact is Meera Iyer, whose office is in Chennai. Replacing only ownership with Pune Reliability Team changes the connected path to Arjun Deshmukh and Hyderabad. The hosting city must remain Bengaluru. The Pro service and release one retain separate owners. These services, people and assignments are fictional. The familiar Indian locations help us demonstrate a real software workflow without making claims about an actual company.

**Technical detail for rehearsal:** The replacement changes one approved ownership record. The alternate team's contact and office records already exist. The three-hop answer changes by following owner, on_call_contact and office_city. An unchanged hosting answer tests another relation on the same entity. The older release and Pro service test version and similarly named entity boundaries.

**Evidence:**

- [docs/INDIA_DEMO.md](../../docs/INDIA_DEMO.md)
- [editguard_chat/examples.py](../../editguard_chat/examples.py)
- [reports/reasonke/depth-final/india-acceptance.json](../../reports/reasonke/depth-final/india-acceptance.json)

### Slide 3: The task and acceptance contract

**Visual cue:** Separate answer quality, locality and operational consistency.

> We define success through observable requirements. A relevant direct or paraphrased question should endorse the corrected answer. A connected question should follow the revised chain. An unaffected question should retain the applicable fact, including its release or plan scope. A competing correction must not silently overwrite active knowledge. Approval and rollback must update fact state and audit history together. We score actual model answers separately from database consistency. Abstentions and errors remain in the evaluation denominator. These requirements give the project a finite, testable objective without promising universal language accuracy.

**Technical detail for rehearsal:** The fact store determines authority, not real-world truth. Scope is represented by canonical IDs and explicit text, rather than a calibrated semantic applicability classifier. Conflicts under the same represented key can be rejected deterministically. Arbitrary contradictions under unrelated keys and ambiguous customer language remain limits. A rule-based clarification must never count as a neural answer success.

**Evidence:**

- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)
- [docs/CHAT_API.md](../../docs/CHAT_API.md)
- [tests/test_chat_store.py](../../tests/test_chat_store.py)

### Slide 4: Method choice and the recorded decisions

**Visual cue:** Distinguish the editing route from the context-delivery mechanism.

> Conventional RAG retrieves context and asks a general instruction model to answer. Parametric editing changes model weights. Our approach keeps the weights fixed, maintains approved corrections externally, and uses a reasoner trained to apply contextual updates across connected facts. Retrieval is still part of the system. The documented development comparison selected bounded retrieval because the deployed Reason-KE++ checkpoint answered twenty-four of twenty-four questions, versus twenty-one with full memory. Qwen also answered twenty-four with retrieval. That development result selected a context policy; it did not establish model superiority or a causal benefit from training.

**Technical detail for rehearsal:** The project first built governance and earlier editor experiments, then integrated the released reasoner on 25 October. We do not claim a complete parametric-method comparison or an originally planned progression. Source-pinned integration established feasibility on a T4. The deployment-policy receipt records three development lineages and twenty-four questions per condition. Final results were not used to select that policy. Reason-KE++ and RAG are complementary layers here, not mutually exclusive labels.

**Evidence:**

- [docs/REASONKE_INTEGRATION.md](../../docs/REASONKE_INTEGRATION.md)
- [docs/PROJECT_JOURNEY_AND_FINISH_PLAN.md](../../docs/PROJECT_JOURNEY_AND_FINISH_PLAN.md)
- [configs/deployment-policy.json](../../configs/deployment-policy.json)
- [reports/reasonke/operations-development-v1/results.json](../../reports/reasonke/operations-development-v1/results.json)

### Slide 5: Reason-KE++ learning pipeline

**Visual cue:** Explain training once, then distinguish it from runtime editing.

> The learning method comes from Wu and colleagues’ Reason-KE++ paper and official implementation. We use their released ReasonKE_plus checkpoint; we did not run its SFT or PPO training. The data generator supplies a rewritten fact and a portability question to a teacher model. Supervised learning then teaches a structured response format. PPO subsequently trains against signals that inspect intermediate work as well as the final answer. This matters because a model can reproduce a reasoning format or reach a correct endpoint while using an incorrect intermediate association. Our independent operations datasets evaluate the deployed system, not this training pipeline.

**Technical detail for rehearsal:** The data-generation code reads requested_rewrite and portability.New Question and calls gpt-4o-mini. The paper's Appendix A reports 718 curated cold-start instances and an 8,296/922 MQuAKE-CF PPO partition. Section 3.2 and Appendix A describe the cold-start source differently in granularity; use the concrete generation code and Appendix A rather than claiming a re-created dataset. The SFT trainer requests assistant-completion-only loss when that TRL collator is available, with a language-model collator fallback. No training-loss curves for our deployment exist. The exact released revision is not linked to the paper's headline score in a model card.

**Evidence:**

- [https://aclanthology.org/2026.findings-acl.1137/](https://aclanthology.org/2026.findings-acl.1137/)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/data_preprocessing/generate_sft_data.py](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/data_preprocessing/generate_sft_data.py)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh)
- [configs/reasonke-model.json](../../configs/reasonke-model.json)

### Slide 6: Structured reasoning and stage-aware rewards

**Visual cue:** Read the structured stages, then the separate reward components.

> Reason-KE++ organizes output into acknowledge, decompose and action stages, followed by a final answer. The reward first rejects malformed structure. For valid output, the released reward code combines a hop-count check, semantic similarity between aligned subquestions, exact subanswer checks and final-answer token F1. PPO uses those reward components to update a policy while limiting large policy changes. Intermediate-step supervision here means scoring the generated stages. It does not mean that Verity verifies every displayed explanation or calls a separate neural verifier during chat. The deployment only runs the trained checkpoint for inference.

**Technical detail for rehearsal:** The pinned reward implementation uses process = 0.2H + 0.4D + 0.4A and total = process + final F1. D uses all-MiniLM-L6-v2 cosine similarity on aligned action-section subquestions when hop count matches. A is the fraction of exact or alias-matched boxed subanswers. Invalid format returns -1. The paper instead writes alpha*process+(1-alpha)*outcome, with alpha=0.5 as its main setting. The code's sum is twice that normalized blend, which can affect reward scale relative to KL. This is documented in the backup slide. Dense component scoring is not evidence of a separate reward on every generated token.

**Evidence:**

- [https://aclanthology.org/2026.findings-acl.1137/](https://aclanthology.org/2026.findings-acl.1137/)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh)


**Handover:** “Member 2 will take the next part.”


## Member 2

### Slide 7: The implemented inference architecture

**Visual cue:** Follow the request across the three computation categories.

> The application separates deterministic state, deterministic retrieval and neural generation. The browser sends a question to the Python API. SQLite stores approved fact versions, chats and audit events. The selector ranks the current approved snapshot and supplies at most four records. A history filter adds compatible conversation pairs. An authenticated private worker runs the pinned checkpoint and returns generated text. The API extracts and persists the answer together with its supplied facts, revision and model metadata. The website can therefore display the exact evidence associated with a response. Source display establishes what the model received, not that every generated sentence is true.

**Technical detail for rehearsal:** The website and worker are separate processes. The application lock serializes mutation and generation operations for a single local operator. The worker has separate request validation and a bearer-token boundary. An unavailable worker produces an explicit unavailable state rather than a template answer. The 200-record retrieval bound is a software guard; neural tests use much smaller worlds.

**Evidence:**

- [editguard_chat/server.py](../../editguard_chat/server.py)
- [editguard_chat/reasonke_backend.py](../../editguard_chat/reasonke_backend.py)
- [editguard_reasonke/server.py](../../editguard_reasonke/server.py)
- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)

### Slide 8: A question through the actual request path

**Visual cue:** Trace request validation, revision capture, generation and atomic turn persistence.

> Consider the office-city question after rollback, at revision eleven. The API validates the message and operating mode under the application operation lock. It captures the knowledge revision, reads active approved facts, and passes the question to the iterative selector. It combines selected records with permitted history and calls the worker. Only after a successful response does SQLite commit the user and assistant messages, evidence and conversation metadata together. A failed model call leaves no dangling user turn. The revision stays fixed across this operation. The long generation call is outside the short SQLite transaction, although the application lock remains held.

**Technical detail for rehearsal:** mutate() acquires the operation lock non-blockingly and reports HTTP 409 if another operation is running. ask() checks model readiness, obtains history and revision, generates before storing, then calls add_turn(). The message transaction and the knowledge-change transaction are different boundaries. This local lock is not a distributed concurrency guarantee. The matched benchmark uses local worker IPC and does not measure the website network path.

**Evidence:**

- [editguard_chat/server.py](../../editguard_chat/server.py)
- [editguard_chat/store.py](../../editguard_chat/store.py)
- [tests/test_chat_reasonke.py](../../tests/test_chat_reasonke.py)
- [tests/test_chat_store.py](../../tests/test_chat_store.py)

### Slide 9: Fact identity, authority and version lineage

**Visual cue:** Distinguish the browser draft from committed approved state.

> A fact has a canonical entity, relation and scope key. The ownership key for the gateway’s second release is distinct from hosting, release one and the Pro service. Each approved version also retains its question, answer, source, aliases, approval metadata and lineage links. A draft remains in the browser until the operator explicitly approves it. A first approval creates an active record and event. A replacement must identify the active predecessor for the same key. A partial unique index prevents two active versions for that represented identity. These controls define authoritative state without turning customer messages into approved facts.

**Technical detail for rehearsal:** The normalized canonical key is encoded through question_key and scope_key. All three canonical identifiers must be supplied together. Without them, legacy records use normalized question/scope text. Aliases are metadata, not a trained entity resolver. Status can change between active, superseded and withdrawn, while fact content/version records and audit events preserve lineage. Approval actor metadata identifies a local research operator, not an independently authenticated enterprise administrator role.

**Evidence:**

- [editguard_chat/store.py](../../editguard_chat/store.py)
- [editguard_chat/workspace.py](../../editguard_chat/workspace.py)
- [editguard_chat/static/app.js](../../editguard_chat/static/app.js)
- [docs/CHAT_API.md](../../docs/CHAT_API.md)

### Slide 10: Atomic supersession and rollback

**Visual cue:** Show the one-revision, two-event rollback boundary.

> The critical guarantee is that fact state, revision and audit events change together. Replacement supersedes the old version and activates the new one at a shared revision. Rollback validates an active replacement and its direct same-key predecessor. Inside BEGIN IMMEDIATE it withdraws the replacement, restores the predecessor, advances the revision once and writes both events. Any exception rolls back every write. Tests inject a late audit-event failure to check that boundary. A competing active approval returns HTTP 409. Historical reads replay earlier events without changing current memory, and knowledge changes invalidate the retriever interface cache.

**Technical detail for rehearsal:** Rollback is idempotent when the predecessor is already active. An obsolete replacement cannot revert a newer successor. WAL, foreign keys, a thread-safe connection and explicit transactions provide local durability/concurrency behavior. The current lexical selector has no persistent embedding or answer cache to serve a withdrawn correction. Transaction tests establish consistency, not factual model correctness or automatic semantic conflict detection across unrelated keys.

**Evidence:**

- [editguard_chat/store.py](../../editguard_chat/store.py)
- [tests/test_chat_store.py](../../tests/test_chat_store.py)
- [tests/test_chat_workspace.py](../../tests/test_chat_workspace.py)
- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)

### Slide 11: Bounded iterative retrieval algorithm

**Visual cue:** Explain the frontier update and the finite stopping rules.

> The ranker tokenizes approved question, scope and answer text and applies a BM25-style score. It removes declared stop words and breaks ties by fact identifier. The query first selects two seed records. Their answer text becomes the search frontier. Each expansion round adds the highest-ranked unseen fact and advances the frontier to that fact’s answer. The selected deployment stops at four records, exhaustion or the round bound. With two seeds and a four-fact cap, at most two additions are normally needed. This is deterministic lexical expansion. It never reads a target answer, depth label or gold supporting identifier during retrieval.

**Technical detail for rehearsal:** The ranker uses k1=1.2 and b=0.75, unique query terms and idf=log(1+(N-df+0.5)/(df+0.5)). The iterative method permits at most three rounds and fills any unused slots from the last frontier ranking. In the concrete India query, hosting and ownership seed the search, then the selected owner answer retrieves its contact and that contact retrieves the office. The inclusion of a hosting distractor is visible rather than hidden. Ranking may still retrieve irrelevant records or fail unfamiliar wording.

**Evidence:**

- [editguard_benchmark/retrieval.py](../../editguard_benchmark/retrieval.py)
- [editguard_depth/retrieval.py](../../editguard_depth/retrieval.py)
- [reports/reasonke/depth-final/india-acceptance.json](../../reports/reasonke/depth-final/india-acceptance.json)

### Slide 12: Retrieval selection before final evaluation

**Visual cue:** Read complete paths, not neural accuracy, on this graph.

> The original two-stage selector expanded its query only once. On the new depth development packet it completed ninety-six of 128 supporting paths, including zero of thirty-two three-hop paths. The iterative policy completed all 128, including all thirty-two three-hop paths. Selection compared six declared candidates before neural profiling and final freeze. An explicit India display-development question exposed a one-seed distractor, so the final choice used two seeds and four facts after checking both display states. These figures measure support retrieval on development data. They are separate from final generated-answer accuracy and do not prove that more supplied context is always better.

**Technical detail for rehearsal:** The original method found 192/224 required fact occurrences. Every candidate completed the packet; the additional two display memory states distinguish the selected policy. Selection revision two documents that adjustment before profiling/freeze. The selected candidate completed 230/230 required occurrences across the 128 packet questions and two display states. Those two states are not independent held-out questions. The depth challenge preserves the original frozen operations retriever and results.

**Evidence:**

- [reports/reasonke/depth-development/retrieval-selection.json](../../reports/reasonke/depth-development/retrieval-selection.json)
- [editguard_depth/retrieval.py](../../editguard_depth/retrieval.py)
- [docs/DEPTH_RUN_PLAN.md](../../docs/DEPTH_RUN_PLAN.md)


**Handover:** “Member 3 will take the next part.”


## Member 3

### Slide 13: Prompt assembly and revision-safe history

**Visual cue:** Distinguish visible history, supplied history and freshly generated comparison.

> The worker serializes approved facts as JSON inside the updated-information block of the latest user message, followed by the current query. The tokenizer applies the checkpoint’s chat template. History includes only complete user-assistant pairs compatible with the current knowledge revision and model. Old pairs remain visible in the website but do not enter a new prompt. For historical comparison, both generations receive the same pre-question history with correction-conditioned pairs removed. This prevents a newer answer from leaking the correction into the older reference side. The comparison generates two new responses; it does not retrieve two weight snapshots or rewrite saved messages.

**Technical detail for rehearsal:** Normal edited chat can retain correction-conditioned pairs from the same revision. The application limits history to eight messages and 6,000 total characters, dropping complete pairs. Worker fields have additional size checks. The two comparison contexts are an approved historical snapshot or no correction facts, and the current snapshot. Revision zero means empty approved memory. JSON delimiters and source text do not establish general prompt-injection resistance.

**Evidence:**

- [editguard_chat/server.py](../../editguard_chat/server.py)
- [editguard_reasonke/runtime.py](../../editguard_reasonke/runtime.py)
- [tests/test_chat_reasonke.py](../../tests/test_chat_reasonke.py)

### Slide 14: Frozen model worker and answer evidence

**Visual cue:** Separate weight storage, compute precision, decoding and extraction.

> Both checkpoints contain approximately 7.615 billion parameters. Our measured configuration uses pinned safetensors, NF4 double-quantized weights, FP16 computation and SDPA attention. Decoding is greedy with one beam, a 4,096-token input ceiling and at most 512 new tokens. A T4 loads one model at a time. The measured loading peak allocated CUDA memory was about 6.59 gibibytes. After generation, exactly one nonempty answer block yields the extracted final text. Malformed or plain output uses a disclosed raw fallback. The response record retains raw text, selected facts, checkpoint revision, timing and token metadata. Nothing in that extraction labels the explanation verified.

**Technical detail for rehearsal:** The pinned ReasonKE_plus revision is ceaedc2ae7c8fed6257953587d991ce550db6106. Qwen's revision is a09a35458c702b33eeacc393d103063234e8bc28. Corrections never change weights. Allocated memory is a PyTorch measurement, not total device usage or a guarantee for another GPU. The benchmark has a hard 240-second parent request watchdog and 1,800-second load watchdog; the 120-second Transformers generation setting is soft. The worker rejects oversize input rather than silently changing the declared budget.

**Evidence:**

- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [configs/instruction-model.json](../../configs/instruction-model.json)
- [editguard_reasonke/runtime.py](../../editguard_reasonke/runtime.py)
- [reports/reasonke/operations-profiles-v1/README.md](../../reports/reasonke/operations-profiles-v1/README.md)

### Slide 15: Fact worlds, lineages and split control

**Visual cue:** Explain a world, a lineage, a question and a generated request.

> Each study contains forty-eight authored fictional fact worlds and 384 question records. Sixteen lineages support development and thirty-two support final evaluation. Eight related variants per final lineage create 256 main questions. We split lineages before question augmentation, so connected variants stay in one partition. The original study has nine facts per world and paths up to two hops. The separate depth study has twelve facts and extends to three hops. Its development and final wording banks differ. Graph audits check labels and supporting paths automatically. Service, team and person identities separate across splits, but simple values and relation vocabulary can repeat.

**Technical detail for rehearsal:** An edit lineage groups original and updated facts plus its related question variants. A lineage is the statistical cluster, not each paraphrase. Each authored pool includes 128 development and 256 final questions. The operations neural policy comparison used a predetermined 24-question development subset, while the depth retrieval-selection diagnostic used the 128-question development packet. Do not pool the two studies or describe all 256 variants as independent entities. Automated review is not independent human semantic review.

**Evidence:**

- [docs/OPERATIONS_DATASET_CARD.md](../../docs/OPERATIONS_DATASET_CARD.md)
- [docs/DEPTH_DATASET_CARD.md](../../docs/DEPTH_DATASET_CARD.md)
- [data/reasonke/operations-v1/manifest.json](../../data/reasonke/operations-v1/manifest.json)
- [data/reasonke/depth-v1/manifest.json](../../data/reasonke/depth-v1/manifest.json)

### Slide 16: Matched conditions and diagnostic populations

**Visual cue:** Read conditions across generators and context policies, then the distinct request totals.

> The original study has five main conditions: Reason-KE++ with initial facts, updated retrieval or full updated memory, and Qwen with updated retrieval or full updated memory. Every condition receives 256 questions. Both updated generators receive the same approved information and identical retrieved fact IDs, with compatible decoding budgets. Each model also answers sixteen preselected gold-support questions. The separate depth challenge has two iterative-retrieval conditions of 256 questions and sixteen oracle questions per model. That produces 1,312 original and 544 depth generations. Oracle diagnostics are smaller separate populations. The initial condition diagnoses staleness against updated truth.

**Technical detail for rehearsal:** Qwen uses its official instruction chat template and explicit short-answer/UNKNOWN instructions; ReasonKE uses the updated-information format. Matched facts and budgets do not make Qwen a verified causal ablation of ReasonKE++ training. Full memory receives all nine current facts, so its failures are not caused by deliberately stale baselines. No parametric editing baseline was measured in these final studies. The original and depth protocols freeze sources, configurations, datasets and scoring before final inference.

**Evidence:**

- [reports/reasonke/operations-final/protocol.json](../../reports/reasonke/operations-final/protocol.json)
- [reports/reasonke/depth-final/protocol.json](../../reports/reasonke/depth-final/protocol.json)
- [editguard_benchmark/models.py](../../editguard_benchmark/models.py)

### Slide 17: Scoring and paired uncertainty

**Visual cue:** Use all assigned outcomes and keep preservation distinct from correctness.

> Primary accuracy is the number of correct endorsed final answers divided by every assigned question. UNKNOWN, execution errors and unexecuted cases receive zero accuracy credit. Coverage separately measures how often the model answers. The frozen scorer matches the entire final span against reviewed controlled aliases for the requested subject and relation. Mentioning the target alongside extra owner text can therefore fail a request for only a person’s name. Preservation uses only locality questions the initial assistant already answered correctly. Factual locality accuracy uses all locality questions. Uncertainty resamples paired whole lineages, preserving correlations between their wording variants.

**Technical detail for rehearsal:** The scorer performs NFKC, whitespace and case normalization and strips limited final punctuation/formatting. It is not a general semantic judge. Aliases were fixed before final inference and audited automatically. The bootstrap uses 10,000 draws, seed 42 and 32 lineages per study. It estimates variation under this controlled design, not real-world representativeness. A narrow false-edited-value metric captures fewer failures than all possible scope mistakes. Correct final text also does not certify the full reasoning trace.

**Evidence:**

- [editguard_benchmark/scoring.py](../../editguard_benchmark/scoring.py)
- [editguard_benchmark/metrics.py](../../editguard_benchmark/metrics.py)
- [editguard_depth/metrics.py](../../editguard_depth/metrics.py)

### Slide 18: Original operations results

**Visual cue:** Read the stacked outcomes and each 256-question denominator.

> The complete original scorecard shows both success and failure. Updated retrieval gives Reason-KE++ 256 correct answers and Qwen 241, with fifteen abstentions. Full updated memory gives Reason-KE++ 231 and Qwen 241. Qwen’s full-memory condition has thirteen strict mismatches and two abstentions. Equal total accuracy can therefore hide different wrong-answer behavior. The retrieved-condition advantage is 5.86 percentage points, with a paired lineage interval from 3.91 to 8.20 points. All assigned requests completed. These results support the chosen bounded retrieval configuration and also show that Reason-KE++ did not win every context condition.

**Technical detail for rehearsal:** R0 has 136/256 correct against updated truth, comprising 12/128 applicable and 124/128 locality questions. It is a staleness diagnostic, not original-handbook accuracy. Both oracle conditions have 16/16 on their separate subsets. Main outcomes sum to 256 per condition. G-RAG answered-only precision is 241/241, but its primary accuracy remains 241/256. G-ALL coverage is 254/256. Later diagnosis preserves all thirteen scored-wrong G-ALL outcomes even when extra text explains some strict failures.

**Evidence:**

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/operations-final/FAILURE_ANALYSIS.md](../../reports/reasonke/operations-final/FAILURE_ANALYSIS.md)


**Handover:** “Member 4 will take the next part.”


## Member 4

### Slide 19: Separate depth challenge results

**Visual cue:** Read the group counts, then qualify the small three-hop difference.

> The separate depth packet gives sixty-four questions per group: locality, one hop, two hops and three hops. Reason-KE++ answers all sixty-four in every group. Qwen scores sixty-four, sixty-four, fifty-seven and sixty-three. Its eight remaining outcomes are UNKNOWN, not wrong nonabstaining answers or crashes. The overall difference is 3.125 percentage points, with a paired interval from 1.56 to 5.08. The three-hop difference is one answer and its interval includes zero. Depth alone does not explain the failures, because the wording constructions also change. This is an independent extension of the task, not a rerun of the original study.

**Technical detail for rehearsal:** The depth main conditions each contain 256 assigned questions. Both sixteen-question oracle groups answered correctly, but none of the eight Qwen failure IDs belongs to that oracle subset. The interval for the three-hop difference is [0,4.6875] points. The 64 two-hop questions consist of 32 direct and 32 paraphrased questions. Qwen's seven two-hop failures are all direct questions in the same wording family; all 32 two-hop paraphrases are correct. No retries or prompt changes were made after inspecting the final failures.

**Evidence:**

- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [reports/reasonke/depth-final/FAILURE_ANALYSIS.md](../../reports/reasonke/depth-final/FAILURE_ANALYSIS.md)

### Slide 20: Reason-KE++ full-memory failure diagnosis

**Visual cue:** Connect the output value to its actual supplied fact and requested identity.

> We inspected all twenty-five full-memory failures. Every one had the required support present, and every wrong final value exactly matched another supplied fact. Fifteen matched the Pro-service owner. The remaining cases matched another release’s owner, an unrelated city or region, or a contact. In the Garnet Relay case, the explanation explicitly names the Pro fact identifier while describing it as the requested service. In the Juniper Scheduler case, a correct owner appears as an intermediate boxed answer, but the model returns that owner’s contact as the team. These are observable association failures. They do not establish an internal attention or training defect.

**Technical detail for rehearsal:** The exact supplied-value groups are 15 Pro owners, 3 release-2 owners, 3 unrelated office cities, 2 previous-owner contacts, 1 hosting region and 1 current-owner contact. Two wrong finals contain the correct expected value as an indexed boxed subanswer, under a conservative parser diagnostic. R-ALL has 16 edit-applicable and 9 locality errors. The separate audit regenerates these counts from raw ledgers without inference or rescoring. Retrieved ReasonKE++ answers these same main questions correctly, but that comparison changes context, not weights.

**Evidence:**

- [reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md](../../reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)
- [scripts/analyze_presentation_failures.py](../../scripts/analyze_presentation_failures.py)

### Slide 21: Qwen failures and strict-format mismatches

**Visual cue:** Distinguish abstention, incorrect fact association and strict answer-contract failure.

> The failure audit also explains Qwen’s different behaviors. In the original retrieved condition, all fifteen nonsuccesses are two-hop paraphrase abstentions despite complete support. In the depth test, seven of nine examples from one direct two-hop template abstain, while the other direct templates score twenty-three out of twenty-three. The final three-hop paraphrase also abstains. Full memory adds ten outputs matching an off-target fact, two correct owner-contact chains with unwanted extra text, and one contaminated owner-contact concatenation, plus two abstentions. We retain the frozen scores. These observations locate failures after support selection but do not prove a causal effect of wording or distractors.

**Technical detail for rehearsal:** The supplementary ledger covers all 63 updated-method nonsuccesses across the two studies: 25 R-ALL, 15 operations G-RAG, 15 G-ALL and 8 depth G-RAG. Every record has complete required support and no token/time-budget flag. Two G-ALL outputs give the correct requested contact alongside its correct owning team and fail the person-only contract. Another combines the Pro owner and correct requested contact. Template concentration is descriptive after evaluation. The original oracle subset overlaps one operations G-RAG failure ID, while the depth oracle overlaps none of its eight failure IDs; neither is a full targeted recovery study.

**Evidence:**

- [reports/reasonke/operations-final/FAILURE_ANALYSIS.md](../../reports/reasonke/operations-final/FAILURE_ANALYSIS.md)
- [reports/reasonke/depth-final/FAILURE_ANALYSIS.md](../../reports/reasonke/depth-final/FAILURE_ANALYSIS.md)
- [output/presentation/analysis/failure-ledger.csv](../../output/presentation/analysis/failure-ledger.csv)

### Slide 22: Retrieval support and preserved locality

**Visual cue:** Explain each chart denominator before comparing percentages.

> Retrieval success and answer success are separate measurements. The original shared selector supplies all 320 required fact occurrences and all sixty-four two-hop paths. The depth selector supplies all 448 occurrences and all 256 main paths. Both generators receive the same selected records, so this is not two independent retrieval studies. For original-study locality, retrieved Reason-KE++ preserves all 124 answers that the initial model already got right, and answers all 128 locality questions correctly. Full memory preserves 117 and answers 119 correctly. Preserving an initially wrong answer would not be factual correctness, which is why we report both denominators.

**Technical detail for rehearsal:** The full-memory condition loses seven R0-correct locality answers but fixes two of the four R0-incorrect locality answers. The narrow false-edited-value endorsement count is 3/128 for full-memory ReasonKE versus 0/128 retrieved, while full memory has nine locality errors overall. The separate depth study has no initial-base condition, so it reports 64/64 locality correctness per model but does not independently estimate baseline-correct preservation. A complete labelled path in a small world does not guarantee large-memory retrieval performance.

**Evidence:**

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md](../../reports/reasonke/operations-final/REASONKE_FAILURE_ANALYSIS.md)

### Slide 23: The measured accuracy–latency tradeoff

**Visual cue:** Read accuracy and latency together, then explain the timing boundary.

> Higher observed coverage has a substantial latency cost. In the original retrieved conditions, Reason-KE++ has a median pipeline time of 15.06 seconds versus Qwen’s 0.92. In the depth study, the medians are 15.94 and 1.12 seconds, with 95th percentiles of 19.28 and 1.38 seconds. All 256 generated requests per condition contribute, including Qwen abstentions. Our saved-token audit also shows much longer Reason-KE++ outputs: median 204 versus 5.5 tokens originally and 212 versus six in the depth study. These are configured system measurements, not normalized compute comparisons or a causal isolation of model architecture.

**Technical detail for rehearsal:** Latency includes context selection, local IPC and generation. It excludes loading, downloads and the browser/network round trip. The token-count difference is consistent with explicit generated reasoning versus short direct answers, but does not by itself decompose the latency cause. Original median-quantile ratio is approximately 16.43 and the depth ratio is approximately 14.24. These are ratios of reported quantiles, not paired per-request speedups. The India HTTP workload measures a different mixed request population, including a two-generation comparison.

**Evidence:**

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)


**Handover:** “Member 5 will take the next part.”


## Member 5

### Slide 24: Correction editor and explicit authority

**Visual cue:** Show the active answer, replacement draft, scope and source.

> The product exposes these mechanisms through a usable workflow. This genuine Verity capture shows the active Bengaluru ownership fact beside a Pune replacement draft, with scope and source fields. Drafting alone does not activate a correction. The operator must approve the replacement and name the predecessor. The library keeps a canonical identity, source and version for inspection. The screenshot workspace restores the audited demonstration receipt rather than issuing new model calls. Our model worker was offline during capture, and the interface reports that state honestly. The editor remains useful for inspecting governed memory independently of inference readiness.

**Technical detail for rehearsal:** The image is an unedited full-viewport website capture. It shows a browser-local draft against restored r11 state. No approval was performed for screenshot creation. The source is a controlled fictional handbook, not an external assertion about a real payment company. The application uses a local research-operator workflow rather than hosted role-based multi-user administration. Customer chat text never silently publishes authoritative knowledge.

**Evidence:**

- [output/presentation/screenshots/provenance.json](../../output/presentation/screenshots/provenance.json)
- [scripts/prepare_verity_screenshot_workspace.py](../../scripts/prepare_verity_screenshot_workspace.py)
- [editguard_chat/static/app.js](../../editguard_chat/static/app.js)

### Slide 25: History, rollback and application acceptance

**Visual cue:** Identify both events at r10 and both at r11, then state the separate acceptance population.

> The history view shows why reversal is more than changing a textbox. Revision ten contains both publication of the replacement and supersession of the predecessor. Revision eleven contains withdrawal of the replacement and restoration of the predecessor. The same revision labels demonstrate the atomic event pairs. In the actual India acceptance workload, fourteen neural answers matched their targets and twelve consistency checks passed. It exercised corrected questions, unchanged scopes, conflict rejection, historical comparison and rollback. Those results cover one fictional display lineage and remain separate from held-out accuracy. The screenshot displays restored saved state, not a new live rollback execution.

**Technical detail for rehearsal:** Thirteen HTTP generation requests produce fourteen model answers because one comparison invokes generation twice. HTTP p50/p95 is 17.11/33.39 seconds over those thirteen mixed requests, not per-answer latency. The acceptance receipt and host audit retain exact request/response state and provenance. A new rollback demonstration requires a running worker for the fresh follow-up answers; historical SQLite state inspection itself does not require the GPU.

**Evidence:**

- [reports/reasonke/depth-final/india-acceptance.json](../../reports/reasonke/depth-final/india-acceptance.json)
- [reports/reasonke/depth-final/india-host-audit.json](../../reports/reasonke/depth-final/india-host-audit.json)
- [output/presentation/screenshots/technical-v2/provenance.json](../../output/presentation/screenshots/technical-v2/provenance.json)

### Slide 26: Chat evidence and stored-response semantics

**Visual cue:** Contrast the answer’s r10 badge with current r11 memory.

> The chat capture displays an actual saved Hyderabad answer generated at revision ten. The top bar reports that current memory is revision eleven. Those labels can differ correctly: rollback changes future context and does not rewrite historical messages. Each response retains the supplied fact records and its own revision, so we can inspect what produced that answer. A fresh query after rollback should use the restored Chennai path. Historical comparison generates two new responses from matched history. The screenshot is explicitly recorded evidence, and no new generation occurred while making the slides.

**Technical detail for rehearsal:** The selected conversation is Recorded: edited three hop, from the published generate_edited_three_hop operation. The saved answer evidence identifies r10 and the Pune/Arjun/Hyderabad chain. The restored screenshot workspace is r11. A displayed historical message is not a cache used for new answers. Without an available worker, send is disabled and no deterministic target substitutes for model output. The original acceptance receipt contains the actual fresh post-rollback Chennai answer separately.

**Evidence:**

- [reports/reasonke/depth-final/india-acceptance.json](../../reports/reasonke/depth-final/india-acceptance.json)
- [output/presentation/screenshots/technical-v2/provenance.json](../../output/presentation/screenshots/technical-v2/provenance.json)
- [editguard_chat/server.py](../../editguard_chat/server.py)

### Slide 27: Results explorer and failure inspection

**Visual cue:** Move from the aggregate outcome to expected, actual and supplied facts.

> The results explorer keeps the evaluation inspectable. Here it filters the twenty-five full-memory Reason-KE++ wrong answers and expands a release-scope failure. The interface shows the expected answer, generated final text, requested canonical identity, supplied facts and raw output. Filtering changes the view, never the scoring denominator. Users can switch between the original and depth studies and export scorecards, figures and raw evidence. This connects aggregate graphs to individual failures instead of displaying only a perfect headline score. The capture uses released saved records and requires no fresh GPU inference.

**Technical detail for rehearsal:** The expanded case asks Beryl Relay release 1, expects Willow unit 192 and endorses Amber unit 72, the release-2 owner. The app retains the raw model revision, prompt digest and pipeline latency. Explanatory prose remains labelled unverified. The screenshot captures the real frontend, including its Verity branding and offline worker state. The underlying frozen raw files and scoring rules are unchanged by this presentation or its post-hoc audit.

**Evidence:**

- [editguard_chat/results.py](../../editguard_chat/results.py)
- [output/presentation/screenshots/technical-v2/provenance.json](../../output/presentation/screenshots/technical-v2/provenance.json)
- [reports/reasonke/operations-final/final-reasonke.json](../../reports/reasonke/operations-final/final-reasonke.json)

### Slide 28: Engineering contribution and reproducibility

**Visual cue:** Name concrete artifacts and explain what reproducibility checks establish.

> Our contribution is the system and the evidence around the reasoner. We implemented governed fact memory, canonical identity, conflict and rollback transactions, revision-safe conversation handling and the private serving boundary. We adapted bounded retrieval, authored independent controlled datasets and built matched finite evaluation with complete ledgers. Protocol hashes, atomic receipts and host audits make the results replayable. Resume executes only unexecuted records under the same identities and preserves completed outcomes. The website, container, CI, data cards, reports and demonstration make that core usable and reviewable. The new failure ledger adds a reproducible description of every updated-method nonsuccess.

**Technical detail for rehearsal:** The current runtime loads published weights rather than reproducing multi-GPU training. Earlier auxiliary-model training remains a separate archived experiment and is not the source of current ReasonKE++ metrics. Source/data/configuration identity checks protect resume. Saved depth checkpoints at 174,197,235 and 247 records agree with the completed ledger. Software regression tests, neural accuracy, mechanical evidence audits and resource measurements have distinct purposes. A hash protects recorded identity, not semantic truth or independent human review.

**Evidence:**

- [docs/CONTRIBUTIONS_AND_EVIDENCE.md](../../docs/CONTRIBUTIONS_AND_EVIDENCE.md)
- [reports/reasonke/depth-final/resume-audit.json](../../reports/reasonke/depth-final/resume-audit.json)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)
- [Dockerfile.chat](../../Dockerfile.chat)
- [scripts/analyze_presentation_failures.py](../../scripts/analyze_presentation_failures.py)

### Slide 29: A reversible editing platform, evaluated

**Visual cue:** Close with the achieved contract, the accuracy–latency cost and the actual limits.

> We set out to make approved factual changes transfer across connected questions while preserving scoped knowledge and supporting reversal. The implemented platform provides that workflow, and the frozen retrieved configurations answered 256 of 256 questions in each controlled study. The matched Qwen conditions answered 241 and 248, while responding substantially faster. Full-memory failures show why context selection matters, and recorded rollback acceptance establishes operational consistency separately. The conclusion is definite but bounded: this is a functioning, auditable editing platform with strong results on its defined task. Larger memories, ambiguous scopes, unfamiliar operational language and fully faithful explanations are not established by these tests.

**Technical detail for rehearsal:** A perfect observed score remains plausible in a small explicit fact world with complete support, short aliases and repeated templates. Each study has only 32 final lineages and automated label review. The result is not 512 independent worlds, a general-chat benchmark or an unrestricted three-hop guarantee. Single-operator deployment, lexical scope matching and absent calibrated confidence are material limits. The principal observed tradeoff is greater answer coverage at higher generation latency. No missing training or test-loss plots are invented for an inference-integration project.

**Evidence:**

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [reports/reasonke/depth-final/india-host-audit.json](../../reports/reasonke/depth-final/india-host-audit.json)


## Optional technical backup slides

Use these only when the panel asks. They are outside the core speaking-time allocation.

### Slide 30: PPO objective and reward implementation

**Visual cue:** Use only if asked about PPO or faithful reproduction.

> PPO compares the current policy’s token probabilities with the previous policy through a likelihood ratio. The clipped objective prevents an estimated advantage from rewarding arbitrarily large probability changes. The reference configuration includes a critic, vLLM rollouts and KL control. The paper expresses a normalized blend of process and outcome rewards, while the pinned released code adds those components. At alpha one half the sum has the same weighting but twice the scale. That difference matters for reproduction claims. Our deployment does not run PPO, and the released checkpoint’s complete training history is not established by these example scripts.

**Technical detail for rehearsal:** The official SFT launcher specifies 10 epochs, learning rate 1e-5, cosine schedule and eight GPU processes. The PPO launcher specifies actor 1e-6, critic 1e-5, KL coefficient 0.001 and 15 epochs. These are published reference settings, not our measured training resources or completed runs. The collator fallback and reward-scale difference are reproducibility caveats. The paper's training hardware/time is not an estimate for the T4. The policy-ratio/advantage formula is the standard clipped surrogate, not a complete accounting of critic and entropy losses.

**Evidence:**

- [https://aclanthology.org/2026.findings-acl.1137/](https://aclanthology.org/2026.findings-acl.1137/)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh)
- [https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py)

### Slide 31: Answer-contract failures in detail

**Visual cue:** Keep contract violations, false associations and explanation quality distinct.

> This backup distinguishes final-answer strictness from factual association. In two Qwen full-memory cases, the output contains the correct owning team and requested contact, but violates the instruction to give only the contact name. Another concatenation combines a Pro-service owner with the correct contact, so its additional text is also factually contaminated. All retain their original wrong scores. Similarly, a correct intermediate boxed answer can coexist with a wrong final answer. The failure ledger retains every case and its evidence, so the panel can inspect these distinctions without replacing the frozen metric with a post-hoc success score.

**Technical detail for rehearsal:** The compatible concatenations are ops-045 and ops-047 two_hop_paraphrase. The contaminated concatenation is ops-005. The new parser finds two R-ALL wrong-final records containing the expected value in an action-section boxed answer; this is a narrow text diagnostic and not a semantic certification of the trace. Twenty-three operations/depth retrieved Qwen abstentions remain literal UNKNOWN. Together with two G-ALL abstentions, the updated-method ledger has 25 abstentions and 38 strict mismatches. The unchanged-base R0 diagnostic is excluded from that failure cohort.

**Evidence:**

- [output/presentation/analysis/failure-ledger.csv](../../output/presentation/analysis/failure-ledger.csv)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)
- [reports/reasonke/operations-final/FAILURE_ANALYSIS.md](../../reports/reasonke/operations-final/FAILURE_ANALYSIS.md)

### Slide 32: Evidence and configuration reference

**Visual cue:** Find the artifact that supports the question being asked.

> Every major claim has a concrete artifact. Model manifests identify exact revisions and budgets. Dataset manifests and frozen protocols bind the task. Raw ledgers retain all assigned outputs, supports and resources. Host replay recomputes extraction and scoring. Derived reports and chart workbooks expose the displayed values. The application acceptance receipt covers real HTTP generations and revision consistency separately. Our supplemental failure audit and screenshot provenance add no new model observations. This evidence chain is how another team member can reproduce the analysis and defend a result without relying only on a slide headline.

**Technical detail for rehearsal:** The current code and protocol sources must retain their recorded hashes. Presentation edits affect documentation and analysis outputs, not the frozen model/data/scoring files. The code checkout should use the version recorded with each protocol. CI and container definitions are reproducibility infrastructure, not proof of enterprise security or throughput. A generated answer with retrieved sources is still model output. The missing checkpoint README does not justify assigning the paper's headline benchmark score to the exact deployed revision.

**Evidence:**

- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [reports/reasonke/operations-final/protocol.json](../../reports/reasonke/operations-final/protocol.json)
- [reports/reasonke/depth-final/protocol.json](../../reports/reasonke/depth-final/protocol.json)
- [reports/reasonke/depth-final/india-acceptance.json](../../reports/reasonke/depth-final/india-acceptance.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)


## Technical questions and defensible answers

### Why is this knowledge editing when the weights do not change?

It is a parameter-preserving editing deployment: an approved replacement changes external authoritative memory and the conditioned model response. Parametric editing is another route, not the definition of every valid edit. Verity supports explicit version, scope, provenance and reversal. It does not claim a weight update for each correction.

### What exactly did our team train in the final Reason-KE++ system?

We did not train the released ReasonKE_plus or Qwen checkpoints. Wu and colleagues provide the Reason-KE++ method and training code. Our final contribution is governed memory, retrieval adaptation, independent data, serving, evaluation and audits. Earlier DistilBERT/FLAN-T5 training is a different archived project phase and cannot be presented as this checkpoint’s training.

### Does _plus prove this is the paper’s 95.48% model?

The configured author-published checkpoint is ReasonKE_plus and the project links it to Reason-KE++. Its exact immutable revision is recorded. A model-card link from that revision to the paper’s particular reported experiment is not available. We therefore use Reason-KE++ as the method name, preserve the exact ID and revision, and report only our own measured scores.

### Is this just RAG with a different name?

Retrieval supplies selected external facts in both updated model conditions. The reasoner is trained for structured use of corrections across connected questions. Governance, history and rollback determine which facts are authoritative. Our matched experiments assess this complete combination and show the importance of bounded context. We do not claim conventional RAG is obsolete or deliberately leave its knowledge stale.

### Why not use a database join or graph traversal?

For this controlled relation schema, a symbolic engine can solve the graph and is useful for ground truth. The evaluated question is whether a learned conversational generator endorses the requested answer from approved context, including wording changes and distractors. The comparison does not establish superiority over a symbolic QA system. We keep deterministic targets out of neural success counts.

### How do SFT and PPO differ?

SFT increases the likelihood of demonstrated structured responses. PPO generates responses, evaluates rewards and updates a policy using a clipped probability-ratio objective and estimated advantages. In Reason-KE++ the reward inspects format, hop count, decomposition similarity, boxed subanswers and final F1. We explain the published design but run the released weights for inference.

### What does intermediate-step supervision mean here?

The reward code parses generated action-stage questions and boxed answers, compares them with labelled steps, and combines those component scores. It is not a guarantee of internal reasoning faithfulness or a separate neural verification call during Verity chat. The code returns a combined response score with components, so do not describe it as an independently verified reward for every output token.

### Why do the paper and code have different reward equations?

The paper writes alpha*process+(1-alpha)*outcome, with alpha0.5 as its main setting. The inspected code returns process+outcome. At equal weighting that is twice the normalized blend, which preserves component weighting but changes scale relative to KL or other loss terms. We disclose this and do not claim a faithful training reproduction.

### Why did the original retriever miss three-hop paths?

It selects two question seeds and performs only one answer-expanded reranking before filling the four slots. A downstream contact’s office may require another expansion through the person’s name. The selected iterative frontier adds one unseen record per round, so the displayed ownership, contact and office chain can enter the context. The improvement was selected on development support labels, not final model outputs.

### Can the four-fact selector handle any four-hop question?

No. Two seeds may include a distractor, leaving only two further additions under the selected four-record cap. Lexical overlap can also fail. The software has a finite candidate and context bound, while the neural evidence covers the specified two- and three-hop task worlds. There is no arbitrary-depth guarantee.

### How is scope enforced?

The store separates canonical entity/relation/scope IDs and rejects competing active versions for the same represented key. Explicit release and entity text enters retrieval and the model prompt. This is not a learned semantic entity resolver or calibrated applicability classifier. Overlapping meanings under different keys can require operator resolution.

### Why keep both an application lock and SQLite transactions?

The application lock fixes the revision across read, retrieval, generation and edit operations for one operator. SQLite transactions atomically commit short state/event or message writes. Holding the operation lock during generation prevents a concurrent edit from changing the revision, while keeping the database transaction short. These are local guarantees rather than distributed consensus.

### What happens if a model call fails?

The application generates before storing a new turn. A failed call leaves no duplicate or dangling new user message. The website reports model unavailability or the request error. It never replaces the failure with the deterministic expected answer. Completed benchmark outcomes remain recorded and are not retried to improve scores.

### Why exclude prior revisions from model history?

An old generated answer can leak a superseded value into a new prompt after an edit. We preserve it in the visible chat but remove the complete user-assistant pair from supplied history. Comparisons also exclude correction-conditioned pairs from both sides and use matched pre-question history, preventing a new correction from entering the reference through a previous answer.

### How do we know the test did not leak?

World/edit lineages split before variants are generated. Service/team/person identities separate across partitions, and the depth study has separate wording banks. Protocol identities freeze before final inference. Templates, relation vocabulary and simple values still repeat, and labels are automatically audited. These controls reduce leakage but do not establish a broad independently authored language benchmark.

### Why can 100% be real but still limited?

The final answer target is a short controlled value in a world with nine or twelve explicit facts. The selector supplied every required path, and each study has only 32 held-out lineages with correlated variants. The recorded score is real for that population. It does not imply arbitrary paraphrase, large-memory, general-chat or explanation reliability. We preserve failures in other conditions rather than inventing hard cases after evaluation.

### Did retrieval cause all full-memory failures?

The controlled difference changes supplied context, and retrieved Reason-KE++ answers all main questions correctly. In full memory, every wrong final value matches another supplied fact, with 15 matching the Pro owner. That is evidence of output association failure under the larger context. It does not reveal the internal attention mechanism or causally isolate a particular distractor without a separate planned ablation.

### Why are three-hop Qwen results better than two-hop results?

The groups differ in wording. Seven direct two-hop failures concentrate in one template with 7/9 abstentions, while other direct templates are 23/23 correct and two-hop paraphrases 32/32. The three-hop group has one abstention. These observations do not support a monotonic depth law and do not prove wording caused failure.

### Do the oracle results repair the observed failures?

They are preselected small diagnostic subsets. Only one original retrieved Qwen failure ID is in its oracle subset; none of the eight depth failures is in its oracle subset. Therefore 16/16 oracle accuracy is not a targeted recovery result for all failures. It helps separate a gold-support generation population from the main retrieval population.

### Why not count a correct contact plus extra owner text as correct?

The frozen request asks for only the person’s name, and the whole final span must match its fixed aliases. Two G-ALL mismatches contain a compatible correct owner/contact chain, so we describe them as answer-contract failures rather than false facts. The primary score stays unchanged. Changing aliases after test inspection would invalidate the frozen comparison.

### Does the system verify reasoning?

It stores raw output and sources, and a separate bounded audit checks extracted indexed subanswers against supplied support. Some explanations contain unsupported claims even when final answers are correct. The interface does not certify arbitrary prose or model-internal reasoning. The post-hoc boxed-text check is an additional parser diagnostic, not an explanation-quality guarantee.

### Why is Reason-KE++ slower than Qwen here?

The measured prompts, checkpoint behavior and output lengths differ. The saved output medians are 204 versus 5.5 tokens originally and 212 versus 6 in depth. This is consistent with generating explicit reasoning versus short answers. It does not causally decompose latency or normalize compute. We report request p50/p95 with their exact measurement boundary and include abstentions.

### What does a paired lineage confidence interval mean?

The bootstrap samples whole lineages with replacement and recomputes the difference between matched conditions for 10,000 draws. Keeping all variants in a sampled lineage preserves their dependence. It describes variation in this controlled population and does not repair unrepresentative data or give a probability that every future question succeeds.

### Can we perform a live demo from the screenshot workspace?

The workspace contains actual saved model responses and audited restored fact state. It makes no new model calls. A live demo needs a ready GPU worker and an explicitly configured fresh demonstration workspace. State inspections can work offline. Do not describe clicking a recorded conversation as fresh inference.

### What is the principal conclusion?

The platform implements reversible governed editing and achieves strong final-answer results on its bounded connected-fact task. Its matched Reason-KE++ retrieval configuration improves observed coverage over the chosen Qwen checkpoint, at substantially higher latency. Full-memory association failures, template-sensitive abstentions, limited worlds and automatic review constrain generalization claims.

## Demonstration and fallback

Use the current [India demonstration guide](../INDIA_DEMO.md). Begin with a ready model worker and a fresh demo workspace. Ask the three-hop office question, approve the ownership replacement, request a fresh answer, check unchanged hosting and historical-release facts, then roll back and request another fresh answer. A live comparison makes two model calls and takes longer than one question.

If the GPU worker is unavailable, identify the saved responses as recorded evidence. The screenshot workspace is a replay of actual audited responses and final state, not a deterministic mock or new inference. Do not click a recorded chat and describe it as a fresh answer. Do not replace a failed live model answer with the expected database target.

## Evidence-based contribution allocation

These areas cover implemented, reviewable work. They are an allocation map, not invented biographical history. Before attaching names, each person must confirm their actual implementation, reproduction, review, analysis or documentation. Several people may collaborate on one area. Presentation assignments are independent.

| Member | Contribution area | Concrete work to describe once confirmed | Evidence |
| --- | --- | --- | --- |
| Member 1 | Task, dataset and ground truth | Fictional fact worlds, edits, controls, question variants, lineage partitions, graph-label audits and dataset documentation. Collaborative review with model and evaluation owners. | editguard_benchmark/data.py; editguard_depth/data.py; dataset cards; tests/test_operations_benchmark.py |
| Member 2 | Reasoner integration and retrieval | Pinned checkpoint loading, measured T4 adaptation, bounded inference, private worker, original and iterative lexical selectors, prompt assembly and history-policy integration. Published model training remains attributed. | editguard_reasonke/runtime.py; editguard_chat/reasonke_backend.py; retrieval modules; raw profiles |
| Member 3 | Evaluation, analysis and reproducibility | Matched fixed protocols, finite execution/resume, endorsed-answer scoring, complete outcome ledgers, paired intervals, latency/token/resource analysis, host audits, failure diagnosis and scientific reports. | editguard_benchmark/; editguard_depth/metrics.py; scripts/analyze_presentation_failures.py; released receipts and derived reports |
| Member 4 | Knowledge state and backend reliability | Canonical identity, explicit approval, supersession/conflicts, event history, atomic rollback, revision-compatible chat, message transactions and strict snapshot export/restore. Collaborative integration review. | editguard_chat/store.py; editguard_chat/workspace.py; editguard_chat/server.py; store/workspace/history tests |
| Member 5 | Product, deployment and presentation | Verity chat and correction editor, source/history and comparison views, results explorer, evidence exports, container/notebook packaging, India demo, UI acceptance and team material. | editguard_chat/static/; editguard_chat/results.py; Dockerfile.chat; notebooks; India receipts; docs/presentation/ |
