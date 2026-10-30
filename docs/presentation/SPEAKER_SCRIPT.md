# EditGuard presentation script

Ten slides for five speakers. Plan for about eight to nine minutes of speech, plus an optional two minute live demonstration. Replace Member 1 to Member 5 with names. The speaking order does not determine contribution ownership.

Read the quoted speech aloud. The visual cues and optional demonstration instructions are for rehearsal. The contribution allocation at the end needs each member to confirm their actual implementation, review, testing or documentation work.

## Speaking order

| Speaker | Slides | Focus | Approximate time |
| --- | --- | --- | --- |
| Member 1 | 1 and 2 | Problem and connected correction example | 1 minute 20 seconds |
| Member 2 | 3 and 4 | Model architecture and safe knowledge changes | 1 minute 40 seconds |
| Member 3 | 5 and 6 | Dataset and fair evaluation | 1 minute 40 seconds |
| Member 4 | 7 and 8 | Measured findings and tradeoffs | 1 minute 55 seconds |
| Member 5 | 9 and 10 | Application demonstration and conclusion | 1 minute 40 seconds |

## Member 1

### Slide 1 EditGuard

**Time:** about 30 seconds. **Visual cue:** introduce the title, then look at the panel.

> Good morning. Our project is EditGuard, a knowledge editing and verification platform for conversational AI. Our goal is to make a factual correction change the assistant's relevant answers, preserve unrelated information, and remain traceable and reversible. We combine an existing learned reasoning model with an approved external memory and a working application. We also measure where that combination succeeds and where it fails.

### Slide 2 A correction changes connected answers

**Time:** about 50 seconds. **Visual cue:** follow the ownership, contact and office example. The example is fictional.

> Consider Kaveri Payment Gateway in release 2. Its owner is Bengaluru Platform Team. The team's contact is Meera Iyer, whose office is in Chennai. Now an administrator changes the owner to Pune Reliability Team. Its contact is Arjun Deshmukh, whose office is in Hyderabad. A correct assistant should update the owner, the contact and the office answer. The office question requires three connected facts. However, the gateway's hosting city should remain Bengaluru, and an older release should retain its own owner. A website can list individual records. Our task is to maintain consistent answers when users query across those records.

**Handover:** “Member 2 will explain how the system applies that change.”

## Member 2

### Slide 3 The model and approved memory

**Time:** about 55 seconds. **Visual cue:** trace the question through context selection and model generation.

> The final system uses the author published ReasonKE_plus checkpoint. Its authors trained the model upstream. Our project integrates and evaluates it. We keep its weights frozen. When an administrator approves a correction, we update the external fact memory without retraining the model. For each question, a bounded lexical selector retrieves relevant approved records. In the depth extension, it expands through answer text to collect connected facts. The private model worker then generates the answer from that context. We store the supplied facts and model output so we can inspect the evidence. The selector is deterministic software, not a newly trained scope classifier.

### Slide 4 Approval history and rollback

**Time:** about 45 seconds. **Visual cue:** point to the lifecycle and history controls.

> Knowledge changes pass through an explicit administrator workflow. Each fact has a source, entity, property and scope identity. A customer's chat message cannot silently become an approved correction. Publishing a replacement records its predecessor and a new knowledge revision. A competing approval for the same active key triggers a conflict instead of choosing silently. Historical comparison generates answers from earlier and current approved states. Rollback restores the previous active version in one database transaction. These controls establish which information the model receives. They do not automatically prove that every generated sentence is true.

**Handover:** “Member 3 will explain how we tested the system fairly.”

## Member 3

### Slide 5 Independent controlled datasets

**Time:** about 50 seconds. **Visual cue:** distinguish the two studies and the development versus final split.

> We authored fictional operational records so that we control the original fact, correction and expected answers. Each study contains 48 graph lineages and 384 questions. We separate 16 development lineages from 32 final lineages before creating related question variants. Each main model condition therefore receives 256 final questions. The original study tests direct, paraphrased, connected and unrelated questions. The separate depth study uses twelve facts per world and includes three hop questions with a separate wording bank for the final split. Labels were audited automatically. Related variants still share a lineage, and wording templates repeat within each split.

### Slide 6 Matched conditions and fixed scoring

**Time:** about 50 seconds. **Visual cue:** identify the matched inputs and separate diagnostic populations.

> The original evaluation contains 1,312 generations across five main conditions and two small gold support diagnostics. The separate depth study contains 544 generations. Updated baselines receive the same approved facts and supporting information. We keep generation settings and answer budgets compatible, using fixed greedy generation and four bit loading on a T4. We score the endorsed short answer against explicit ground truth. An UNKNOWN response receives zero accuracy credit and stays in the denominator. We measure retrieval separately from generation and use preselected gold support questions as diagnostics. We freeze the final configuration and scoring before the final run.

**Handover:** “Member 4 will show the results and their practical meaning.”

## Member 4

### Slide 7 Selected context improves this task

**Time:** about 55 seconds. **Visual cue:** compare retrieved context with full memory, not just the tallest bar.

> In the original study, ReasonKE with updated retrieved facts answered all 256 main questions correctly. Qwen with the same retrieval answered 241 correctly and abstained on fifteen. More context did not always help. ReasonKE with the full updated memory scored 231 out of 256, while Qwen full memory scored 241. That result prevents us from claiming ReasonKE wins every comparison. Retrieval supplied the required support, but the generator could still confuse facts or abstain. We also measured preservation separately: ReasonKE retrieval retained all 124 previously correct locality answers and answered all 128 locality controls correctly. Preservation and factual accuracy use different denominators.

### Slide 8 Three hop results and latency

**Time:** about 60 seconds. **Visual cue:** show the depth comparison, then the latency values.

> In the separate depth study, ReasonKE scored 256 out of 256 and Qwen scored 248, with eight UNKNOWN responses. On three hop questions, the scores were 64 out of 64 and 63 out of 64. The overall difference was 3.125 percentage points. A paired analysis resampling whole lineages gave a 95 percent interval of about 1.56 to 5.08 points. The three hop difference interval included zero, so that subset supports a weaker comparison. There is also a clear speed cost. Warm median latency was 15.94 seconds for ReasonKE and 1.12 seconds for Qwen. The stronger observed answer score therefore comes with substantially slower responses.

**Handover:** “Member 5 will show how these components work together in the application.”

## Member 5

### Slide 9 The application workflow

**Time:** about 50 seconds, excluding an optional live demonstration. **Visual cue:** identify Chat, Knowledge and Research results.

> The application provides normal text chat, a knowledge editor, source display, history, comparison and rollback. Research results shows saved experiments and raw outputs. In the demonstration, we ask for the original owner and contact's office, approve the ownership replacement, and ask again using different wording. We also check hosting and the older release, then roll back and request a fresh answer. Our recorded India application acceptance workload completed fourteen actual model answers correctly and passed twelve transaction and provenance consistency checks. This is one fictional display lineage. We report it separately from held out benchmark accuracy.

### Slide 10 Findings and limits

**Time:** about 50 seconds. **Visual cue:** end with the achieved outcome and the limits beside it.

> Our completed product supports approved, traceable and reversible knowledge changes, with reproducible measurements of connected answers. Our contribution includes the application, knowledge lifecycle, independent datasets, context selection adaptations and evaluation infrastructure. The observed 100 percent scores are real results on small controlled tests. Each world has few explicit facts, the selector supplied complete support, and templates repeat. Those conditions limit what the score establishes. We retain the full memory failures, abstentions and latency cost. The conclusion is that this design met its defined short answer contract in these tests. Broader operational language, larger memories and ambiguous or missing information require further validation. Thank you.

## Optional live demonstration

Use this after slide 9 if the panel permits extra time. Keep the model loaded and check its ready status before presenting. Use a fresh demonstration workspace so the initial handbook is revision 9.

1. In Chat ask: **In which Indian city is the office of the on-call contact for the team that owns Kaveri Payment Gateway in release 2?** Initial target: Chennai.
2. In Knowledge choose **Draft the example ownership update**, inspect it, then explicitly approve replacement. Drafting alone changes nothing.
3. Ask the office question again. Updated target: Hyderabad. Ask **Which Indian city hosts Kaveri Payment Gateway in release 2?** Its target remains Bengaluru.
4. Show the source and history. Roll back the replacement and request a fresh office answer. Restored target: Chennai. Existing stored chat answers do not rewrite themselves.

**During generation:** “The new approval changes memory. The model generates a fresh answer from the selected approved state.”

**If the GPU session is unavailable:** “The live worker is unavailable. We will show the recorded application evidence and its saved outputs.” Use saved evidence explicitly. Do not present stored answers as a live generation.

## Short answers for panel questions

**How is this knowledge editing if it uses retrieval?**

“This implementation uses external memory to change the model system's answers while keeping weights frozen. Its operational contribution is scoped approval, versioning, history and rollback around a learned reasoner. It uses a memory based route to knowledge editing. It does not perform a weight editing method such as ROME.”

**What did your team train?**

“The final ReasonKE and Qwen checkpoints were trained upstream. We integrated and evaluated them. An earlier auxiliary model study remains archived separately. We do not claim its training as training of this final deployment.”

**Why is accuracy 100 percent?**

“It is 256 correct out of 256 assigned questions in each small controlled main test. The worlds have explicit facts and complete retrieved support, with repeated task constructions. That is a bounded observed result. ReasonKE full memory still made errors, and arbitrary user dialogue has no such guarantee.”

**Why use an LLM when a database can return facts?**

“A structured lookup is sufficient for a single fixed record. We evaluate natural language questions that require selecting a scope and combining related records. Our database establishes approved state. The model generates the requested answer. We keep their correctness evidence separate.”

**Did the oracle experiment fix Qwen's eight depth failures?**

“We did not test that claim. The sixteen oracle questions were preselected, and none of those eight failure IDs overlaps that subset. Both models passed the diagnostic, but it does not demonstrate recovery on those failures.”

**Can you guarantee a new paraphrase or a longer chain?**

“No. We demonstrated the tested direct and paraphrased questions through three hops. Novel wording, larger graphs and incomplete evidence need separate evaluation.”

**How do you stop a bad correction?**

“Only explicit approval activates it. Canonical keys prevent competing active versions in the same represented scope. Sources and history remain visible, and rollback is transactional. Semantic contradictions written under unrelated keys can still require administrator judgement.”

**What belongs to your project rather than the upstream repository?**

“The governed website, transactional memory and history, independent fictional datasets, retrieval adaptations, finite matched experiments, scoring and uncertainty, raw output audits and reproducible reports. The upstream authors own the learned model method and training.”

## Contribution allocation for the five members

This maps completed project work into five substantive ownership areas. Confirm actual work before attaching names or using past tense. A speaker can present any slides regardless of their contribution area. Implementation, review, reproduction, testing and documentation can be shared and should be credited accurately.

| Member | Contribution area | Work to describe once confirmed | Reviewable evidence and collaboration |
| --- | --- | --- | --- |
| Member 1 | Dataset and experiment design | Controlled fictional graph design, original and corrected truth, direct and paraphrased questions, scope and locality controls, lineage splits, supporting fact audit and dataset limitations. | `editguard_benchmark/data.py`, `editguard_depth/`, dataset cards and benchmark tests. Collaborate with Members 2 and 3 on retrieval contracts and scoring. |
| Member 2 | Model integration and context selection | Pinned ReasonKE and Qwen inference, measured T4 configuration, bounded prompts, private worker, original and iterative selection, revision consistent conversation context. Describe model integration rather than claiming upstream model training. | `editguard_reasonke/runtime.py`, retrieval modules, profile receipts and model card. Collaborate with Members 1, 3 and 4 on evidence, budgets and approved context. |
| Member 3 | Evaluation and reproducibility | Frozen protocols, finite runs and resume, endorsed answer scoring, all outcome denominators, retrieval and oracle diagnostics, paired lineage intervals, resource measurements, audits, graphs, reports, CI and container verification. | `editguard_benchmark/`, `editguard_depth/metrics.py`, `scripts/audit_operations_evidence.py`, released study evidence and `.github/workflows/ci.yml`. Collaborate across the team on fair claims and regression evidence. |
| Member 4 | Knowledge lifecycle and backend reliability | Canonical identity, explicit approval, replacement and conflicts, immutable events, historical views, atomic rollback, conversation transactions and validated workspace restore. | `editguard_chat/store.py`, `editguard_chat/workspace.py`, API documentation and store/recovery tests. Collaborate with Members 2 and 5 on revision safe generation and interface behavior. |
| Member 5 | Application and demonstration | Chat, correction editor, searchable knowledge library, source/history display, conflict choices, historical comparison, result explorer, scientific downloads and India demonstration workflow. | `editguard_chat/static/`, `editguard_chat/server.py`, `docs/INDIA_DEMO.md` and recorded application acceptance evidence. Collaborate with Members 1, 3 and 4 on examples, metric labels and complete user flows. |
