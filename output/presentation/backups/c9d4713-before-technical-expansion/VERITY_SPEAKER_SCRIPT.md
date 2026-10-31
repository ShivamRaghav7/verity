# Verity mid-project review: timed five-speaker script

**Total: 10:00. Five speakers, 2:00 each. Thirteen slides, no extra introduction or closing slide.**

Read only the quoted speech. The remaining text is rehearsal guidance. The schedule includes visual reading, pointing and handovers. Rehearse at about 125 words/minute and keep a visible timer. Q&A follows the ten-minute presentation. The website section uses genuine recorded captures; fresh GPU generation is an optional demonstration afterward.

Replace Member 1–5 with names. Presentation assignments need not match contribution ownership. Identify the reused checkpoint once in configuration/Q&A. The main story is our approved-memory application and measured comparisons.

## Fixed speaking order

| Speaker | Clock | Slides | Main subject |
| --- | --- | --- | --- |
| Member 1 | 00:00–02:00 | 1–2 | Objective and sourced Air India motivation |
| Member 2 | 02:00–04:00 | 3–4 | Implemented architecture and retrieval |
| Member 3 | 04:00–06:00 | 5–8 | Governed corrections and website walkthrough |
| Member 4 | 06:00–08:00 | 9–11 | Experimental design, findings and latency |
| Member 5 | 08:00–10:00 | 12–13 | Engineering conclusion and next research milestone |

## Member 1

### Slide 1: Verity

**Clock: 00:00–00:25. Slot: 25 seconds. Speech: 42 words.**

> Verity is our platform for updating the facts used by an AI assistant. An approved correction should transfer to direct and connected questions, with a clear source, history and reversal. This review covers our implemented system, measured findings and next training milestone.

**Visual action and pacing:** Introduce the product and objective. Advance at 00:25. At 125 words/minute, leave approximately 5 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Current implementation: governed external memory, deterministic retrieval and fixed neural generators. The reused ReasonKE_plus checkpoint is identified in configuration notes. The new training milestone is future work. This is a mid-project review, not a claim of a newly trained final model.

**Evidence:**

- [docs/CURRENT_ARCHITECTURE.md](../../docs/CURRENT_ARCHITECTURE.md)
- [docs/CONTRIBUTIONS_AND_EVIDENCE.md](../../docs/CONTRIBUTIONS_AND_EVIDENCE.md)

### Slide 2: Air India: one dated update, connected questions

**Clock: 00:25–02:00. Slot: 95 seconds. Speech: 150 words.**

> Consider a familiar change. In December 2021, the government described Air India as government-owned. On twenty-seven January 2022, the completed handover transferred its shares to Talace, a wholly owned subsidiary of Tata Sons. An assistant using the older record can continue presenting the old owner unless its reference information is updated. The connected question is more interesting: which company is the parent of the new owner? The answer follows the ownership record to Talace, then to Tata Sons. Its official contact page supplies Mumbai as the city. We use these official sources in the website demonstration. The ownership example is explicitly dated to the 2022 handover, because later ownership arrangements changed. A question about ownership before the handover should still use the earlier government record. Updating the selected snapshot must preserve that historical scope. The engineering problem is maintaining the correct reference state and using it consistently across related questions.

**Visual action and pacing:** Point to the two dates and parent chain. Allow a deliberate reading pause. Hand over at 02:00. At 125 words/minute, leave approximately 23 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Primary sources: CCI/PIB 20 December 2021 reports government ownership; Ministry of Finance/PIB 27 January 2022 reports transfer to Talace and its Tata Sons parent. Tata confirms the handover. The corporate-contact fact is from the official Tata contact page checked on 31 October 2026, not a claim about its historical 2022 office. A later Tata merger release documents subsequent change. The demo's selected snapshot advances from the prior dated record to the completed handover. It is not a claim that the model's pretraining necessarily gives a stale answer. Photo is the genuine image embedded in the PIB handover announcement.

**Evidence:**

- [https://www.pib.gov.in/pressreleasepage.aspx?lang=2&prid=1783550&reg=48](https://www.pib.gov.in/pressreleasepage.aspx?lang=2&prid=1783550&reg=48)
- [https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1792950&lang=2&reg=48](https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1792950&lang=2&reg=48)
- [https://www.tata.com/newsroom/business/air-india-handover](https://www.tata.com/newsroom/business/air-india-handover)
- [https://www.tata.com/contact-us](https://www.tata.com/contact-us)
- [https://www.tata.com/newsroom/business/air-india-vistara-merger-completion](https://www.tata.com/newsroom/business/air-india-vistara-merger-completion)
- [data/demos/air-india-handover.json](../../data/demos/air-india-handover.json)

## Member 2

### Slide 3: Implemented architecture

**Clock: 02:00–03:05. Slot: 65 seconds. Speech: 93 words.**

> The architecture separates approved knowledge from generated answers. A question enters the Python API. The application captures the current memory revision, selects relevant approved records and filters conversation history to avoid stale context. A private GPU worker generates the response. The application then stores the answer together with the facts supplied, model identity and memory revision. SQLite manages the approved versions and audit events. Our current comparison uses two frozen seven-billion-parameter generators with matched retrieved information. This gives us a working, inspectable baseline platform on which the next training stage can be evaluated.

**Visual action and pacing:** Trace the editable flow once. Identify where approval and generation meet. Advance at 03:05. At 125 words/minute, leave approximately 20 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** The reused generator is YukinoKaren/ReasonKE_plus at ceaedc2ae7c8fed6257953587d991ce550db6106. The comparison is Qwen/Qwen2.5-7B-Instruct at a09a35458c702b33eeacc393d103063234e8bc28. Both have 7,615,616,512 parameters. Actual inference uses NF4 double quantization, FP16 compute, SDPA, greedy decoding, 4,096 input tokens and 512 new tokens on T4, one model at a time. New approvals change external memory. Application operation lock spans generation; short SQLite transactions persist changes/turns. Evidence shows supplied inputs, not a certification of all generated prose.

**Evidence:**

- [editguard_chat/server.py](../../editguard_chat/server.py)
- [editguard_chat/store.py](../../editguard_chat/store.py)
- [editguard_chat/reasonke_backend.py](../../editguard_chat/reasonke_backend.py)
- [editguard_reasonke/runtime.py](../../editguard_reasonke/runtime.py)
- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [configs/instruction-model.json](../../configs/instruction-model.json)

### Slide 4: Retrieval across connected facts

**Clock: 03:05–04:00. Slot: 55 seconds. Speech: 98 words.**

> Connected questions need facts that may not share all the original question's words. We first rank approved records using a BM25-style lexical score and choose two seeds. Their stored answer text becomes the next search query. We add the highest-ranked unseen fact, then continue from its answer. The deployed iterative policy stops at four facts. In the Air India example, the selected owner leads to its parent, and the parent leads to the contact city. This is transparent, bounded retrieval. Generation remains a separate step, which lets us distinguish missing support from a wrong answer despite complete support.

**Visual action and pacing:** Follow owner, parent, city. State the four-fact budget. Hand over at 04:00. At 125 words/minute, leave approximately 8 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Original study used its preserved two-stage lexical selector. The separate depth study/application policy uses iterative answer-frontier expansion, two seeds, at most three rounds and cap four. It examines approved id/question/scope/answer text, never hidden target or support IDs. Scores are not probabilities. The active memory guard is 200 records; evaluation worlds are smaller. On depth development, iterative support improved complete paths from 96/128 to 128/128 before final freeze. This is development retrieval evidence, separate from final neural accuracy.

**Evidence:**

- [editguard_benchmark/retrieval.py](../../editguard_benchmark/retrieval.py)
- [editguard_depth/retrieval.py](../../editguard_depth/retrieval.py)
- [reports/reasonke/depth-development/retrieval-selection.json](../../reports/reasonke/depth-development/retrieval-selection.json)
- [data/demos/air-india-handover.json](../../data/demos/air-india-handover.json)

## Member 3

### Slide 5: A correction has an explicit lifecycle

**Clock: 04:00–04:30. Slot: 30 seconds. Speech: 49 words.**

> A correction begins as a draft with an answer, source and scope. Publication explicitly approves it. Replacing an active fact requires naming its predecessor for the same canonical identity. Competing approval is rejected. Rollback withdraws the replacement and restores its predecessor in one transaction, while retaining the audit history.

**Visual action and pacing:** Point to approve, supersede and rollback. Advance at 04:30. At 125 words/minute, leave approximately 6 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Canonical entity/relation/scope identifiers map to the stored key. A partial unique index allows one active version per represented key. BEGIN IMMEDIATE groups state, revision and audit events. Rollback advances the revision once and writes withdrawn/restored events together. Historical scope and the selected local snapshot are distinct; rollback does not reverse the actual Air India event. Same-key conflict checks do not detect every semantic contradiction under different keys.

**Evidence:**

- [editguard_chat/store.py](../../editguard_chat/store.py)
- [tests/test_chat_store.py](../../tests/test_chat_store.py)
- [data/demos/air-india-handover.json](../../data/demos/air-india-handover.json)

### Slide 6: Website: approve the sourced update

**Clock: 04:30–05:00. Slot: 30 seconds. Speech: 44 words.**

> This is the real Verity correction editor. The selected reference record starts with the earlier government-ownership snapshot. We enter Talace as the updated owner, attach the dated government announcement, inspect the scope and explicitly publish the replacement. The active record changes only after approval.

**Visual action and pacing:** Point to source/date, scope and approval. Advance at 05:00. At 125 words/minute, leave approximately 9 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Use genuine captured website state from the isolated sourced-demo workspace. The fixture uses official sources, not fictional Kaveri data. It remains a product demonstration, separate from held-out evaluation. The snapshot moves from 20 December 2021 to 27 January 2022; it does not assert present-day ownership. The presenter describes recorded captures, without waiting for live GPU inference inside this slot.

**Evidence:**

- [data/demos/air-india-handover.json](../../data/demos/air-india-handover.json)
- [scripts/prepare_sourced_demo.py](../../scripts/prepare_sourced_demo.py)
- [docs/CHAT_API.md](../../docs/CHAT_API.md)

### Slide 7: Website: ask a connected question

**Clock: 05:00–05:35. Slot: 35 seconds. Speech: 59 words.**

> Back in Chat, we ask for the parent of Air India's owner in the selected snapshot. The approved ownership fact supplies Talace, and the parent record supplies Tata Sons. The response retains its model and memory revision, and the interface shows the source records supplied with it. Those records make the answer inspectable and help us diagnose any mismatch.

**Visual action and pacing:** Point to the question, generated answer and source evidence. Advance at 05:35. At 125 words/minute, leave approximately 7 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Capture an actual generated response from the sourced demonstration. Display the exact output rather than replacing it with a target. Source cards establish supplied context; they do not independently verify arbitrary prose. Keep the screenshot caption recorded, and retain the complete demonstration receipt, including any mismatches. Do not fold this familiar real example into the controlled benchmark accuracy.

**Evidence:**

- [scripts/run_sourced_demo.py](../../scripts/run_sourced_demo.py)
- [editguard_chat/server.py](../../editguard_chat/server.py)
- [data/demos/air-india-handover.json](../../data/demos/air-india-handover.json)

### Slide 8: Website: history and reversible state

**Clock: 05:35–06:00. Slot: 25 seconds. Speech: 43 words.**

> The history records the approval and replacement. A conflicting publication is blocked, and rollback restores the earlier local snapshot with an auditable event pair. Stored answers retain the revision used when they were generated. A new question uses the currently active reference state.

**Visual action and pacing:** Point to the paired rollback events. Hand over at 06:00. At 125 words/minute, leave approximately 4 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Show actual history from the sourced demo. A later reapproval may leave the active demonstration on the handover snapshot while preserving its rollback event. Explain this if the current state and a stored answer have different revision numbers. Reopening a stored message is not fresh inference. Rollback restores local memory, not real-world ownership.

**Evidence:**

- [scripts/run_sourced_demo.py](../../scripts/run_sourced_demo.py)
- [editguard_chat/store.py](../../editguard_chat/store.py)
- [editguard_chat/server.py](../../editguard_chat/server.py)

## Member 4

### Slide 9: Experimental design

**Clock: 06:00–06:45. Slot: 45 seconds. Speech: 79 words.**

> For controlled measurement, we created fictional operations handbooks with explicit facts and expected answers. Each study has forty-eight fact worlds, split into sixteen development and thirty-two final lineages before question variants are generated. The original study reaches two hops; the separate depth challenge reaches three. Main conditions have 256 assigned questions each. We compare the actual generator and context configurations under matched inputs and scoring. Abstentions remain in the denominator, and uncertainty resamples whole lineages. Labels were automatically audited.

**Visual action and pacing:** Distinguish sourced demonstration from controlled experiment. Advance at 06:45. At 125 words/minute, leave approximately 7 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Each study authors 384 question records: 128 development and 256 final, eight variants per lineage. Original worlds have nine facts; depth has twelve. Original final workload: five main conditions * 256 + two oracle subsets * 16 = 1,312 generations. Depth: two main * 256 + two oracle * 16 = 544. Updated retrieved/full-memory configurations receive current facts. Oracle subsets are diagnostics. Automated audit is disclosed; no independent human semantic review. New Air India demonstration is sourced real material, not the data behind these scores.

**Evidence:**

- [reports/reasonke/operations-final/protocol.json](../../reports/reasonke/operations-final/protocol.json)
- [reports/reasonke/depth-final/protocol.json](../../reports/reasonke/depth-final/protocol.json)
- [docs/OPERATIONS_DATASET_CARD.md](../../docs/OPERATIONS_DATASET_CARD.md)
- [editguard_depth/data.py](../../editguard_depth/data.py)

### Slide 10: Measured answers on controlled fact worlds

**Clock: 06:45–07:30. Slot: 45 seconds. Speech: 76 words.**

> With retrieved current facts, ReasonKE_plus answered 256 of 256 main questions in each study. Qwen answered 241 in the original and 248 in depth; the remaining responses were UNKNOWN. These perfect scores are bounded by small fact worlds and controlled wording. Full memory tells another part of the story: original ReasonKE_plus fell to 231 of 256, with twenty-five wrong associations despite complete support. Our failure audit identifies the supplied distractor values instead of hiding those cases.

**Visual action and pacing:** Read counts rather than saying universal 100% accuracy. Advance at 07:30. At 125 words/minute, leave approximately 9 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Original R-RAG 256/256, G-RAG 241/256 with 15 abstentions; depth R-RAG 256/256, G-RAG 248/256 with 8 abstentions. Original R-ALL 231/256, G-ALL 241/256 (13 strict mismatches and 2 abstentions). R-ALL's 25 wrong final values all match another supplied value, including 15 Pro-owner values. All 63 updated-method nonsuccesses across the released conditions have complete support and no recorded budget-hit flags. Paired retrieved differences: original +5.859375 pp [3.90625,8.203125]; depth +3.125 pp [1.5625,5.078125]. These compare fixed systems, not a causal training ablation.

**Evidence:**

- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)

### Slide 11: The accuracy–latency tradeoff

**Clock: 07:30–08:00. Slot: 30 seconds. Speech: 44 words.**

> The higher coverage has a cost. Depth median latency was fifteen point nine four seconds for ReasonKE_plus and one point one two for Qwen. This includes selection, local communication and generation, including abstentions. The slower configuration answered eight more of the 256 requests correctly.

**Visual action and pacing:** Name the timing boundary and eight additional correct answers. Hand over at 08:00. At 125 words/minute, leave approximately 9 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Depth warm pipeline n=256 each: Reason p50/p95 15.93582438600015/19.282786326999826 seconds; Qwen 1.1189718869995886/1.3847337005004192. Excludes cold loading/downloads/browser network. Output medians 212 vs 6 tokens; this is consistent with differing output lengths but not a causal decomposition. Ratios of quantiles are not paired speedups. Do not promise training will improve this tradeoff.

**Evidence:**

- [reports/reasonke/depth-final/results.json](../../reports/reasonke/depth-final/results.json)
- [output/presentation/analysis/failure-audit.json](../../output/presentation/analysis/failure-audit.json)

## Member 5

### Slide 12: Our engineering contribution

**Clock: 08:00–08:50. Slot: 50 seconds. Speech: 80 words.**

> We have built the complete maintenance path: approved versioned memory, bounded connected retrieval, revision-aware chat, atomic rollback and an evidence explorer. We also created the controlled datasets, fixed evaluation protocols, recoverable execution ledgers and failure analysis. In the original retrieved ReasonKE_plus condition, all 124 baseline-correct locality answers were retained. Our current conclusion is that the platform works on its measured bounded task and exposes its accuracy–latency tradeoff. The next step is to turn this baseline into a matched model-adaptation study.

**Visual action and pacing:** Summarise the implemented mechanisms and current conclusion. Advance at 08:50. At 125 words/minute, leave approximately 12 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Concrete contribution lies in implemented platform and evaluation, with reused model weights accurately attributed. Original baseline-correct retention 124/124 is separate from all-locality factual accuracy 128/128. Current memory/retrieval guards do not establish production-scale accuracy. Five-person contribution allocation belongs in the script/guide and must reflect actual implementation/review/reproduction rather than invented ownership. The product is locally deployable with separate GPU dependencies and container/notebook delivery.

**Evidence:**

- [docs/CONTRIBUTIONS_AND_EVIDENCE.md](../../docs/CONTRIBUTIONS_AND_EVIDENCE.md)
- [reports/reasonke/operations-final/results.json](../../reports/reasonke/operations-final/results.json)
- [Dockerfile.chat](../../Dockerfile.chat)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)

### Slide 13: Next Milestone: Model Training and Broader Evaluation

**Clock: 08:50–10:00. Slot: 70 seconds. Speech: 113 words.**

> Our next milestone is a focused adaptation study. First, we will construct source-grounded edit examples with reviewed intermediate steps and create fresh lineage-separated development and test partitions. We will profile a compact pilot, train supervised adaptation, then add PPO using process and final-answer feedback. The comparison will hold the model family, retrieval, data access and inference budgets constant across the instruction baseline, SFT and SFT-plus-PPO stages. Broader evaluation will vary wording, entities, memory size and reasoning depth while measuring locality, abstention, latency and resources. We will preserve the existing results as regression evidence and use a fresh frozen test for the new research question: does training improve transfer and robustness under matched conditions?

**Visual action and pacing:** Point to the three matched stages and broader tests. Finish at exactly 10:00. At 125 words/minute, leave approximately 16 seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.

**Rehearsal detail:** Planned, not completed. Begin with a measured pilot before choosing the full training size, GPU budget or optimiser-step cap. Prepare a finite run question, profiled runtime/memory, input/artifact identities, checkpoint/resume verification and stopping rule before any substantial user-run command. SFT/PPO stages should share a verified base; keep the reused ReasonKE_plus system as a separate reference, not a causal parent claim. Use new unseen test material because existing released tests have been inspected. Reward design must be validated for formatting shortcuts and wrong-but-plausible intermediate steps. No numerical gain or runtime is promised.

**Evidence:**

- [docs/presentation/VERITY_MIDPROJECT_RESEARCH_ROADMAP.md](../../docs/presentation/VERITY_MIDPROJECT_RESEARCH_ROADMAP.md)
- [configs/reasonke-model.json](../../configs/reasonke-model.json)
- [configs/instruction-model.json](../../configs/instruction-model.json)

## Clock discipline

- At 01:55, 03:55, 05:55 and 07:55, the next speaker gets ready. Hand over on the two-minute boundary.
- Advance on the listed clock marks. The final milestone slide ends at 10:00.
- Use the captured website views during the timed presentation. Do not launch an unbounded live generation or perform a fresh model download in the slot.
- If the panel interrupts, ask whether to pause the presentation clock. Do not silently add Q&A time to a strict ten-minute slot.

## Short technical Q&A

**Is the current implementation RAG?** Yes. It combines approved external memory, bounded lexical retrieval and fixed generators. Our implemented contribution is governance, context/history handling, the application, independent controlled evaluation and evidence engineering.

**Which model is reused?** `YukinoKaren/ReasonKE_plus`, pinned at `ceaedc2ae7c8fed6257953587d991ce550db6106`. The comparison is `Qwen/Qwen2.5-7B-Instruct` at its pinned revision. The detailed upstream training tutorial is outside this review.

**Where is our new model training?** SFT/PPO adaptation is the next milestone. The slide describes a matched study, beginning with reviewed data and profiling, then a fresh frozen broader evaluation.

**Is the Air India example current ownership?** It is explicitly a dated historical snapshot around the 27 January 2022 handover. Later arrangements changed. The corporate-contact fact comes from the official page checked on 31 October 2026.

**Does 256/256 guarantee arbitrary questions?** It is the observed final-span score on a controlled packet with 32 final lineages and small worlds. The broader next study addresses generalisation and larger memory.

**Why does full memory fail?** All 25 original ReasonKE_plus wrong finals match other supplied values despite complete required support. That describes visible wrong associations; it does not prove an internal causal mechanism.

**Does rollback change neural weights?** It atomically restores approved memory and events. Fresh requests use that state. Saved messages retain their original revision and output.

**Will training beat the baseline?** That is the research question. We will measure accuracy, coverage, locality, latency and resources under matched conditions rather than promise a gain.

## Evidence-based contribution allocation

Attach names only to implementation, reproduction, review, analysis or documentation actually performed. These work areas can overlap and are independent of speaking roles.

Proposed lead allocation for implementation/review/reproduction. Confirm actual work before treating this as historical credit; members can collaborate across rows.

| Lead | Work area | Concrete contribution | Reviewable evidence |
| --- | --- | --- | --- |
| Member 1 | Data and sources | Controlled worlds, questions, splits, label audits and the sourced Air India fixture | data modules, dataset cards, data/demos/air-india-handover.json |
| Member 2 | Retrieval and model integration | Pinned GPU worker, bounded context selection, iterative expansion and prompt/history interfaces | runtime, reasonke_backend.py, retrieval modules |
| Member 3 | Backend reliability | Canonical identity, explicit approval, supersession/conflicts, events and atomic rollback | store.py, workspace.py, server.py, regression tests |
| Member 4 | Evaluation and analysis | Fixed matched protocols, ledgers/resume, scoring, uncertainty, latency and failure diagnosis | benchmark/depth packages, raw receipts and audit scripts |
| Member 5 | Product and reproducibility | Verity chat/editor/explorer, real-source walkthrough, containers/notebooks, CI and presentation material | static/, results.py, sourced-demo scripts and docs |
