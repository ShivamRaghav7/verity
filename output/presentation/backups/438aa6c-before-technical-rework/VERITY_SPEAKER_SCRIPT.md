# Verity technical presentation script

Twelve slides for five speakers. Plan for approximately eleven minutes of speech, with an optional two minute demonstration after slide 12. Read the quoted paragraphs aloud. The timing and visual cues are rehearsal instructions. Replace Member 1 through Member 5 with names. Speaking assignments and contribution ownership are separate.

Verity is the current website and presentation name. Older source modules, evidence receipts and historical screenshots retain the earlier EditGuard name so their recorded identities remain traceable.

## Speaking order

| Speaker | Slides | Focus | Approximate time |
| --- | --- | --- | --- |
| Member 1 | 1 and 2 | Problem and precise edit contract | 1 minute 30 seconds |
| Member 2 | 3 through 5 | Inference architecture, model configuration and retrieval | 2 minutes 50 seconds |
| Member 3 | 6 and 7 | Transaction guarantees and controlled datasets | 1 minute 50 seconds |
| Member 4 | 8 through 10 | Experimental protocol and measured results | 3 minutes |
| Member 5 | 11 and 12 | Failure analysis, application evidence and conclusion | 1 minute 50 seconds |

## Member 1

### Slide 1 Verity

**Time:** about 30 seconds. **Visual cue:** introduce the project and its bounded goal.

> Good morning. Verity is a knowledge editing and verification platform for conversational AI. We study how an approved factual correction changes the assistant's answers across connected records, while preserving unrelated information. Our final implementation combines a frozen learned reasoner with versioned external memory. The project includes a transactional editing workflow, a working website and reproducible model comparisons. We evaluate factual answers separately from database correctness and retain the failures in our evidence.

### Slide 2 The edit contract

**Time:** about one minute. **Visual cue:** follow the service, team, person and office relations. Identify the unchanged hosting fact.

> An edit has a subject, relation and scope. In our fictional example, the subject is Kaveri Payment Gateway, the relation is owner, and the scope is release 2. Initially, its owner is Bengaluru Platform Team. That team's contact is Meera Iyer, whose office is in Chennai. We replace only the ownership fact with Pune Reliability Team. Its contact is Arjun Deshmukh, whose office is in Hyderabad. The office answer should therefore change through three relations. Hosting city, the Pro service and release 1 should remain unchanged. We call those unaffected questions locality controls. Rollback should restore the original reasoning path without deleting the history.

**Handover:** “Member 2 will explain how the model receives and uses this approved state.”

## Member 2

### Slide 3 End to end inference

**Time:** about 50 seconds. **Visual cue:** trace one question through the website API, memory selector and worker.

> The browser sends a question to the Python application API. The API reads the active approved facts from SQLite at a specific knowledge revision. A revision identifies a memory state, while the model checkpoint identifies fixed neural weights. The context selector chooses supporting facts and the controller adds permitted conversation history. Only complete conversation pairs from the applicable revision enter a new prompt. A separate private GPU worker generates the answer. The application stores its raw output, extracted answer, supplied fact identifiers, sources and model revision. This provenance makes the request inspectable. Retrieving a source does not certify every sentence the model generates.

### Slide 4 Model configuration and hardware

**Time:** about 55 seconds. **Visual cue:** distinguish weight storage, compute precision and generation limits.

> We integrate the author published ReasonKE_plus checkpoint and compare it with Qwen2.5 7B Instruct. Both contain approximately 7.615 billion parameters. Their authors trained them upstream. We pin exact revisions and keep their weights frozen. NF4 double quantization compresses weight storage, while FP16 supplies the computation precision. SDPA is PyTorch's attention implementation. On the T4, the measured peak allocated CUDA memory during loading was about 6.59 gibibytes. We load one model at a time. The prompt allows at most 4,096 tokens, and generation allows 512 new tokens, using greedy decoding with one beam. These limits and the measured profile define our actual configuration.

### Slide 5 Context selection for longer paths

**Time:** about one minute. **Visual cue:** compare the original two stage retrieval with iterative answer text expansion.

> The original selector ranks facts against the question, takes two seeds, adds their answer text to the query, and reranks to fill four context slots. On the new development packet, it supplied complete support for 96 of 128 questions, but zero of 32 three hop questions. We therefore evaluated a finite family of iterative selectors before freezing the final test. The selected policy keeps two question ranked seeds and expands through answer text for up to three rounds, with a four fact cap. It supplied complete support for all 128 development questions, including all 32 three hop paths. Those are retrieval diagnostics, not neural answer accuracy. Ordinary retrieval never receives expected answers or gold support identifiers. It also supplies no calibrated applicability probability.

**Handover:** “Member 3 will explain how approved facts stay consistent and how we constructed the test data.”

## Member 3

### Slide 6 Transactional knowledge lifecycle

**Time:** about 55 seconds. **Visual cue:** show the active Bengaluru ownership record beside the unapproved Pune replacement draft, then explain the atomic rollback pseudocode.

> Each fact uses the canonical key entity identifier, relation identifier and scope identifier. A partial unique database index permits at most one active version for that key. To replace a fact, the request must name the current predecessor through supersedes. An unresolved competing approval receives HTTP 409 and leaves memory unchanged. Rollback uses a BEGIN IMMEDIATE transaction: withdraw the replacement, restore its direct predecessor, and write two audit events at one new revision. The whole change commits or rolls back together. Historical reads replay events without changing live memory. These rules protect represented identities. Two contradictory statements under unrelated keys can still require administrator judgement.

### Slide 7 Data and leakage controls

**Time:** about 55 seconds. **Visual cue:** distinguish a graph lineage, a question variant and a generation request.

> We authored two separate fictional datasets. Each contains 48 graph lineages and 384 question records. A lineage groups one fact world and its related correction variants. Sixteen lineages support development and 32 support final evaluation. We split lineages before question augmentation, so related variants stay together. Eight variants per final lineage produce 256 main questions per condition. The original study uses nine facts per world. The depth study uses twelve and adds three hop questions with separate development and final wording banks. A deterministic graph audit checks labels and supporting paths. There is no independent human semantic review, and templates still repeat within each split.

**Handover:** “Member 4 will connect that data design to the fixed evaluation protocol and results.”

## Member 4

### Slide 8 Experimental protocol and scoring

**Time:** about one minute. **Visual cue:** explain the request totals and the separate main and oracle populations.

> The original study evaluates five main conditions on 256 questions, plus sixteen preselected oracle questions for each model. That gives 1,312 generations. The separate depth study evaluates two models on 256 main questions plus sixteen oracle questions each, giving 544. Oracle means we supply known supporting facts as a diagnostic. Updated baselines receive the same approved information, with compatible generation settings and budgets. The stale initial condition is only a diagnostic reference. We score the endorsed short answer for the requested subject and relation. Merely mentioning a target value is insufficient. UNKNOWN abstentions receive zero primary credit and remain in the denominator. Configuration and scoring freeze before final inference.

### Slide 9 Original study results

**Time:** about 55 seconds. **Visual cue:** compare retrieved and full memory conditions, then distinguish retention from factual locality accuracy.

> With updated retrieved facts, ReasonKE scored 256 out of 256 and Qwen scored 241, with fifteen UNKNOWN abstentions. With full updated memory, ReasonKE scored 231 while Qwen scored 241. ReasonKE therefore did not win every comparison. All 25 ReasonKE full memory failures had the required support available and endorsed an answer from another supplied fact. That identifies an output association error without proving its internal cause. We also separate two locality measures. ReasonKE retrieval retained all 124 baseline correct controls, and it answered all 128 locality questions correctly. Retention measures preservation of previously correct answers. Factual accuracy also includes controls the initial assistant answered incorrectly.

### Slide 10 Depth results and uncertainty

**Time:** about one minute. **Visual cue:** read the per depth denominators, uncertainty interval and latency values.

> In the depth main test, ReasonKE scored 256 out of 256. Qwen scored 248 with eight abstentions. Each locality and depth group contains 64 questions. Qwen scored 64, 64, 57 and 63 across locality, one, two and three hops. ReasonKE scored 64 in every group. The overall difference is 3.125 percentage points. We estimate uncertainty by resampling whole lineages, keeping correlated variants together. The 95 percent interval is about 1.56 to 5.08 points. The three hop interval includes zero, so that comparison remains uncertain. Warm median and 95th percentile latency were 15.94 and 19.28 seconds for ReasonKE, versus 1.12 and 1.38 for Qwen. Higher observed answer coverage has a substantial speed cost.

**Handover:** “Member 5 will explain the perfect score's limits and show what the completed product establishes.”

## Member 5

### Slide 11 Failure analysis and the perfect score

**Time:** about one minute. **Visual cue:** connect the small controlled task to the retained failures rather than promising arbitrary accuracy.

> A perfect observed score is plausible here because each world has few explicit facts, retrieval supplied every required supporting path, answers are short, and task templates repeat. It does not establish universal natural language accuracy. The depth Qwen failures also show why depth alone is insufficient: seven of nine instances of one direct two hop wording template produced UNKNOWN, while the other direct templates scored 23 out of 23. All eight main failures had complete support. The preselected oracle set contains none of those failure identifiers, so it does not demonstrate their recovery. Separately, the explanation audit found unsupported intermediate claims. A correct final answer does not verify the whole reasoning trace.

### Slide 12 Application evidence and conclusion

**Time:** about one minute, excluding the optional demonstration. **Visual cue:** identify the recorded Chennai answer after rollback, then distinguish the application acceptance counts from held-out results.

> The completed website supports chat, explicit approval, conflict resolution, historical comparison and rollback. The screenshot displays a saved three hop answer after rollback, rather than a fresh generation. The finite India application workload produced fourteen correct neural answers and passed twelve provenance and transaction consistency checks. It represents one fictional display lineage and remains separate from held out accuracy. Our contribution is the governed application, versioned memory, retrieval adaptation, independent datasets and reproducible evaluation. The upstream authors supply the learned checkpoint and training. The evidence supports this system's defined short factual contract under the tested conditions. Larger memories, ambiguous scopes, missing evidence and independently reviewed operational language need further validation. We can now demonstrate a correction and its reversal. Thank you.

## Optional live demonstration

Allow approximately two additional minutes after slide 12. Start with an idle, ready model worker and a fresh demonstration workspace. See [the detailed demonstration guide](../INDIA_DEMO.md). A fresh approved handbook reaches revision 9. The replacement reaches revision 10 and atomic rollback reaches revision 11.

1. In Chat ask: **In which Indian city is the office of the on-call contact for the team that owns Kaveri Payment Gateway in release 2?** The initial expected answer is Chennai.
2. In Knowledge choose **Draft the example ownership update**, inspect the Pune Reliability Team replacement, then explicitly approve it. Drafting changes nothing.
3. Request a fresh office answer. The updated target is Hyderabad. Ask the hosting city question separately. Its target remains Bengaluru.
4. Show sources and history. If time permits, compare approved revision 9 with the current revision. This generates two new answers, so it adds latency.
5. Roll back the replacement and request another fresh office answer. The restored target is Chennai. Existing stored messages do not rewrite themselves.

**During generation:** “Approval changes the selected memory state. The fixed model now generates a fresh answer from that state.”

**If the GPU worker is unavailable:** “The live inference worker is unavailable. We will use the recorded application evidence and identify its saved responses.” Do not present a stored answer as a live generation. If a fresh answer fails, inspect its supplied facts and raw output rather than substituting the deterministic target.

## Technical questions to rehearse

**Is the editor changing neural weights?** No. The final platform changes approved external memory and conditions a frozen, upstream trained generator on retrieved facts. Parametric methods such as ROME modify weights and represent a different editing route.

**Is this a faithful SERAC implementation?** No. The final deployment has no newly trained standalone scope classifier or SERAC counterfactual model. Earlier auxiliary model experiments remain archived separately.

**Why does the project need a learned model?** The model interprets the factual question and generates an answer across supplied relations. SQLite determines approved state. A simple fixed lookup can solve a single known key, and a symbolic graph engine could be another useful baseline. We evaluate the actual neural outputs rather than counting database answers as model successes.

**What improved in our retrieval implementation?** The original four fact policy failed all 32 development three hop paths. The selected iterative policy completed all 32 before final freeze. This is a development support recall improvement. It does not by itself prove a neural accuracy improvement or guarantee arbitrary longer paths.

**Why use two seeds?** A packet only selection initially favoured a smaller policy. A declared display development question exposed a distractor. We compared six bounded candidates before profiling and final freeze, then selected two seeds and four total facts. That choice did not use final model outputs.

**What does a 95 percent interval mean here?** We repeatedly resample the 32 complete graph lineages and recompute the paired difference. It expresses variation under this finite controlled sampling design. It is not a 95 percent probability that every future question will succeed, and it does not repair unrepresentative data.

**Does perfect retrieval mean perfect answers?** No. It means all labelled required supporting facts reached the prompt. Both final models received complete support, but Qwen still abstained. Full memory also contained the support and ReasonKE made errors.

**Why are three hop Qwen results better than two hop?** The groups use different wording constructions. Seven direct two hop failures concentrate in one template. The observations do not establish a monotonic relationship between depth and accuracy.

**Is the Qwen comparison a causal measurement of ReasonKE training?** No. Shared information and compatible settings make a useful system comparison. The exact checkpoint ancestry is not established, so we cannot isolate the causal effect of one training method.

**Are explanations verified?** Only a bounded set of extracted indexed subanswers received a separate support audit. Unassessed prose remains, and no complete trace was certified fully verified. The interface displays evidence without labelling arbitrary prose verified.

**What do transactions guarantee?** A represented replacement or rollback changes state and audit events atomically. They do not guarantee semantic truth or detect every contradiction expressed under another key.

**Can we run the measured configuration on the laptop?** The fixed loader's measured T4 loading allocation exceeds the inspected laptop VRAM. No measured local run supports a promised two hour estimate. That does not prove every alternative model or loader is impossible.

## Contribution allocation for the five members

These are five substantive ownership areas covering completed project work. Confirm each person's actual implementation, review, reproduction, analysis or documentation before attaching a name. Presentation roles remain independent of these areas.

| Member | Contribution area | Work to describe once confirmed | Reviewable evidence |
| --- | --- | --- | --- |
| Member 1 | Dataset and experiment design | Fictional graph construction, original and updated truth, direct and paraphrased questions, scope/locality controls, lineage splits, graph label audits and dataset limitations. Shared review with Members 2 and 3. | `editguard_benchmark/data.py`, `editguard_depth/data.py`, dataset cards and dataset tests. |
| Member 2 | Model integration and context selection | Pinned checkpoint loading, T4 profiling, bounded generation, private worker, original and iterative selectors and revision consistent model context. Attribute upstream model training accurately. Shared review with Members 1, 3 and 4. | `editguard_reasonke/runtime.py`, retrieval modules, model card and raw profiles. |
| Member 3 | Evaluation and reproducibility | Fixed protocols, finite execution and resume, endorsed answer scoring, complete denominators, oracle/retrieval separation, paired lineage intervals, latency/resource measurement, receipt audits, scientific graphs, reports, CI and deployment checks. | `editguard_benchmark/`, `editguard_depth/metrics.py`, audit scripts, released receipts and `.github/workflows/ci.yml`. |
| Member 4 | Knowledge lifecycle and backend reliability | Canonical identity, explicit approval, competing version rejection, supersession, immutable events, historical reads, atomic rollback, conversation transactions and validated workspace restore. Shared integration with Members 2 and 5. | `editguard_chat/store.py`, `editguard_chat/workspace.py`, API documentation and store/recovery tests. |
| Member 5 | Application and demonstration | Chat interface, correction editor, knowledge search, source/history display, conflict choices, before/after comparison, results explorer, scientific downloads and the India demonstration workflow. Shared verification with Members 1, 3 and 4. | `editguard_chat/static/`, `editguard_chat/server.py`, demonstration guide and application acceptance receipts. |
