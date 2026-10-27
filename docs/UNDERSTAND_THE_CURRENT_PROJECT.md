# Understanding EditGuard from the beginning

EditGuard is a knowledge-maintenance application for an AI assistant. You can
approve a fact change, ask questions using it, inspect the supplied evidence,
compare an earlier state, and undo the change. We also built a controlled
experiment to measure when those answers are correct. This guide explains the
actual system and development history, without inventing a past plan or claiming
that every question succeeds.

The current website's default display facts are the clearly fictional
[India-themed handbook](INDIA_DEMO.md). The Dahlia examples below explain the
completed original benchmark and remain available through its legacy API.
Changing display names does not change the measured results. A new depth study
adds three-hop paths and held-out wording; it is reported separately.

## The problem, using one concrete example

Imagine an internal operations handbook with these facts:

```text
Dahlia Relay, release 2 → owned by Saffron unit 14 → contact Mara Voss
Copper unit 74 → contact Orin Dane
```

Now ownership moves to Copper unit 74. Asking “Who owns Dahlia Relay?” should
return Copper. Asking “Who should I contact for its owning team?” should return
Orin, even though we changed only the ownership fact. The assistant must connect
two relationships. Saffron's own contact remains Mara; release 1, the service's
hosting region and the similarly named Dahlia Relay Pro must keep their separate
answers.

The authoritative handbook is still necessary. An assistant can help users ask
natural-language questions across its relationships instead of knowing where
each entry lives. That creates a maintenance problem: the assistant may select
an obsolete fact, connect the wrong team, or apply an update too widely. Our
project makes those changes controllable and measures these failure modes. The
entire example handbook is fictional; we are not operating a real on-call service.

## What “editing knowledge” means here

A model's **weights** are numerical parameters learned during training. A **fact**
is a piece of information such as “Copper unit 74 owns Dahlia Relay in release 2.”
There are different ways to change a model's behavior. Some methods update
weights. Our current method stores approved facts outside the model and supplies
relevant ones when asking a question. This is external-memory, in-context
knowledge editing. Adding or withdrawing an edit does not retrain the model.

The current generator is the author's released **ReasonKE_plus** checkpoint. It
was trained upstream to use updated information and produce connected answers.
We adapted its deployment to a T4, integrated governed memory and built the
application and evaluation. We did not invent ReasonKE or train its released
weights. The official Qwen2.5-7B-Instruct checkpoint is our comparison generator.
Their exact training ancestry is unverified, so a difference in scores cannot
prove which upstream training step caused it.

Earlier, we did train our own **DistilBERT applicability classifier** and
**FLAN-T5 short-answer model** for a bounded support editor. Those trained models,
losses and results remain documented in [their model card](MODEL_CARD.md). They
are a separate experiment; the current ReasonKE website does not secretly use
that classifier. Training loss measures the training objective, not percentage
accuracy: a loss of 0.48 does not mean 48% wrong. Different objectives' loss values
cannot be compared directly. Current ReasonKE experiments measure inference,
not another training run.

## What we actually did, in order

The repository records several iterations, rather than one uninterrupted success.

| Actual phase | What, why and how |
| --- | --- |
| 11 October 2026 | Built the first correction-and-rollback workflow with explicit facts and transactions. This established who approves information and how to reverse a change before relying on neural answers. |
| 13–20 October | Trained and evaluated the support editor's two auxiliary models. Updated RAG performed better on that study's stated request-bound metric: 649/720 versus 613/720. These are earlier support results, not current chat accuracy. |
| 21–24 October | Explored visual editing and ran a comparative visual study. Some methods matched edited targets while damaging unrelated answers, showing why edit success alone was insufficient. The visual evidence remains separately archived. |
| 25 October | Built ordinary conversational interaction, persistent history, sourced approvals and before/after comparison; corrected stale-history behavior. Integrated the released ReasonKE checkpoint with pinned revisions and recorded real T4 website generations. |
| 26 October | Authored the connected operations dataset, profiled both checkpoints, completed the matched development comparison, selected retrieved context, and froze a larger held-out evaluation. Expanded canonical identities, historical comparison, atomic rollback, recovery, raw-answer inspection and scientific exports. |
| 27 October | Restored the user-requested pause checkpoint, verified that all 483 saved outputs stayed unchanged, completed the remaining 301 ReasonKE and 528 Qwen requests, independently replay-audited the full study on the laptop, and generated final scientific reports. No completed failure was retried for a better answer. |

The [evidence-backed history](PROJECT_JOURNEY_AND_FINISH_PLAN.md) links earlier
commits and studies. It is a dated planning/history document; current behavior is
described in [the architecture](CURRENT_ARCHITECTURE.md) and [demo guide](OPERATIONS_DEMO.md).

## What happens inside the current application

**Approval and identity.** An operator enters a question/fact, answer, scope and
source. Canonical IDs identify its entity, property and release. Two phrasings
with the same IDs share one history. A conflicting active value requires explicit
replacement. Customer messages never approve information automatically. SQLite
stores fact versions, approval events, conversations and a knowledge revision.

**Retrieval.** A lexical retriever selects question-relevant facts, expands the
search using the first selected answers, then supplies up to four facts. This
helps fetch the contact associated with a selected owner. It never reads hidden
test answers. The software accepts up to 200 active facts, but our model study
uses nine-fact worlds; 200 is a software bound, not a measured accuracy claim.

**Generation.** A separate authenticated GPU worker receives the question,
selected facts and bounded conversation history. The trained neural model
generates text. The application extracts the final answer and retains raw output,
model revision, fact IDs and timing. A worker failure produces an explicit error,
not a database/template answer credited to the model.

**Consistency.** Old chat messages remain visible after an edit. Stale generated
answers are excluded from new model context. Comparisons give both sides matched
history, preventing the newer answer from leaking into the earlier side.

**Governance.** Replacement, withdrawal and restoration preserve history. Atomic
rollback withdraws a replacement and restores its predecessor in one transaction
and one revision. If a late database write fails, neither half is committed.
These deterministic guarantees concern state changes; they do not guarantee a
correct generated answer.

## The dataset and our measured results

We independently authored **48 fictional fact worlds**, each with nine facts,
one ownership change and eight questions. This gives **384 question records**.
We split whole edit lineages into 16 development and 32 final worlds before
making wording variants. An edit **lineage** is one underlying change and all
related questions; paraphrases from it cannot be scattered across splits. Labels
were audited automatically, without independent human semantic review. Shared
templates limit broad language-generalization claims.

The completed development run uses three predetermined development worlds:
24 identical questions in each of five configurations, giving **120 actual model
requests**. It does not execute the entire development partition.

| Configuration | What the model receives | Correct / assigned |
| --- | --- | ---: |
| R0 | ReasonKE with initial facts | 12/24 |
| R-RAG | ReasonKE with retrieved updated facts | 24/24 |
| R-ALL | ReasonKE with all current facts | 21/24 |
| G-RAG | Official Qwen with the same retrieved facts | 24/24 |
| G-ALL | Official Qwen with all current facts | 22/24 |

R0 is a **staleness diagnostic** scored against updated truth. Its 12 changed-fact
answers matched old truth; its 12 unaffected answers were correct. Updated RAG
gets properly updated information, not deliberately stale documents.

We selected retrieved context for the existing ReasonKE deployment because its
development score was 24/24 versus 21/24 with full memory. Qwen retrieval also
scored 24/24 and was faster. We therefore claim neither a ReasonKE advantage nor
universal accuracy. ReasonKE full memory confused old/new contacts twice and
applied the new owner to an older release once. Qwen full memory chose the
similarly named service once and abstained once. All outcomes remain in
[the raw development evidence](../reports/reasonke/operations-development-v1/README.md).

### The completed held-out study

The **1,312-request final run completed** under the frozen settings. It asked the
same 256 held-out questions in five main configurations (1,280 calls), plus the
same sixteen preselected oracle questions to each model (32 calls). Therefore
1,312 is the number of model calls, not 1,312 independent questions. Each main
condition contains 128 applicable and 128 locality questions from 32 held-out
lineages. Test answers were not used to change settings or scoring.

| Configuration | Correct / assigned | Scored wrong | Abstained |
| --- | ---: | ---: | ---: |
| R0: ReasonKE, initial facts against updated targets | 136/256 | 120 | 0 |
| R-RAG: ReasonKE, updated retrieved facts | 256/256 | 0 | 0 |
| R-ALL: ReasonKE, full updated memory | 231/256 | 25 | 0 |
| G-RAG: Qwen, same updated retrieved facts | 241/256 | 0 | 15 |
| G-ALL: Qwen, full updated memory | 241/256 | 13 | 2 |

Both oracle diagnostics scored 16/16. All 1,312 requests generated outputs;
execution errors and unexecuted cases were zero. Wrong answers and abstentions
remain in their denominators. Here 256/256 means every assigned answer matched
our fixed short-answer rules on this dataset. It does not mean every future
natural-language question will succeed.

The retrieval comparison's 15-answer difference came entirely from Qwen saying
`UNKNOWN` on fifteen two-hop paraphrases. Qwen's other 241 retrieved-context
answers were correct. Necessary supporting facts were supplied in all fifteen
cases. This distinguishes abstaining from confidently answering incorrectly.
Some full-memory scored failures also contain a correct person plus extra team
text; the fixed only-name/whole-answer rule still rejects them. We disclose that
scoring limitation without changing aliases after testing.

R0's final locality accuracy was 124/128. ReasonKE retrieval retained all those
124 correct answers and also answered the four previously wrong locality cases
correctly, giving 128/128 locality accuracy. ReasonKE full memory retained
117/124 baseline-correct locality answers and scored 119/128 on all locality
questions. Preserving an error would not count as factual success.

Read the [final study](../reports/reasonke/operations-final/README.md) and
[inspectable failure analysis](../reports/reasonke/operations-final/FAILURE_ANALYSIS.md).

We also completed a separate **16-generation real website workload**: twelve
direct/paraphrase/connected/restoration answers and four historical-comparison/
atomic-rollback answers. All sixteen matched their targets, with all eleven
recorded history/rollback consistency checks passing and no cleanup errors.
The laptop independently replayed their raw-output extraction, model identity
and scores. These use development worlds, so they demonstrate the integrated
workflow rather than adding sixteen independent held-out facts. The twelve
website requests had p50/p95 15.42/18.78 seconds, including storage and worker
HTTP. The primary benchmark's warm pipeline times have a different scope.

## What the numbers mean

| Term | Meaning and denominator |
| --- | --- |
| Applicable-edit accuracy | Correct answers to questions the change should affect, divided by all assigned applicable questions. This includes connected contact answers, not only the directly edited owner. |
| Locality accuracy | Correct answers to all assigned questions outside the edit's scope: another property, similarly named service, older release or unrelated fact. |
| Retention / preservation | Among locality questions R0 already answered correctly, how many remain correct after editing? Preserving an already wrong answer is not factual correctness. A zero eligible denominator is undefined. |
| Retrieval recall | Required supporting facts retrieved divided by all required facts. Complete two-hop support asks whether both needed facts were retrieved. Correct retrieval can still lead to wrong generation. |
| Oracle-support accuracy | A separate diagnostic giving the model the gold supporting facts. It isolates a generation question and is not the normal retrieval pipeline's score. |
| Coverage | Nonabstaining answers divided by all assigned questions. Answered-only precision uses only answered cases; primary accuracy still includes every assigned case. |
| Abstention | An empty answer or recognized response such as UNKNOWN. It receives zero primary accuracy credit. Execution errors and unexecuted cases also remain in the denominator. |
| False edited-value endorsement | An out-of-scope question answered with the inapplicable edited value. This detects one specific error type, not every possible wrong application. |

Answer matching checks the **whole endorsed final answer** against fixed aliases.
Mentioning the target in a rejected possibility or intermediate explanation is
insufficient. Strict matching can reject a valid longer form; raw outputs remain
available without changing test aliases afterward.

**Paired bootstrap uncertainty** compares methods on the same questions, then
resamples whole lineages to estimate an interval for their difference. Questions
sharing a fact are correlated. The final study has 32 independent lineages,
compared with three in the bounded development run. ReasonKE retrieval minus
Qwen retrieval was +5.86 percentage points, with a 95% interval of +3.91 to
+8.20 points. This estimates a matched configuration difference in this bounded
population, not confidence that an individual future answer is correct.

**p50 latency** is the median request time; **p95** is the time below which roughly
95% of measured requests fall. Final ReasonKE R-RAG had 15.06/16.71 seconds,
versus Qwen G-RAG 0.92/1.20 seconds, with 256 requests each. These warm pipeline
measurements include retrieval, local worker communication and generation, not
cold loading or the browser round trip. The separate HTTP workload measures the
website path. Different latency scopes must not be mixed.

**GPU memory** is reported in bytes or GiB; 1 GiB is 1,073,741,824 bytes. Allocated
memory holds active tensors; reserved memory also includes PyTorch's memory pool.
Neither is automatically total-system VRAM usage. NF4 stores weights compactly
while FP16 is the computation format. Download size, GPU memory and parameter
count measure different things. Loading time is separate from warm inference.

## Navigate the website and show a meaningful change

Open the pinned Colab launcher from [README](../README.md), choose T4, run its
setup and worker cells, and wait for **ready**. Keep the runtime connected. The
current checkpoint accepts text; ordinary conversation is supported, while the
study measures short factual questions rather than unrestricted dialogue.

1. In **Knowledge**, use a fresh workspace and explicitly approve the fictional
   operations handbook. Nine facts create revision 9. Existing knowledge history
   is never overwritten by this helper.
2. In **Chat**, ask who owns Dahlia Relay in release 2: the intended initial answer
   is Saffron unit 14. Ask its owner's contact: Mara Voss. Ask its region or
   release-1 owner to establish unaffected controls.
3. In **Knowledge**, choose **Draft the example ownership update**, review Copper
   unit 74 and its source, then explicitly approve replacement. Revision becomes
   10. Drafting alone changes nothing authoritative.
4. Ask again and try different wording. Intended updated answers are Copper unit
   74 and Orin Dane. Region remains West sector; release 1 remains Silver unit
   194; Dahlia Relay Pro remains Cedar unit 134. Inspect actual output rather than
   silently replacing a failed answer with the target.
5. On a user question, choose **Compare**, select **Approved memory at a revision**,
   enter 9 and run it. The same checkpoint generates against old and current
   approved memory. Without-context comparison is also available; it is not a
   different foundation-model checkpoint.
6. In Knowledge, press **Roll back to previous version** on the active replacement.
   This restores Saffron atomically at revision 11. Ask the owner/contact again.
   Expand **Approval and rollback history**; old messages remain historical.

Search the fact library by text or canonical identity. **Show inactive** exposes
old versions. Publishing a competing canonical key without explicit replacement
is blocked. These rules detect identified conflicts, not every semantic overlap.

Before Colab ends, **Export workspace JSON**. It saves text chats, fact versions
and events. Restore it into an empty workspace after checking the preview. The
server validates integrity, references and event/state agreement transactionally.
A checksum detects corruption, not truth or authorship. Keep chat exports private.

## Inspect evidence instead of trusting a headline score

**Research results** labels development/final phases and shows all outcome counts,
category scores, retrieval, preservation, timing, graphs and downloadable reports.
The earlier visual study is explicitly separate.

Under **Inspect the saved answers**, filter by condition, outcome or category,
then choose **Show saved answers**. Expand a record to read the question, target,
generated answer, supplied facts, raw output, model revision and prompt digest.
Filtering changes what you see, not the reported denominator. These are saved
experiment outputs, not fresh live chat responses.

The **explicit subanswer-support audit** checks recognizable indexed, boxed
subanswers against supplied/current facts. It labels claims supported,
contradicted, unsupported or unassessable, retaining surrounding unassessed prose.
The final audit inspected 32 preselected traces, sixteen per context policy.
Retrieved context had 32/32 extracted claims supported. Full memory had 26
supported, three contradicted and three unsupported extracted claims. All 128
surrounding/preamble/outside-action segments remain unassessed. This never
certifies the whole explanation or faithful internal reasoning, and changes no
primary answer scores. Read [the complete final audit](../reports/reasonke/operations-final/explanation-audit.json).

## What we contributed, and what remains limited

Our contribution is the governed editing application, connected-fact dataset,
bounded deployment, fair comparison runner, reproducible scoring/auditing,
scientific analysis and recovery/testing machinery. Earlier auxiliary-model
training is an additional separate ML contribution. The
[contribution evidence map](CONTRIBUTIONS_AND_EVIDENCE.md) links each substantial
work package to code, tests and receipts.

Passing software tests demonstrates specific state and API contracts; it is not
a neural accuracy score. Model failures are measured observations to explain and
retain. The current system has automated labels, shared templates, bounded
contexts, strict short-answer scoring, no calibrated answer-confidence probability
and a single-operator deployment. Retrieved sources show what information was
supplied, not that all generated prose is verified. The defensible outcome is a
working reversible editing product and its actual measured behavior: the selected
retrieval configuration met the direct/connected-answer and locality objective
on the frozen controlled test, with higher answer coverage and higher latency
than the instruction-model retrieval baseline. The full-memory ReasonKE
configuration scored below Qwen full memory, so the specialized checkpoint was
not best under every policy. This is a concrete bounded conclusion, not a claim
of universal accuracy. Live editing and rollback are supported separately by
the operational website receipts rather than folded into the test score.
