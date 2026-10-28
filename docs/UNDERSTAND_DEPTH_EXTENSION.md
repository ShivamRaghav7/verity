# Understanding the three-hop extension

This guide explains why we extended EditGuard, what the new numbers mean and
how to demonstrate the work. It describes the actual work rather than an
invented development story. For the full earlier project, start with
[Understanding the current project](UNDERSTAND_THE_CURRENT_PROJECT.md).

**Status at this checkpoint, 28 October 2026:** the 16 GPU profiling requests and
48 replacement development requests are complete and backed up. The final
protocol was frozen and published before final inference began. All **544 final
requests have completed**, and their saved outputs have been replay-audited on
the laptop. ReasonKE with iterative retrieval scored **256/256** on the main
questions; Qwen with the same retrieval scored **248/256**, with eight
abstentions. The completed earlier study remains a separate result. The new
14-generation India website acceptance check also completed: 14/14 actual model
answers matched, all twelve consistency checks passed, and rollback restored the
nine initial facts. Its raw outputs were replay-audited separately on the laptop;
this one display lineage adds no held-out benchmark observations.

An earlier temporary Colab runtime expired before its development outputs were
exported. Those unknown outcomes are disclosed rather than counted as measured
results. Its saved profiles were restored unchanged for the replacement run.
A later accidental browser disconnection did **not** stop the final evaluator:
the same GPU process and saved ledger were still active when checked. We
reconnected to it and backed up the evidence without restarting inference.

## 1. Why extend a system that scored 100%?

The earlier ReasonKE retrieval configuration answered 256/256 questions
correctly on its controlled operations benchmark. We preserve that result.
Those questions were short, used repeated wording patterns and belonged to
32 fictional edit lineages. Success there does not establish that every
paraphrase, longer chain or real company handbook will work.

We therefore asked a specific new question: **when one ownership fact changes,
can the assistant follow three connected facts and preserve unrelated facts?**
The purpose is to measure a useful boundary, not to manufacture failures or
make a perfect score look less suspicious. A perfect score on the new test
would also need to be reported with its limitations.

Here is the demonstration chain:

```text
Kaveri Payment Gateway -> owner team -> on-call person -> office city
```

A one-hop question asks for the owner. A two-hop question asks for that team's
contact. A three-hop question asks for that contact's office city. Changing the
owner should change the appropriate connected answers without changing every
fact about the service or people.

The Indian cities are real place names. The service, teams, people and their
assignments are fictional. For example, a team named after Bengaluru can have
a contact whose office is in Chennai. This prevents guessing the office city
from the team's name. These are controlled demonstration facts, not statements
about an actual company or government service.

## 2. What happened, in order?

| Date / step | What we did | Why it matters |
| --- | --- | --- |
| 27 October: preserve the completed study | Kept the original questions, outputs, configuration and scorecard unchanged. | The extension cannot retroactively improve or replace the earlier evidence. |
| 27 October: build a new dataset | Created new fictional graphs and separate development/final wording banks. | Longer questions alone would add little if they merely repeated the original examples. |
| 27 October: inspect retrieval | Checked whether the old retriever supplied every fact needed by each development question. | A model cannot reliably follow a chain if the third fact never reaches it. |
| 27 October: select bounded expansion | Compared six declared retrieval configurations on development data and two declared display states. | We selected a small context policy before seeing new final model answers. |
| 27 October: prepare the experiment | Added source/data hashes, exact-plan resume, fixed scoring, staged notebooks and regression tests. | An interrupted run should not become a reason to silently discard bad outcomes. |
| 28 October morning: profile on a T4 | Ran eight development questions per checkpoint and saved both raw ledgers locally. | Actual time and GPU allocation can guide the larger run. |
| 28 October: recover an interruption | The later temporary runtime expired. The profiles survived locally; development outputs were not backed up. | We disclose the evidence gap and do not claim the missing run succeeded. |
| 28 October: complete replacement development | Restored the saved profiles byte-for-byte, then recorded all 24 development requests per model. Both models answered 24/24 correctly. | These small development checks establish that the declared workflow executes; they are not held-out accuracy. |
| 28 October: freeze and publish | Bound source, data, settings, scoring and all four profile/development ledgers in a protocol, and published it before starting final inference. | Final answers cannot legitimately guide a change to this configuration or its scoring rules. |
| 28 October: start the separate final run | Started the fixed 272 requests per checkpoint sequentially on the T4, with atomic outcome saves. | All 544 assigned outcomes must remain in the evidence, including failures and abstentions. |
| 28 October: recover a page disconnection | Reattached the notebook to the still-running evaluator and saved another laptop backup. | A disconnected browser page is different from an expired or terminated compute runtime; this event required no restart. |
| 28 October: complete final inference | Recorded all 272 requests per model, retaining Qwen's eight abstentions. No request remained unexecuted or had an execution error. | A completed experiment includes unsuccessful answers rather than only its successful examples. |
| 28 October: audit and report | Recomputed scores from both complete raw ledgers, checked frozen input hashes and produced a separate scorecard and scientific figures. | The measured results can be traced to saved outputs rather than copied from the upstream paper. |
| 28 October: complete the application check | Recorded 14/14 correct actual model answers through the website; all twelve consistency checks passed, including rejected conflict and atomic rollback. Replayed the receipt on the laptop and opened the model-ready website with its completed depth dashboard. | Live application evidence is retained separately from benchmark accuracy; state checks do not become neural answers. |

The run plan is in [DEPTH_RUN_PLAN.md](DEPTH_RUN_PLAN.md). Recovery did not
change the approved facts, question labels or model settings. The frozen
[protocol](../reports/reasonke/depth-final/protocol.json) records the exact input
hashes. The replacement development execution is identified as such; the lost
execution is not part of a claimed successful denominator.

## 3. What is in the dataset?

A **lineage** is one fictional world with an original ownership fact and its
correction. Each world contains twelve facts and eight related questions.
Questions from the same world are kept together when splitting the data.

| Partition | Worlds / lineages | Questions | Purpose |
| --- | ---: | ---: | --- |
| Development | 16 | 128 | Retrieval selection and declared development checks |
| Final | 32 | 256 | Held-out measurement after freezing settings |
| Total | 48 | 384 | The complete new controlled packet |

Each world has two one-hop questions, two two-hop questions, two three-hop
questions and two locality controls. Consequently, the final set has
**64 questions in each of these four groups**, including 64 three-hop questions.
Each pair includes a direct wording and a paraphrase where applicable.

The controls ask about a historical release or another property of the same
service. They are labelled **depth 0** to distinguish them from the chain
affected by the edit. This label does not mean they require no information:
each control still needs its own explicit fact.

The split was made before wording augmentation. Development and final entity
names are disjoint, and their authored question-template banks are separate.
Templates still repeat within each split; cities, relation vocabulary and
generic release labels can repeat. Labels were checked automatically against
the structured graphs. We do not claim independent human semantic review.
See the [dataset card](DEPTH_DATASET_CARD.md) for the precise checks.

**256 questions are not 256 independent worlds.** Eight answers from one world
share facts and can fail together. The independent grouping for uncertainty
calculations is the 32 final lineages. The original and new studies have
different populations and retrieval policies, so we report their scores
separately rather than pooling them into one larger accuracy percentage.

## 4. What changed in retrieval?

Retrieval chooses which approved facts reach the model. The earlier two-stage
retriever completed 96/128 new development support paths, including 0/32
three-hop paths. That was a retrieval diagnostic, not proof that the neural
model could not answer a three-hop question.

The selected extension uses **at most four facts, two initial lexical seeds and
up to three expansion rounds**:

1. Match the customer's question against approved fact text to select seeds.
2. Use selected answer text, such as a team name, to look for a connected fact.
3. Continue through newly selected answer text, within the fixed bounds.
4. Supply the selected facts and question to the neural answer generator.

Ordinary retrieval does not receive the expected answer, gold support IDs or
the question's depth label. It is bounded lexical expansion, not a perfect
semantic scope classifier.

The selected policy supplied complete support for all 128 packet development
questions and both declared India display states: **130/130 paths**, containing
**230/230 required fact occurrences**. The two display states are additional
development checks, not independent held-out worlds.

Those percentages describe **support availability**. They do not mean the model
generated 130 correct answers. A model can receive the right facts and still
select the wrong person, abstain, include an extra answer or produce malformed
output. We measure retrieval and generation separately to locate such failures.

## 5. What models and GPU measurements are involved?

We compare the pinned upstream **ReasonKE_plus** checkpoint with the same-size
official **Qwen2.5-7B-Instruct** checkpoint. Both receive the same selected facts
and generation budgets. Both models were trained upstream; this extension
does not train their weights. We also cannot claim that a difference between
them is caused solely by ReasonKE training, because an exact parent-checkpoint
relationship has not been verified.

The model sources are [YukinoKaren/ReasonKE_plus](https://huggingface.co/YukinoKaren/ReasonKE_plus)
and [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
The frozen protocol binds their exact revisions and runtime configuration.

The deployment uses NF4 quantized weights with FP16 computation. Quantization
reduces memory use. An approved correction changes external memory and the
context supplied to the generator; it does not rewrite the model's weights.
The project contribution includes approved-memory management, bounded retrieval,
evaluation, provenance, recovery and the application. It should not be presented
as authorship of the upstream model or algorithm.

The completed profile used eight development requests per checkpoint:

| Checkpoint | Correct / assigned | Mean request time | Slowest observed request |
| --- | ---: | ---: | ---: |
| ReasonKE | 8/8 | 15.9742 s | 17.9072 s |
| Qwen | 8/8 | 1.08893 s | 2.4864 s |

These are small development profiles, not final accuracy estimates. Request
time includes context selection, local communication and generation; model
loading is separate. The measured peak loading allocation was approximately
**6.59 GiB**, above the laptop GPU's roughly 6 GiB capacity for this fixed loader.
The T4 therefore hosts the model runs. This does not prove that every possible
7B implementation is impossible on the laptop.

Multiplying the profile mean by a later request count gives a planning estimate.
For example, 272 × 15.9742 seconds is about 72.4 minutes of ReasonKE request
time. The slowest-sample extrapolation is about 81.2 minutes. Neither number
includes loading or guarantees that unseen questions take the same time.
We have no new training-loss curve because this stage performs inference,
not training.

## 6. How should the final numbers be read?

Each model has 256 ordinary final questions plus **16 preselected three-hop
oracle diagnostics**: 272 requests per model and **544 assigned requests**
across the two models. The same question appearing under two models does not
become two independent questions.

**Why the main accuracy denominator is 256, not 272:** the extra sixteen
requests deliberately receive gold support, whereas the main questions use
ordinary retrieval. Pooling these easier diagnostic conditions would change
the question being measured. Similarly, 544 counts model requests, not 544
different questions or independently authored graphs.

An oracle diagnostic supplies the known required supporting facts. It helps
separate generation from retrieval failure. It does not directly supply a
target answer field, and it must never be described as ordinary retrieval
performance. Its question subset is selected before inference rather than
chosen afterwards to repair specific failures.

- **Accuracy:** correct final answers divided by every assigned question.
- **Coverage:** answered questions divided by every assigned question.
- **Wrong:** an answer was generated but failed the declared factual/whole-answer rule.
- **Abstained:** the model declined or returned the declared unknown response.
- **Execution error:** the request failed to produce a usable model response.
- **Unexecuted:** an assigned request has no completed outcome.
- **Fact recall:** required supporting fact occurrences supplied to the model,
  divided by all required occurrences.
- **Complete-path retrieval:** questions for which every required fact was supplied.
- **p50 / p95 latency:** the median request time and the time below which 95% of
  measured request times fall. These have their own disclosed sample counts.

Abstentions, errors and unexecuted questions receive zero primary accuracy
credit. They do not disappear from the denominator. A correct name mentioned
only in an explanation or as a rejected possibility is insufficient; scoring
checks the endorsed final answer against the requested subject, relation and
declared answer aliases.

Paired uncertainty compares the two models on the same questions and resamples
whole lineages. A narrow or zero-width interval on a small controlled packet
does not establish universal certainty. Retrieved provenance also does not
verify all generated explanation text.

### The completed final results

| Matched condition | Correct / assigned | Accuracy | Wrong | Abstained | Errors / unexecuted |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReasonKE + iterative retrieval | 256/256 | 100% | 0 | 0 | 0 / 0 |
| Qwen + the same iterative retrieval | 248/256 | 96.875% | 0 | 8 | 0 / 0 |
| ReasonKE + preselected gold support | 16/16 | 100% | 0 | 0 | 0 / 0 |
| Qwen + preselected gold support | 16/16 | 100% | 0 | 0 | 0 / 0 |

The main questions break down as follows:

| Question group | ReasonKE correct / assigned | Qwen correct / assigned |
| --- | ---: | ---: |
| Locality controls: another property or historical scope | 64/64 | 64/64 |
| One hop: owner team | 64/64 | 64/64 |
| Two hops: on-call person | 64/64 | 57/64 |
| Three hops: office city | 64/64 | 63/64 |

All eight Qwen failures were abstentions: seven direct two-hop questions and
one paraphrased three-hop question. **Every main question received all its
required support**: 256/256 complete paths per model, or 512/512 model-request
paths. There are 448/448 required fact occurrences per model. This repeated
retrieval outcome comes from one shared selector, not two independent retrieval
experiments. Missing retrieval support therefore does not explain these eight
abstentions. The gold-support subset was preselected and contains different
context conditions; its 16/16 result must not be presented as a retry that fixed
the eight failures.

ReasonKE's observed main accuracy advantage is **3.125 percentage points**.
Resampling the 32 paired lineages gives a 95% interval of **+1.5625 to +5.078125
points**. This describes uncertainty across this controlled lineage population;
it is not a guarantee for arbitrary questions or an isolated effect of model
training. ReasonKE's three-hop advantage is only 1/64, and its corresponding
paired interval includes zero. The evidence does not establish that one model
is universally better at every three-hop task.

The cost is higher latency: ReasonKE's main-request p50/p95 was
**15.94/19.28 seconds**, versus **1.12/1.38 seconds** for Qwen, with 256 generated
requests per model. These warm pipeline times include retrieval, local worker
communication and generation. They exclude model loading, downloads and browser
network time. Accuracy and speed are separate product tradeoffs.

The practical conclusion is bounded: the selected four-fact retrieval policy
and ReasonKE checkpoint handled every assigned question in this new controlled
packet, including 64 three-hop questions and all locality controls. A ceiling
score leaves the model's failure boundary unresolved. The packet still uses
fictional short facts, repeated within-split templates, automated labels and
32 held-out graph lineages. It does not measure unrestricted dialogue, messy
documents, arbitrary editing scopes or four-hop reasoning. We did not modify
test questions or aliases after seeing the results, and we did not invent
failures to make the graph look realistic.

Inspect the [audited scorecard](../reports/reasonke/depth-final/results.json),
[host audit](../reports/reasonke/depth-final/host-audit.json) and
[measured report and figures](../reports/reasonke/depth-final/derived/REPORT.md).
These results remain separate from the completed operations-v1 experiment.

## 7. How to demonstrate it on the website

Use the India chat launcher with the iterative policy and a fresh workspace.
The website requires a ready model worker for neural chat; an unavailable-model
badge is an infrastructure state, not a generated answer.

1. In **Knowledge**, approve the fictional India handbook. Nine facts produce
   knowledge revision 9. Inspect their sources and separate release scopes.
2. In **Chat**, ask the direct owner, the owner's on-call contact and that
   contact's office city. The initial targets are Bengaluru Platform Team,
   Meera Iyer and Chennai.
3. Return to **Knowledge** and draft the example ownership update. Inspect it
   and explicitly approve replacement with Pune Reliability Team. Drafting
   alone publishes nothing.
4. Repeat the questions. The intended updated targets are Pune Reliability
   Team, Arjun Deshmukh and Hyderabad. The hosting city stays Bengaluru; release
   1 and the Pro service retain their separate owners.
5. Use historical comparison against revision 9, then roll back the replacement.
   The transaction restores the previous owner and retains the audit trail.
   Inspect actual generated responses and supplied facts throughout.
6. In **Research results**, use the separate depth-challenge view. A deployment
   containing the completed depth evidence shows the audited depth groups,
   outcome counts, retrieval support, latency and exports separately from the
   original operations scorecard. An older pinned launcher may lack this new
   evidence; its unmeasured display is not a new failed evaluation.

For the comparison in step 5, open **Compare original & corrected**, choose
**Approved memory at a revision**, enter `9`, and choose **Run comparison**.
This asks the model afresh using the same question with two approved-memory
states. It does not simply replay the answer shown earlier in the chat. Choosing
**Without correction context** is a different comparison: the model is asked
without the fictional handbook, so it may not know these invented facts.

In Chat, ordinary conversation is also generated by the model. The controlled
factual benchmark does not score every possible conversational response. Expand
the supplied-fact evidence to inspect what reached the model; a retrieved source
supports a fact but does not certify every sentence in a generated explanation.

The [India demonstration guide](INDIA_DEMO.md) gives the exact example questions.
Export the workspace before ending a temporary Colab runtime. Export experiment
evidence separately: saved facts and rollback correctness are software checks,
whereas a correct neural answer requires a recorded model generation.
