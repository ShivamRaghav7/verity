# Understanding the three-hop extension

This guide explains why we extended EditGuard, what the new numbers mean and
how to demonstrate the work. It describes the actual work rather than an
invented development story. For the full earlier project, start with
[Understanding the current project](UNDERSTAND_THE_CURRENT_PROJECT.md).

**Status at this checkpoint, 28 October 2026:** both eight-request GPU profiles
completed and were backed up. A subsequent development run lost its temporary
Colab runtime before its results were downloaded. Its last observed ledger had
0/24 ReasonKE requests recorded. Those missing outcomes cannot be treated as
measured results. We are restoring the profiles and preparing a replacement
development run. The new final evaluation has not run, so this document claims
no held-out three-hop answer accuracy.

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
| Next stages | Complete replacement development, freeze its inputs, then run the separate final population once and audit the saved outputs. | Final results must follow the frozen plan rather than guide settings or scoring. |

The run plan is in [DEPTH_RUN_PLAN.md](DEPTH_RUN_PLAN.md). Recovery does not
change the approved facts, question labels or model settings. The completed
profiles should be restored rather than regenerated. The replacement
development execution must be identified as such; the lost execution is not
part of a claimed successful denominator.

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

**Final measurement section: unmeasured at this checkpoint.** The eventual
separate scorecard, raw receipts, audit and figures belong under
`reports/reasonke/depth-final/`. That output location is not evidence that the
run has already completed.

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
6. In **Research results**, use the separate depth-challenge view. Until audited
   final evidence is released, it displays an explicit unmeasured state. Once
   released, it shows depth groups, outcome counts, retrieval support, latency
   and exports separately from the original operations scorecard.

The [India demonstration guide](INDIA_DEMO.md) gives the exact example questions.
Export the workspace before ending a temporary Colab runtime. Export experiment
evidence separately: saved facts and rollback correctness are software checks,
whereas a correct neural answer requires a recorded model generation.
