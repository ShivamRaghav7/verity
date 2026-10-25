# EditGuard: running and explaining the ReasonKE website

EditGuard lets you chat with a language model, approve a replacement fact, and
inspect how that fact changes later answers. You can compare answers and undo
the correction. The current backend is the author's trained
[`YukinoKaren/ReasonKE_plus`](https://huggingface.co/YukinoKaren/ReasonKE_plus)
checkpoint, pinned to revision `ceaedc2ae7c8fed6257953587d991ce550db6106`.

## What happens when you edit knowledge

The model already has learned knowledge in its numerical weights. Our
administrator's correction is stored separately in a versioned database. For
an edited question, the application supplies active approved facts alongside
the question. The trained model then generates an answer using that context.
Publishing a correction does **not** change its weights.

This is in-context knowledge editing with external memory. The upstream
research trained the model to reason using updated information. Our work is
the application and deployment around it: explicit approval, sources, version
records, conflict controls, comparison, withdrawal, a bounded model API, and
reproducible receipts of actual generations. We did not invent ReasonKE or
train this released checkpoint. See the
[official research repository](https://github.com/YukinoshitaKaren/Reason-KE).

## Start the website

1. Open the current pinned Colab launcher linked at the top of the
   [README](../README.md). The notebook file is
   [04_reasonke_chat.ipynb](../notebooks/04_reasonke_chat.ipynb).
2. In Colab, choose **Runtime → Change runtime type → T4 GPU**. Run the
   hardware, setup, and worker-launch cells in order. Setup creates an isolated
   environment; it does not start model training.
3. Read the progress output. The worker downloads the pinned checkpoint,
   loads it, and generates the five predefined development answers. Avoid
   starting another worker while this one is running.
4. Run the website cell. It waits for the model and opens the website through
   Colab's port viewer. Check that the website reports the model as ready.
5. Keep the Colab runtime connected while using the website. Its URL belongs
   to this runtime and expires when the runtime ends. A fresh runtime needs
   the setup and launch steps again; an old website URL cannot restart it.

The first measured T4 session took 338.68 seconds for download/loading and
69.19 seconds for its five diagnostic generations. Individual generations
took about 11–16 seconds in that session. These measurements help set
expectations; they are not a time guarantee for another session.

The local `run-chat.ps1` launcher uses the older Ollama backend. Use the
ReasonKE Colab launcher for this demonstration.

## Navigate the website

| Area | What to do there |
| --- | --- |
| **Chat** | Start a conversation, ask text questions, inspect generated answers, and compare an answer. **New chat** starts a separate conversation. |
| **Use approved knowledge** | Keep this enabled to supply active facts. Disable it to ask without correction context. Both modes use the same trained checkpoint. |
| **Knowledge** | Publish a correction with its answer, scope and source. Inspect active records, publish replacement versions, withdraw a record, or show inactive records to restore one. |
| **Knowledge r…** | The database revision. Publishing, withdrawing, or restoring knowledge advances the revision; previous messages remain historical records. |
| **Research results** | The preserved **earlier visual-model experiment**. Its numbers are not ReasonKE chat accuracy. The ReasonKE receipts linked below are separate evidence. |

This backend accepts text. Its image attachment control is disabled. A
customer's statement in Chat does not silently become approved knowledge;
publishing in Knowledge is the explicit administrator action.

## A short, repeatable demonstration

Use this deliberately fictional change to make the difference obvious. Say
aloud that Mumbai is a **counterfactual demonstration target**; India's actual
capital is New Delhi. Do not present the edited answer as a real-world fact.

1. Start with no active corrections. In Chat, ask:

   **What is the capital of India?**

   The original answer observed in our live checks was **New Delhi**.

2. Open Knowledge and enter these four fields exactly:

   | Field | Value |
   | --- | --- |
   | Question or fact to correct | `What is the capital of India?` |
   | Corrected answer | `Mumbai` |
   | Where should it apply? | `India; capital; controlled counterfactual demonstration` |
   | Source or reason for the update | `Controlled counterfactual demonstration. This record defines the approved answer within the demonstration.` |

   Click **Approve & publish**. The source describes the controlled exercise;
   it is not a real government source. Keep this note unambiguous. In an
   earlier attempt, putting the competing answer inside the source note
   caused the model to reject the intended override. That failure is retained
   in the development record.

3. Return to Chat with **Use approved knowledge** enabled. Ask the original
   question again. Then start a **New chat** and ask:

   **Which city serves as India's capital?**

   **Mumbai** was observed for both forms after the unambiguous correction.
   The fresh conversation checks the paraphrase without relying on the prior
   assistant answer.

4. Use **Compare original & corrected** on the paraphrase. The application
   generates both sides afresh with matched question history: one without
   supplied corrections and one with approved facts. Our live comparison
   produced **New Delhi / Mumbai**. This compares context conditions of the
   **same** trained model, not a newly trained model against its foundation
   checkpoint.

5. In Knowledge, click **Withdraw** on the demonstration correction. Ask the
   paraphrase again. It returned **New Delhi** after withdrawal in the live
   check. Withdrawal removes the fact from future prompts; it does not erase
   the earlier edited messages.

6. For ordinary conversation, ask:

   **Explain overfitting in two short sentences.**

   The model generates an explanation rather than looking up a stored answer.
   Read its content; a nonempty answer alone is not an accuracy measurement.

## Versions, conflict handling and undo

On an active correction, **Publish a new version** fills the form and marks
which record the new approval will supersede. Review the answer and source,
then explicitly approve the replacement. The old record remains in history.

Publishing another active record with the same case/whitespace-normalized
question and scope is blocked until you explicitly select the version to
replace. This deterministic conflict check does not detect every contradiction
between differently worded facts. Write precise scopes and inspect related
records yourself.

Select **Show inactive** to inspect withdrawn or superseded records. **Restore
version** activates the selected record again. If another record is active in
that same exact scope, withdraw it first. Restoring an old record changes
future correction context; it does not restore old model weights because those
weights were never changed.

## Save evidence before Colab closes

The notebook's optional live acceptance cell runs seven real generations:
original, corrected direct, fresh paraphrase, both comparison sides,
post-withdrawal, and ordinary conversation. Run it against isolated memory
before publishing your own active facts. It refuses to clear existing active
knowledge and withdraws only its own demonstration correction.

The final export cell downloads receipts, configuration inputs and logs.
Inspect `acceptance.json` for the five model checks and
`live-acceptance.json` for the website workflow, including failed or unexecuted
cases. This evidence export is not a complete backup of all personal chats.

For a private backup of the current website database, add and run this Colab
cell while the runtime is still available:

```python
import sqlite3
from google.colab import files

backup = RUN / 'chat-backup.sqlite3'
with sqlite3.connect(RUN / 'chat/chat.sqlite3') as source_db:
    with sqlite3.connect(backup) as destination_db:
        source_db.backup(destination_db)
files.download(str(backup))
```

This uses SQLite's backup operation to include committed data consistently.
Keep the downloaded database private if your chats contain personal material.
Model weights are downloaded again or reused from cache, not included in the
database backup.

## Explain the results accurately

The [first real T4 receipt](../reports/reasonke/t4-development-1/README.md)
contains five predefined development questions: a direct edit, a paraphrase,
a two-hop question, an unrelated control, and a similar-name boundary. All
five final answers matched their targets. The facts were supplied explicitly,
so this checks generation with evidence, not retrieval quality. Five chosen
questions do not establish general accuracy or reproduce a published score.

The raw explanations also contained unsupported statements despite correct
final answers. A source shown under an answer means it was supplied to the
model; it does not verify every generated sentence. Preserve that distinction
when presenting the demo.

The working deployment and these development checks establish an integration
milestone. A broader frozen evaluation of this ReasonKE backend and its final
results report remain separate work; the older visual-study report cannot
substitute for them.
