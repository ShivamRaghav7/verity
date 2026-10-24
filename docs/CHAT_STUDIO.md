# EditGuard Chat Studio

Chat Studio is the everyday website: have a normal text conversation, optionally
attach an image, publish a correction in **Knowledge**, and return to the same
conversation to inspect the difference. It uses a local instruction-following
model through Ollama. It is separate from the earlier BLIP-2 research workbench.

## Start it on Windows

Open PowerShell in the project folder. For the first setup, run:

```powershell
cd "C:\Users\shiva\OneDrive\Desktop\projekt-KE"
.\scripts\run-chat.ps1 -Setup
```

The setup prepares the local application dependencies and the two named Ollama
models: `qwen3-vl:2b-instruct` for generated conversation and optional image questions, and
`all-minilm:22m` for text embeddings used by correction retrieval. Initial model
downloads require an internet connection and disk space. The script's output is
the place to check setup errors; a website being reachable does not by itself
mean the model is ready.

Use the explicit **`qwen3-vl:2b-instruct`** tag. In our development check, the
generic `qwen3-vl:2b` tag resolved to a thinking checkpoint. Requesting
`think: false` still produced no visible answer within the fixed 384-token
generation budget. We selected the explicit instruction tag to remove that
default-tag ambiguity. The original abstention remains a development
observation; it is not removed from the history or counted as a success, and no
new chat accuracy is inferred from changing this setting. The frozen BLIP-2
study and its recorded results remain unchanged.

For later launches, use:

```powershell
.\scripts\run-chat.ps1
```

Open **http://127.0.0.1:8004/** and check the model-status badge. Leave the
PowerShell server and Ollama running while using the site. Stop the application
with **Ctrl+C** in its server terminal. Restarting the website preserves saved
chats and correction records in `runtime/chat/`; do not delete that folder to
restart the program. Those local runtime files are excluded from Git.

This path uses the laptop's local Ollama service rather than the temporary Colab
URL. Hardware speed, available memory and image size affect performance. The
recorded BLIP-2 T4 timings are not estimates for this new chat model.

## Use the three tabs

**Chat.** Start a new conversation or reopen one in the conversation list. Type a
normal message; an image is optional. For example, ask for an explanation,
continue with a follow-up, or attach an image and ask what it shows. The model
generates the response. It has no live web search and can make factual mistakes.

**Knowledge.** This is the local administrator workflow. Enter the factual
question being corrected, its intended answer, a narrow subject/scope and the
source you are relying on. Review the fields and publish the correction. A
customer chat message alone never becomes an authoritative correction. Records
show their version, source and active or inactive state. Withdraw a correction
to stop supplying it to later generations; restore it to make it active again.

**Results.** Inspect the preserved visual-editing comparison and its denominators.
This is saved evidence from the earlier BLIP-2/VLKEB experiment. It is not an
accuracy score for the new Qwen conversation model, the new text retriever or a
particular answer you just generated.

## A short end-to-end demonstration

Use a controlled fictional fact so an edited association is not presented as
real-world truth.

1. In Chat, ask: **What is the launch city of the fictional Northstar project?**
   Inspect the original answer rather than assuming it knows this invented fact.
2. In Knowledge, enter that question, the answer **The fictional Northstar
   project's launch city is Pune**, the scope **Fictional Northstar project /
   launch**, and the source **Controlled demonstration handbook, revision 1
   (fictional)**. Publish it explicitly.
3. Return to the same conversation and ask the question again. Inspect the
   generated answer and its correction/source evidence. A retrieved record is
   evidence of which input was supplied, not a certificate that the entire
   response is correct.
4. Try a paraphrase and then an unrelated question. An edit should apply to its
   intended subject and relation; observe any mistaken applications rather than
   hiding them.
5. Use **Compare** for a user message to generate an original-model answer and a
   correction-enabled answer from matched pre-question history. Compare produces
   two new responses; it does not relabel a desired target as model output.
6. Withdraw the record in Knowledge and ask again. The new request uses the
   current correction revision. Old answers remain visible as historical chat
   messages; withdrawal does not rewrite what was previously said.

Publishing a changed answer for the same normalized question and scope requires
an explicit replacement of its current active record. This creates another
version and preserves the previous record. Exact case/whitespace conflict checks
do not detect every contradiction between differently worded facts. Retrieval
also asks for clarification when competing high-scoring corrections cannot be
resolved safely by its heuristic.

## What actually changes when knowledge is edited

The default method is **external correction memory plus retrieval-augmented
generation**. Publishing writes an approved, versioned record to SQLite. For a
new question, a pretrained embedding model compares it to active correction
questions and scopes. A relevant selected record is supplied as delimited data
to the conversation model, which generates the answer. The interface records
the selected correction, source, retrieval decision and knowledge revision.

The base model's weights remain unchanged in Chat Studio. No per-correction
training run is needed for this method. Withdrawing a correction removes it
from later retrieval, while the audit history remains available. This is an
external-memory editing workflow; it must not be described as a permanent
parameter update, newly trained foundation model or a reproduction of MEND.

The `.72` similarity cutoff is a chat development setting. It was not calibrated
by the old visual experiment, and a similarity score is not a probability of
correct applicability. A lexical fallback, if selected instead of the embedding
encoder, is named explicitly. Narrow scopes and clear questions help, but the
system does not guarantee arbitrary entity, version or paraphrase boundaries.

## Our contribution and the comparison with parameter editing

We reused pretrained Qwen and MiniLM models, just as the research workbench reused
BLIP-2 and the authors' MEND network. The application contribution is the complete
maintenance workflow: persistent conversation, separate operator approval,
versioned correction memory, scoped retrieval, source evidence, explicit
conflict handling, withdrawal/restoration and matched before/after generation.
The model response and stored target remain distinct throughout that workflow.

The preserved research contribution is the controlled five-method comparison,
integrity-checked assets, documented model adaptations and exact parameter
rollback. See [the measured visual results](VLKEB_RESULTS.md) and
[the original workbench guide](VLKEB_WORKBENCH.md). That parameter-editing
workbench is still a separate experiment; it is not silently running inside
Chat Studio's correction-memory method.

Updated RAG was a useful default to build on, not a universally proven winner.
In the earlier small visual study, updated RAG matched 54/60 edited targets but
falsely applied corrections to 13/40 unrelated cases. Fine-tuning matched more
edited targets while harming unrelated answers; adapted MEND traded target
success for better preservation in a very small baseline-correct subset. Those
tradeoffs motivate inspecting a correction before trusting its effects.

## Practical limits

- Normal conversation and image questions are generated by a compact pretrained
  model; answers may be incomplete, wrong or sensitive to wording.
- Retrieved sources and operator approval do not independently establish factual
  truth. The source is the administrator's stated provenance.
- Conversation context is bounded. Previously correction-conditioned assistant
  prose is excluded from later model prompts to reduce stale correction leakage,
  although it stays visible in the chat history.
- This is a local single-operator academic application, not an internet-hosted
  multi-user service with accounts and roles. Keep its loopback binding; do not
  expose the server by changing network settings.
- The preserved 20-edit/500-answer report and its automated label-review limits
  remain unchanged. New conversational performance needs its own measured
  evaluation; no accuracy number is inferred from the old report.

Official references: [Qwen3-VL through Ollama](https://ollama.com/library/qwen3-vl:2b-instruct),
[Ollama chat API](https://docs.ollama.com/api/chat), and
[image input documentation](https://docs.ollama.com/capabilities/vision).
