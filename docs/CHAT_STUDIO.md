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

## What was measured locally

The [second local acceptance rehearsal](../reports/chat-studio-local-2/README.md)
used the actual `qwen3-vl:2b-instruct` and `all-minilm:22m` models. Its
[raw receipt](../reports/chat-studio-local-2/live-acceptance.json) contains ten
fresh model generations: normal text, a follow-up, the initial unknown fact,
direct and paraphrased corrected questions, an unrelated question, two comparison
responses, a post-withdrawal question and a generated geometric-image question.
All 13 operational checks and six narrowly defined fixture assertions passed.
These checks overlap the same outputs; they are not 19 independent cases or a
representative chat-accuracy score. The 44 unit tests using contract fixtures are
separate engineering evidence.

The correction's `LUMEN-73` value is synthetic test data, not a real credential.
After withdrawal, the model stopped endorsing that supplied value but generated
an unsupported alternative, `ASTER-001`. The defined non-endorsement assertion
passed; the answer was still not established as factually correct. Withdrawal
controls which memory is supplied, not whether every later sentence is true.

The [first actual-model rehearsal](../reports/chat-studio-local-1/live-acceptance.json)
also passed its 13 operational checks, yet its direct corrected response repeated
an earlier refusal. The [qualitative review](../reports/chat-studio-local-1/qualitative-review.json)
retains that failure. The context policy was then changed to exclude all
prior-revision assistant answers from later generation prompts, while preserving
visible history. This was development and debugging, not repeated tuning of the
frozen visual test set. See the second run's
[code and model identity record](../reports/chat-studio-local-2/run-context.json):
its recorded base Git commit alone is not the complete identity of the working
implementation used during the run.

The separate [browser observations](../output/chat-studio-2026-10-25/browser-observations.json)
show a controlled counterfactual Australia example: original Canberra, supplied
counterfactual Melbourne, and Canberra again after withdrawal. This is a GUI
workflow demonstration, not a real-world factual update or an additional sample
in the acceptance run or the 20-edit visual benchmark.

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
- Conversation context is bounded. All assistant answers from earlier knowledge
  revisions, including old base-model answers or refusals, are excluded from
  later model prompts. Correction-conditioned assistant prose is also excluded.
  Those messages remain visible in history. A knowledge change can therefore
  reduce continuity of details supplied only in an earlier assistant response.
- This is a local single-operator academic application, not an internet-hosted
  multi-user service with accounts and roles. Keep its loopback binding; do not
  expose the server by changing network settings.
- The preserved 20-edit/500-answer report and its automated label-review limits
  remain unchanged. New conversational performance needs its own measured
  evaluation; no accuracy number is inferred from the old report.

Official references: [Qwen3-VL through Ollama](https://ollama.com/library/qwen3-vl:2b-instruct),
[Ollama chat API](https://docs.ollama.com/api/chat), and
[image input documentation](https://docs.ollama.com/capabilities/vision).

## Optional Linux website container

`Dockerfile.chat` packages the website, Pillow, the model-identity configuration
and the saved visual evaluation report. It runs as an unprivileged user and
keeps new conversations and correction records under `/app/runtime/chat`.
It does **not** install Ollama or contain the model weights. The base image
`python:3.12-slim` is a floating minor-version tag, not an immutable image digest;
the separately recorded Qwen and embedding model digests are checked by the
application when those models are installed.

On a Linux machine with the configured Ollama models already prepared on the
host, build and start the website from the repository root:

```bash
docker build --file Dockerfile.chat --tag editguard-chat:local .
docker volume create editguard-chat-data
docker run --rm --name editguard-chat --network host \
  --mount type=volume,src=editguard-chat-data,dst=/app/runtime/chat \
  editguard-chat:local
```

Open `http://127.0.0.1:8004/` on that machine. Native Linux host networking lets
the container reach the host's loopback Ollama service on port 11434 while the
website retains its own loopback-only binding. An ordinary bridge-mode `-p`
mapping cannot reach a server bound to the container's loopback interface. No
public tunnel or broader bind address is required. The named volume retains
the database and images after the container stops; preserve that volume to keep
the history. Its initial contents inherit the image's writable runtime directory.
If a volume already exists with incompatible ownership, correct its ownership
deliberately rather than deleting saved history or running the application as root.

This is an optional native Linux packaging path. It is not a claim of Windows
Docker Desktop host-network/GPU compatibility or model inference inside the
container. Ollama owns model execution and its hardware configuration on the
host. The native Windows launch above is the project's measured Chat Studio
path. No local Docker or GPU-container execution was performed while preparing
this file; the CI build/import/HTTP checks are separate engineering evidence and
do not establish model quality.
