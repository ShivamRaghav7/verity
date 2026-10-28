# India-themed demonstration handbook

The website's starter example now uses familiar Indian city names and invented
service/team/person assignments. **Kaveri Payment Gateway, its teams, people and
ownership changes are fictional.** The city labels make the example relatable;
they do not assert a real company's operations or a government policy.

This display example is separate from the completed operations-v1 benchmark.
Changing it does not rename or rescore any of the 1,312 recorded model requests.
The separate 544-request depth evaluation is complete and replay-audited at this
checkpoint (28 October 2026): ReasonKE scored 256/256 and Qwen 248/256 on the
main retrieval questions. The display walkthrough below is a separate
application workflow. Its live acceptance check completed **14/14 actual model
answers** and all **12 consistency checks**, including conflict rejection and
atomic rollback. The [raw receipt](../reports/reasonke/depth-final/india-acceptance.json)
and [separate replay audit](../reports/reasonke/depth-final/india-host-audit.json)
retain the evidence. This is one fictional display lineage, not fourteen new
held-out factual lineages or an arbitrary-language guarantee.

## Open the website

Use the [India notebook](../notebooks/07_india_chat.ipynb) on an idle T4 and wait
for its model-ready check before opening the website link. Do not launch its
model worker alongside the depth evaluation in the same runtime. A page that
loads with an unavailable model can display the interface and stored facts, but
cannot demonstrate a real neural answer.

The sidebar contains **Chat**, **Knowledge** and **Research results**. Knowledge
controls the approved facts; Chat uses the model to answer; Research results
shows saved experiment evidence rather than evaluating the current conversation.
The knowledge revision badge identifies the memory state, not a model version.

## Start in a fresh workspace

In **Knowledge**, choose **Approve fictional India handbook**. This explicit
administrator action approves nine facts and creates revision 9. It preserves
existing chats. A workspace that already has knowledge history is not replaced;
use a new `--storage` directory for this example and keep the previous directory.
Withdrawing every old fact does not erase its history or make it a fresh workspace.

The nine initial facts are:

| Fact | Initial value |
| --- | --- |
| Kaveri Payment Gateway release-2 owner | Bengaluru Platform Team |
| Bengaluru Platform Team contact | Meera Iyer |
| Pune Reliability Team contact | Arjun Deshmukh |
| Kaveri Payment Gateway release-2 hosting city | Bengaluru |
| Kaveri Payment Gateway Pro release-2 owner | Hyderabad Integration Team |
| Kaveri Payment Gateway release-1 owner | Chennai Legacy Team |
| Konkan Records Office city | New Delhi |
| Meera Iyer's office city | Chennai |
| Arjun Deshmukh's office city | Hyderabad |

All nine records carry explicit fictional provenance. The city of a person's
office deliberately differs from the city in their team's name, so a three-hop
answer requires the office fact rather than a team-name shortcut.

## Show the edit and its connected answers

1. Ask **Which team owns Kaveri Payment Gateway in release 2?** The initial target
   is Bengaluru Platform Team.
2. Ask **Who is the on-call contact for the team that owns Kaveri Payment Gateway
   in release 2?** The two-hop initial target is Meera Iyer.
3. Ask **In which Indian city is the office of the on-call contact for the team
   that owns Kaveri Payment Gateway in release 2?** The three-hop initial target
   is Chennai: service → team → person → office city.
4. In Knowledge, choose **Draft the example ownership update**, inspect the
   proposed Pune Reliability Team owner, and explicitly approve replacement.
   Drafting alone approves nothing. The revision becomes 10.
5. Repeat the questions. The intended new answers are Pune Reliability Team,
   Arjun Deshmukh and Hyderabad. Hosting city stays Bengaluru; release-1 owner
   stays Chennai Legacy Team; the Pro service keeps its separate owner.
6. Compare the contact/office question against approved memory at revision 9,
   then **Roll back to previous version**. Atomic rollback restores the original
   owner at revision 11, so the intended contact and office become Meera/Chennai.

To compare a historical state, choose **Compare original & corrected** under an
answer, set **Compare against** to **Approved memory at a revision**, enter `9`,
and choose **Run comparison**. Each side is a fresh model generation with the
same question and different approved memory. **Without correction context**
removes this fictional handbook instead; it is not the pre-edit revision.

To publish a different correction, enter its question/fact, corrected answer,
scope and source in Knowledge's editor, then choose **Approve & publish**.
Expand **Canonical fact identity** for entity, property and version/plan IDs;
fill all three together for catalogue facts so paraphrased corrections share
the same version history. A draft is not authoritative. An unresolved overlapping
conflict needs administrator resolution; the assistant should not silently
choose one answer. Inspect **Approval and rollback history** to see revisions
and retained fact versions.

These targets are controlled facts, not a promise that a generation will match.
Inspect actual answers and supplied facts. The bounded retriever may omit a
required third-hop fact; a failure can come from retrieval or generation. Never
substitute the table's deterministic target for the model's response.

The optional `--context-policy iterative` setting uses the same bounded lexical
answer-text expansion function as the separate depth challenge. Its fact cap is
selected on development data and reported in each answer's evidence. It supplies
approved facts to the neural generator, without reading an expected answer or
gold support IDs. The CLI default remains `retrieved`, preserving the original
study's two-stage four-fact policy. Use `--backend reasonke --context-policy
iterative` explicitly for the new depth-oriented demonstration; it is a separate
serving policy, not a retroactive change to the 1,312-request study.

## Save and interpret the demonstration

The final experiment and application receipts, audits, graphs and PDF are now
saved on the laptop and in Git. You can close Colab without losing those
measurements. Live chat stops when its runtime ends. To use it again, open the
released India notebook on an available T4 and run its cells; this loads the
pinned checkpoint and website without repeating the benchmark. Approve the
fictional handbook again in a fresh workspace, or restore your private exported
workspace to recover its conversations and edit history.

Before ending the temporary runtime, open Knowledge's **Keep your workspace**
area and choose **Export workspace JSON**. Store this file privately: it contains
text conversations as well as approved facts, versions and history. Restoration
is an explicit administrator action into an empty workspace. Experiment ledgers
and scorecards are exported separately; restoring a chat is not an evaluation.

In Research results, keep the original operations scorecard and the separate
depth view distinct. A deployment containing the new audited evidence displays
256/256 ReasonKE main answers, 248/256 Qwen main answers and their separate
16-request gold-support diagnostics. An older pinned notebook may not package
the new scorecard and can therefore display an unmeasured state. A software
regression test proves that an API or transaction behaves as specified; only a
saved actual model response can establish a neural answer outcome. The wider
interpretation, including abstentions and the slower ReasonKE latency, is explained in
[Understanding the three-hop extension](UNDERSTAND_DEPTH_EXTENSION.md).

## API compatibility

- `GET /api/examples/india` returns the display-only India definition without
  changing knowledge.
- `POST /api/examples/india/approve` uses the administrator workflow and refuses
  nonempty knowledge history.
- Legacy `/api/examples/operations` and `/api/examples/operations/approve` retain
  the original Dahlia example for reproducing the recorded operational study.

The frontend defaults to India. The old benchmark and its operational receipts
remain tied to the original example, rather than retroactively relabelled.
