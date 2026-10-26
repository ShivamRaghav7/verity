# The current application: a complete ownership-change demonstration

Start the pinned ReasonKE Colab launcher from the repository README. Use a T4,
run the setup and worker cells, and wait until the model status is ready before
opening the website. Model loading generates no answers by default; the optional
startup and seven-generation checks have explicit switches. Keep the notebook
connected while demonstrating. This checkpoint accepts text, not images.

This walkthrough uses fictional operations data. It demonstrates a change that
affects both a directly stored answer and a connected question: a service changes
owners, so the person to contact also changes. It is not a real on-call directory.

## Establish the initial handbook

Open **Knowledge** in a new workspace and press **Approve fictional operations
handbook**. Review its fictional-data notice first. This explicitly adds nine
facts and advances knowledge to revision 9. It works only before any knowledge
history exists; it preserves existing conversations. It never replaces another
workspace's knowledge. New customer messages cannot approve facts.

The example service is **Dahlia Relay**. In release 2 its initial owner is **Saffron
unit 14**, whose contact is **Mara Voss**. Another fact already says that **Copper
unit 74** has contact **Orin Dane**. The distinction matters: an ownership change
should change which team's contact is selected, not overwrite either team's
contact fact.

In **Chat**, ask these questions in fresh conversations when comparing independent
factual outputs. Ordinary conversation and follow-ups are supported, but this
controlled factual workflow is what the current accuracy study measures.

| Question | Expected initial answer |
| --- | --- |
| Which team owns Dahlia Relay in release 2? | Saffron unit 14 |
| Who is the on-call contact for the team that owns Dahlia Relay in release 2? | Mara Voss |
| Which region hosts Dahlia Relay in release 2? | West sector |
| Which team owns Dahlia Relay in release 1? | Silver unit 194 |
| Which team owns Dahlia Relay Pro in release 2? | Cedar unit 134 |

Read the generated answer and expand its supplied facts/source evidence. The
model can still make an error; retrieved sources do not verify arbitrary prose.
Every stored answer retains its model and knowledge revisions.

## Publish an actual change

Return to **Knowledge** and press **Draft the example ownership update**. This
fills the editor without activating anything. It proposes **Copper unit 74** as
the release-2 owner, with a source and the same canonical entity/property/scope
IDs. Review it, then press **Approve & replace version**. Revision becomes 10.

Ask the direct question again and use different wording, for example **Name the
team responsible for Dahlia Relay, release 2**. The intended answer is now Copper
unit 74. The connected on-call question should now answer **Orin Dane**, using
both the changed ownership and the existing contact fact. The region, release-1
owner, Pro-service owner and Saffron team's own contact remain unchanged.

The website selects up to four approved facts through two-stage lexical retrieval
and supplies them to the trained ReasonKE checkpoint. It does not query the
ground-truth answer field, return a database template as model output, or update
weights for this correction. The software accepts up to 200 active records;
the measured model study uses nine-fact worlds, so accuracy at 200 is unmeasured.

## Compare against the actual previous handbook

For an existing user question, press **Compare**. By default it generates without
correction context and with current context. Those are two new generations of
the same checkpoint, not two weight versions.

In the dialog choose **Approved memory at a revision**, enter **9**, and press
**Run comparison**. The left uses the original approved handbook; the right uses
the current one. Comparing the contact question should expose the change from
Mara Voss to Orin Dane. Both sides receive matched pre-question history without
previous correction-conditioned answers leaking between them. The comparison
does not modify memory or overwrite the conversation.

## Inspect conflict controls and roll back

Search **Knowledge** for `Dahlia Relay` or its canonical ID. Facts about another
property or release have different identities. Publishing a contradictory value
for the same canonical key without explicitly replacing the active version is
blocked; the conflict box lets the operator select the existing version to replace.
This is deterministic identity-based conflict control, not a guarantee of
detecting every semantic contradiction in arbitrary text.

To undo the ownership update, press **Roll back to previous version** on its
active card. One transaction withdraws the replacement and restores Saffron at
revision 11; both events share that revision, with no exposed empty state.
Alternatively, withdraw the replacement, turn on **Show inactive**, and restore
the original version manually. These two actions advance to revisions 11 and 12.
Asking again
should use Saffron/Mara. Restoring while a competing version is still active is
rejected; no silent priority rule chooses one. Expand **Approval and rollback
history** to inspect all events. Old messages remain historical evidence and
are not rewritten as if they were newly generated.

## Save work and inspect the experiment

**Export workspace JSON** saves text conversations, corrections, versions and
events. Keep it privately because it contains your chats. To restore, launch an
empty workspace, choose the JSON, inspect the counts, and explicitly approve
restoration. The server checks the checksum, references, state transitions and
revision history in a transaction. Corrupt input changes nothing; existing work
cannot be overwritten. A checksum proves integrity, not truth or authorship.

**Research results** opens the current controlled text experiment. Development
and final evidence have distinct labels. It includes all outcome counts,
category denominators, required-fact retrieval, preservation, false corrected-
value endorsements, warm latency, scientific figures, SVGs, a CSV and JSON export.
The archived visual study uses a different model and dataset. Its accuracy is
never substituted for the current assistant's performance.

For the completed development study and its limitations, read
[the measured evidence](../reports/reasonke/operations-development-v1/README.md).
For schemas and endpoints see [the API](CHAT_API.md). The local administrator
website is a single-operator research application; it does not provide a hosted
multi-user authorization service.
