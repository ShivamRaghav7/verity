# India-themed demonstration handbook

The website's starter example now uses familiar Indian city names and invented
service/team/person assignments. **Kaveri Payment Gateway, its teams, people and
ownership changes are fictional.** The city labels make the example relatable;
they do not assert a real company's operations or a government policy.

This display example is separate from the completed operations-v1 benchmark.
Changing it does not rename or rescore any of the 1,312 recorded model requests.

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

These targets are controlled facts, not a promise that a generation will match.
Inspect actual answers and supplied facts. The existing bounded retriever may
omit a required third-hop fact; a failure can come from retrieval or generation.
Never substitute the table's deterministic target for the model's response.

## API compatibility

- `GET /api/examples/india` returns the display-only India definition without
  changing knowledge.
- `POST /api/examples/india/approve` uses the administrator workflow and refuses
  nonempty knowledge history.
- Legacy `/api/examples/operations` and `/api/examples/operations/approve` retain
  the original Dahlia example for reproducing the recorded operational study.

The frontend defaults to India. The old benchmark and its operational receipts
remain tied to the original example, rather than retroactively relabelled.
