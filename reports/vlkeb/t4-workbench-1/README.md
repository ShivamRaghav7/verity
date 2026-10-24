# Real T4 application rehearsal

This is our measured application lifecycle evidence, separate from the sealed
20-edit / 500-answer comparison. The application ran at source commit
`e0324c39e545e8222ed99ca6148bdebf9c790465` on a Tesla T4. The scientific sources
still matched the pre-inference seal.

All **15 recorded lifecycle checks passed**. They cover inactive candidate
previews, explicit operator approval, rejection of a conflicting activation,
JSON export, exact restoration of all six editable tensors, clean shutdown of
the owned server, and a fresh server retaining history without automatically
reactivating a patch. The final captured state was model-ready with no active
patch. A temporary Colab runtime may subsequently expire.

The fixed development fixture comes from the author's training annotations.
It is not an additional held-out result. The FT preview generated the target
`Albany, New York`, while the MEND preview generated `depton, texas`. FT also
changed the one unrelated probe into a repeated answer. Correct lifecycle
behavior is not a claim of correct or safe arbitrary model edits.

| File | Meaning |
| --- | --- |
| [workflow-evidence.json](workflow-evidence.json) | Fifteen assertions, observed preview answers, export digest, restart and conflict receipts. |
| [application-snapshots.json](application-snapshots.json) | Raw status/history snapshots from seven stages, with session tokens excluded. |
| [browser-observations.json](browser-observations.json) | Eleven browser observations of the operator workflow. |
| [exported-while-active.json](exported-while-active.json) | Exact JSON bytes exported while the FT patch was active; this is a historical state. |
| [host-verification.json](host-verification.json) | Independent local byte/arithmetic checks of the transferred records, not another GPU run or human factual review. |
| [context/first-server.log](context/first-server.log) | First owned server's execution log. |
| [context/restarted-server.log](context/restarted-server.log) | Fresh server's execution log. |

The evidence transfer archive was 13,140 bytes, with SHA256
`957925d08a275de95a74e3d807944809f281b04129cf83aa32707938a57d4e4d`.
Model weights, parameter-patch binaries and upload bytes are excluded from Git.
Screenshots are in
[the application proof folder](../../../output/vlkeb-workbench-2026-10-24/).

The restart-to-observed-ready duration includes polling delay and must not be
used as a benchmark model-load time. The rehearsal establishes neither
independent human factual review, production multi-user security, permanent
hosting nor GPU execution inside Docker.
