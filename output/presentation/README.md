# Verity technical presentation

The current deck has twelve slides for approximately eleven minutes of speech,
plus an optional two-minute live demonstration after slide 12. It retains the
five-speaker format and uses the new Verity name.

## Current materials

- `Verity_Technical_Presentation.pptx`: portable backup of the revised live deck.
  Text, three technical tables and two scientific charts remain native and
  editable. The chart data workbooks contain snapshots of the displayed values.
  Slides 6 and 12 include genuine Verity website screenshots of the correction
  editor and a recorded three-hop response after rollback.
- [Verity speaker script](../../docs/presentation/VERITY_SPEAKER_SCRIPT.md): exact
  speech, timings, visual cues, handovers, optional demo and contribution map.
- [Verity team handbook](../../docs/presentation/VERITY_TEAM_HANDBOOK.md): a
  standalone technical onboarding reference covering the method, implementation,
  datasets, scoring, results, limitations, operation, glossary and panel Q&A.

| Speaker | Slides | Focus |
| --- | --- | --- |
| Member 1 | 1–2 | Problem and edit contract |
| Member 2 | 3–5 | Architecture, model configuration and retrieval |
| Member 3 | 6–7 | Transaction invariants and data construction |
| Member 4 | 8–10 | Experimental protocol and measured findings |
| Member 5 | 11–12 | Failure analysis, product evidence and conclusion |

Speaking assignments are independent of contribution ownership. Confirm each
member's actual work before assigning names to the contribution allocation.

## Technical coverage

1. Verity and the bounded goal
2. The edit contract and three-hop dependency
3. Inference architecture and evidence boundary
4. Frozen checkpoints, quantization and measured hardware
5. Two-stage versus iterative answer-frontier retrieval
6. Canonical keys, conflict rejection and atomic rollback
7. Lineage-first data splits and label limitations
8. Frozen comparisons, oracle diagnostics and endorsed-answer scoring
9. Original study, full-memory failures and locality measures
10. Depth results, paired uncertainty and latency
11. Failure analysis and the limits of 100%
12. Website acceptance and the project's contribution

## Authoring and evidence

`../../scripts/build_verity_presentation_session.mjs` exports
`build(presentation)`. Invoke it inside an Artifact Session callback on an empty
presentation. It reads the Markdown script and attaches the speech and evidence
references to every slide's notes. The live slides were rendered and inspected
before exporting the portable backup. Package checks additionally verify the
native chart values against their workbook snapshots. Native PowerPoint opening
is a separate application check and is not claimed by those validations.

No training or inference ran to create the revision. The documented evidence
replay commands and software regression commands were checked on the CPU.
Protocol-bound source, labels, raw outputs and scoring remain unchanged.

## Website screenshot provenance

The current frontend displays **Verity**. Historical Python package names,
workspace schemas and frozen receipts retain their EditGuard identities.
Website captures display approved facts, saved results and actual recorded India
acceptance responses in the current interface. The isolated workspace comes from
[scripts/prepare_verity_screenshot_workspace.py](../../scripts/prepare_verity_screenshot_workspace.py),
which audits the published receipt and preserves response text, timestamps,
memory revisions and generation provenance. Recorded chat titles identify the
replay. This creates zero model calls and no additional accuracy observations.
A displayed saved response does not establish live model readiness. Captions and
speaker notes must distinguish recorded output from a fresh neural demonstration.

The full-resolution [knowledge editor](screenshots/verity-knowledge-editor.jpg),
[recorded chat](screenshots/verity-recorded-chat.jpg) and supplementary
[research results](screenshots/verity-research-results.jpg) captures are available
separately. [Capture provenance](screenshots/provenance.json) records their hashes,
receipt source and offline worker status.

## Earlier introductory version

`EditGuard_Presentation.pptx`, `EditGuard_Speaker_Script.docx` and
`presentation-manifest.json` preserve the earlier ten-slide introductory package.
Use the Verity material above for the technical presentation. Existing reports,
source modules and saved UI screenshots can retain the historical EditGuard
name without changing their measured result or protocol identity.
