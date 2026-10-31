# Verity technical presentation

The current deck has **29 core slides and three optional technical backup slides**. Allow approximately **24–26 minutes**, including transitions, for five speakers. The model identifier is `YukinoKaren/ReasonKE_plus`; the method name is **Reason-KE++**. Runtime corrections change approved external memory while checkpoint weights remain frozen.

## Current materials

- `Verity_Technical_Presentation.pptx`: expanded deck with editable architecture/sequence diagrams, tables and five native scientific charts.
- [Matching five-speaker script](../../docs/presentation/VERITY_SPEAKER_SCRIPT.md): exact speech, detailed rehearsal notes, sources, optional backups, technical Q&A and contribution allocation at the end.
- [Team handbook](../../docs/presentation/VERITY_TEAM_HANDBOOK.md): standalone system, data, findings, operation and limitations.
- [Reason-KE++ verification](../../docs/presentation/REASONKE_PLUS_PLUS_TECHNICAL_NOTES.md): primary-paper and official-code references for learning and identity.
- [Failure diagnosis](analysis/README.md), [audit JSON](analysis/failure-audit.json) and [63-record ledger](analysis/failure-ledger.csv): saved nonsuccesses, zero new inference and unchanged frozen scores.

| Speaker | Core slides | Focus |
| --- | --- | --- |
| Member 1 | 1–6 | Problem, contract, method and Reason-KE++ learning |
| Member 2 | 7–12 | Architecture, transactions and retrieval |
| Member 3 | 13–18 | Serving, data, protocol, scoring and original findings |
| Member 4 | 19–23 | Depth results, failure diagnosis, preservation and latency |
| Member 5 | 24–29 | Product, acceptance, contribution and conclusion |

Backup slides cover PPO, strict mismatch cases and evidence provenance. Speaking roles are independent of contribution ownership. Confirm each person's actual implementation/review/reproduction before attaching names.

## Technical coverage

The revision separates the published SFT/PPO design from runtime editing, traces the actual request/transaction path and links aggregate results to individual failures. Graphs show development support recall, original outcome composition, depth accuracy, locality preservation and latency. Controlled populations, automated review, repeated wording and the perfect score's limits remain explicit. No new training, inference, prompt tuning or test rescoring was performed for this deck.

## Genuine website views

Slides 24–27 show the editor, source/history view, recorded three-hop chat and saved-results explorer. [Capture provenance](screenshots/technical-v2/provenance.json) records original JPEG hashes, slide mapping and audited receipt source. The worker was unavailable during capture. Saved responses and restored state are labelled as recorded evidence.

[The replay utility](../../scripts/prepare_verity_screenshot_workspace.py) audits the actual India receipt and restores message text, timestamps, revisions, identities and generation provenance into a fresh isolated workspace. It creates no extra accuracy observations.

## Authoring and verification

[Canonical slide content](../../docs/presentation/verity-technical-content.json) drives speaker notes and the Markdown script. Run `python -m scripts.render_verity_speaker_script` to regenerate the script. [The native builder](../../scripts/build_verity_technical.mjs) exports `build(presentation)` for the bundled `@oai/artifact-tool` runtime and reads released metrics directly. This authoring dependency is separate from the lightweight application environment.

Every slide is rendered and checked for readability, directions, data, images and note/script agreement. Native chart workbooks contain numeric snapshots derived from the receipts, rounded to twelve significant digits for Excel compatibility. Displayed labels use fewer digits. Original measurements remain unchanged. Package/import checks alone do not establish target-application rendering; the manifest distinguishes them from actual PowerPoint inspection.

Reproduce failure analysis with `python -m scripts.analyze_presentation_failures --output NEW_DIRECTORY`. Existing destinations are refused. Frozen sources, raw outputs and protocols stay intact. The original and depth studies remain separate populations.

## Preserved versions

The preceding twelve-slide package is in [backups/438aa6c-before-technical-rework/](backups/438aa6c-before-technical-rework/) with byte-for-byte copies and a hash manifest. The earlier `EditGuard_Presentation.pptx` remains. The user-edited historical Word script has not been replaced.
