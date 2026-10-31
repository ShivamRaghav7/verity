# Verity mid-project presentation

Current deliverable: **13 slides, exactly 10:00, five two-minute speaking slots.**

- [Presentation](Verity_Mid_Project_Presentation.pptx): editable diagrams, one table, two native charts, genuine sourced website captures and detailed notes.
- [Timed five-speaker script](../../docs/presentation/VERITY_SPEAKER_SCRIPT.md): exact speech, clock marks, rehearsal cues, Q&A and evidence-based responsibility allocation.
- [Updated study guide](../../docs/presentation/VERITY_SLIDE_BY_SLIDE_DEFENCE_GUIDE.md): explains every slide, mechanisms, choices, findings and likely panel questions.
- [Next research milestone](../../docs/presentation/VERITY_MIDPROJECT_RESEARCH_ROADMAP.md): reviewed data, profiled SFT/PPO adaptation, matched training-stage comparisons and fresh broader evaluation.
- [Release manifest](midproject/manifest.json): package, assets, evidence hashes and verification boundaries.

## Story and timing

| Speaker | Slides | Clock | Focus |
| --- | --- | --- | --- |
| Member 1 | 1–2 | 00:00–02:00 | Objective and verified Air India handover example |
| Member 2 | 3–4 | 02:00–04:00 | Implemented architecture and connected retrieval |
| Member 3 | 5–8 | 04:00–06:00 | Approval, history, conflicts, rollback and recorded walkthrough |
| Member 4 | 9–11 | 06:00–08:00 | Controlled experiments, measured findings and latency |
| Member 5 | 12–13 | 08:00–10:00 | Engineering conclusion and next milestone |

PowerPoint slide advance timings total 600 seconds and match the script. Use the recorded walkthrough within the clock; fresh GPU inference is optional after the talk. Rehearse with a timer. PowerPoint's Use Timings setting controls whether the native schedule is followed.

The completed architecture is governed retrieval augmented generation with frozen generators. The reused `YukinoKaren/ReasonKE_plus` identity is stated in configuration notes. Model-training adaptation is the next milestone. The main deck omits the detailed upstream training tutorial.

## Sourced real website demonstration

The real example concerns the **27 January 2022 Air India handover**, with earlier government ownership retained in historical scope. The selected local snapshot advances from December 2021 to the handover. The Tata Sons contact-city record is from its official page checked on 31 October 2026; it is not a claim about a historical 2022 office. Government and Tata sources are recorded in [the fixture](../../data/demos/air-india-handover.json).

A fresh isolated workspace ran seven actual model requests through the application API on a Colab T4. All seven matched the declared demonstration targets. A competing publication returned HTTP 409 without changing the revision; rollback restored the predecessor; the handover was reapplied for the active demo. These are selected product acceptance cases, **not a new held-out real-world accuracy benchmark**.

- [Complete receipt and exported state snapshots](midproject/air-india-demo-receipt.json).
- [Original screenshot provenance](midproject/screenshots/provenance.json).
- [Official photo provenance](midproject/assets/photo-provenance.json).
- Preparation: `python -m scripts.prepare_sourced_demo --output NEW_WORKSPACE`.
- Finite recording: `python -m scripts.run_sourced_demo --base-url LOCAL_WEBSITE --manifest NEW_WORKSPACE/sourced-demo-manifest.json --output NEW_RECEIPT.json`.

The editor screenshot is a prepared draft; the recorded API workflow performs approval. Chat and history show actual persisted generation and events. Original screenshot pixels remain unaltered; slide annotations are separate editable text. Stored responses are labelled recorded and are not described as fresh generation.

## Experimental evidence

The unchanged original and depth studies use controlled fictional operations handbooks. Main conditions have 256 assigned questions each. The deck compares actual generator/context configurations and preserves abstentions in the denominator. Full-memory ReasonKE_plus failures and Qwen abstentions remain visible. The perfect retrieved scores are scoped to small controlled worlds and wording.

[Failure diagnosis](analysis/README.md), [audit JSON](analysis/failure-audit.json), and [63-record ledger](analysis/failure-ledger.csv) retain the individual nonsuccesses. No benchmark inference, test rescoring or training was performed for this revision. The seven sourced demonstration calls are separate.

## Authoring and checks

[Canonical content](../../docs/presentation/verity-midproject-content.json) drives speaker notes and script. Regenerate the script with `python -m scripts.render_verity_midproject_script`. [The native builder](../../scripts/build_verity_midproject.mjs) exports `build(presentation)` for the bundled first-party Artifact Tool runtime and reads released metrics directly. [The timing utility](../../scripts/time_verity_midproject.py) applies the declared native PowerPoint schedule to a fresh exported package.

Charts have embedded editable numeric workbook snapshots derived from unchanged receipts. Every slide is rendered and visually reviewed. Structural, font, geometry, chart/workbook and first-party import checks are recorded separately from actual PowerPoint inspection. Rendering/structure checks alone do not prove factual validity.

## Preserved versions

[The immediate backup](backups/72ff983-before-midproject/) contains the old technical deck, content, script, guide, handbook, builders and hash manifest. `Verity_Technical_Presentation.pptx` also remains at its original path as an older edition. Its upstream tutorial is not part of the current mid-project deck. Earlier backups remain in place. The user-edited historical Word script was not replaced.
