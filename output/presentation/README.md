# Five speaker EditGuard presentation

The deck has ten slides for an approximately eight to nine minute talk. Each
member presents two consecutive slides. A live application demonstration can
add about two minutes after slide 9.

- `EditGuard_Presentation.pptx` is the portable PowerPoint backup of the live
  presentation. Its charts and text remain editable. Each slide's notes contain
  the exact speech and relevant source references.
- `EditGuard_Speaker_Script.docx` is the printable nine-page rehearsal script.
  It includes speaking order, timings, exact speech, handovers, an optional demo,
  short panel answers and the contribution allocation at the end.
- `../../docs/presentation/SPEAKER_SCRIPT.md` is the versioned script source.

Use Member 1 through Member 5 until the team assigns names. Presentation order
and contribution ownership are separate. Confirm actual implementation, review,
test reproduction and documentation before recording each person's contribution.

## Narrative

1. Project and goal
2. A connected ownership correction
3. Model and approved external memory
4. Approval, conflicts, history and rollback
5. Independent controlled datasets
6. Fair comparisons and fixed scoring
7. Original study findings
8. Separate depth findings and latency cost
9. Application workflow and recorded acceptance
10. Bounded conclusion and project contribution

The perfect observed scores remain explicitly bounded. The deck retains the
full-memory failures and baseline abstentions, separates both studies and the
website workload, and attributes the final checkpoints' training upstream.

## Authoring

`../../scripts/build_presentation_session.mjs` exports a `build(presentation)`
function. Invoke it on an empty presentation inside an Artifact Session callback
so that edits sync to the live viewer. It reads local screenshots and the script
source, adds two native charts and attaches the exact speech to the slide notes.
Render and review the live slides before exporting the portable backup.

Rebuild the printable script with an isolated Python that has `python-docx`:

```powershell
python -m scripts.build_speaker_script_docx
```

A rebuilt document requires refreshed Word fields and another layout review.
The published copy already contains refreshed page numbers. Both deliverables
use saved evidence and make zero model calls. Neither authoring workflow changes
the frozen experiments, labels or scoring rules.
