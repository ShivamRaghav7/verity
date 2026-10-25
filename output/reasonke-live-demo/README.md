# Live website demonstration — 25 October 2026

This folder records an assistant-operated, user-visible demonstration through
the actual website controls after the user closed the previous Colab session.
These are observed UI results and screenshots, not a raw backend receipt,
held-out benchmark, or new general accuracy measurement.

The source-pinned Colab launcher at `f6983b695ec2796de49026ac7e6361e51d9fbee9`
loaded application source `cb5fb1442e4b7c198ffc85309c772c6a5ad2f0cc` and the
author checkpoint `YukinoKaren/ReasonKE_plus`, revision
`ceaedc2ae7c8fed6257953587d991ce550db6106`, on a fresh T4 runtime. No training
or per-edit weight update ran. The demonstrator used mouse clicks, form input,
the website's Compare control and its withdrawal/history controls.

| Step | Observed model output or application behavior |
| --- | --- |
| Normal conversation: explain overfitting in two sentences | Generated a relevant explanation about fitting training noise and poor generalization |
| Ask India's capital with no active corrections | New Delhi |
| Approve Mumbai in a labelled synthetic demonstration | One active correction at revision 1 |
| Repeat the direct question | Mumbai |
| Ask “Which city serves as India's capital?” in a fresh chat | Mumbai |
| Compare that paraphrase using the website | New Delhi without correction context; Mumbai with approved knowledge |
| Withdraw the synthetic India correction | Zero active corrections at revision 2; inactive history retains the record and Restore version control |
| Repeat the paraphrase after withdrawal | New Delhi |
| Approve two fictional catalogue facts | MiraQuay telescope inventor = Tovan Vale; Tovan Vale birth city = Lyris; two active facts at revision 4 |
| Ask “In which city was the inventor of the MiraQuay telescope born?” in a fresh chat | Lyris, with both supplied fact records displayed |

The factual demonstration uses a counterfactual target: India's actual capital
is New Delhi. Mumbai was withdrawn. The two fictional catalogue facts remain
active for the user to try. Source and scope fields identify their fictional
origin. No assertion about arbitrary paraphrases or arbitrary multi-hop edits
follows from these selected examples.

The screenshots show real generated answers and actual UI state. Both Compare
sides use the same author-trained checkpoint; only supplied correction context
changes. Displaying a source does not verify every generated sentence.

- `01-normal-chat.jpg`: ordinary generated conversation.
- `02-original-answer.jpg`: the original factual response.
- `03-approved-correction.jpg`: active synthetic correction and provenance.
- `04-live-comparison.jpg`: actual generated New Delhi / Mumbai comparison.
- `05-withdrawn-history.jpg`: inactive record, zero active facts, restore control.
- `06-after-withdrawal.jpg`: restored New Delhi answer at revision 2.
- `07-two-hop-answer.jpg`: Lyris with the two approved fictional facts.

The website URL is temporary and depends on the Colab runtime remaining
connected. See [the walkthrough](../../docs/REASONKE_DEMO.md) for relaunch and
backup instructions. Separate machine-readable deployment and acceptance
evidence remains in [the earlier receipt](../../reports/reasonke/t4-restart-1/README.md).
