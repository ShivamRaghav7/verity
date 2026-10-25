# First real ReasonKE deployment check

This is a **five-question development diagnostic**, not a held-out benchmark,
an estimate of arbitrary-chat accuracy, or a reproduction of the published
95.48% result. All supporting fictional facts were supplied to the model.
There was no retrieval experiment and no training in this run.

The author checkpoint `YukinoKaren/ReasonKE_plus` at
`ceaedc2ae7c8fed6257953587d991ce550db6106` ran on a Tesla T4 with NF4 weights
and FP16 computation. All five final answers matched the exact predefined
targets, with no execution errors or empty answers:

| Predefined question type | Generated final answer | Generation seconds |
| --- | --- | ---: |
| Direct fictional inventor fact | Tovan Vale | 15.02 |
| Independent paraphrase | Tovan Vale | 13.62 |
| Two-hop inventor → birthplace | Lyris | 13.83 |
| Unrelated France capital control | Paris | 11.00 |
| Similar product name boundary | Serin Moss | 15.66 |

The five generations took 69.19 seconds including receipt handling. Download
and model loading took 338.68 seconds in this session. Peak CUDA allocation
during load was 7,071,596,544 bytes; load peak reservation was 7,140,802,560
bytes. Per-question peaks are in the raw receipt. These are PyTorch CUDA
measurements, not a promise about total system VRAM or another computer.

## Evidence and provenance

- `acceptance.json`: raw model outputs, extracted answers, exact matching,
  every assigned denominator, timings, model revision and measured memory.
- `inventory.json`: actual Colab hardware/software and source revisions.
- `model-manifest.json` and `predefined-cases.json`: exact inputs; their SHA-256
  digests were checked against the receipt when exported.
- `live-session.json`: an intermediate snapshot of controlled website checks.
  It is not a second benchmark or the final state of all later interactions.

The inference code loaded from `3bbca199230b5db8ae6deac3bffae0958f1643e7`.
The website was updated separately for Colab's current external domain and
private Host rewriting. The inference modules, model manifest and five inputs
did not change during those website compatibility fixes.

## Observed limitations

Correct final answers did **not** establish faithful explanations. The direct
answer's generated explanation invented that the telescope was named after its
inventor. The similar-name example contained an unsupported intermediate scope
statement despite returning the intended final answer. The generated steps must
therefore remain unverified, and final-answer matching must be reported
separately from reasoning quality.

An additional live counterfactual tried replacing India's capital with Mumbai
inside a clearly labelled synthetic demonstration. Its first source note also
stated the real-world answer, New Delhi. The model returned New Delhi despite
the correction. After replacing that contradictory source note, it returned
Mumbai for the direct question and a paraphrase in a fresh conversation. The
first failure remains in history. This shows sensitivity to supplied context;
it does not establish reliable arbitrary factual overrides.

The live comparison generated New Delhi without supplied correction context
and Mumbai with it. Both sides used the same author-trained checkpoint, not an
unedited foundation-model baseline. See the saved comparison screenshot in
`output/reasonke-2026-10-25/live-comparison.jpg`.

Withdrawal was subsequently observed in the live interface at knowledge revision
3: the same paraphrase returned New Delhi again. The ordinary conversation
request "Explain overfitting in two short sentences" then produced a relevant
two-sentence explanation. These later observations are recorded separately in
`browser-observations.json`; they were not included in the intermediate database
snapshot before the temporary Colab session ended. The saved conversation image
is `output/reasonke-2026-10-25/live-conversation-and-withdrawal.jpg`.

The checkpoint has no model card proving that this exact revision produced the
paper's score. New corrections update external memory, not model weights. The
weights and method were trained upstream and must be credited accordingly.
