# Panel rehearsal: one visual correction

Use the [workbench notebook](../notebooks/03_visual_editing_workbench.ipynb)
and [setup guide](VLKEB_WORKBENCH.md). Start one GPU worker, wait for **Model
ready**, and keep its log. The Colab viewer depends on the connected runtime
and notebook access. It is a private notebook demonstration; this is not a
publicly hosted service. Docker CI establishes CPU packaging, not a GPU demo.

Allow time for model loading and candidate generation before the panel session.
Do not run the evaluation or another full-model worker alongside the workbench.
Keep exported evidence before ending a temporary Colab runtime.

## Fixed example and what the saved evidence actually shows

Select **direct** from **Development image**. Use exactly:

- Question: `What is the name of the city depicted in the image?`
- Desired short answer: `Albany, New York`
- Editing method: **Last-layer fine-tuning (15 steps, adapted baseline)**.

The target is the authors' counterfactual benchmark label. This fixture comes
from **TRAIN** and may have been seen during MEND training. It is not a newly
verified fact or an unseen test case. FT performs 15 AdamW updates at `0.0002`
on layer 31's fc1/fc2 weights using FP32 master optimizer copies.

The [saved T4 FT smoke](../reports/vlkeb/t4-ft-last-layer-1/run/smoke.json)
passed eight mechanical edit/persistence/restoration checks and recorded:

| Query | Saved edited answer | Interpretation |
|---|---|---|
| direct | `Albany, New York` | Exact target match |
| alternate image | `Albany, New York` | Exact target match on the other photograph |
| paraphrase | `Albany` | Incomplete under the fixed whole-answer rule |
| neighbor | Repeated `Albany, Albany, …` | Unrelated answer changed; a harmful side effect |

These are observations from one saved run, not promised live outputs or a
general success rate. Read the actual new outputs even when they differ.

## Rehearse the operator workflow

1. Click **Generate current answer**. Show the selected image/source and actual
   base output. The saved run's direct base answer was `detroit, michigan`;
   do not paste it into the live output or treat the target as factual proof.
2. Click **Preview correction**. Wait for completion. Show before/candidate
   answers, target-match warning, and the fixed unrelated probe's before/after
   answers. Explain the neighbor damage. Check **Exact preview restoration:
   passed**: the saved candidate is inactive and the base weights are restored.
3. Open **Patch history** and inspect this candidate's method, image hash,
   patch hash, parameter hashes and probe record. Download its evidence JSON.
   Downloading the parameter patch is optional and approximately 600 MiB.
4. Explain that a maintainer could leave this harmful candidate inactive. For
   the bounded activation/rollback demonstration, deliberately click **Approve
   and activate** and acknowledge the warning. Approval records an operator
   decision; it does not certify correctness or erase the side effect.
5. Generate the direct question again. Select **alternate image**, then
   **paraphrase**, then **neighbor**, generating each current answer. Their
   fixture questions are supplied automatically. Changing a query clears stale
   displayed answers; its old base answer is not reused as a new comparison.
6. Show the one-active-patch restriction: **Preview correction** and further
   approval controls are disabled. In the API rehearsal, attempting to activate
   a candidate while one is active must return HTTP `409`. No patch stacking is
   demonstrated. Keep the session token private.
7. Click **Roll back active edit**. Inspect the exact-restoration result and
   `rolled_back` history record. Return to **direct** and generate again to
   inspect the restored base behavior. Download the final evidence JSON.

## Report, restart and closing claim

Open **Evaluation** and click **Refresh evaluation**. Show only the configured
local `evaluation.json`, its completion counts, denominators, abstentions/errors,
locality and measured latency. Download that report. The default directory is
`artifacts/vlkeb/eval20`; use `--evaluation-dir` at startup if the actual report
is elsewhere. An absent report must display no numerical scores. Author-reported
paper numbers and this demonstration cannot substitute for the final experiment.

During a separate restart rehearsal, retain the same history directory and
restart only the known owning process after it stops. Wait for **Model ready**.
A fresh process loads original model weights and has no active patch; a record
left active before restart becomes `inactive_after_restart`. It is never silently
reapplied. Show retained history and generate the original query again.

Say: “We integrated a published editor and an adapted FT baseline with an
auditable preview, approval and exact rollback workflow. This example shows
both a successful target change and unrelated damage.” Add benchmark numbers
only from the completed saved report. Record the live GPU/browser rehearsal
separately; the historical smoke alone does not establish that deployment check.
