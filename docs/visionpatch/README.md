# VisionPatch: a beginner's guide

VisionPatch repairs a specific mistake inside an image-recognition model: a printed word can outweigh the object shown in the photograph. The bounded study is complete: it includes a development pilot, fixed-configuration held-out evaluation, rollback checks, and a working local demonstration. It is a research prototype, not a generally reliable image assistant.

The original customer-support EditGuard version remains preserved separately. This visual project does not reuse its factual-answer scores or claim that its trained language models perform visual recognition.

## What the model does

The pretrained model is CLIP ViT-B/32. It converts an image into a numerical representation and compares it with representations of fixed text labels. We give it eight candidates: airplane, camera, cup, elephant, grand piano, laptop, watch, and cell phone. It selects the candidate with the highest similarity. It must choose from these labels even if an uploaded picture contains something else.

An encoder extracts the image representation. A final projection matrix transforms that representation into the space used for comparing images with text. We freeze both encoders and change only that projection. A patch is the saved numerical change plus its original parameter snapshot and provenance. It is not a handwritten table mapping uploaded photographs to answers.

Our controlled mistake uses the word **camera** printed on a white label. The task is to recognize the main object rather than the misleading word. A matching white label without text helps reveal errors caused by covering the object instead of reading the word.

## What we did, and why

1. **Established the research question.** We chose a documented visual model-editing problem with observable class predictions. The method is inspired by associative-memory editing; we do not claim a new editing algorithm or a faithful reproduction of the original classifier-editing paper.
2. **Pinned the model and source data.** We recorded a CLIP checkpoint revision, the official Caltech101 archive checksum, and source attribution. Pinning means a later run can request the same version rather than silently using changed inputs.
3. **Separated source photographs before rendering variants.** We selected 144 originals across eight classes. Each class has two support, eight retention, and eight probe photographs. An exact image hash identifies its source bytes. An original and all its variants stay together, preventing the same source photograph from teaching and testing the patch.
4. **Built the correction examples.** Four support photographs, two cups and two laptops, each have printed-label and blank-label versions. We teach the changed projection to make the printed version's representation resemble the matched blank version. The correction does not require memorizing the true labels of probe photographs.
5. **Added a retention reference.** Sixty-four different clean photographs define representation directions that the patch should disturb less. This is a penalty encouraging preservation, not a mathematical guarantee of unchanged predictions.
6. **Profiled and ran the experiment.** An earlier GPU forward-pass profile measured whether the model fit. The actual pilot used CPU when CUDA was unavailable. Its closed-form edit solves a small numerical optimization problem; it does not require a long GPU training run.
7. **Compared fair alternatives.** We ran unchanged CLIP, our projection edit, gradient fine-tuning of the same projection with the same examples, and a privileged masking control that knows where the synthetic label was placed.
8. **Checked transfer and restoration.** Probe images were unseen during fitting; their labels appear at the top rather than the support images' bottom position. Both learned approaches reached 63/64 correct printed-label predictions, compared with 42/64 originally. Rollback restored original parameters and predictions exactly. These are development results, and fine-tuning tied the edit.
9. **Froze and evaluated the final configuration.** We sealed the same learned weights, settings, scoring rules, and an 80-original final manifest before final inference. None of the 144 development image hashes appears in that manifest. Original CLIP recognized 49/80 printed-label images correctly; editing, fine-tuning, and privileged masking each recognized 80/80. All retained 80/80 clean predictions, including ten genuine cameras. This perfect count in a small controlled set is not a guarantee for new photographs. Final evaluation did not train or select another configuration.
10. **Exercised the actual HTTP workflow.** A repeated CPU upload check demonstrated original camera prediction, patched airplane prediction, and exact rollback. Twelve warm requests measured p50 243.25 ms and p95 326.93 ms; the first request took 11.74 s with lazy encoder loading. This is one repeated development image, not a broad recognition or concurrency benchmark.

[RESULTS.md](RESULTS.md) explains every reported denominator and preserves the development error even though the different final images were all recognized correctly after repair. [PROTOCOL.md](PROTOCOL.md) defines the evaluation rules. Source labels are inherited from Caltech101; automated checking is disclosed and is not independent human review.

## Open and navigate the demonstration

From this repository in PowerShell, using the prepared isolated ML environment:

```powershell
.\.venv-ml\Scripts\python.exe -m visionpatch
```

Open [the local demonstration](http://127.0.0.1:8002). Keep the terminal running. Stop it with Ctrl+C. The launcher binds to the local machine; this service has no public-user authentication.

- **Experiment:** inspect the saved development comparison and whether original or patched parameters are active. The metric cards summarize the saved comparison, so activating rollback changes live recognition rather than rewriting historical measurements. Final held-out results are a separate report; the interactive gallery remains the development gallery.
- **Administrator:** check “I approve activating this experimental patch,” then choose **Apply learned patch**. This activates an already fitted patch; it does not retrain the model. The projection digest identifies the current parameter state.
- **Image gallery:** choose misleading printed label, blank label control, or original photograph. Filter by category and use **Show more photographs**. “Source” is the dataset label; “Original” and “Patched” are model predictions. Inspect genuine cameras as well as repaired cups and laptops.
- **Rollback:** restore the saved original projection and refresh the gallery. Exact restoration means the original parameter bytes return; it does not mean every original prediction was correct.
- **Try a photograph:** upload a JPEG, PNG, or WebP, at most 5 MiB decoded and 20 megapixels. Prefer one main object from the eight candidates. Optionally add the experimental camera label. This runs the actual frozen encoder, while the prepared gallery reuses cached encoder features. First-upload loading adds latency. The preview shows the original upload; the selected transformation is applied on the server. No upload has an independently established correct answer.
- **Patch history:** see session activation and rollback events. History is currently in memory and resets when the process restarts.
- **Export experiment JSON:** open machine-readable status, model metadata, and the pilot report. Interactive API documentation is available at [local API docs](http://127.0.0.1:8002/docs).

Scores are similarities between fixed candidates, not calibrated probabilities that an answer is correct. The application does not generate captions or answer unrestricted image questions.

## Reproduce the development experiment

The prepared `.venv-ml` already contains the measured runtime. Dependency declarations are in `requirements-vision.txt`; preserve the existing environment when reproducing its recorded results. A different machine or runtime needs its own measurements.

```powershell
.\.venv-ml\Scripts\python.exe -m visionpatch.prepare --allow-download
.\.venv-ml\Scripts\python.exe -m visionpatch.profile --device cpu --allow-download --output reports/visionpatch/encoder-profile-repeat.json
.\.venv-ml\Scripts\python.exe -m visionpatch.pilot --device cpu --artifacts artifacts/visionpatch/development-repeat --report reports/visionpatch/pilot-repeat.json
.\.venv-ml\Scripts\python.exe -m unittest tests.test_visionpatch_api -v
```

The first two commands may download the checksum-checked source archive and pinned model. Repeat output paths must be unused: prior reports are retained. The original app reads `development-v1` artifacts and `pilot-v1.json`; a separate repeat is an independent comparison, not an automatic replacement of the demonstrated patch.

The profile stops after two warmup and eight measured batches. The pilot extracts the finite manifest's features, fits one closed-form patch, takes exactly 25 fine-tuning updates, evaluates development probes, and checks rollback. Existing matching feature caches can be reused. There is no optimizer resume for the 25-update pilot; restart a failed repeat in a new output directory. The closed-form solve has no training checkpoint sequence.

Local photographs and model weights are ignored by Git; a repository clone alone does not include those artifacts. Relevant files are `data/generated/visionpatch/development-v1/manifest.json`, `artifacts/visionpatch/development-v1/`, and `reports/visionpatch/pilot-v1.json`.

The completed final record is `reports/visionpatch/final-v1.json`, bound to `reports/visionpatch/final-seal-v1.json` and `data/generated/visionpatch/final-v1/manifest.json`. The implementation separates preparation/sealing from inference:

```powershell
.\.venv-ml\Scripts\python.exe -m visionpatch.finalize --device cpu --prepare-only
.\.venv-ml\Scripts\python.exe -m visionpatch.finalize --device cpu
```

These describe the recorded final workflow; existing final evidence is protected rather than overwritten. A fresh-machine reproduction must first recreate the trained artifacts and their exact bound hashes. Use new `--dataset`, `--artifacts`, `--seal`, and `--report` paths for an explicitly labeled reproduction; the same final photographs are already observed and must not become a fresh tuning set.

The small controlled study is finished. New fonts, other distractor words, natural text-heavy photographs, arbitrary candidate vocabularies, and deployment conditions would be optional future studies, not unfinished claims hidden inside these results. None is established by the present measurements.
