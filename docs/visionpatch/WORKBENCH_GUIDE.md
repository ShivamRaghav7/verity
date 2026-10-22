# VisionPatch: understand the project and demonstrate it

Written for someone starting without machine-learning background. This guide describes the implemented visual project and its recorded v2 experiment, as of 22 October 2026. The website is a local research application. The measurements describe the controlled experiment, not a promise about arbitrary uploaded photographs.

## 1. The project in ordinary language

Imagine a photograph of a cup with a paper label saying **camera**. A person asked to identify the object would usually say **cup**. An image-recognition model can instead be distracted by the written word and choose **camera**.

VisionPatch lets an administrator show the model that mistake and supply a matching photograph with the misleading word removed. It fits a small change to the model's numerical parameters. It then checks whether the correction helped and what other predictions changed. The administrator can approve the candidate, activate it, test the current model on an image, and restore an earlier version.

The research question is more demanding than “can we repair one image?” It is:

> Can a targeted edit reduce a misleading word's influence, generalize to other photographs and layouts, preserve useful object and text recognition, and coexist with later edits?

The answer from our experiment is qualified: targeted recognition improved substantially, but repairs did not eliminate every error, some unrelated predictions worsened, and applying edits in different orders affected the outcome. Those findings are part of the project.

The model chooses among **eight categories**: airplane, camera, cup, elephant, grand piano, laptop, watch and cell phone. It does not generate captions or answer unrestricted visual questions. If you upload an unsupported object, it still has to choose one of these eight labels. A plausible-looking answer is not evidence that the object belongs to the supported vocabulary.

## 2. Why this is a model-editing project

The original model already contains a learned association between the appearance of written words, pictures and object descriptions. We intervene in that learned association after pretraining by changing model parameters using a small set of correction examples. Subsequent images run through the changed parameters. The application does not look up an uploaded photograph in a table of manually written answers.

The most precise description is **visual model editing of a learned image–text association**. It connects to knowledge-editing research through targeted post-training intervention, generalization, preservation, sequential interference and reversibility. It does not insert a new factual proposition such as “a company changed its policy.” It does not demonstrate arbitrary visual factual knowledge editing, a novel algorithm, or faithful reproduction of a named editing method. A panel should hear this distinction clearly.

We previously built the separate customer-support **EditGuard** project. This visual project changes the problem and architecture. We do not present EditGuard's two trained auxiliary language models or its language-answer scores as components or results of VisionPatch. Both projects remain available as separate work.

## 3. The small amount of machine learning you need to know

| Term | Meaning here | Why it matters |
|---|---|---|
| Model | A program whose numerical parameters were learned from examples | Its predictions can be wrong even when the program runs correctly |
| Pretrained | Trained previously, before our experiment | We start with CLIP rather than training a vision model from nothing |
| CLIP | A model that represents images and text so their similarity can be compared | It supplies the original image and text understanding |
| Encoder | The part that turns an image or text into a list of numbers | Both encoders stay fixed in our work |
| Feature / representation / embedding | A numerical description produced inside the model | Similarity is computed using these numbers, not human-readable rules |
| Projection | A learned matrix that transforms one representation into another | We edit CLIP's final image projection |
| Parameter / weight | A number stored inside the model | A patch changes actual weights |
| Frozen | Left unchanged during our fitting | Most of CLIP remains exactly the same |
| Patch | A saved parameter update, its parent snapshot and associated metadata | It can be inspected, activated and rolled back |
| Fine-tuning | Updating parameters using repeated optimization steps | It is a comparison method that also changes the image projection |
| Generalization | Working on examples that did not teach the correction | A repaired teaching image alone does not establish this |
| Retention / preservation | Keeping useful behavior after editing | A targeted repair should not silently damage other abilities |
| Interference | One edit changes the effect of another | It appears when we apply several edits sequentially |
| Ground truth | The answer against which a prediction is checked | In the study this comes from source categories or the word we rendered |
| Baseline | A comparison method | Improvement needs a clear reference and fair conditions |
| Ablation | Removing one component for comparison | Object-only editing tests what happens without explicit text protection |
| Inference | Running an already fitted model to obtain a prediction | Uploading a photograph for recognition does not train a new patch |

For one prediction, the image encoder produces 768 numbers. The image projection transforms them into 512 numbers. Frozen text representations describe the eight candidate labels. We compare the projected image with those text representations and return the most similar candidate.

The editable matrix has 512 × 768 = **393,216 parameters**. That is a small part of the pretrained model. A study edit uses four correction pairs and stores a low-rank update, with rank at most four. The interactive single-pair workflow has rank at most one. “Low rank” means the change is represented using a limited number of directions; it does not guarantee that the behavioral effect is small.

```text
Photograph → frozen image encoder → image projection → similarity to eight text candidates
                                        ↑
                              only this matrix is edited

Eight candidate descriptions → frozen text encoder and text projection
```

The displayed scores are relative similarity scores. They are **not calibrated probabilities that the prediction is correct**. A high winning score can accompany a wrong answer.

## 4. What happened, in order: what, why and how

The order below is a development timeline, not a claim that every step required a separate day.

### Step 1 — Preserve the earlier work

**What:** We kept the original EditGuard project and the first VisionPatch experiment available separately.

**Why:** A stronger second study should not erase the earlier implementation or rewrite its measured results.

**How:** Versioned code and saved evidence distinguish the original application, the first visual pilot, and the v2 workbench/study. The pilot website at `/` and the new correction studio at `/workbench` use separate active-patch workflows. Activating a workbench patch does not rewrite the pilot's historical report.

### Step 2 — Establish a controlled visual failure

**What:** We used photographs from Caltech101 and placed a synthetic printed word on a white label.

**Why:** We need to know the intended object and isolate the misleading text's effect.

**How:** Each source can have a clean image, a white label without text, and an otherwise matching label containing a word. The blank label is important: covering part of a photograph can itself make recognition harder. Comparing the printed label with its blank counterpart helps distinguish the text's effect from the effect of covering the object.

The real photographs come from an external, attributed dataset. Our controlled variants, split rules and evaluation tasks form the project-specific benchmark. We did not create all source photographs ourselves, and we do not call source labels independently human reviewed by our team.

### Step 3 — Build and test the first small visual edit

**What:** The initial visual version studied one misleading word, **camera**, using four support pairs and 64 separate clean retention photographs.

**Why:** A small working experiment establishes that the edit changes real predictions before we build a larger interface and study.

**How:** The editor made printed-label representations resemble their matching blank-label representations while penalizing changes in retention directions. On 64 development photographs, original CLIP got 42 printed-label questions correct; the projection edit and projection fine-tuning each got 63 correct. On a separate sealed 80-photograph final set, those learned methods and privileged masking each got 80/80, versus 49/80 for unchanged CLIP.

Those are v1 counts on different images and conditions. They do not imply universal reliability or superiority over fine-tuning. The v1 development error remains documented even though its separate final set had no edited errors.

### Step 4 — Identify what the easy demonstration did not prove

**What:** We asked whether the repair merely makes the model ignore text more generally.

**Why:** Written words can be useful. A model that ignores every word may recognize a cup more easily but become worse when asked to read a legitimate label. Also, one repair may affect another.

**How:** V2 adds three edit words, a controlled text-reading task, held-out fonts and positions, other-word measurements, and all six orders of applying three edits.

### Step 5 — Separate fitting, development choices and final checking

**What:** We prepared 80 fitting originals, 32 new validation originals and 48 new final originals.

**Why:** If the same photograph teaches an edit and tests it, the score may measure memorization. If final results guide repeated setting changes, that final set becomes another development set.

**How:** The 80 fitting photographs reuse only old v1 support and retention data: 16 support originals plus 64 clean retention originals. All 224 previously observed v1 originals are excluded from selecting the new validation and final images. Exact source identifiers and content hashes are checked. Every variant of an original stays in the same split.

Validation is used to choose between declared settings. Final data are used after those choices and weights are sealed. The final command performs inference and scoring, not additional fitting.

Exact duplicate separation is checked. Similar-looking photographs, the same physical object appearing in separate photos, and exposure during CLIP's original pretraining have not been independently ruled out.

### Step 6 — Make the v2 tasks harder in a controlled way

**What:** We use **camera**, **laptop** and **watch** as misleading words.

**Why:** A single word and one label layout are too narrow to test selective behavior meaningfully.

**How:** Training labels use DejaVu Sans at the bottom; validation uses DejaVu Serif at the top; final uses DejaVu Sans Bold at the top and center. The final font and center position do not occur in fitting or validation. Literal word cards use separate template splits and test useful reading with prompts that explicitly ask about the word rather than the object.

### Step 7 — Fit matched methods under a finite plan

**What:** We compare five methods: unchanged CLIP, object-only editing, text-protected editing, projection fine-tuning, and oracle masking.

**Why:** Improvement over an unchanged model is insufficient to show that a particular editing design helps.

**How:** Each independent edit gets four source/reference pairs. The protected method additionally uses 32 training word cards alongside 64 clean retention photos. Fine-tuning receives the same correction and protection information. It takes 25 optimization updates rather than training the entire model.

There are two declared text-protection strengths and two fine-tuning learning rates. Validation selects one setting for each method globally across all three words. The selected protected setting is `protected-text-1`; the selected fine-tuning learning rate is 0.0001. The object-only method is a fixed comparison. The development budget is **33 fits**: 27 closed-form edits and six 25-update fine-tunes, including the sequential study. We did not expand the grid to chase final results.

A closed-form edit solves numerical equations directly instead of taking hundreds of gradient steps. It is still a fitted parameter update learned from examples. Cached feature fitting is fast because the expensive encoder stays frozen. Training time alone is not a measure of project depth.

The **oracle** knows exactly where we placed the synthetic label and which task is being asked. It substitutes the corresponding blank-label result for attacked object tasks and leaves legitimate text-reading inputs unchanged. This is privileged information. Its score does not demonstrate an implemented automatic text detector or a practical learned defense.

### Step 8 — Profile before running and record the actual run

**What:** We measured a small GPU run before the full bounded study.

**Why:** Runtime and memory estimates should come from the available machine rather than a guess.

**How:** The profile timed encoder batches and a small fit. The actual development stage took about **20.98 seconds** and the final evaluation about **19.14 seconds** in the recorded CUDA environment. Final image rendering and feature extraction accounted for about 3.90 seconds. Final peak CUDA allocated memory was 645,448,704 bytes, about **616 MiB**; reserved memory and total system memory are different measurements. These timings exclude environment installation and downloads and are not website response-latency benchmarks.

The profile estimates were lower than the full measured run because short operation timings omit reporting, disk work and other overhead. We retain actual full-run measurements rather than presenting the optimistic estimate as observed time.

### Step 9 — Seal, evaluate and keep every outcome

**What:** We saved hashes binding the chosen artifacts, configuration, inputs and scoring code, then scored the frozen final tasks.

**Why:** A reader should be able to distinguish choices made before final inference from reactions to final results.

**How:** The final report records that final-based selection was not performed. It stores per-example predictions, denominators, grouped uncertainty and all sequential orders. Failed recognition cases remain in the counts. “No training performed” in the final report describes the final evaluation stage; the development stage had already fitted the models.

### Step 10 — Turn the experiment into an auditable application

**What:** The new correction studio fits real candidates from supplied pairs and stores them in a persistent registry.

**Why:** Merely switching a precomputed demonstration patch does not let an administrator author a new correction or handle conflicting model versions.

**How:** Candidate fitting, approval and activation are separate actions. SQLite stores metadata, active lineage and audit events; artifact files store the numerical patch and parent snapshot. Activation checks the expected version. Rollback restores saved parameter bytes. Live recognition reads the registry's active projection.

## 5. Understand the datasets before reading the scores

| Data role | What it is used for | What it cannot prove |
|---|---|---|
| Support pair | A misleading image and its matching blank reference that teach a correction | Generalization to new images |
| Retention reference | Examples used during fitting to discourage damage | Independent preservation accuracy |
| Validation | Held-out examples used to select declared settings | Unbiased final performance after that selection |
| Final | Examples evaluated after settings and weights are sealed | Performance on every real-world image |
| Uploaded live photograph | An image run through original and active projections | Accuracy without independently established ground truth |

There are **48 distinct final source photographs**, six per category. Each produces 13 object inputs: one clean image, plus a blank and a printed-label variant for each of three words at each of two positions. Therefore 48 × 13 = **624 object questions**. Separately, eight words across three final templates produce **24 text questions**. Total: **648 questions**, not 648 independent photographs.

An independent edit's **target-label accuracy** uses 48 photographs × two positions = **96 inputs** for its own word. Its **other-word accuracy** uses the other two words: 48 × two positions × two words = **192 inputs**. Clean accuracy uses the 48 clean photographs once. Text accuracy uses 24 cards.

For example, 91/96 means 91 correct predictions and five wrong predictions. It does not mean 91 edits succeeded. The same underlying photograph appears at two positions, so statistical calculations must keep those observations together.

## 6. Read the actual v2 result

All counts below come from `reports/visionpatch/study-v2/final.json`. Each word's edit in this table is fitted independently from the original projection; this is not yet the three-edit active sequence.

| Edited word | Unchanged target accuracy | Object-only edit | Text-protected edit | Projection fine-tuning | Privileged masking |
|---|---:|---:|---:|---:|---:|
| camera | 55/96 | 92/96 | 91/96 | 92/96 | 96/96 |
| laptop | 31/96 | 88/96 | 77/96 | 86/96 | 96/96 |
| watch | 68/96 | 92/96 | 92/96 | 92/96 | 96/96 |

The protected method improves camera accuracy from 57.29% to 94.79%: an increase of **37.50 percentage points**. Its laptop improvement is from 32.29% to 80.21%, or 47.92 percentage points. These are within this controlled dataset. They do not prove the protected method is best: object-only editing and fine-tuning did better on some target tasks.

All independent models retained **48/48 clean object accuracy**. Controlled text accuracy was **24/24** for protected editing and fine-tuning for all three words. The object-only watch edit reduced text accuracy to **22/24**. This is evidence that protecting text can matter in this example; it is too small and controlled to establish broad reading preservation.

### Accuracy, repair and preservation answer different questions

For the protected camera edit:

- **Target accuracy 91/96:** correct answers across all camera-label object inputs.
- **Repair rate 36/41:** corrected inputs among 41 baseline errors on attacked, non-camera objects whose clean versions were originally correct. The matched blank control is reported separately. This is about 87.80%. Its denominator is not 96.
- **Clean preservation 48/48:** retained correct clean answers among those the original model got right.
- **Text preservation 24/24:** retained correct word-card answers among those the original model got right.
- **Other-word accuracy 90/192:** correct answers on laptop- and watch-label inputs after the camera edit; unchanged CLIP got 99/192 on that same subset.
- **New other-word errors 17/99:** of the 99 other-word answers the baseline got right, 17 became wrong. Some previously wrong answers also improved, so 17 new errors does not imply accuracy simply fell from 99 to 82. The observed final correct count is 90.

Preserving an answer only means it stayed the same and correct on the defined baseline-correct subset. A model that repeats an originally wrong answer has not become factually correct. This is why the report includes both overall accuracy and preservation.

No valid study question is quietly dropped for abstention. This fixed-candidate model does not abstain; it returns one of the eight labels. The report records zero abstentions, not a selective success rate obtained by removing difficult cases.

### What the uncertainty numbers mean

The camera protected edit's paired accuracy difference versus base is 0.375, or +37.50 percentage points. Its recorded grouped bootstrap interval is approximately **+26.04 to +48.96 percentage points**.

The calculation repeatedly resamples the 48 source photographs, keeping a photograph's related variants together, and recomputes the difference on the same questions for both models. It measures sampling variation within this small controlled study. It does not account for every bias, unknown pretraining overlap, new domains or different fonts. Text intervals use only eight word groups, so they are especially limited. An interval of zero text change on these cards does not prove that all text recognition is unaffected.

### What sequential editing shows

There are six orders for three words. Each next edit is fitted on its actual current parent state. For example, camera → laptop → watch means edit camera first, fit laptop from that updated model, and finally fit watch from that parent.

After all three protected edits, camera-label accuracy ranged from **86/96 to 92/96**, laptop-label accuracy from **51/96 to 74/96**, and watch-label accuracy from **90/96 to 92/96**, depending on the order. Every final sequence retained 48/48 clean object and 24/24 controlled text accuracy. All six recorded exact rollback checks passed.

This reveals a limitation: clean-image and simple text preservation can look perfect while earlier attack repairs interfere. The experiment includes all six orders rather than selecting a flattering one using final results. Exact rollback is an engineering property separate from recognition accuracy: the restored original model can still make its original mistakes.

## 7. Start the website

In PowerShell:

```powershell
cd "C:\Users\shiva\OneDrive\Desktop\projekt-KE"
.\scripts\run-visionpatch.ps1
```

Keep that terminal running. Open **http://127.0.0.1:8002/workbench**. Stop the server with Ctrl+C when finished. The script uses the prepared isolated Python environment. The first model-dependent request can take longer because the model is loaded lazily. This does not retrain CLIP.

The prepared local machine includes downloaded model and data artifacts. A Git clone alone does not include the ignored photographs and model weights; fresh setup is described in the project documentation. If a server is already running on port 8002, use that instance or stop the old terminal process before starting another.

The app binds to the local machine. The administrator fields record an asserted identity; they do not authenticate a person. This is a local research tool, not a public multi-user production service.

## 8. Navigate the correction studio

The left menu scrolls to sections of the workbench:

| Menu item | What to do there |
|---|---|
| Pilot experiment | Opens the preserved first visual demonstration at `/` |
| Correction studio | Load a supplied development pair or upload your own matched images |
| Candidate evaluation | Inspect a newly fitted candidate before approving it |
| Live recognition | Compare original and currently active projections on the same image |
| Version history | Inspect saved candidates, approve/activate, rollback and view the audit trail |
| Selection study | Read the separate recorded validation/final experiment and scientific figures |

The top cards show **registry revision**, **active updates** and **active projection fingerprint**. A fingerprint is a cryptographic digest of parameter shape, dtype and exact bytes. It identifies a parameter state; it is not an accuracy score. The revision protects against acting on an outdated version.

### A complete provided-pair demonstration

1. Under **Start with a provided development pair**, choose **cup · misleading “camera”** and click **Load pair**. The two pictures, expected category, printed word, title and source are filled in. Inspect both pictures: the reference should depict the same scene with the misleading word blanked. The provided illustrations deliberately use observed training-support failures so that the demonstration starts with an actual mistake. This selection makes them teaching illustrations, not an unbiased accuracy sample.
2. Enter **Your name / reviewer identifier** and a concrete **Reason for the correction**, such as “The printed camera label distracts recognition from the cup; reduce its effect while preserving the object.” The source and reference are support examples, not independent test images.
3. Before fitting, use **Live recognition → Compare correction source** if you want to record the original/current prediction. This uses the active workbench state. If earlier patches are already active, the current prediction may already be repaired; do not pretend that this is a new baseline failure. Use **Restore base** with a reason when deliberately starting a fresh demo from the original model.
4. Click **Fit & preview candidate**. The server encodes the supplied images, fits a new numerical update on the current parent, evaluates reference examples and saves a candidate. Preview does **not** activate it.
5. In **Candidate evidence**, read the source's before/after labels, reference prediction, clean and text reference counts, new errors, repairs and warnings. A wrong reference prediction, a failed support repair or collateral errors deserves attention. A candidate is not automatically reliable because fitting completed. Use **Export candidate JSON** to keep the full evidence.
6. Fill or check **Administrator / reviewer** and **Reason** in the version registry. Tick **I reviewed the evidence and approve this experimental candidate**, then click **Approve candidate**. Approval records a decision; the active projection remains unchanged.
7. Click **Activate approved patch**. The server checks that its parent and the registry revision still match, then changes the active projection transactionally.
8. Use **Compare correction source**, or upload another supported-category image and click **Compare uploaded photograph**. Original and current projections receive the same encoded image. The expected category entered for authoring is not supplied as the live recognition answer. A newly uploaded image has no independently reviewed ground truth in that request, so inspect the actual picture instead of treating a changed label as proof of correctness.
9. Open **Version history → Audit trail**. Check who proposed, approved and activated the correction, why they did it, and which state is active.
10. Click **Rollback latest** to restore the immediate saved parent. With multiple active edits, **Restore base** removes the entire active chain and restores the original projection. Saved candidates and audit records remain. Run recognition again to observe restored behavior.

Supplied pairs are convenient teaching examples. The independent frozen study supplies generalization evidence. Repeatedly choosing provided pairs until one looks impressive is not a substitute for the study.

### Uploading a new pair

Use **A / Misleading image** for the failure and **B / Matched reference** for the same photograph with the misleading word removed. Acceptable formats are PNG, JPEG and WebP, up to 5 MiB each and 20 megapixels. Images are converted to a 224 × 224 crop, so keep the main object and relevant label visible.

Choose the actual object from the eight supported categories. Supply the word, a title, image provenance, your identifier and the reason. The expected category is used to report whether the support/reference predictions agree with your intention; it does not replace the image-derived training target. An unrelated “correct example” photograph can teach an unintended transformation. The application cannot certify that your pair is semantically valid.

Each interactive candidate uses one pair. The formal study uses four pairs per word and separately selected settings. Do not quote the study's accuracy as a guarantee for a one-pair candidate you just created.

### Conflicts are handled explicitly

Suppose two candidates were fitted from the same active parent. Activating one changes that parent. The other is now based on a different parameter state and displays **Branch conflict**. It cannot silently overwrite or add itself to the new branch. Refresh the registry, review the active state and fit a new candidate from that state if appropriate.

A stale browser tab can also receive a state-conflict message when its expected revision is outdated. The interface refreshes the registry. This is intended protection, not a recognition failure. It does not resolve disagreements about the correct object or the validity of a reference; those require administrator judgment.

### What survives a restart

The workbench registry persists in `runtime/visionpatch/workbench/workbench.sqlite3`; numerical snapshots and patches live in the adjacent `patches` directory. The text reference cache lives in that workbench directory too. Restarting the server reconstructs the active lineage and checks artifact integrity. Back up the complete workbench directory together with the required model/data artifacts; copying only the database omits parameter files.

Do not edit or remove the registry/artifact files to perform a normal rollback. Use the interface. Restore base retains the history. The preserved pilot page has its own earlier session-based activation/history behavior; that is separate from the persistent workbench.

## 9. Understand the charts and exported files

The **Selection study** section contains counts and independent-result tables. The selectivity figure compares target repair, clean recognition and useful text recognition across methods. The interference figure compares sequential application orders. Read the axes and denominators before comparing bar heights. High clean scores alone do not establish that unrelated attacked inputs were preserved.

**Study JSON** opens the recorded result, including per-example outputs and intermediate sequential stages. It describes fitted study artifacts, not the current arbitrary interactive candidate. Activating or rolling back an interactive candidate does not recalculate or change these historical measurements.

Useful repository records:

| File | Purpose |
|---|---|
| `configs/visionpatch-study-v2.json` | Declared finite experiment settings |
| `docs/visionpatch/STUDY_V2.md` | Split, task, fitting and scoring protocol |
| `docs/visionpatch/ARCHITECTURE_V2.md` | Components, numerical edit and registry design |
| `reports/visionpatch/study-v2/profile.json` | Measured preflight and its limitations |
| `reports/visionpatch/study-v2/development.json` | All validation candidates, selection and fitted-stage measurements |
| `reports/visionpatch/study-v2/final-seal.json` | Artifact/input/code fingerprints fixed before final inference |
| `reports/visionpatch/study-v2/final.json` | Frozen final counts, observations, uncertainty and sequential outcomes |
| `reports/visionpatch/final-v1.json` | Separate original visual final result |
| `/docs` on the local website | Interactive API reference for the running service |

Large source images and model weights remain outside Git. Committed reports make measurements inspectable; they are not substitutes for the artifacts needed to rerun inference.

## 10. What “finished” and “failure” mean

A **recognition failure** is a wrong model prediction. A **regression** is previously correct behavior becoming wrong. A **software failure** is broken execution, integrity checking or an unintended state change. A **pending action** may be an administrator's unapproved candidate rather than unfinished model training. These are different things.

The recorded v2 study has completed final inference and scoring. Its wrong predictions remain research findings. We cannot honestly make a completed report have no model failures by deleting difficult examples or calling them successful. Likewise, an unapproved candidate remains a candidate until an administrator deliberately approves it; automatic approval would defeat the workflow.

The workbench's operational checks, including transactions and exact restoration, are tested separately from whether every image is recognized correctly. A robust implementation can reject stale edits correctly while its model remains imperfect. Future improvements require a new protocol and fresh held-out evidence; this observed final set must not become a tuning target presented as unseen again.

The defensible outcome is a runnable parameter-editing platform plus a reproducible study of effectiveness, preservation and interference. It is not a claim of 100% reliable visual understanding, automatic semantic verification, superiority to every baseline, or deployment-ready authentication.

## 11. A short panel explanation you can use

“VisionPatch studies targeted repair of a pretrained vision–language model. CLIP sometimes follows a misleading word printed on a photograph instead of recognizing the object. We freeze its encoders and fit a small change to the final image projection from matched misleading and blank-reference images. We check both the intended repairs and collateral effects, including useful text reading and interference between successive edits.

“Our v2 evaluation holds out 48 source photographs, a font and a placement condition after validation-only configuration selection. The protected camera edit improved target accuracy from 55/96 to 91/96 while retaining the recorded clean and controlled text counts. However, other-word predictions suffered some regressions, and sequential results depended on edit order. We report those limits alongside the gains.

“The application lets an administrator create a real candidate, inspect its evidence, approve and activate it separately, test the actual active parameters and restore saved states. This is a controlled visual association-editing prototype, not a general-purpose image assistant or a claim of a new state-of-the-art algorithm.”

If asked why not simply delete every visible word, explain that some tasks require reading words and that our oracle masking control has information a practical detector would need to recover. If asked why the fitting is quick, explain the frozen encoder, cached features and small direct projection solve. If asked whether all predictions are now correct, show a documented failure and explain what it reveals. Those answers demonstrate understanding more convincingly than a universal-accuracy claim.
