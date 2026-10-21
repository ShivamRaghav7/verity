# VisionPatch: Targeted Repair of Visual Misidentification

Status: the first development pilot is complete; see [measured results](RESULTS.md) and the [beginner's guide](README.md). This proposal defines the bounded research direction, not a final validation claim. The preserved EditGuard V1 is a separate project version.

## The problem and intended user

A pretrained image classifier can recognize an object correctly, then change its prediction when an unrelated printed word is placed in the photograph. For a model maintainer, this is a recurring model defect: the model gives excessive influence to text that should not determine the object's identity.

VisionPatch investigates whether a small parameter edit can reduce that defect on unfamiliar photographs while retaining useful recognition. The immediate user is a researcher or maintainer diagnosing a deployed visual model. A future inventory or image-search application is a motivating example, not a validated commercial deployment.

The first task is deliberately specific: classify the main object in a single-object image, even when a synthetic label names another candidate object. We do not claim to solve arbitrary visual understanding, document reading, captions, medical recognition, or malicious-image security.

## Why this is model editing

The correction changes a saved model parameter matrix. The patched model is evaluated on new images without retrieving a matching correction or receiving the true object label. The experiment asks whether a targeted repair generalizes and how much unrelated behavior it changes.

This fits the broader literature on visual model editing. Santurkar and colleagues studied internal prediction-rule rewrites, including a CLIP typographic example. Our initial method instead edits CLIP's final visual projection using paired representations and a retention penalty. It is an adaptation, not a faithful reproduction of their spatial, concept-localized method, a new algorithm claim, or a reproduction of SERAC. [Primary paper](https://arxiv.org/abs/2112.01008), [authors' illustrated explanation](https://gradientscience.org/editing/).

This first implementation is a visual classifier. CLIP compares image and text representations; it does not generate explanatory answers. A captioning or question-answering model would require a separate integration and evaluation. [CLIP paper](https://arxiv.org/abs/2103.00020).

## Smallest architecture

1. Pin a pretrained CLIP ViT-B/32 checkpoint and preprocessing revision. Freeze its image encoder, text encoder, class prompts, and scoring rule.
2. Extract pre-projection visual features from support pairs: an image with a misleading printed label and the same image with an identically positioned blank label.
3. Fit a regularized, low-rank additive update to `visual_projection`. Encourage attacked features to recover the matched blank-label representation, while penalizing changes on a separate clean retention set. The matched target keeps label occlusion fixed while changing its text.
4. Save the update, parent checkpoint digest, data manifest, hyperparameters, and measurements. Apply the same patch to every inference image; there is no true-class-based routing.
5. Compare original and patched predictions, inspect failures, activate a patch, and restore the exact parent weights through rollback.

The patch is learned from examples through an explicit optimization objective. It is not handwritten answer replacement. A local projection fine-tuning baseline receives the same support and retention examples.

## Data and fair comparisons

Begin with a bounded subset of the [official Caltech101 record](https://data.caltech.edu/records/mzrjq-6wc02). Its [official API metadata](https://data.caltech.edu/api/records/mzrjq-6wc02) declares CC BY 4.0; preserve creator credit, source, and transformation details. Download images into an ignored local directory rather than committing them. Exact sample counts and split hashes belong in the generated manifest. Record source labels and audit limitations without claiming independent human review.

Create misleading-text and blank-label variants only after splitting original source images. The initial vocabulary is airplane, camera, cup, elephant, grand piano, laptop, watch, and cell phone. The printed distractor is `camera`, and paired cup and laptop support images fit the edit. Use Pillow's default font, with bottom-position support labels and top-position probe labels. This tests position transfer, not unfamiliar-font generalization. Keep genuine camera images in evaluation: suppressing the distractor score everywhere is a failure, not a successful repair. The development manifest contains 144 originals: two support, eight retention, and eight probe images per class. Only four of the 16 support slots are used to fit the patch.

Compare unchanged CLIP, targeted projection editing, equal-example projection fine-tuning, and an oracle blank-region masking control. The control knows the synthetic overlay location and therefore has privileged information; report it as such. Text-bearing images are a separate challenge because ignoring all written text may destroy useful behavior.

## Finite feasibility gate

The first 24–48 hours resolve whether this model, dataset subset, and edit location support a useful experiment. Measure clean recognition, actual label-induced failures, edit transfer, collateral errors, latency, and memory. Run development data only. Do not choose classes or methods using final-test results.

Proceed only if there are enough observable failures to evaluate repair and the implementation runs within measured hardware limits. A result where fine-tuning or masking works better remains a valid finding. If the gate fails, document the cause and reconsider the edit location once; do not grow the project indefinitely.

Success means a reproducible, bounded repair experiment with clear tradeoffs, an inspectable application, and honest reporting. It does not require perfect accuracy or superiority over every baseline.
