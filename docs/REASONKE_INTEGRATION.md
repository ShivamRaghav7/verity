# Released ReasonKE model integration

The implementation choice is the author's public `YukinoKaren/ReasonKE_plus`
checkpoint, revision `ceaedc2ae7c8fed6257953587d991ce550db6106`. Its first real
T4 deployment generated matching final answers for five predefined fixtures;
this is an inference adaptation, not a claim of a completed paper reproduction.
The previous website and all its reports remain preserved at Git commit
`1e4bba9dd6d805d158f5bffef3f69607ab4939d1` on
`codex/chat-knowledge-studio`. This integration is a separate branch.

## What editing means here

An administrator publishes a replacement fact in versioned external memory.
The pretrained, edit-conditioned reasoner receives the active approved facts
with the question. It was trained upstream to use updated facts and perform
multi-hop reasoning. Publishing or withdrawing a fact changes the supplied
memory; it does not perform a new weight update or train a model in our project.
Compare runs the same checkpoint with and without supplied corrections. Neither
side is a claim to reproduce the original pre-SFT Qwen model.

The initial deployment supplies all active corrections, with finite limits.
This avoids inheriting the earlier website's uncalibrated similarity cutoff,
but applicability and final answer quality still depend on the learned model.
The application refuses oversized fact memory rather than silently dropping
records. Scope, source, approval, conflict handling and withdrawal remain
application controls; they do not certify arbitrary generated prose.

## What the upstream release establishes

- [Official code](https://github.com/YukinoshitaKaren/Reason-KE), inspected at
  `d515c6ac5ec747029484af65103a6270fa693058`, contains Reason-KE and Reason-KE++.
- [The paper](https://arxiv.org/abs/2511.12661) reports 95.48% in its multi-hop
  MQuAKE-CF-3k setting. This is a published result, not our measurement or a
  universal/current ranking across all knowledge editing tasks.
- [The author checkpoint](https://huggingface.co/YukinoKaren/ReasonKE_plus)
  is public and contains 7,615,616,512 parameters. The source safetensors total
  15,231,233,024 bytes. Its model card does not explicitly associate this exact
  revision with the reported score and supplies no checkpoint license metadata.
  The upstream code's Apache-2.0 license alone does not establish weight terms.
- The released evaluator counts a case successful when any question variant
  succeeds and retrieves using supplied edit facts. Those conditions must be
  disclosed; they are not per-question accuracy or real chat retrieval quality.
- The full SFT/PPO training configuration uses eight GPUs. Reusing released
  weights avoids attempting that training on a single T4.

## Deployment adaptation and stopping conditions

Use an isolated Colab environment; do not change existing local ML environments.
The loader uses pinned safetensors, disables remote model code, uses NF4 weights
with FP16 computation, and does not retain unnecessary hidden states or scores.
These are adaptations whose memory, latency and answer quality need actual
measurement. Four-bit storage is not evidence that full inference fits the GPU.
The download still transfers the original approximately 15.23 GB of weights.

The first check answers: **Can this released checkpoint load on the available
GPU and generate useful corrected answers in our actual request format?**

Inputs are the pinned model manifest and five predefined fixtures in
`configs/reasonke-acceptance.json`. Expected artifacts are a JSON receipt with
raw generations, per-question exact-match diagnostics, versions, elapsed time,
and peak CUDA allocation/reservation. Every fixture, including failures and
empty answers, stays in the receipt. The run ends after five generations or an
explicit exception; there is no training, optimizer or indefinite retry loop.
The initial real T4 run measured 338.68 seconds for download/loading and 69.19
seconds for the five-generation diagnostic. Load peak reserved CUDA memory was
7,140,802,560 bytes. These are session observations, not estimates for another
run. See [the receipt and limitations](../reports/reasonke/t4-development-1/README.md).
Interrupted model downloads can reuse the Hugging Face cache. Completed fixture
records must remain inspectable. No training checkpoint/resume claim applies.

These five development fixtures are not a representative accuracy estimate,
unseen test set, or proof of arbitrary multi-hop reasoning. No paper metric is
inserted into the application's measured-results dashboard.

The [cold-restart and website receipt](../reports/reasonke/t4-restart-1/README.md)
records a second T4 session and the complete publish, paraphrase, compare and
withdraw workflow. Its six factual checks matched; its seventh ordinary-chat
check measured only nonempty generation. The [demo walkthrough](REASONKE_DEMO.md)
explains the interface, model contribution, evidence, and how to save the database.

## Serving

The worker binds to loopback and protects generation/status with an ephemeral
bearer token. Its health route reveals no model output or secret. The existing
website selects this backend explicitly; it never silently substitutes Ollama
when the worker fails. Both services can run inside the same Colab session and
the website can be opened with Colab's authenticated port viewer. This requires
no public tunnel or Google Drive mount. Runtime storage is temporary; export
evidence and approved memory before the session ends.

Model loading and live generation have separate real evidence in the linked
T4 receipt. HTTP and simulated-client tests remain software-contract evidence.
The small diagnostic and live comparison do not establish unrestricted
conversation accuracy, arbitrary edit reliability or faithful explanations.

Open the [source-pinned Colab notebook](https://colab.research.google.com/github/ShivamRaghav7/verity/blob/f6983b695ec2796de49026ac7e6361e51d9fbee9/notebooks/04_reasonke_chat.ipynb)
or upload `notebooks/04_reasonke_chat.ipynb` to Colab. Select T4 and run the
hardware, setup and launch cells, then inspect the progress cell until the
worker is ready. The website cell can also wait for readiness. An optional cell
runs `scripts.verify_reasonke_live` against isolated memory, preserves all seven
assigned outcomes, and withdraws only its own correction. This covers actual
generation through the complete website API, separate from software fixtures.
The notebook's full source SHA is recorded inside it; keep the Colab runtime
running while using the website. Its URL expires when the runtime ends.

The website's **Research results** tab preserves the previous visual-model
study. It does not contain accuracy numbers for ReasonKE. Use the linked
ReasonKE deployment receipts for this backend until its broader evaluation is
completed; the two protocols must not be mixed.
