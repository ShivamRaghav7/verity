# Reason-KE++: method, deployed identity and verification notes

This note supports the expanded Verity presentation. It separates the published learning design from this project's inference deployment. The final platform uses `YukinoKaren/ReasonKE_plus`, pinned at `ceaedc2ae7c8fed6257953587d991ce550db6106`. Corrections change approved external memory, not checkpoint weights. Verity did not execute this checkpoint's SFT/PPO training.

The method name in the presentation is **Reason-KE++**. The precise model identifier stays visible in configuration/evidence. The release does not provide a model-card association between this exact revision and the paper's particular headline experiment, so we do not claim to reproduce its reported score. The code below is reference code, not a complete training-provenance certificate for the weights.

## Sources verified

- [Primary Reason-KE++ paper, Wu et al., Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1137/), especially Sections 3.1–3.4 and Appendix A.
- [Official repository](https://github.com/YukinoshitaKaren/Reason-KE/tree/d515c6ac5ec747029484af65103a6270fa693058), inspected at immutable commit `d515c6ac5ec747029484af65103a6270fa693058`.
- [Teacher-data generator](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/data_preprocessing/generate_sft_data.py).
- [SFT trainer](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py) and [launcher](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh).
- [Stage-aware reward](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py) and [PPO launcher](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh).
- [Our pinned deployment manifest](../../configs/reasonke-model.json) and [existing integration record](../REASONKE_INTEGRATION.md).

## Learning design and code-level checks

The paper describes cold-start SFT followed by PPO with process and final-answer signals. Its Appendix A reports 718 curated cold-start examples and an 8,296/922 PPO partition. These are reference-study dataset counts, not Verity data. The model emits acknowledgment, decomposition and action stages inside a reasoning block, followed by a final answer. Intermediate-step checks supply training feedback; they do not certify explanations during deployment. [Primary paper](https://aclanthology.org/2026.findings-acl.1137/)

The data generator constructs an edit from `requested_rewrite`, pairs it with `portability.New Question`, and requests a teacher response using `gpt-4o-mini`. The SFT trainer loads a prepared dataset and uses TRL. It attempts completion-only assistant loss, but has a general language-model collator fallback. That fallback is a reproduction caveat rather than evidence of which path trained the released weights. [Generator](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/data_preprocessing/generate_sft_data.py), [trainer](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.py)

The inspected reward code rejects invalid tags/structure with −1. For valid output it adds final token F1 and a process score with hop, decomposition and boxed-answer components. The decomposition implementation compares aligned action-section subquestions using MiniLM cosine similarity when hop count matches. Subanswers use exact or alias matches. A combined response score with component diagnostics is not an independently verified reward for every generated token. [Reward implementation](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py)

The SFT launcher specifies 10 epochs and learning rate `1e-5`. The PPO launcher uses a critic, vLLM rollouts, actor learning rate `1e-6`, critic learning rate `1e-5`, KL coefficient `0.001` and 15 epochs. Both launchers describe eight-GPU reference runs. These are published settings, not measured Verity training runs or an estimate for our T4 inference configuration. [SFT launcher](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/train/sft.sh), [PPO launcher](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/reasonke.sh)

### Reward scaling difference

The paper writes a normalized blend `alpha * process + (1-alpha) * outcome`, with `alpha=0.5` as its main setting. The pinned code returns `process + outcome`. The sum preserves equal component weighting but doubles that normalized scale, potentially changing its relation to KL. The backup slide and Q&A disclose this difference. We do not silently claim exact agreement or a faithful training reproduction. [Paper](https://aclanthology.org/2026.findings-acl.1137/), [code](https://github.com/YukinoshitaKaren/Reason-KE/blob/d515c6ac5ec747029484af65103a6270fa693058/ReasonKE%2B%2B/verl/verl/utils/reward_score/my_reward.py)

## What the presentation does and does not claim

The training design explains why this model family is relevant. Verity's experimental claims come only from its independent released operations/depth ledgers. There are no new training-loss curves, PPO rewards or weight-edit measurements in this presentation. The exact checkpoint ancestry relative to the chosen official Qwen comparison is unverified, so this is a matched system comparison rather than a causal ablation of SFT versus PPO.

Our implemented adaptations include a T4 inference loader, approved versioned memory, a bounded lexical context policy, iterative answer-text expansion for the depth study, revision-compatible conversation history and audited persistence. These are documented in the [architecture](../CURRENT_ARCHITECTURE.md), [handbook](VERITY_TEAM_HANDBOOK.md) and [matching script](VERITY_SPEAKER_SCRIPT.md). The new [failure audit](../../output/presentation/analysis/failure-audit.json) describes saved outputs without changing any frozen score.

Attribution stays concise in the oral script on slide 5 and in the technical references. Slide faces explain mechanisms and evidence rather than repeatedly restating credit. Contribution ownership at the end of the script describes implemented areas and asks team members to confirm actual work before assigning names.
