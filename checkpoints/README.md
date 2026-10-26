# Paused evaluation checkpoint

The evaluation was cleanly interrupted at the user's request on 26 October 2026.
All background completion threads stopped; no second model or website check was
started. The current in-flight question may need executing again because it had
no saved completed answer. Completed correct, wrong, abstained and error records
are never retried.

`editguard-paused-evaluation.zip` is downloaded here on the laptop, outside
temporary Colab storage. It contains all profiles/development receipts, the frozen
protocol and the interrupted final ReasonKE receipt. Its SHA-256 is:

`b684760aba108aea917338475f1ff242fc464c5e8fd9c8fb625d5d6d53d0348a`

Every enclosed JSON hash and the frozen protocol digest were verified after
download. **483/784 ReasonKE requests are recorded; 301 ReasonKE and 528 Qwen
requests remain**, followed by the separate sixteen website generations.
The Qwen final job had not started. This is progress accounting, not final accuracy.

## Resume later

Keep the ZIP. Code is already pushed on `codex/reasonke-integration`. In a fresh
T4 Colab runtime, use the pinned setup from notebook 05, upload this ZIP, and
restore it into a new directory using:

```text
python -m scripts.restore_operations_checkpoint /content/editguard-paused-evaluation.zip --directory /content/editguard-reasonke/artifacts/operations-resumed
python -u -m scripts.run_operations_pipeline --phase final --protocol /content/editguard-reasonke/artifacts/operations-resumed/protocol.json --directory /content/editguard-reasonke/artifacts/operations-resumed
```

Use the setup virtual environment's Python executable, not an unrelated Python.
The restore tool verifies the exact frozen source/data/configuration identities
before writing. The pipeline already passes `--resume` to each model runner,
skipping all completed results and running only unexecuted records. Do not run
profiles or development again, refreeze the protocol, tune against partial final
results, or start another GPU worker alongside it.

The original inference source is `e4f82a3e6d25b99464327dded2052719446db028`.
Later application commits retain identical frozen inference-file hashes; the
restore verifier checks those hashes rather than assuming any branch is equivalent.
A new Colab runtime may need to download/reload model weights again. No training
or completed-answer recomputation is required.

After the resumed pipeline completes, use the versioned
`scripts.finalize_operations_release` workflow for integrity/claim audits,
scientific exports, sixteen isolated website generations and a fresh demo.
The prior notebook's background threads are stopped; they do not resume themselves.
