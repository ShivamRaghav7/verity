# Exact original demonstration data

`controlled-v1-source.zip` preserves the eleven original files byte for byte.
The companion manifest binds its archive and source hashes. Bundling and the
reconstruction check did not interpret held-out question contents.

This archive contains fictional generated data with pending source review flags.
It is not itself eligible training or final-evaluation data. A separate final
protocol and audited packet establish the final-only workflow, with automated
review and no human-review claim.

From a clean checkout, reconstruct the ignored original directory once:

```powershell
.\.venv\Scripts\python.exe scripts\final_data_bundle.py restore --archive data\final\controlled-v1-source.zip --output data\generated\controlled-v1
```

The tool refuses existing directories, escaping/extra/duplicate archive entries
and any manifest or source-file hash mismatch. Do not regenerate the original
files with a newer formatter or overwrite the archive to change final questions.
