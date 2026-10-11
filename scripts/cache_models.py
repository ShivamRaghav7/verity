"""Download pinned public checkpoints into the project's isolated cache."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def selected_models(config, auxiliary, selection):
    """Resolve a finite download list before importing optional ML packages."""
    roles = {"base": ["base"], "embedding": ["embedding"], "both": ["base", "embedding"],
             "scope": ["scope"], "all": ["base", "embedding", "scope", "answer"]}[selection]
    models = []
    by_identity = {}
    for role in roles:
        if role in ("base", "embedding"):
            model_id, revision = config[f"{role}_model_id"], config[f"{role}_revision"]
        else:
            model_id, revision = auxiliary[f"{role}_candidate"], auxiliary["model_revisions"][role]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Checkpoint model identifiers must be nonempty strings")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Checkpoint revisions must be exact commit hashes")
        identity = (model_id, revision)
        if identity in by_identity:
            by_identity[identity]["roles"].append(role)
            continue
        record = {"roles": [role], "model_id": model_id, "revision": revision,
                  "loader": "seq2seq" if role in ("base", "answer") else "encoder"}
        models.append(record)
        by_identity[identity] = record
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference.json")
    parser.add_argument("--aux-config", type=Path, default=ROOT / "configs/models.json")
    parser.add_argument("--model", choices=["base", "embedding", "both", "scope", "all"], default="both",
                        help="all caches both frozen and auxiliary backbones; duplicate identities download once")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        auxiliary = json.loads(args.aux_config.read_text(encoding="utf-8")) if args.model in ("scope", "all") else {}
        models = selected_models(config, auxiliary, args.model)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
    for item in models:
        model_id, revision = item["model_id"], item["revision"]
        options = {"revision": revision, "cache_dir": config["cache_dir"], "local_files_only": False}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
        loader = AutoModelForSeq2SeqLM if item["loader"] == "seq2seq" else AutoModel
        model = loader.from_pretrained(model_id, **options)
        print(json.dumps({"status": "cached", "roles": item["roles"], "model_id": model_id, "revision": revision,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "tokenizer": type(tokenizer).__name__}), flush=True)
        del model, tokenizer


if __name__ == "__main__":
    main()
