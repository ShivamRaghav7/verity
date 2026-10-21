"""Download/checksum and prepare a finite development-only image manifest."""

import argparse
import json
from pathlib import Path

from .data import download_source, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--archive", type=Path, default=Path("artifacts/visionpatch/caltech-101.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/visionpatch/development-v1"))
    args = parser.parse_args()
    download_source(args.archive, allow_download=args.allow_download)
    manifest = prepare_dataset(args.archive, args.output)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "images": len(manifest["images"]),
                      "manifest": str(args.output / "manifest.json"), "final_test": False}, indent=2))


if __name__ == "__main__":
    main()
