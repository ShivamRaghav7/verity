"""Launch the isolated local VisionPatch demo on loopback port 8002."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--legacy-api", action="store_true", help="Enable archived V1 API for reproduction only")
    args = parser.parse_args()
    # The prepared demonstration must not wait for Hub metadata or telemetry.
    # Explicit checkpoint downloads belong to the separate preparation commands.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.legacy_api:
        os.environ["VISIONPATCH_ENABLE_LEGACY_API"] = "1"
    import torch
    import uvicorn

    torch.set_num_threads(4)
    uvicorn.run("visionpatch.app:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
