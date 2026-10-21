"""Launch the isolated local VisionPatch demo on loopback port 8002."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    import torch
    import uvicorn

    torch.set_num_threads(4)
    uvicorn.run("visionpatch.app:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
