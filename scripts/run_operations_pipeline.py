"""Execute the two model jobs sequentially, preserving bounded logs and receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("profile", "development", "final"), required=True)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/operations"))
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.phase == "final" and args.protocol is None:
        parser.error("Final stage requires --protocol")
    args.directory.mkdir(parents=True, exist_ok=True)
    for model in ("reasonke", "instruction"):
        output = args.directory / f"{args.phase}-{model}.json"
        command = [sys.executable, "-u", "-m", "editguard_benchmark.run", "--model", model,
                   "--phase", args.phase, "--output", str(output), "--resume"]
        if args.protocol:
            command += ["--protocol", str(args.protocol)]
        print(json.dumps({"starting_model": model, "phase": args.phase, "output": str(output)}), flush=True)
        with (args.directory / f"{args.phase}-{model}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=31000)
        if result.returncode:
            raise SystemExit(f"{model} job stopped; inspect its preserved receipt and log. No second model was loaded.")
    print("Both sequential jobs finished. Raw outcomes are preserved; this message does not imply correct answers.", flush=True)


if __name__ == "__main__":
    main()
