"""Prepare an isolated, pinned VLKEB environment without changing notebook Python.

The default invocation prints a plan and makes no filesystem/network changes.
Only --fetch-upstream fetches code. Only --install downloads an isolated Python,
CUDA dependencies, and the small uv bootstrap tool. No model weights are fetched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/VLKEB/VLKEB.git"
UPSTREAM_REVISION = "10951b7b3788928f578b73f07eac9e1eaa0316f3"
PYTHON_VERSION = "3.11.9"
UV_VERSION = "0.9.5"
TORCH_INDEX = "https://download.pytorch.org/whl/cu117"
ENV_MARKER = ".editguard-vlkeb-environment.json"


def run(command: list[str], *, env: dict[str, str] | None = None,
        capture: bool = False, timeout: int = 1800) -> str:
    result = subprocess.run(command, check=True, text=True, env=env,
                            capture_output=capture, timeout=timeout)
    return result.stdout.strip() if capture else ""


def verify_upstream(path: Path) -> dict[str, str]:
    if not (path / ".git").is_dir():
        raise ValueError(f"Not an isolated Git checkout: {path}")
    revision = run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Upstream checkout must be {UPSTREAM_REVISION}; found {revision}")
    dirty = run(["git", "-C", str(path), "status", "--porcelain"], capture=True)
    if dirty:
        raise ValueError("Upstream checkout contains changes/untracked files; refusing to overwrite it")
    return {"url": UPSTREAM_URL, "revision": revision, "path": str(path)}


def fetch_upstream(path: Path) -> dict[str, str]:
    if path.exists():
        return verify_upstream(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "--quiet", str(path)])
    run(["git", "-C", str(path), "remote", "add", "origin", UPSTREAM_URL])
    run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", UPSTREAM_REVISION])
    run(["git", "-C", str(path), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    return verify_upstream(path)


def uv_wheel_suffix(system: str, machine: str) -> str:
    if machine.lower() not in {"x86_64", "amd64"}:
        raise ValueError("This reproduction bootstrap currently supports x86_64 only")
    if system == "Linux":
        return "py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    if system == "Windows":
        return "py3-none-win_amd64.whl"
    raise ValueError(f"Unsupported uv tool platform: {system}")


def extract_uv(wheel: Path, target: Path, system: str) -> Path:
    """Extract only the binary, checking member identity instead of unpacking paths."""
    filename = "uv.exe" if system == "Windows" else "uv"
    with zipfile.ZipFile(wheel) as archive:
        # PyPI uv wheels place their executables under the wheel's scripts path.
        members = [item for item in archive.infolist()
                   if item.filename.endswith("/scripts/" + filename)]
        if len(members) != 1:
            raise ValueError("Expected exactly one uv executable in the official wheel")
        binary = target / filename
        binary.write_bytes(archive.read(members[0]))
    binary.chmod(0o755)
    return binary


def ensure_uv(tool_dir: Path) -> tuple[Path, dict[str, str]]:
    """Use only this task's tool; never install into notebook/site packages."""
    system = platform.system()
    suffix = uv_wheel_suffix(system, platform.machine())
    tool_dir.mkdir(parents=True, exist_ok=True)
    binary = tool_dir / ("uv.exe" if system == "Windows" else "uv")
    receipt = tool_dir / "uv-wheel.json"
    if binary.exists():
        if not receipt.is_file():
            raise ValueError(f"Existing uv has no bootstrap receipt: {binary}")
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        if metadata.get("version") != UV_VERSION:
            raise ValueError("Existing uv receipt has a different version")
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if metadata.get("binary_sha256") != digest:
            raise ValueError("Existing uv executable does not match its receipt")
        return binary, metadata
    with urllib.request.urlopen(f"https://pypi.org/pypi/uv/{UV_VERSION}/json", timeout=60) as response:
        package = json.load(response)
    candidates = [item for item in package["urls"] if item["filename"].endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError("Cannot identify the pinned official uv wheel for this platform")
    release = candidates[0]
    wheel = tool_dir / release["filename"]
    expected_digest = release["digests"]["sha256"]
    if not wheel.exists() or hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_digest:
        with urllib.request.urlopen(release["url"], timeout=60) as response, wheel.open("wb") as output:
            shutil.copyfileobj(response, output)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("Official uv wheel failed its PyPI SHA-256 check")
    binary = extract_uv(wheel, tool_dir, system)
    metadata = {"version": UV_VERSION, "url": release["url"], "wheel_sha256": expected_digest,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if run([str(binary), "--version"], capture=True).split()[:2] != ["uv", UV_VERSION]:
        raise ValueError("Downloaded uv executable reports an unexpected version")
    return binary, metadata


def validate_environment_target(path: Path) -> None:
    if path == ROOT or path == Path(sys.prefix).resolve():
        raise ValueError("Refusing to replace the project/current Python environment")
    if path.exists():
        marker = path / ENV_MARKER
        if not marker.is_file():
            raise ValueError(f"Existing target is not owned by this bootstrap: {path}")
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("kind") != "editguard-vlkeb" or data.get("python") != PYTHON_VERSION:
            raise ValueError("Existing environment marker is incompatible")


def install_environment(path: Path, tool_dir: Path, requirements: Path) -> dict:
    if platform.system() != "Linux" or platform.machine().lower() != "x86_64":
        raise ValueError("--install targets Linux x86_64 (Colab); the existing Windows environment is untouched")
    validate_environment_target(path)
    uv, tool_receipt = ensure_uv(tool_dir)
    isolated = dict(os.environ)
    isolated.update({"UV_PYTHON_INSTALL_DIR": str(tool_dir / "python"),
                     "UV_CACHE_DIR": str(tool_dir / "cache"), "UV_NO_MODIFY_PATH": "1"})
    print("Preparing the isolated Python interpreter...", flush=True)
    run([str(uv), "python", "install", PYTHON_VERSION], env=isolated)
    marker = {"kind": "editguard-vlkeb", "python": PYTHON_VERSION}
    if not path.exists():
        run([str(uv), "venv", "--python", PYTHON_VERSION, str(path)], env=isolated)
        (path / ENV_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    python = path / "bin" / "python"
    exact_version = run([str(python), "-c", "import platform; print(platform.python_version())"], capture=True)
    if exact_version != PYTHON_VERSION:
        raise ValueError(f"Expected Python {PYTHON_VERSION}; found {exact_version}")
    # Explicit public indexes and exact CUDA local-version pins avoid inheriting
    # whatever torch happens to be installed in the changing Colab notebook image.
    print("Installing pinned dependencies into the isolated environment...", flush=True)
    run([str(uv), "pip", "install", "--quiet", "--python", str(python),
         "--index-url", "https://pypi.org/simple", "--extra-index-url", TORCH_INDEX,
         "--index-strategy", "unsafe-best-match", "--requirement", str(requirements)], env=isolated)
    print("Checking dependencies and imports...", flush=True)
    run([str(uv), "pip", "check", "--python", str(python)], env=isolated)
    check = '''import importlib.metadata as m, json, platform, torch, torchvision, transformers, higher, timm, omegaconf, iopath, fairscale, pandas, accelerate
print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"torchvision":torchvision.__version__,"transformers":transformers.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'''
    imports = json.loads(run([str(python), "-c", check], capture=True))
    freeze = run([str(uv), "pip", "freeze", "--python", str(python)], capture=True, env=isolated)
    (path / "resolved-requirements.txt").write_text(freeze + "\n", encoding="utf-8")
    result = {"status": "dependencies_installed", "utc": datetime.now(timezone.utc).isoformat(),
              "environment": str(path), "python_executable": str(python), "imports": imports,
              "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
              "uv": tool_receipt, "model_weights_downloaded": False,
              "full_upstream_import_tested": False,
              "limitations": "Dependency imports are not evidence of a successful model load or edit. Run the separate upstream bridge import check before weights."}
    (path / "bootstrap-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_plan(args: argparse.Namespace) -> dict:
    return {"status": "plan_only" if not (args.install or args.fetch_upstream) else "explicit_setup_requested",
            "upstream": {"url": UPSTREAM_URL, "revision": UPSTREAM_REVISION,
                         "directory": str(args.upstream_dir)},
            "python": PYTHON_VERSION, "uv": UV_VERSION, "environment": str(args.env_dir),
            "requirements": str(ROOT / "requirements-vlkeb.txt"), "torch_index": TORCH_INDEX,
            "fetch_upstream_requested": args.fetch_upstream, "install_requested": args.install,
            "model_weights_downloaded": False,
            "stopping_condition": "Pinned code fetched and/or isolated dependency imports checked; no training or editing starts",
            "linux_cuda_install_tested": False,
            "note": "--install can download several GB of CUDA packages, plus isolated Python. No setup runtime/memory claim has been measured."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=ROOT / "tmp" / "vlkeb-upstream")
    parser.add_argument("--env-dir", type=Path, default=ROOT / "artifacts" / "vlkeb" / "environment")
    parser.add_argument("--tool-dir", type=Path, default=ROOT / "artifacts" / "vlkeb" / "tools")
    parser.add_argument("--fetch-upstream", action="store_true", help="Fetch exact pinned code; refuses changes to an existing checkout")
    parser.add_argument("--install", action="store_true", help="Install isolated Linux Python/CUDA dependencies; never installs models")
    args = parser.parse_args(argv)
    args.upstream_dir = args.upstream_dir.resolve()
    args.env_dir = args.env_dir.resolve()
    args.tool_dir = args.tool_dir.resolve()
    print(json.dumps(build_plan(args), indent=2))
    try:
        if args.fetch_upstream:
            print(json.dumps({"upstream_verified": fetch_upstream(args.upstream_dir)}, indent=2))
        if args.install:
            print(json.dumps(install_environment(args.env_dir, args.tool_dir, ROOT / "requirements-vlkeb.txt"), indent=2))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"VLKEB setup stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
