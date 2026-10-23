"""Prepare only pinned VLKEB smoke assets. Default is a network-free plan.

Uses Python's standard library. Does not import torch, deserialize checkpoints,
run models, train anything, fetch final evaluation rows, or install packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def relative_path(root: Path, value: str) -> Path:
    """Reject escaping paths on either Windows or POSIX, including symlinks."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative asset path: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative asset path: {value!r}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes its root: {value!r}")
    return result


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("assets"):
        raise ValueError("Unsupported or empty VLKEB asset manifest")
    names = set()
    for asset in manifest["assets"]:
        relative_path(Path.cwd(), asset["path"])
        if asset["path"] in names:
            raise ValueError(f"Duplicate asset path: {asset['path']}")
        names.add(asset["path"])
        if not asset["url"].startswith("https://"):
            raise ValueError("Pinned asset URLs must use HTTPS")
        if not isinstance(asset.get("size_bytes"), int) or asset["size_bytes"] < 0:
            raise ValueError(f"Invalid byte count: {asset['path']}")
        digest = asset.get("sha256") or asset.get("md5")
        length = 64 if asset.get("sha256") else 32
        if not digest or not re.fullmatch(rf"[0-9a-f]{{{length}}}", digest):
            raise ValueError(f"Missing or invalid pinned checksum: {asset['path']}")
    return manifest


def verify_asset(path: Path, asset: dict) -> dict:
    if path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"Wrong byte count for {asset['path']}")
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False) if asset.get("md5") else None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            sha.update(chunk)
            if md5:
                md5.update(chunk)
    if asset.get("sha256") and sha.hexdigest() != asset["sha256"]:
        raise ValueError(f"SHA256 mismatch for {asset['path']}")
    if md5 and md5.hexdigest() != asset["md5"]:
        raise ValueError(f"Published MD5 mismatch for {asset['path']}")
    return {"path": asset["path"], "size_bytes": path.stat().st_size,
            "sha256": sha.hexdigest(), "upstream_checksum_verified": True,
            "verification_algorithm": "sha256" if asset.get("sha256") else "md5",
            "url": asset.get("url"), "revision": asset.get("revision")}


def download_asset(root: Path, asset: dict, *, opener=urlopen) -> dict:
    """Resume a partial response only after strict Content-Range validation."""
    target = relative_path(root, asset["path"])
    if target.exists():
        return verify_asset(target, asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = relative_path(root, asset["path"] + ".part")
    expected_size = asset["size_bytes"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Oversized partial file: {partial}; move it aside before retrying")
    if expected_size == 0 and not partial.exists():
        partial.touch()
    if offset != expected_size:
        headers = {"User-Agent": "EditGuard-VLKEB-reproduction/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(asset["url"], headers=headers)
        with opener(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status == 206:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or tuple(map(int, match.groups())) != (offset, expected_size - 1, expected_size):
                    raise ValueError(f"Invalid resumed Content-Range for {asset['path']}: {content_range}")
            elif status == 200:
                # A server may ignore Range. Restart rather than append duplicate bytes.
                offset = 0
            else:
                raise ValueError(f"Unexpected HTTP status {status} for {asset['path']}")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != expected_size - offset:
                raise ValueError(f"Unexpected HTTP byte count for {asset['path']}")
            received = offset
            next_progress = received + 64 * CHUNK_BYTES
            with partial.open("ab" if offset else "wb") as handle:
                while chunk := response.read(CHUNK_BYTES):
                    received += len(chunk)
                    if received > expected_size:
                        raise ValueError(f"Response exceeds pinned size for {asset['path']}")
                    handle.write(chunk)
                    if received >= next_progress:
                        print(f"{asset['path']}: {received:,}/{expected_size:,} bytes "
                              f"({100 * received / expected_size:.1f}%)", file=sys.stderr, flush=True)
                        next_progress = received + 64 * CHUNK_BYTES
                handle.flush()
                os.fsync(handle.fileno())
            if received != expected_size:
                raise ValueError(f"Incomplete download for {asset['path']}; rerun to resume {partial}")
    record = verify_asset(partial, asset)
    os.replace(partial, target)
    return record


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"Refusing to write metadata through a symlink: {partial}")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def build_fixture(root: Path, manifest: dict) -> dict:
    """Derive a fixture solely from the hash-checked upstream training JSON."""
    spec = manifest["fixture"]
    source_path = relative_path(root, spec["source_path"])
    source_asset = next(a for a in manifest["assets"] if a["path"] == spec["source_path"])
    verify_asset(source_path, source_asset)
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    row = rows[spec["primary_index"]]
    required = [row["image"], row["image_rephrase"], row["m_loc"]]
    other_image = rows[spec["additional_image_index"]]["image"]
    required = list(dict.fromkeys(required + [other_image]))
    for image in required:
        relative_path(root, image)
    fixture = {"schema_version": 1, "purpose": "development_smoke_only",
               "upstream_split": "train", "source_revision": spec["source_revision"],
               "source_indices": spec["source_indices"],
               "source_train_sha256": source_asset["sha256"],
               "disclosure": spec["disclosure"], "record": row,
               "additional_same_entity_image": other_image,
               "required_images": required, "image_directory": spec["image_directory"]}
    return fixture


def extract_fixture_images(root: Path, manifest: dict, fixture: dict, *,
                           verify_only: bool = False, archive_verified: bool = False) -> list[dict]:
    """Copy only declared regular images after the whole archive is verified.

    No extract/extractall calls; no archive filenames become output paths.
    Duplicate suffix matches and nonregular selected entries are rejected.
    """
    spec = manifest["fixture"]
    archive_path = relative_path(root, spec["archive_path"])
    archive_asset = next(a for a in manifest["assets"] if a["path"] == spec["archive_path"])
    archive_record = ({"sha256": archive_asset["sha256"]} if archive_verified
                      else verify_asset(archive_path, archive_asset))
    image_root = relative_path(root, spec["image_directory"])
    required = set(fixture["required_images"])
    found = {}
    if not verify_only:
        image_root.mkdir(parents=True, exist_ok=True)
    # Read-only archive iteration: full compressed archive must be available.
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name:
                raise ValueError(f"Unsafe archive member: {member.name!r}")
            candidates = [r for r in required if name.as_posix() == r or name.as_posix().endswith("/" + r)]
            if not candidates:
                continue
            if len(candidates) != 1 or candidates[0] in found:
                raise ValueError(f"Ambiguous or duplicate image archive member: {member.name}")
            image = candidates[0]
            if not member.isfile() or member.size > MAX_IMAGE_BYTES:
                raise ValueError(f"Invalid selected image member: {member.name}")
            destination = relative_path(image_root, image)
            partial = relative_path(image_root, image + ".part")
            if not verify_only:
                destination.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            received = 0
            with archive.extractfile(member) as source:
                target = partial.open("wb") if not verify_only else None
                try:
                    while chunk := source.read(CHUNK_BYTES):
                        received += len(chunk)
                        sha.update(chunk)
                        if target:
                            target.write(chunk)
                finally:
                    if target:
                        target.close()
            if received != member.size:
                raise ValueError(f"Truncated image member: {member.name}")
            checksum = sha.hexdigest()
            found[image] = {"path": f"{spec['image_directory']}/{image}",
                            "size_bytes": member.size, "sha256": checksum,
                            "verified_source_archive_sha256": archive_record["sha256"],
                            "archive_member": member.name}
            if verify_only:
                verify_asset(destination, found[image])
    if set(found) != required:
        raise ValueError(f"Required fixture images missing from archive: {sorted(required - found.keys())}")
    for image in (() if verify_only else required):
        destination = relative_path(image_root, image)
        partial = destination.with_name(destination.name + ".part")
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != found[image]["sha256"]:
            raise ValueError(f"Existing image differs from the verified archive: {destination}")
        os.replace(partial, destination)
    return [found[image] for image in fixture["required_images"]]


def verify_assets(manifest: dict, asset_root: Path, *, group: str = "all") -> dict:
    """Network-free preflight API. Never deserialize a checkpoint before success.

    Verifies every selected byte against the pinned manifest. Image checksums are
    recomputed from selected entries of the verified archive, not trusted solely
    because a mutable local fixture JSON claims them.
    """
    root = asset_root.resolve()
    records, missing, errors = [], [], []
    selected = [a for a in manifest["assets"] if group == "all" or a["group"] == group]
    if not selected:
        return {"verified": False, "assets": [], "missing": [],
                "errors": ["No assets selected"], "fixture_prepared": False,
                "reproduction_executed": False}
    for asset in selected:
        try:
            path = relative_path(root, asset["path"])
            if not path.is_file():
                missing.append(asset["path"])
                continue
            records.append(verify_asset(path, asset))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{asset.get('path', '<unknown>')}: {error}")
    fixture_prepared = False
    if group in ("all", "fixture") and not missing and not errors:
        try:
            fixture = build_fixture(root, manifest)
            fixture_path = relative_path(root, manifest["fixture"]["output_path"])
            if not fixture_path.is_file():
                missing.append(manifest["fixture"]["output_path"])
            else:
                saved = json.loads(fixture_path.read_text(encoding="utf-8"))
                for key in fixture:
                    if saved.get(key) != fixture[key]:
                        raise ValueError(f"Prepared fixture metadata mismatch: {key}")
                derived = extract_fixture_images(root, manifest, fixture, verify_only=True,
                                                archive_verified=True)
                if saved.get("image_integrity") != derived:
                    raise ValueError("Fixture image integrity differs from the verified source archive")
                fixture_prepared = True
        except (OSError, ValueError, KeyError, tarfile.TarError) as error:
            errors.append(f"fixture: {error}")
    return {"verified": not missing and not errors, "assets": records,
            "missing": missing, "errors": errors, "fixture_prepared": fixture_prepared,
            "reproduction_executed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs/vlkeb-assets.json")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "artifacts/vlkeb/assets")
    parser.add_argument("--group", choices=("core", "fixture", "all"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Explicitly download selected pinned files and prepare training-only fixture")
    mode.add_argument("--verify", action="store_true", help="Read and checksum local files; never access the network")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        root = args.root.resolve()
        assets = [a for a in manifest["assets"] if args.group == "all" or a["group"] == args.group]
        plan = {"mode": "download" if args.download else "verify" if args.verify else "dry_run",
                "root": str(root), "network_requested": args.download,
                "selected_asset_count": len(assets),
                "selected_download_bytes": sum(a["size_bytes"] for a in assets),
                "selected_download_gib": round(sum(a["size_bytes"] for a in assets) / 1024**3, 3),
                "files": [{"path": a["path"], "size_bytes": a["size_bytes"],
                           "present": relative_path(root, a["path"]).is_file()} for a in assets],
                "disclaimer": "Download sizes are not VRAM, runtime estimates or reproduction evidence. Checkpoint deserialization and model execution are separate steps."}
        if not args.download and not args.verify:
            print(json.dumps(plan, indent=2))
            return 0
        if args.verify:
            result = verify_assets(manifest, root, group=args.group)
            print(json.dumps(result, indent=2))
            return 0 if result["verified"] else 1
        records = []
        for asset in assets:
            print(f"{plan['mode']}: {asset['path']}", file=sys.stderr, flush=True)
            if args.download:
                records.append(download_asset(root, asset))
        fixture_prepared = False
        if args.group in ("all", "fixture"):
            fixture = build_fixture(root, manifest)
            if args.download:
                fixture["image_integrity"] = extract_fixture_images(root, manifest, fixture)
                write_json(relative_path(root, manifest["fixture"]["output_path"]), fixture)
            fixture_prepared = True
        result = {"schema_version": 1, "status": "verified", "mode": plan["mode"],
                  "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "assets": records, "fixture_prepared": fixture_prepared,
                  "reproduction_executed": False, "elapsed_note": "No model or training executed"}
        if args.download:
            write_json(root / "asset-records.json", result)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"Asset preparation stopped: {error}\nPinned files were not substituted with newer revisions. If a source has disappeared, report its URL and resolve provenance before retrying. Interrupted .part files can resume; a checksum mismatch requires moving the corrupt file aside.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
