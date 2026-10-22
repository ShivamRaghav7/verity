"""Pinned download, interrupted response and safe fixture checks; no real assets."""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_vlkeb_assets import (
    build_fixture, download_asset, extract_fixture_images, load_manifest, main,
    relative_path, verify_asset, verify_assets, write_json,
)


class FakeResponse(BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}


class VLKEBAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"a synthetic pinned checkpoint, never deserialized"
        self.asset = {"id": "weights", "group": "core", "path": "model/weights.bin",
                      "url": "https://example.invalid/pinned/weights.bin", "revision": "fixed",
                      "size_bytes": len(self.payload),
                      "sha256": hashlib.sha256(self.payload).hexdigest()}

    def test_manifest_has_every_required_file_and_no_final_rows(self):
        manifest = load_manifest(Path(__file__).resolve().parents[1] / "configs/vlkeb-assets.json")
        paths = {asset["path"] for asset in manifest["assets"]}
        self.assertTrue({"opt-2.7b/pytorch_model.bin", "bert-base-uncased/pytorch_model.bin",
                         "eva_vit_g.pth", "blip2_pretrained_opt2.7b.pth", "mend.pt",
                         "upstream-train.json", "upstream-images.tar.gz"}.issubset(paths))
        self.assertFalse(any("eval.json" in asset["url"] for asset in manifest["assets"]))
        self.assertEqual(sum(asset["size_bytes"] for asset in manifest["assets"]), manifest["total_download_bytes"])
        self.assertFalse(manifest["weights_downloaded_during_preparation"])

    def test_dry_run_does_not_network_or_create_root(self):
        target = self.root / "no-created-directory"
        with patch("scripts.prepare_vlkeb_assets.urlopen", side_effect=AssertionError("network")), redirect_stdout(StringIO()) as output:
            self.assertEqual(main(["--root", str(target)]), 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "dry_run")
        self.assertFalse(target.exists())

    def test_download_checks_pinned_content_and_cached_file(self):
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return FakeResponse(self.payload)
        result = download_asset(self.root, self.asset, opener=opener)
        self.assertTrue(result["upstream_checksum_verified"])
        self.assertEqual((self.root / self.asset["path"]).read_bytes(), self.payload)
        download_asset(self.root, self.asset, opener=lambda *args, **kwargs: self.fail("cached network call"))
        self.assertEqual(len(requests), 1)

    def test_resume_appends_only_strict_matching_range(self):
        part = self.root / (self.asset["path"] + ".part")
        part.parent.mkdir()
        part.write_bytes(self.payload[:9])
        def opener(request, timeout):
            self.assertEqual(request.get_header("Range"), "bytes=9-")
            return FakeResponse(self.payload[9:], status=206,
                                headers={"Content-Range": f"bytes 9-{len(self.payload)-1}/{len(self.payload)}",
                                         "Content-Length": str(len(self.payload) - 9)})
        download_asset(self.root, self.asset, opener=opener)
        self.assertEqual((self.root / self.asset["path"]).read_bytes(), self.payload)

    def test_ignored_range_restarts_instead_of_appending(self):
        part = self.root / (self.asset["path"] + ".part")
        part.parent.mkdir()
        part.write_bytes(b"old prefix")
        download_asset(self.root, self.asset, opener=lambda *args, **kwargs: FakeResponse(self.payload))
        self.assertEqual((self.root / self.asset["path"]).read_bytes(), self.payload)

    def test_bad_range_checksum_and_short_response_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            download_asset(self.root, self.asset, opener=lambda *args, **kwargs: FakeResponse(self.payload, status=206,
                           headers={"Content-Range": f"bytes 1-{len(self.payload)}/{len(self.payload)+1}"}))
        broken = dict(self.asset, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            download_asset(self.root, broken, opener=lambda *args, **kwargs: FakeResponse(self.payload))
        part = self.root / (self.asset["path"] + ".part")
        part.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            download_asset(self.root, self.asset, opener=lambda *args, **kwargs: FakeResponse(self.payload[:3], headers={"Server": "fake"}))
        self.assertEqual(part.read_bytes(), self.payload[:3])
        self.assertFalse((self.root / self.asset["path"]).exists())

    def test_published_md5_is_verified_and_local_sha256_recorded(self):
        asset = dict(self.asset)
        del asset["sha256"]
        asset["md5"] = hashlib.md5(self.payload, usedforsecurity=False).hexdigest()
        result = download_asset(self.root, asset, opener=lambda *args, **kwargs: FakeResponse(self.payload))
        self.assertEqual(result["verification_algorithm"], "md5")
        self.assertEqual(result["sha256"], self.asset["sha256"])

    def test_escaping_paths_rejected_on_any_host(self):
        for unsafe in ("../file", "/file", "C:/file", "dir\\file", ".", ""):
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                relative_path(self.root, unsafe)

    def fixture_manifest(self, extra_members=None):
        row = {"image": "m.one/edit.jpg", "image_rephrase": "m.one/other.jpg",
               "m_loc": "m.two/neighbor.jpg", "src": "What city?", "rephrase": "Which city?",
               "pred": "old", "alt": "new", "loc": "text local", "loc_ans": "answer",
               "m_loc_q": "What country?", "m_loc_a": "different"}
        source = json.dumps([row, dict(row, image="m.one/extra.jpg")]).encode()
        (self.root / "upstream-train.json").write_bytes(source)
        train_asset = {"path": "upstream-train.json", "group": "fixture", "size_bytes": len(source),
                       "sha256": hashlib.sha256(source).hexdigest(), "url": "https://example.invalid/train"}
        images = {"VLKEB_images/mmkb_images/" + image: image.encode()
                  for image in [row["image"], row["image_rephrase"], row["m_loc"], "m.one/extra.jpg"]}
        images["VLKEB_images/unused.jpg"] = b"do not extract"
        images.update(extra_members or {})
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path, payload in images.items():
                member = tarfile.TarInfo(path)
                if payload is None:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/elsewhere"
                    archive.addfile(member)
                else:
                    member.size = len(payload)
                    archive.addfile(member, BytesIO(payload))
        archive_bytes = buffer.getvalue()
        (self.root / "upstream-images.tar.gz").write_bytes(archive_bytes)
        archive_asset = {"path": "upstream-images.tar.gz", "group": "fixture", "size_bytes": len(archive_bytes),
                         "sha256": hashlib.sha256(archive_bytes).hexdigest(), "url": "https://example.invalid/archive"}
        return {"assets": [train_asset, archive_asset], "fixture": {"source_path": "upstream-train.json",
                "archive_path": "upstream-images.tar.gz", "source_revision": "fixed-train-revision",
                "source_indices": [0, 1], "primary_index": 0, "additional_image_index": 1,
                "output_path": "train-fixture.json", "image_directory": "images", "disclosure": "training-seen smoke only"}}

    def test_only_four_training_images_extracted_then_independently_verified(self):
        manifest = self.fixture_manifest()
        fixture = build_fixture(self.root, manifest)
        fixture["image_integrity"] = extract_fixture_images(self.root, manifest, fixture)
        write_json(self.root / "train-fixture.json", fixture)
        self.assertEqual(len(list((self.root / "images").rglob("*.jpg"))), 4)
        self.assertFalse((self.root / "images/unused.jpg").exists())
        self.assertEqual(fixture["upstream_split"], "train")
        with patch("scripts.prepare_vlkeb_assets.urlopen", side_effect=AssertionError("network")):
            report = verify_assets(manifest, self.root)
        self.assertTrue(report["verified"], report)
        self.assertTrue(report["fixture_prepared"])
        # A tampered image plus rewritten local fixture hash cannot fool archive verification.
        entry = fixture["image_integrity"][0]
        image = self.root / entry["path"]
        image.write_bytes(b"tampered bytes")
        entry["size_bytes"] = image.stat().st_size
        entry["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        write_json(self.root / "train-fixture.json", fixture)
        self.assertFalse(verify_assets(manifest, self.root)["verified"])

    def test_corrupt_archive_rejected_before_extracting_images(self):
        manifest = self.fixture_manifest()
        fixture = build_fixture(self.root, manifest)
        archive = self.root / "upstream-images.tar.gz"
        archive.write_bytes(b"not verified archive")
        with self.assertRaises(ValueError):
            extract_fixture_images(self.root, manifest, fixture)
        self.assertFalse((self.root / "images").exists())

    def test_archive_traversal_duplicate_and_symlink_are_rejected(self):
        for malicious in ({"../../escaped.jpg": b"escape"},
                          {"duplicate/m.one/edit.jpg": b"duplicate"},
                          {"VLKEB_images/mmkb_images/m.one/edit.jpg": None}):
            with self.subTest(malicious=list(malicious)):
                manifest = self.fixture_manifest(malicious)
                fixture = build_fixture(self.root, manifest)
                with self.assertRaises(ValueError):
                    extract_fixture_images(self.root, manifest, fixture)

    def test_missing_files_are_reported_and_do_not_create_root(self):
        absent = self.root / "absent"
        result = verify_assets({"assets": [self.asset]}, absent, group="core")
        self.assertFalse(result["verified"])
        self.assertEqual(result["missing"], [self.asset["path"]])
        self.assertFalse(absent.exists())

    def test_manifest_rejects_unpinned_or_non_https_files(self):
        path = self.root / "manifest.json"
        for mutation in ({"sha256": None}, {"url": "http://unsafe.invalid"}, {"path": "../escape"}):
            asset = dict(self.asset, **mutation)
            write_json(path, {"schema_version": 1, "assets": [asset]})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
