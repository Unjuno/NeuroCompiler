"""Unit tests for release-bundle integrity verification (standard library only)."""
from __future__ import annotations
import csv
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_release_bundle import verify_bundle  # deliberately fails before implementation


def sample(tmp: Path, *, tamper: bool = False, missing: bool = False, unsafe: bool = False):
    inner = io.BytesIO()
    with ZipFile(inner, "w", ZIP_DEFLATED) as z:
        z.writestr("source/readme.txt", "reproducible research\n")
        if unsafe:
            z.writestr("../traversal.txt", "unsafe path")
    original = inner.getvalue()
    fields = ["stage", "archive", "sha256", "bytes", "entries", "uncompressed_bytes", "crc_test_passed", "github_storage"]
    manifest = tmp / "source_archives.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"stage":"stage-x", "archive":"research.zip", "sha256": hashlib.sha256(original).hexdigest(),
                    "bytes":len(original), "entries":2 if unsafe else 1,
                    "uncompressed_bytes":sum(len(x) for x in [b'reproducible research\n']+([b'unsafe path'] if unsafe else [])),
                    "crc_test_passed":"True", "github_storage":"GitHub Release asset (NOT git)"})
    outer = tmp / "bundle.zip"
    with ZipFile(outer, "w", ZIP_DEFLATED) as z:
        z.writestr("README_RELEASE.md", "archive snapshot")
        z.writestr("archives/source_archives.csv", manifest.read_bytes())
        if not missing:
            z.writestr("original_archives/research.zip", (original[:20] + bytes([original[20] ^ 1]) + original[21:]) if tamper else original)
    return outer,manifest


class ReleaseBundleTests(unittest.TestCase):
    def test_valid_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            outer, manifest=sample(Path(d))
            result=verify_bundle(outer,manifest)
            self.assertEqual(result['verified_archives'],1)
            self.assertEqual(result['verified_files'],1)

    def test_tampered_original_fails(self):
        with tempfile.TemporaryDirectory() as d:
            outer, manifest=sample(Path(d),tamper=True)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_bundle(outer,manifest)

    def test_missing_original_fails(self):
        with tempfile.TemporaryDirectory() as d:
            outer, manifest=sample(Path(d),missing=True)
            with self.assertRaisesRegex(ValueError,"Missing archive"):
                verify_bundle(outer,manifest)

    def test_unsafe_inner_entry_fails(self):
        with tempfile.TemporaryDirectory() as d:
            outer, manifest=sample(Path(d),unsafe=True)
            with self.assertRaisesRegex(ValueError,"Unsafe archive member"):
                verify_bundle(outer,manifest)

    def test_outer_archive_sha256_option(self):
        with tempfile.TemporaryDirectory() as d:
            outer, manifest=sample(Path(d))
            with self.assertRaisesRegex(ValueError,"Bundle SHA256 mismatch"):
                verify_bundle(outer,manifest,expected_bundle_sha256='0'*64)


if __name__ == "__main__":
    unittest.main()
