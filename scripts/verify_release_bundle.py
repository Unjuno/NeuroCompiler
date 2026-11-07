#!/usr/bin/env python3
"""Validate a NeuroCompiler preservation ZIP against the tracked archive manifest.

This verifies ZIP structure, SHA-256, sizes, member counts and CRCs. It does
NOT certify provenance, code safety, licensing, or scientific reproducibility.
Never extract or load research checkpoints during verification.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile, BadZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "archives" / "source_archives.csv"


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and ":" not in p.parts[0] and ".." not in p.parts and "\\" not in name


def _safe_zip_members(z: ZipFile) -> None:
    names = [x.filename for x in z.infolist()]
    for name in names:
        if not _safe_name(name):
            raise ValueError(f"Unsafe archive member: {name}")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate ZIP entry names")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_rows(manifest_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(StringIO(manifest_text))
    required = {"archive", "sha256", "bytes", "entries", "uncompressed_bytes"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("Invalid source archives manifest")
    rows = list(reader)
    if not rows or len(set(row["archive"] for row in rows)) != len(rows):
        raise ValueError("Empty or duplicate source archives manifest")
    for row in rows:
        if not _safe_name(row["archive"]) or "/" in row["archive"]:
            raise ValueError("Unsafe archive filename")
        digest = row["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Malformed SHA256 in source archives manifest")
    return rows


def verify_bundle(bundle: Path, manifest: Path = DEFAULT_MANIFEST,
                  expected_bundle_sha256: str | None = None) -> dict[str, object]:
    if expected_bundle_sha256:
        if _sha256_file(bundle) != expected_bundle_sha256.lower():
            raise ValueError("Bundle SHA256 mismatch")
    expected = manifest.read_bytes().decode("utf-8-sig")
    rows = _source_rows(expected)
    verified_files = 0
    with ZipFile(bundle) as z:
        _safe_zip_members(z)
        names = set(z.namelist())
        if "archives/source_archives.csv" not in names:
            raise ValueError("Missing archive manifest inside release bundle")
        embedded_manifest = z.read("archives/source_archives.csv").decode("utf-8-sig")
        if embedded_manifest != expected:
            raise ValueError("Embedded archive manifest does not match repository manifest")
        for row in rows:
            name = f"original_archives/{row['archive']}"
            if name not in names:
                raise ValueError(f"Missing archive: {name}")
            info = z.getinfo(name)
            if info.file_size != int(row["bytes"]):
                raise ValueError(f"Size mismatch: {name}")
            # CRC verification happens during z.read; SHA confirms exact bytes.
            data = z.read(name)
            if sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"SHA256 mismatch: {name}")
            try:
                with ZipFile(BytesIO(data)) as nested:
                    _safe_zip_members(nested)
                    entries = [x for x in nested.infolist() if not x.is_dir()]
                    if len(entries) != int(row["entries"]):
                        raise ValueError(f"Entry count mismatch: {name}")
                    if sum(x.file_size for x in entries) != int(row["uncompressed_bytes"]):
                        raise ValueError(f"Uncompressed size mismatch: {name}")
                    bad = nested.testzip()
                    if bad is not None:
                        raise ValueError(f"CRC mismatch: {name}/{bad}")
                    verified_files += len(entries)
            except BadZipFile as exc:
                raise ValueError(f"Invalid nested ZIP: {name}") from exc
        # Validate root ZIP entries as well, including the release's sidecars.
        bad_outer = z.testzip()
        if bad_outer is not None:
            raise ValueError(f"Outer archive CRC mismatch: {bad_outer}")
    return {
        "bundle": str(bundle),
        "bundle_sha256": _sha256_file(bundle),
        "verified_archives": len(rows),
        "verified_files": verified_files,
        "sha256_match": expected_bundle_sha256 is not None,
        "status": "PASS",
        "scope": "file integrity only; not scientific reproducibility or license clearance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-bundle-sha256", default=None)
    args = parser.parse_args()
    try:
        result = verify_bundle(args.bundle, args.manifest, args.expected_bundle_sha256)
    except (ValueError, OSError, BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
