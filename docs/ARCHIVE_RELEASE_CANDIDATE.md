# Research archive release candidate — 2026-11-07 snapshot

**Status: prepared, NOT yet published as a GitHub Release.** This is a preservation archive, not a software API or a scientific reproducibility certificate.

- **Suggested GitHub release tag:** `research-archive-2026-11-07`
- **Suggested title:** NeuroCompiler — preserved research archives (2026-11-07)
- **One release asset:** `NeuroCompiler_complete_research_release.zip`
- **Size:** 214,856,850 bytes (204.90 MiB)
- **Whole-bundle SHA-256:** `b5bc2b4f2777aafba5f4a9c6c3ce1531bd63b112b7c752e41668efc1b03bf6a1`
- **Contents:** 9 original source ZIP archives (1,512 nested files), an older lightweight repository snapshot and metadata/checksums.
- **Archive coverage:** original experiment bundles for stages 0–8 and independent Stage 9R. Historical Stage 9 source/raw files were not recovered. Some historical stages were bundled together.

The nested archive manifest is [`archives/source_archives.csv`](../archives/source_archives.csv). Source ZIPs contain 210 PyTorch `.pt` files and 105 `.npz` files. Never execute untrusted code or unpickle `.pt` checkpoints merely to validate integrity.

### Verify the exact release candidate before uploading

From the current repository root, with the archive in an accessible local path:

```bash
python3 scripts/verify_release_bundle.py \
  --bundle /path/to/NeuroCompiler_complete_research_release.zip \
  --expected-bundle-sha256 b5bc2b4f2777aafba5f4a9c6c3ce1531bd63b112b7c752e41668efc1b03bf6a1
```

Expected: `"status": "PASS"`, `"verified_archives": 9`, `"verified_files": 1512`.

This checks the outer SHA-256, the nine original archive SHA-256 values and sizes, ZIP entry names, counts, uncompressed sizes and nested CRCs. It checks **integrity**, not who authored files, their license, or whether published results are scientifically reproducible.

### Prepublication review gate

**Do not publish the 205 MiB asset until the project owner has reviewed:**

1. Copyright and permission to distribute historical code, generated media, data, and serialized checkpoints. The public repository currently has no selected `LICENSE`.
2. Privacy/secrets in all raw text and binary data. The 2026-11-08 limited scan found no common credential strings or suspicious filenames among 1,512 nested entries, but it is not a complete audit.
3. Deserialization warning for `.pt` files and third-party dependencies.
4. The archive is a *2026-11-07 snapshot*. It is not byte-identical to the GitHub `main` tree after subsequent documentation/CI updates.

### Release note text (ready to copy when clearance is granted)

> **NeuroCompiler research preservation archive (snapshot 2026-11-07).**
> 
> Includes original Stage 0–8 experiment archives and the independent Stage 9R implementation, together with raw metrics, saved checkpoints, code and supplementary materials. The original historical Stage 9 run is not available. This is a research snapshot; it does not establish a production PDE solver or general superiority to classical numerical methods.
> 
> SHA-256 (`NeuroCompiler_complete_research_release.zip`): `b5bc2b4f2777aafba5f4a9c6c3ce1531bd63b112b7c752e41668efc1b03bf6a1`.
> 
> Verify locally using `scripts/verify_release_bundle.py` and the repository's `archives/source_archives.csv`. PyTorch `.pt` checkpoint loading may execute untrusted deserialization paths; validate provenance before opening.

See [the release procedure](RELEASE_OPERATIONS.md) for the remaining GitHub UI steps.
