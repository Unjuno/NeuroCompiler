# Publish the original research archive on GitHub Releases

**As of 2026-11-08: preparation and integrity verification complete; no GitHub Release or assets have been published.** The current GitHub connection can read repository metadata and commit files but does not expose an operation for creating a Release and uploading binary assets. No workaround should put raw 205 MiB content into Git history.

## Release procedure (GitHub UI)

1. Open [`Unjuno/NeuroCompiler` → Releases](https://github.com/Unjuno/NeuroCompiler/releases), then **Draft a new release**.
2. Choose **Create new tag** `research-archive-2026-11-07` based on `main`. The tag will point to the current repository tree; the attached archive itself is explicitly a **2026-11-07 preservation snapshot**, not a matching source snapshot of the tag.
3. Set release title **NeuroCompiler — preserved research archives (2026-11-07)**.
4. Copy the [prepared release notes](ARCHIVE_RELEASE_CANDIDATE.md#release-note-text-ready-to-copy-when-clearance-is-granted).
5. Upload exactly one asset, `NeuroCompiler_complete_research_release.zip` (214,856,850 bytes). Do **not** commit it to `main`. The file is supplied separately as a downloadable conversation artifact.
6. Before publishing, complete the copyright, privacy, and serialized-model security review documented in [the release candidate](ARCHIVE_RELEASE_CANDIDATE.md#prepublication-review-gate).
7. **Only after approval**, publish the Release. Download the asset from GitHub and run `python3 scripts/verify_release_bundle.py --bundle /path/to/downloaded.zip --expected-bundle-sha256 b5bc2b4f2777aafba5f4a9c6c3ce1531bd63b112b7c752e41668efc1b03bf6a1`.
8. Record the actual Release URL, Git commit SHA, verified checksum, and rights decision in `docs/CORRECTIONS.md` and amend this document's status.

## Important distinctions

- GitHub **Public** visibility is not a reuse license. No license text has been selected on behalf of the owner.
- SHA-256 and CRC checks confirm file integrity only; they do not demonstrate scientific reproducibility or absence of confidential information.
- The `repository/NeuroCompiler_public_repo.zip` nested in the archive is the *old* 2026-11-07 repository snapshot. The latest `main` contains later release-preparation documentation.
- For reproducible experiments, inspect the [reproducibility notes](REPRODUCIBILITY.md), recorded environment versions and original stage-specific inputs.
