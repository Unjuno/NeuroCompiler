# Archive security and provenance — scoped audit (2026-11-08)

**Status: limited automated inspection complete, rights review pending.** This document records what was actually checked, not a statement of universal safety.

- Container: `/mnt/data/NeuroCompiler_complete_research_release.zip` (214,856,850 bytes).
- Whole-bundle SHA-256: `b5bc2b4f2777aafba5f4a9c6c3ce1531bd63b112b7c752e41668efc1b03bf6a1`.
- All nine source archives matched their entries in `archives/source_archives.csv`; CRC passed. The standalone release verifier also passed all nine nested archives and 1,512 files.
- Inspected nested file extensions: `.csv` 732, `.json` 261, `.pt` 210, `.npz` 105, `.py` 62, `.log` 58, `.txt` 46, `.png` 21, `.md` 17.
- Checked filenames for conspicuous private-key, credential, `.env`, `.git`, `.ssh` and similar patterns: no flagged names.
- Checked UTF-8-compatible text files up to 500 KiB each using basic patterns for GitHub tokens, OpenAI keys, AWS access keys, private-key headers, and long key/password assignments: no matches.
- This **does not** inspect every binary value, large text file, embedded model metadata, or all personal information and is **not a license audit**.

Before distribution of the raw archives, the project owner must confirm permissions and review potentially sensitive task data/logs and unsafe checkpoint deserialization. Do not treat the scan as an assurance that the release is legally or operationally safe.
