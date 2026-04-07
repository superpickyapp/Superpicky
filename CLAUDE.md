# CLAUDE.md (Claude / Anthropic Coding Agents)

Use `scripts_dev/AI_CODING_RULES.md` as the single source of truth for this repository.

## Always Enforce

- UTF-8 safety first; do not introduce Chinese text corruption.
- ExifTool Chinese metadata writes must use UTF-8 temp files (`-XMP:Title<=tmp.txt`) instead of inline CLI values.
- Keep changes cross-platform (Windows + macOS).
- Any persistent external process must have deterministic cleanup on task/app exit.
- Packaged CUDA failures: prioritize packaging/runtime diagnosis before algorithm refactors.
- Keep Windows Torch/CUDA packaging with `upx=False` unless explicitly requested and validated.

## Minimum Verification

- `py -3 -m py_compile` for changed Python modules.
- Metadata write/read-back check for non-ASCII fields.
- Packaged app smoke test when `.spec` or runtime packaging behavior changes.

