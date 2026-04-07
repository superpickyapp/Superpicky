# AGENTS.md (Codex / OpenAI Coding Agents)

Follow `scripts_dev/AI_CODING_RULES.md` as the project baseline.

## Mandatory Project Constraints

- Keep files in UTF-8; avoid introducing mojibake.
- For ExifTool non-ASCII metadata writes, prefer UTF-8 temp-file redirection (`-Tag<=file`) over inline command args.
- Preserve Windows/macOS compatibility for paths and subprocess behavior.
- For SQLite in threaded code: either serialize shared-connection access with a lock or use per-thread connections; never assume `check_same_thread=False` is enough.
- Do not directly access private DB connection internals from business code (e.g., `report_db._conn.*`); add thread-safe wrapper methods instead.
- Ensure transaction handling is consistent and defensive (avoid unsynchronized mixed transaction styles; commit only when valid).
- Ensure persistent external processes (like `exiftool -stay_open`) have explicit shutdown and are closed on exit.
- For packaged-only CUDA issues, first suspect packaging/runtime differences.
- In Windows PyInstaller spec for Torch/CUDA, keep `upx=False` unless explicitly re-validated.

## Validation Minimum

- Run `py -3 -m py_compile` on changed Python files.
- For metadata changes: write + read-back verification with Chinese sample values.
- For `.spec` changes: packaged startup smoke test.
- For DB/threading changes: run a small multi-thread write/read stress check and confirm no transaction-state errors.
