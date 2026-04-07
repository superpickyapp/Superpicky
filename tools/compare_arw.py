#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare ARW files and generate diagnostic reports.

Usage:
  python tools/compare_arw.py <good.arw> <bad.arw>
  python tools/compare_arw.py <good.arw> <bad.arw> -o report.txt
  python tools/compare_arw.py <good.arw> <bad.arw> --max-diff 200
  python tools/compare_arw.py --probe --probe-source <source.arw> -o probe_report.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STRUCTURAL_TAGS = [
    "FileType",
    "FileTypeExtension",
    "MIMEType",
    "FileSize",
    "PreviewImageStart",
    "PreviewImageLength",
    "ThumbnailOffset",
    "ThumbnailLength",
    "JpgFromRawStart",
    "JpgFromRawLength",
    "OtherImageStart",
    "OtherImageLength",
    "HiddenDataOffset",
    "HiddenDataLength",
    "SR2SubIFDOffset",
    "StripOffsets",
    "StripByteCounts",
]


VOLATILE_TAGS = {
    "SourceFile",
    "FileName",
    "Directory",
    "FileSize",
    "FileCreateDate",
    "FileAccessDate",
    "FileModifyDate",
    "FileInodeChangeDate",
    "FilePermissions",
}


PROBE_PROFILES: List[Tuple[str, List[str]]] = [
    ("rating_overwrite", ["-Rating=3", "-overwrite_original"]),
    ("rating_in_place", ["-Rating=3", "-overwrite_original_in_place"]),
    ("sony_rating_overwrite", ["-Sony:Rating=3", "-overwrite_original"]),
    ("sony_rating_in_place", ["-Sony:Rating=3", "-overwrite_original_in_place"]),
    ("xmp_rating_overwrite", ["-XMP:Rating=3", "-overwrite_original"]),
    ("xmp_rating_in_place", ["-XMP:Rating=3", "-overwrite_original_in_place"]),
    (
        "superpicky_full_overwrite",
        [
            "-Rating=3",
            "-XMP:Pick=0",
            "-XMP:Label=Red",
            "-XMP:City=664.43",
            "-XMP:State=04.99",
            "-XMP:Country=BEST",
            "-XMP:Description=SP probe caption",
            "-overwrite_original",
        ],
    ),
    (
        "superpicky_full_in_place",
        [
            "-Rating=3",
            "-XMP:Pick=0",
            "-XMP:Label=Red",
            "-XMP:City=664.43",
            "-XMP:State=04.99",
            "-XMP:Country=BEST",
            "-XMP:Description=SP probe caption",
            "-overwrite_original_in_place",
        ],
    ),
]


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _run_exiftool(exiftool: str, args: List[str], timeout: int = 120) -> Dict[str, Any]:
    cmd = [exiftool] + args
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=timeout,
            creationflags=creationflags,
            cwd=str(Path(exiftool).resolve().parent),
        )
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "cmd": cmd,
        }

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": _decode(proc.stdout or b""),
        "stderr": _decode(proc.stderr or b""),
        "cmd": cmd,
    }


def _detect_exiftool(user_path: Optional[str]) -> Optional[str]:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)

    from_path = shutil.which("exiftool")
    if from_path:
        return from_path

    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "exiftools_win" / "exiftool.exe",
        root / "exiftools_mac" / "exiftool",
        root / "exiftool.exe",
        root / "exiftool",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_info(file_path: str) -> Dict[str, Any]:
    st = os.stat(file_path)
    return {
        "path": str(Path(file_path).resolve()),
        "size_bytes": st.st_size,
        "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256(file_path),
    }


def _parse_json_list(payload: str) -> Dict[str, Any]:
    if not payload.strip():
        return {}
    parsed = json.loads(payload)
    if isinstance(parsed, list):
        return parsed[0] if parsed else {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _normalize(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _normalize(v[k]) for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_normalize(i) for i in v]
    return v


def _short(v: Any, limit: int = 180) -> str:
    if isinstance(v, (dict, list)):
        text = json.dumps(v, ensure_ascii=False, sort_keys=True)
    else:
        text = str(v)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _is_volatile_key(key: str) -> bool:
    if key in VOLATILE_TAGS:
        return True
    if key.startswith("ExifTool:Warning"):
        return True
    if ":" in key:
        suffix = key.rsplit(":", 1)[-1]
        if suffix in VOLATILE_TAGS:
            return True
    return False


def _collect_structure(exiftool: str, file_path: str) -> Dict[str, Any]:
    args = ["-json", "-n"] + [f"-{tag}" for tag in STRUCTURAL_TAGS] + [file_path]
    res = _run_exiftool(exiftool, args, timeout=40)
    if not res["ok"]:
        return {"_error": f"exiftool failed ({res['returncode']}): {res['stderr'].strip()}"}
    try:
        data = _parse_json_list(res["stdout"])
    except Exception as exc:
        return {"_error": f"json parse failed: {exc}"}
    return {k: data.get(k) for k in STRUCTURAL_TAGS}


def _collect_metadata(exiftool: str, file_path: str) -> Dict[str, Any]:
    args = [
        "-json",
        "-n",
        "-a",
        "-u",
        "-G1",
        "-s",
        "-api",
        "LargeFileSupport=1",
        file_path,
    ]
    res = _run_exiftool(exiftool, args, timeout=120)
    if not res["ok"]:
        return {"_error": f"exiftool failed ({res['returncode']}): {res['stderr'].strip()}"}
    try:
        return _parse_json_list(res["stdout"])
    except Exception as exc:
        return {"_error": f"json parse failed: {exc}"}


def _collect_validate(exiftool: str, file_path: str) -> Dict[str, Any]:
    args = ["-validate", "-warning", "-error", "-a", "-u", "-s", "-G1", file_path]
    res = _run_exiftool(exiftool, args, timeout=60)
    text = "\n".join(
        [
            line.strip()
            for line in (res["stdout"] + "\n" + res["stderr"]).splitlines()
            if line.strip()
        ]
    )
    lines = text.splitlines() if text else []
    warnings = [ln for ln in lines if "warning" in ln.lower()]
    errors = [ln for ln in lines if "error" in ln.lower()]
    return {
        "returncode": res["returncode"],
        "warnings": warnings,
        "errors": errors,
        "lines": lines,
    }


def _diff_dicts(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str]]:
    left_keys = set(left.keys())
    right_keys = set(right.keys())
    added = sorted(right_keys - left_keys)
    removed = sorted(left_keys - right_keys)
    common = sorted(left_keys & right_keys)
    changed = [k for k in common if _normalize(left[k]) != _normalize(right[k])]
    return added, removed, changed


def _pick_changes(
    keys: Iterable[str], left: Dict[str, Any], right: Dict[str, Any], limit: int
) -> List[str]:
    keys = list(keys)
    total = len(keys)
    rows: List[str] = []
    for idx, key in enumerate(keys, start=1):
        if idx > limit:
            rows.append(f"... ({total - limit} more changed tags omitted)")
            break
        rows.append(f"- {key}")
        rows.append(f"  good: {_short(left.get(key))}")
        rows.append(f"  bad : {_short(right.get(key))}")
    return rows


def _format_delta(before: Any, after: Any) -> str:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        delta = after - before
        sign = "+" if delta >= 0 else ""
        return f" (delta {sign}{delta})"
    return ""


def build_report(
    good_path: str,
    bad_path: str,
    exiftool_path: str,
    max_diff: int,
) -> str:
    now = dt.datetime.now().isoformat(timespec="seconds")
    exiftool_ver = _run_exiftool(exiftool_path, ["-ver"], timeout=10)
    exiftool_version = exiftool_ver["stdout"].strip() if exiftool_ver["ok"] else "unknown"

    good_info = _file_info(good_path)
    bad_info = _file_info(bad_path)

    good_validate = _collect_validate(exiftool_path, good_path)
    bad_validate = _collect_validate(exiftool_path, bad_path)

    good_struct = _collect_structure(exiftool_path, good_path)
    bad_struct = _collect_structure(exiftool_path, bad_path)

    good_meta = _collect_metadata(exiftool_path, good_path)
    bad_meta = _collect_metadata(exiftool_path, bad_path)

    good_meta_filtered = {k: v for k, v in good_meta.items() if not _is_volatile_key(k)}
    bad_meta_filtered = {k: v for k, v in bad_meta.items() if not _is_volatile_key(k)}
    added, removed, changed = _diff_dicts(good_meta_filtered, bad_meta_filtered)

    struct_added, struct_removed, struct_changed = _diff_dicts(good_struct, bad_struct)

    out: List[str] = []
    out.append("=== ARW Compare Report ===")
    out.append(f"Generated: {now}")
    out.append(f"ExifTool: {exiftool_path}")
    out.append(f"ExifTool version: {exiftool_version}")
    out.append("")
    out.append("[Files]")
    out.append(f"good: {good_info['path']}")
    out.append(f"bad : {bad_info['path']}")
    out.append(f"good size/sha256: {good_info['size_bytes']} / {good_info['sha256']}")
    out.append(f"bad  size/sha256: {bad_info['size_bytes']} / {bad_info['sha256']}")
    out.append(f"good mtime: {good_info['mtime']}")
    out.append(f"bad  mtime: {bad_info['mtime']}")
    out.append("")
    out.append("[Validate]")
    out.append(
        f"good: rc={good_validate['returncode']}, warnings={len(good_validate['warnings'])}, errors={len(good_validate['errors'])}"
    )
    out.append(
        f"bad : rc={bad_validate['returncode']}, warnings={len(bad_validate['warnings'])}, errors={len(bad_validate['errors'])}"
    )
    if good_validate["warnings"] or good_validate["errors"]:
        out.append("good validate lines:")
        out.extend([f"- {ln}" for ln in (good_validate["errors"] + good_validate["warnings"])[:20]])
    if bad_validate["warnings"] or bad_validate["errors"]:
        out.append("bad validate lines:")
        out.extend([f"- {ln}" for ln in (bad_validate["errors"] + bad_validate["warnings"])[:20]])
    out.append("")
    out.append("[Structural Tags]")
    out.append(f"changed={len(struct_changed)}, added={len(struct_added)}, removed={len(struct_removed)}")
    for tag in struct_changed:
        out.append(f"- {tag}")
        out.append(f"  good: {_short(good_struct.get(tag))}")
        out.append(f"  bad : {_short(bad_struct.get(tag))}")
    for tag in struct_added[:20]:
        out.append(f"- ADDED {tag}: {_short(bad_struct.get(tag))}")
    for tag in struct_removed[:20]:
        out.append(f"- REMOVED {tag}: {_short(good_struct.get(tag))}")
    out.append("")
    out.append("[All Metadata Diff]")
    out.append(f"changed={len(changed)}, added={len(added)}, removed={len(removed)}")
    out.append(f"(volatile file tags ignored: {', '.join(sorted(VOLATILE_TAGS))})")
    if changed:
        out.append("changed tags:")
        out.extend(_pick_changes(changed, good_meta_filtered, bad_meta_filtered, max_diff))
    if added:
        out.append("added tags in bad:")
        for key in added[:max_diff]:
            out.append(f"- {key}: {_short(bad_meta_filtered.get(key))}")
        if len(added) > max_diff:
            out.append(f"... ({len(added) - max_diff} more added tags omitted)")
    if removed:
        out.append("removed tags from bad:")
        for key in removed[:max_diff]:
            out.append(f"- {key}: {_short(good_meta_filtered.get(key))}")
        if len(removed) > max_diff:
            out.append(f"... ({len(removed) - max_diff} more removed tags omitted)")
    out.append("")
    out.append("[Hint]")
    out.append("If Structural Tags show major offset/length changes (Preview/JpgFromRaw/Strip/HiddenData),")
    out.append("that is a strong signal why Sony Image Edge Viewer rejects the written ARW.")

    return "\n".join(out) + "\n"


def build_probe_report(
    source_path: str,
    exiftool_path: str,
    max_diff: int,
    probe_workdir: str,
) -> str:
    now = dt.datetime.now().isoformat(timespec="seconds")
    exiftool_ver = _run_exiftool(exiftool_path, ["-ver"], timeout=10)
    exiftool_version = exiftool_ver["stdout"].strip() if exiftool_ver["ok"] else "unknown"

    source_info = _file_info(source_path)
    baseline_struct = _collect_structure(exiftool_path, source_path)
    baseline_validate = _collect_validate(exiftool_path, source_path)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_dir = Path(probe_workdir).expanduser().resolve() / f"arw_probe_{stamp}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    baseline_copy = probe_dir / "base.ARW"
    shutil.copy2(source_path, baseline_copy)

    results: List[Dict[str, Any]] = []
    for name, profile_args in PROBE_PROFILES:
        target = probe_dir / f"{name}.ARW"
        shutil.copy2(source_path, target)

        run_res = _run_exiftool(exiftool_path, profile_args + [str(target)], timeout=120)
        struct = _collect_structure(exiftool_path, str(target))
        validate = _collect_validate(exiftool_path, str(target))

        _, _, struct_changed = _diff_dicts(baseline_struct, struct)
        changes_detail = []
        for tag in struct_changed[:max_diff]:
            before = baseline_struct.get(tag)
            after = struct.get(tag)
            changes_detail.append(f"{tag}: {_short(before)} -> {_short(after)}{_format_delta(before, after)}")

        results.append(
            {
                "name": name,
                "file": str(target),
                "write_ok": run_res["ok"],
                "write_rc": run_res["returncode"],
                "write_stderr": run_res["stderr"].strip(),
                "struct_changed_count": len(struct_changed),
                "struct_changed": struct_changed,
                "changes_detail": changes_detail,
                "validate_warnings": len(validate["warnings"]),
                "validate_errors": len(validate["errors"]),
                "validate_rc": validate["returncode"],
            }
        )

    results_sorted = sorted(
        results,
        key=lambda r: (r["struct_changed_count"], r["validate_errors"], r["validate_warnings"], r["name"]),
    )

    out: List[str] = []
    out.append("=== ARW Probe Report ===")
    out.append(f"Generated: {now}")
    out.append(f"ExifTool: {exiftool_path}")
    out.append(f"ExifTool version: {exiftool_version}")
    out.append("")
    out.append("[Source]")
    out.append(f"source: {source_info['path']}")
    out.append(f"size/sha256: {source_info['size_bytes']} / {source_info['sha256']}")
    out.append(f"mtime: {source_info['mtime']}")
    out.append("")
    out.append("[Baseline Validate]")
    out.append(
        f"rc={baseline_validate['returncode']}, warnings={len(baseline_validate['warnings'])}, errors={len(baseline_validate['errors'])}"
    )
    out.append("")
    out.append("[Probe Workdir]")
    out.append(str(probe_dir))
    out.append("")
    out.append("[Profiles]")
    for row in results_sorted:
        out.append(
            f"- {row['name']}: write_rc={row['write_rc']}, struct_changed={row['struct_changed_count']}, "
            f"validate_warnings={row['validate_warnings']}, validate_errors={row['validate_errors']}"
        )
        if not row["write_ok"] and row["write_stderr"]:
            out.append(f"  write_stderr: {_short(row['write_stderr'])}")
        if row["changes_detail"]:
            out.append("  changed tags:")
            for detail in row["changes_detail"][:max_diff]:
                out.append(f"  - {detail}")
            if len(row["struct_changed"]) > max_diff:
                out.append(f"  - ... ({len(row['struct_changed']) - max_diff} more changed tags omitted)")
    out.append("")
    out.append("[Hint]")
    out.append(
        "If all profiles change offset-like tags (Preview/Thumbnail/JpgFromRaw/Strip/HiddenData/SR2SubIFD),"
    )
    out.append("direct ARW writing is likely unsafe for Sony Image Edge Viewer in this environment.")

    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare or probe ARW files for metadata write compatibility.")
    parser.add_argument("good", nargs="?", help="Reference ARW path (known-good) or source ARW in probe mode")
    parser.add_argument("bad", nargs="?", help="Candidate ARW path (cannot be opened by viewer)")
    parser.add_argument("-o", "--output", help="Write report to file")
    parser.add_argument("--exiftool", help="Path to exiftool executable")
    parser.add_argument("--max-diff", type=int, default=120, help="Max tags to print for each diff section")
    parser.add_argument("--probe", action="store_true", help="Run built-in write-profile probe on one ARW source")
    parser.add_argument("--probe-source", help="Source ARW path for probe mode")
    parser.add_argument(
        "--probe-workdir",
        default=".",
        help="Directory to place probe copies (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    exiftool = _detect_exiftool(args.exiftool)
    if not exiftool:
        print(
            "ERROR: exiftool not found. Provide --exiftool or place exiftool in PATH/exiftools_win/exiftools_mac.",
            file=sys.stderr,
        )
        return 2

    if args.probe:
        src = args.probe_source or args.good
        if not src:
            print("ERROR: probe mode requires --probe-source <source.arw> (or positional <good>).", file=sys.stderr)
            return 2
        src_path = str(Path(src).expanduser())
        if not os.path.exists(src_path):
            print(f"ERROR: probe source file not found: {src_path}", file=sys.stderr)
            return 2
        report = build_probe_report(
            source_path=src_path,
            exiftool_path=exiftool,
            max_diff=args.max_diff,
            probe_workdir=args.probe_workdir,
        )
    else:
        if not args.good or not args.bad:
            print("ERROR: compare mode requires <good.arw> and <bad.arw>.", file=sys.stderr)
            return 2

        good_path = str(Path(args.good).expanduser())
        bad_path = str(Path(args.bad).expanduser())
        if not os.path.exists(good_path):
            print(f"ERROR: good file not found: {good_path}", file=sys.stderr)
            return 2
        if not os.path.exists(bad_path):
            print(f"ERROR: bad file not found: {bad_path}", file=sys.stderr)
            return 2

        report = build_report(good_path=good_path, bad_path=bad_path, exiftool_path=exiftool, max_diff=args.max_diff)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report saved: {out_path}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
