#!/usr/bin/env python3
"""
bundle_py_to_json.py

Recursively collect all .py files under a root directory and dump them
to a JSON file as a list of { "path": <relative path>, "content": <file text> }.

Usage:
  python bundle_py_to_json.py --root . --out code_dump.json
  python bundle_py_to_json.py --root C:/Users/Me/project --out project_code.json \
      --exclude .git __pycache__ venv .venv dist build .mypy_cache .pytest_cache

Notes:
- Reads files as UTF-8 with errors='replace' so it never crashes on encoding.
- Skips common junk dirs by default; add more with --exclude.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any

DEFAULT_EXCLUDES = {
    ".git", "__pycache__", "venv", ".venv", "env", ".env",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    ".tox", ".ruff_cache", ".coverage", ".gitlab", ".github"
}

def iter_py_files(root: Path, exclude_dirs: Iterable[str]) -> Iterable[Path]:
    exclude = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn

def read_text_safe(p: Path) -> str:
    # Normalize newlines and tolerate odd encodings
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return f.read().replace("\r\n", "\n").replace("\r", "\n")

def bundle(root: Path, out_path: Path, exclude_dirs: Iterable[str]) -> Dict[str, Any]:
    files: List[Dict[str, str]] = []
    for p in sorted(iter_py_files(root, exclude_dirs)):
        rel = p.relative_to(root).as_posix()
        try:
            content = read_text_safe(p)
        except Exception as e:
            content = f"<<ERROR READING FILE: {e}>>"
        files.append({"path": rel, "content": content})
    payload: Dict[str, Any] = {
        "root": str(root.resolve()),
        "count": len(files),
        "files": files,
    }
    # Write JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

def main():
    ap = argparse.ArgumentParser(description="Bundle .py files into a JSON archive.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root directory to scan")
    ap.add_argument("--out", type=Path, default=Path("code_dump.json"), help="Output JSON file")
    ap.add_argument(
        "--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDES),
        help="Directory names to exclude (not paths; exact folder names)."
    )
    args = ap.parse_args()

    payload = bundle(args.root, args.out, args.exclude)
    print(f"Wrote {payload['count']} files to {args.out}")

if __name__ == "__main__":
    main()