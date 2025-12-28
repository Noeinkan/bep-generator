"""Utility to report which fields are still missing aiPrompt in helpContentData.js.

Reads the canonical list from fields_without_aiPrompt.json and checks each field block
for presence of an 'aiPrompt:' entry.

Run from repo root or ml-service.
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    # This script lives in ml-service/scripts/
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _repo_root()
    js_path = root / "src" / "data" / "helpContentData.js"
    fields_path = root / "fields_without_aiPrompt.json"

    fields = json.loads(fields_path.read_text(encoding="utf-8"))["fieldsList"]
    lines = js_path.read_text(encoding="utf-8").splitlines()

    start_idx: dict[str, int] = {}
    for i, line in enumerate(lines):
        if line.startswith("  ") and line.rstrip().endswith(": {"):
            name = line.strip().split(":", 1)[0]
            start_idx[name] = i

    def find_field_end(start: int) -> int | None:
        brace = 0
        in_field = False
        for i in range(start, len(lines)):
            s = lines[i]
            brace += s.count("{") - s.count("}")
            if "{" in s and not in_field:
                in_field = True
            if in_field and brace == 0 and s.strip().endswith("},"):
                return i
        return None

    def has_aiprompt(start: int, end: int) -> bool:
        for i in range(start, end + 1):
            if "aiPrompt:" in lines[i]:
                return True
        return False

    missing: list[str] = []
    not_found: list[str] = []

    for item in fields:
        name = item["fieldName"]
        if name not in start_idx:
            not_found.append(name)
            continue
        end = find_field_end(start_idx[name])
        if end is None:
            not_found.append(name)
            continue
        if not has_aiprompt(start_idx[name], end):
            missing.append(name)

    print(f"Fields still missing aiPrompt: {len(missing)}")
    for n in missing:
        print(f"- {n}")

    print(f"\nFields not found or parse failed: {len(not_found)}")
    for n in not_found:
        print(f"- {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
