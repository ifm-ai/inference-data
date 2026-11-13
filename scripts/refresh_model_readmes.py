#!/usr/bin/env python3
"""Refresh each model README's totals line and per-cell table from the tree.

The README prose is hand-maintained; only the two data statements
(``**N points / N leaves / N MB.**`` and the cell table rows) are rewritten,
so regenerating after a campaign is mechanical and the validator's
README_TOTALS_STALE / README_ROWS_STALE checks stay satisfied.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TOTAL_RE = re.compile(r"\*\*(\d+) points / (\d+) leaves / ([\d.]+) MB\.\*\*")
ROW_RE = re.compile(r"^\| `([^`]+)` \| (\d+) \| (\d+) \|$", re.M)


def refresh(readme: Path, points: int, leaves: int, mb: float,
            per_cell: dict[str, tuple[int, int]]) -> bool:
    text = readme.read_text()
    new_total = f"**{points} points / {leaves} leaves / {mb} MB.**"
    text, n = TOTAL_RE.subn(new_total, text, count=1)
    if n != 1:
        print(f"{readme}: no totals line matched", file=sys.stderr)
        return False

    def row(match: re.Match) -> str:
        cell = match.group(1)
        p, l = per_cell.get(cell, (int(match.group(2)), int(match.group(3))))
        return f"| `{cell}` | {p} | {l} |"

    text, n = ROW_RE.subn(row, text)
    if n != len(per_cell):
        print(f"{readme}: table rows {n} != cells {len(per_cell)}", file=sys.stderr)
        return False
    readme.write_text(text)
    return True


def main() -> int:
    perf = Path(__file__).resolve().parent.parent / "perf"
    ok = True
    for readme in sorted(perf.glob("k2-horizon-*/README.md")):
        model = readme.parent
        points = leaves = 0
        size = 0
        per_cell: dict[str, tuple[int, int]] = {}
        for placement in sorted(p for p in model.iterdir() if p.is_dir()):
            for runtime in sorted(p for p in placement.iterdir() if p.is_dir()):
                for dtype in sorted(p for p in runtime.iterdir() if p.is_dir()):
                    cp = cl = 0
                    for point in dtype.iterdir():
                        if not point.name.startswith("isl"):
                            continue
                        for run in point.iterdir():
                            if not run.name.startswith("c"):
                                continue
                            cp += 1
                            for f in run.iterdir():
                                if f.name.endswith(".json.gz"):
                                    cl += 1
                                    size += f.stat().st_size
                    key = dtype.relative_to(model).as_posix()
                    per_cell[key] = (cp, cl)
                    points += cp
                    leaves += cl
        mb = round(size / 1e6, 1)
        if refresh(readme, points, leaves, mb, per_cell):
            print(f"{readme.relative_to(perf.parent)}: {points} points / {leaves} leaves / {mb} MB")
        else:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
