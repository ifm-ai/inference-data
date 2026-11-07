#!/usr/bin/env python3
"""Validate a published inference-data tree and report every inconsistency.

The dataset makes promises that no existing script checked: every point holds
exactly three independently decompressible repetitions, every record keeps the
same shape, every point's ``c<N>`` directory is served by a server that admits
``N`` requests, every cell's receipt matches its directory, each model README
states the totals the tree actually holds, and a placement measured in one
dtype covers the same point shapes in its sibling dtype. ``reproduce.py``
re-emits launch commands and the publisher only writes; neither reads the
shipped tree back. This tool does.

Exit status is 0 when nothing is reported, 1 when any ERROR is reported, and 2
when only WARNINGs are reported. Each finding carries a stable code so a known
condition can be allowed with ``--allow`` instead of being re-explained.

Usage:
    python3 scripts/validate_dataset.py [--dataset-root perf] [--json]
                                       [--max-report 40] [--allow CODE[,CODE]]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = (1, 2, 3)

# A server may stop this many output tokens short of a maximum-length request.
# Both runtimes do it consistently; a longer shortfall means the workload did
# not complete as requested.
SHORTFALL_TOLERANCE = 3

# A record has this shape everywhere in the tree except where an older
# publisher path was used. Both key sets are accepted only under --allow.
STD_RECORD_KEYS = frozenset(
    {
        "concurrency",
        "observed_running",
        "qualification",
        "scope",
        "streams",
        "token_ids_requested",
        "upstream",
    }
)
# The capacity plan writes a leaner record through the single-request path.
LEAN_RECORD_KEYS = frozenset(
    {"prompt_token_ids", "qualification", "raw_sse", "scope", "upstream"}
)

LEAF_RE = re.compile(r"^run(\d+)_results\.json\.gz$")
PLACEMENT_RE = re.compile(r"^(\d+)x(\d+)$")
POINT_RE = re.compile(r"^isl(\d+)-osl(\d+)$")
CONC_RE = re.compile(r"^c(\d+)$")
README_TOTAL_RE = re.compile(
    r"\*\*(\d+) points / (\d+) leaves / ([\d.]+) MB\.\*\*"
)
README_ROW_RE = re.compile(r"^\| `([^`]+)` \| (\d+) \| (\d+) \|$", re.M)

# Findings that are real but explainable, reported as WARNING not ERROR.
ERROR = "ERROR"
WARNING = "WARNING"


class Report:
    def __init__(self, allowed: set[str]) -> None:
        self.allowed = allowed
        self.counts: Counter[str] = Counter()
        self.items: list[tuple[str, str, str]] = []

    def add(self, severity: str, code: str, detail: str) -> None:
        if code in self.allowed:
            return
        self.counts[code] += 1
        self.items.append((severity, code, detail))

    def error(self, code: str, detail: str) -> None:
        self.add(ERROR, code, detail)

    def warn(self, code: str, detail: str) -> None:
        self.add(WARNING, code, detail)

    @property
    def errors(self) -> int:
        return sum(1 for s, _, _ in self.items if s == ERROR)

    @property
    def warnings(self) -> int:
        return sum(1 for s, _, _ in self.items if s == WARNING)


def nearest_rank(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[min(rank - 1, len(ordered) - 1)]


def load_leaf(path: Path, report: Report, rel: str) -> dict | None:
    try:
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as error:
        report.error("LEAF_UNREADABLE", f"{rel}: {type(error).__name__}: {error}")
        return None
    if not isinstance(data, dict):
        report.error("LEAF_NOT_OBJECT", f"{rel}: top level is {type(data).__name__}")
        return None
    return data


def check_record_shape(data: dict, rel: str, report: Report) -> None:
    keys = frozenset(data)
    if keys == STD_RECORD_KEYS:
        return
    if keys == LEAN_RECORD_KEYS:
        # The capacity/single-request path emits this. It omits the identity
        # fields and the admission audit a reader may rely on.
        report.warn(
            "RECORD_LEAN_SHAPE",
            f"{rel}: record omits concurrency/observed_running/streams/token_ids_requested",
        )
        return
    missing = sorted(STD_RECORD_KEYS - keys)
    extra = sorted(keys - STD_RECORD_KEYS)
    report.error(
        "RECORD_SHAPE", f"{rel}: missing {missing or '-'} extra {extra or '-'}"
    )


def check_upstream(
    data: dict, rel: str, isl: int, osl: int, concurrency: int, report: Report
) -> None:
    upstream = data.get("upstream")
    if not isinstance(upstream, dict):
        report.error("UPSTREAM_MISSING", f"{rel}: no upstream object")
        return
    input_lens = upstream.get("input_lens")
    output_lens = upstream.get("output_lens")
    if not isinstance(input_lens, list) or not input_lens:
        report.error("INPUT_LENS_MISSING", f"{rel}: input_lens absent or empty")
    elif input_lens[0] != isl:
        report.error(
            "INPUT_LEN_MISMATCH", f"{rel}: input_lens {input_lens} vs isl {isl}"
        )
    if not isinstance(output_lens, list) or not output_lens:
        report.error("OUTPUT_LENS_MISSING", f"{rel}: output_lens absent or empty")

    if isinstance(input_lens, list) and input_lens and len(input_lens) != concurrency:
        report.error(
            "REQUEST_COUNT_MISMATCH",
            f"{rel}: {len(input_lens)} requests recorded for c{concurrency}",
        )

    itls = upstream.get("itls_us")
    if not isinstance(itls, list):
        report.error("ITLS_MISSING", f"{rel}: upstream.itls_us absent")
        return
    stats = upstream.get("itl_us_stats")
    for index, (series, stat) in enumerate(zip(itls, stats or [])):
        if not isinstance(series, list):
            report.error("ITLS_NOT_LIST", f"{rel}: itls_us[{index}] is not a list")
            continue
        if isinstance(output_lens, list) and index < len(output_lens):
            achieved = output_lens[index]
            if series and len(series) != max(achieved - 1, 0):
                # The runtime reported fewer intervals than it generated
                # tokens. Timing coverage is then incomplete for this request.
                report.warn(
                    "ITLS_SHORT",
                    f"{rel}: request {index} has {len(series)} intervals for "
                    f"{achieved} output tokens",
                )
        if stat is None:
            continue
        if stat.get("count") != len(series):
            report.error(
                "ITL_STATS_COUNT",
                f"{rel}: request {index} stats count {stat.get('count')} "
                f"vs {len(series)} intervals",
            )
        if series and stat.get("min") != min(series):
            report.error("ITL_STATS_MIN", f"{rel}: request {index} min mismatch")
        if series and stat.get("max") != max(series):
            report.error("ITL_STATS_MAX", f"{rel}: request {index} max mismatch")
        if series and stat.get("p50") != nearest_rank(series, 0.50):
            report.error("ITL_STATS_P50", f"{rel}: request {index} p50 mismatch")


def check_identity(
    data: dict, rel: str, concurrency: int, point: str, report: Report
) -> None:
    recorded = data.get("concurrency")
    if recorded is None:
        report.warn(
            "CONCURRENCY_UNRECORDED",
            f"{rel}: record carries no concurrency; directory says c{concurrency}",
        )
    elif recorded != concurrency:
        report.error(
            "CONCURRENCY_MISMATCH",
            f"{rel}: record concurrency {recorded} vs directory c{concurrency}",
        )
    observed = data.get("observed_running")
    if isinstance(observed, int) and observed > concurrency:
        report.error(
            "ADMISSION_EXCEEDS_CONCURRENCY",
            f"{rel}: observed_running {observed} > c{concurrency}",
        )
    got = data["upstream"].get("output_lens") if isinstance(data.get("upstream"), dict) else None
    if isinstance(got, list) and got:
        osl = int(POINT_RE.match(point).group(2))
        shortfall = osl - got[0]
        if shortfall > SHORTFALL_TOLERANCE:
            # The runtimes stop a couple of tokens short of a maximum-length
            # request by design. A longer shortfall means the workload did not
            # complete as requested.
            report.warn(
                "OUTPUT_SHORTFALL",
                f"{rel}: requested osl {osl}, server returned {got[0]} "
                f"(short by {shortfall}, tolerance {SHORTFALL_TOLERANCE})",
            )
        elif shortfall > 0:
            report.warn(
                "OUTPUT_SHORTFALL_WITHIN_TOLERANCE",
                f"{rel}: requested osl {osl}, server returned {got[0]} "
                f"(short by {shortfall})",
            )
        elif shortfall < 0:
            report.error(
                "OUTPUT_OVER_REQUESTED",
                f"{rel}: requested osl {osl}, server returned {got[0]}",
            )


def check_cell(
    cell_dir: Path, placement: str, runtime: str, dtype: str, report: Report
) -> tuple[int, int, int]:
    """Validate one cell. Return (points, leaves, bytes).

    ``cell_dir`` is the ``<dtype>`` directory; ``placement``, ``runtime`` and
    ``dtype`` are its three ancestors' names.
    """
    rel_cell = cell_dir.as_posix()
    env_path = cell_dir / "environment.json"
    points = leaves = size = 0
    if not env_path.is_file():
        report.error("CELL_NO_ENVIRONMENT", f"{rel_cell}: no environment.json")
        env = None
    else:
        try:
            env = json.loads(env_path.read_text())
        except ValueError as error:
            report.error("CELL_ENVIRONMENT_INVALID", f"{rel_cell}: {error}")
            env = None

    match = PLACEMENT_RE.match(placement)
    total_devices = None
    if not match:
        report.error("CELL_BAD_PLACEMENT", f"{rel_cell}: {placement!r} is not <nodes>x<gpus>")
    else:
        nodes, per_node = int(match.group(1)), int(match.group(2))
        total_devices = nodes * per_node
        if nodes < 1 or per_node < 1:
            report.error("CELL_BAD_PLACEMENT", f"{rel_cell}: non-positive placement")

    launches: dict[str, dict] = {}
    if env is not None:
        cell = env.get("cell") or {}
        if cell.get("topology") != placement:
            report.error(
                "CELL_TOPOLOGY",
                f"{rel_cell}: cell.topology {cell.get('topology')!r} vs {placement!r}",
            )
        if cell.get("runtime") != runtime:
            report.error(
                "CELL_RUNTIME",
                f"{rel_cell}: cell.runtime {cell.get('runtime')!r} vs {runtime!r}",
            )
        if cell.get("dtype") != dtype:
            report.error(
                "CELL_DTYPE",
                f"{rel_cell}: cell.dtype {cell.get('dtype')!r} vs {dtype!r}",
            )
        if total_devices is not None and cell.get("gpus") != total_devices:
            report.error(
                "CELL_GPUS",
                f"{rel_cell}: cell.gpus {cell.get('gpus')} vs {total_devices}",
            )
        system = env.get("system") or {}
        if total_devices is not None:
            if system.get("device_count") != total_devices:
                report.error(
                    "CELL_DEVICE_COUNT",
                    f"{rel_cell}: device_count {system.get('device_count')} vs {total_devices}",
                )
            devices = system.get("devices") or []
            if len(devices) != total_devices:
                report.error(
                    "CELL_DEVICE_LIST",
                    f"{rel_cell}: len(devices) {len(devices)} vs {total_devices}",
                )
        if match and int(match.group(1)) > 1:
            if cell.get("nodes") != int(match.group(1)):
                report.error("CELL_NODES", f"{rel_cell}: cell.nodes not {match.group(1)}")
            if cell.get("gpus_per_node") != int(match.group(2)):
                report.error(
                    "CELL_GPUS_PER_NODE", f"{rel_cell}: cell.gpus_per_node not {match.group(2)}"
                )
        launches = env.get("server_launch") or {}
        if not isinstance(launches, dict) or not launches:
            report.error("CELL_NO_LAUNCH", f"{rel_cell}: server_launch empty")

    for point_dir in sorted(
        p for p in cell_dir.iterdir() if p.is_dir() and POINT_RE.match(p.name)
    ):
        pm = POINT_RE.match(point_dir.name)
        isl, osl = int(pm.group(1)), int(pm.group(2))
        for conc_dir in sorted(
            p for p in point_dir.iterdir() if p.is_dir() and CONC_RE.match(p.name)
        ):
            concurrency = int(CONC_RE.match(conc_dir.name).group(1))
            rel_point = f"{conc_dir.as_posix()}"
            points += 1
            if str(concurrency) not in launches and env is not None:
                report.error(
                    "POINT_NO_LAUNCH",
                    f"{rel_point}: no server_launch entry for c{concurrency}",
                )
            found = sorted(
                int(LEAF_RE.match(entry.name).group(1))
                for entry in conc_dir.iterdir()
                if LEAF_RE.match(entry.name)
            )
            extra = sorted(
                entry.name
                for entry in conc_dir.iterdir()
                if not LEAF_RE.match(entry.name)
            )
            for name in extra:
                report.error("POINT_EXTRA_FILE", f"{rel_point}: unexpected {name}")
            if found != list(RUNS):
                report.error(
                    "POINT_RUN_SET",
                    f"{rel_point}: runs {found} (expected {list(RUNS)})",
                )
            per_run_itl: dict[int, tuple[int, ...] | None] = {}
            per_run_out: dict[int, tuple[int, ...] | None] = {}
            for run in found:
                leaf = conc_dir / f"run{run}_results.json.gz"
                rel = leaf.as_posix()
                size += leaf.stat().st_size
                data = load_leaf(leaf, report, rel)
                if data is None:
                    continue
                leaves += 1
                check_record_shape(data, rel, report)
                check_upstream(data, rel, isl, osl, concurrency, report)
                check_identity(data, rel, concurrency, point_dir.name, report)
                upstream = data.get("upstream") or {}
                itls = upstream.get("itls_us")
                per_run_itl[run] = (
                    tuple(len(x) for x in itls) if isinstance(itls, list) else None
                )
                outs = upstream.get("output_lens")
                per_run_out[run] = tuple(outs) if isinstance(outs, list) else None
            # Repetitions of one point must agree on what they produced.
            shapes = {v for v in per_run_itl.values() if v is not None}
            if len(shapes) > 1:
                report.warn(
                    "RUNS_DISAGREE_ITL",
                    f"{rel_point}: interval counts differ between runs {sorted(shapes)}",
                )
            outcomes = {v for v in per_run_out.values() if v is not None}
            if len(outcomes) > 1:
                report.error(
                    "RUNS_DISAGREE_OUTPUT",
                    f"{rel_point}: output_lens differ between runs {sorted(outcomes)}",
                )
    return points, leaves, size


def point_shapes(cell_dir: Path) -> set[str]:
    """Every ``isl<I>-osl<O>/c<N>`` shape a cell holds, as one comparable set."""
    return {
        f"{point_dir.name}/{conc_dir.name}"
        for point_dir in cell_dir.iterdir()
        if point_dir.is_dir() and POINT_RE.match(point_dir.name)
        for conc_dir in point_dir.iterdir()
        if conc_dir.is_dir() and CONC_RE.match(conc_dir.name)
    }


def check_dtype_parity(
    model_dir: Path,
    shapes: dict[tuple[str, str], dict[str, set[str]]],
    report: Report,
) -> None:
    """Compare point coverage between dtypes within one placement and runtime.

    ``shapes`` maps ``(placement, runtime)`` to ``{dtype: {point/conc, ...}}``.
    A dtype pair must cover the same point shapes: if a BF16 run exists for a
    shape, the same shape must exist for the other dtype. Missing coverage in
    either direction is reported separately, because "not yet measured" and
    "measured in one dtype only" need different follow-up.
    """
    for (placement, runtime), by_dtype in sorted(shapes.items()):
        if len(by_dtype) < 2:
            dtypes = sorted(by_dtype)
            counts = {d: len(by_dtype[d]) for d in dtypes}
            report.error(
                "DTYPE_PAIR_INCOMPLETE",
                f"{model_dir.as_posix()}/{placement}/{runtime}: only "
                f"{dtypes[0]} measured ({counts[dtypes[0]]} points); "
                f"no matching {'fp8' if dtypes[0] == 'bf16' else 'bf16'} cell",
            )
            continue
        dtypes = sorted(by_dtype)
        for index, left in enumerate(dtypes):
            for right in dtypes[index + 1:]:
                left_set, right_set = by_dtype[left], by_dtype[right]
                if left_set == right_set:
                    continue
                cell = f"{model_dir.as_posix()}/{placement}/{runtime}"
                if left_set - right_set:
                    report.error(
                        "DTYPE_MISSING_COVERAGE",
                        f"{cell}: {len(left_set - right_set)} point(s) measured in "
                        f"{left} but not {right}: "
                        + ", ".join(sorted(left_set - right_set)),
                    )
                if right_set - left_set:
                    report.error(
                        "DTYPE_MISSING_COVERAGE",
                        f"{cell}: {len(right_set - left_set)} point(s) measured in "
                        f"{right} but not {left}: "
                        + ", ".join(sorted(right_set - left_set)),
                    )


def check_readme(
    model_dir: Path,
    points: int,
    leaves: int,
    size: int,
    per_cell: dict[str, tuple[int, int]],
    report: Report,
) -> None:
    rel = f"{model_dir.as_posix()}/README.md"
    readme = model_dir / "README.md"
    if not readme.is_file():
        report.error("README_MISSING", f"{rel}: model has no README")
        return
    text = readme.read_text()
    match = README_TOTAL_RE.search(text)
    actual_mb = round(size / 1e6, 1)
    actual = (points, leaves, actual_mb)
    if not match:
        report.error("README_NO_TOTALS", f"{rel}: no totals line")
    else:
        claimed = (int(match.group(1)), int(match.group(2)), float(match.group(3)))
        if claimed != actual:
            report.error(
                "README_TOTALS_STALE",
                f"{rel}: states {claimed}, tree holds {actual}",
            )
    rows = {
        m.group(1): (int(m.group(2)), int(m.group(3)))
        for m in README_ROW_RE.finditer(text)
    }
    if rows != per_cell:
        only_readme = sorted(set(rows) - set(per_cell))
        only_tree = sorted(set(per_cell) - set(rows))
        differing = sorted(
            c for c in set(rows) & set(per_cell) if rows[c] != per_cell[c]
        )
        detail = []
        if only_readme:
            detail.append(f"listed but absent {only_readme}")
        if only_tree:
            detail.append(f"present but unlisted {only_tree}")
        if differing:
            detail.append(
                "counts differ "
                + ", ".join(f"{c} {rows[c]}->{per_cell[c]}" for c in differing)
            )
        report.error("README_ROWS_STALE", f"{rel}: " + "; ".join(detail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", default="perf")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--max-report", type=int, default=40)
    parser.add_argument(
        "--allow",
        default="",
        help="comma-separated finding codes to accept silently",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    allowed = {code.strip() for code in args.allow.split(",") if code.strip()}
    report = Report(allowed)

    model_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    total_points = total_leaves = total_size = 0
    for model_dir in model_dirs:
        model_points = model_leaves = model_size = 0
        per_cell: dict[str, tuple[int, int]] = {}
        shapes: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        cell_dirs = sorted(
            p for p in model_dir.iterdir() if p.is_dir() and PLACEMENT_RE.match(p.name)
        )
        if not cell_dirs:
            report.error("MODEL_NO_CELLS", f"{model_dir.as_posix()}: no <nodes>x<gpus> cells")
        for placement_dir in cell_dirs:
            for runtime_dir in sorted(p for p in placement_dir.iterdir() if p.is_dir()):
                for dtype_dir in sorted(p for p in runtime_dir.iterdir() if p.is_dir()):
                    points, leaves, size = check_cell(
                        dtype_dir,
                        placement_dir.name,
                        runtime_dir.name,
                        dtype_dir.name,
                        report,
                    )
                    model_points += points
                    model_leaves += leaves
                    model_size += size
                    key = f"{placement_dir.name}/{runtime_dir.name}/{dtype_dir.name}"
                    per_cell[key] = (points, leaves)
                    shapes[(placement_dir.name, runtime_dir.name)][dtype_dir.name].update(
                        point_shapes(dtype_dir)
                    )
        check_readme(model_dir, model_points, model_leaves, model_size, per_cell, report)
        check_dtype_parity(model_dir, shapes, report)
        total_points += model_points
        total_leaves += model_leaves
        total_size += model_size

    if args.json:
        print(
            json.dumps(
                {
                    "models": len(model_dirs),
                    "points": total_points,
                    "leaves": total_leaves,
                    "bytes": total_size,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "findings": [
                        {"severity": s, "code": c, "detail": d}
                        for s, c, d in report.items
                    ],
                },
                indent=2,
            )
        )
    else:
        for severity, code, detail in report.items[: args.max_report]:
            print(f"{severity} {code}: {detail}")
        if len(report.items) > args.max_report:
            print(f"... {len(report.items) - args.max_report} more findings")
        print()
        print(
            f"{len(model_dirs)} models / {total_points} points / {total_leaves} leaves "
            f"/ {total_size / 1e6:.1f} MB"
        )
        print(f"{report.errors} error(s), {report.warnings} warning(s)")

    if report.errors:
        return 1
    return 2 if report.warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())