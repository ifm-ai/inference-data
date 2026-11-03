#!/usr/bin/env python3
"""Add the missing concurrency-1 launch receipt to published cells.

A published cell records one ``server_launch`` entry per shipped concurrency.
Wave 5 added concurrency-1 points to cells whose ``environment.json`` carried
only a ``"2"`` entry, so the points ship without a launch receipt and
``scripts/reproduce.py`` cannot emit their server command.

The campaign archive still holds the launch that served them. This reads that
receipt, rewrites the checkpoint path to the ``<models-root>/`` placeholder
used by every published entry, and inserts it as ``server_launch["1"]``.

The archive record is accepted only when it is the launch those points ran on:

* every attempt in the config that holds one of the cell's published points
  must agree on the server configuration (dtype, max-num-seqs, tp size,
  context length, prefix caching, served name) — ports differ per attempt and
  are not part of the comparison;
* every published concurrency-1 point must be covered by an attempt;
* the archived ``--max-num-seqs`` must equal the concurrency.

Any disagreement is fatal: a receipt that does not describe the measurement is
worse than a missing one. Existing ``server_launch["1"]`` entries are never
overwritten.

It also repairs two cells whose ``server_launch["2"]`` is byte-identical to
their ``"1"`` entry: their c2 points were measured by the maxc-phase1
search-capacity sweep, whose c2/ attempt holds the receipt that actually
served them (--max-num-seqs 2). The duplicate is replaced only when the
attempt's request shapes equal the cell's published c2 points, so a receipt
that cannot be shown to serve the measurement is never installed.

Usage:
    backfill_launch_receipts.py --repo <worktree> --incoming <archive .incoming>
        [--dry-run] [--models-root /mnt/weka/shrd/k2m/micah.villmow/models]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Published directory -> campaign config whose attempts served it.
CONFIGS = {
    "k2-horizon-3.7b/1x4/vllm/bf16": "3.7b-bf16-vllm-g4",
    "k2-horizon-3.7b/1x8/vllm/bf16": "3.7b-bf16-vllm-g8",
    "k2-horizon-32b/1x8/vllm/bf16": "32b-bf16-vllm-g8",
    "k2-horizon-32b/1x8/vllm/fp8": "32b-fp8-vllm-g8",
    "k2-horizon-7b/1x4/vllm/bf16": "7b-bf16-vllm-g4",
    "k2-horizon-7b/1x4/vllm/fp8": "7b-fp8-vllm-g4",
    "k2-horizon-7b/1x8/vllm/bf16": "7b-bf16-vllm-g8",
    "k2-horizon-7b/1x8/vllm/fp8": "7b-fp8-vllm-g8",
    "k2-horizon-900m/1x4/vllm/bf16": "900m-bf16-vllm-g4",
    "k2-horizon-900m/1x8/vllm/bf16": "900m-bf16-vllm-g8",
    "k2-horizon-MoVA-36b-a4b/1x8/vllm/bf16": "36b-mova-bf16-vllm-g8",
    "k2-horizon-MoVA-36b-a4b/1x8/vllm/fp8": "36b-mova-fp8-vllm-g8",
}

POINT_RE = re.compile(r"isl(\d+)-osl(\d+)")
REQUEST_SEQ_RE = re.compile(r"isl-(\d+)-osl-(\d+)")

# Cells whose published "2" entry is a copy of their "1" entry. Their c2
# points came from the maxc-phase1 search-capacity sweep, whose c2/ attempt
# holds the receipt that actually served it. Keys: cell dir -> receipt path
# scaffold relative to the maxc-phase1 performance root.
REPAIRS_C2 = {
    "k2-horizon-900m/1x2/vllm/bf16": (
        "900m-bf16-vllm-g2-tp-ep-maxc/search-capacity/c1/attempt-2593794/c2"
    ),
    "k2-horizon-MoVA-36b-a4b/1x4/vllm/bf16": (
        "36b-mova-bf16-vllm-g4-tp-ep-maxc/search-capacity/c1/attempt-2593840/c2"
    ),
}


def server_config(argv: list[str]) -> tuple:
    """Server settings that must agree across every attempt of a cell.

    The port and the checkpoint path vary per attempt and with the archive
    location, so neither takes part in the comparison.
    """
    def value(flag: str) -> str | None:
        return argv[argv.index(flag) + 1] if flag in argv else None

    return (
        value("--dtype"),
        value("--max-num-seqs"),
        value("--tensor-parallel-size"),
        value("--max-model-len"),
        "--no-enable-prefix-caching" in argv,
        value("--served-model-name"),
    )


def published_points(cell: Path) -> set[tuple[int, int]]:
    points = set()
    for conc in cell.glob("isl*/c1"):
        match = POINT_RE.fullmatch(conc.parent.name)
        if match:
            points.add((int(match.group(1)), int(match.group(2))))
    return points


def attempts_covering(config_dir: Path, points: set[tuple[int, int]]) -> list[Path]:
    covering = []
    for attempt in sorted(config_dir.iterdir()):
        if not attempt.is_dir():
            continue
        served = set()
        for name in os.listdir(attempt):
            match = REQUEST_SEQ_RE.search(name)
            if match:
                served.add((int(match.group(1)), int(match.group(2))))
        if served & points:
            covering.append(attempt)
    return covering


def build_entry(argv: list[str], models_root: str) -> dict:
    """Published launch entry: placeholder path, argv plus rendered shell."""
    published = [
        arg.replace(models_root.rstrip("/") + "/", "<models-root>/") for arg in argv
    ]
    shell = " ".join(published)
    return {"argv": published, "shell": shell}


def attempt_points(attempt: Path) -> set[tuple[int, int]]:
    """The (isl, osl) workpoints one search-capacity attempt measured.

    The request files name the shapes only as decode-N / prefill-N; their
    ``sequence_length`` and ``output_tokens`` fields hold the real counts.
    A point is the rest of the context not spent on output:
    ``isl = sequence_length - output_tokens``. Prefill and boundary requests
    collapse to the same pair, decode to its own.
    """
    points = set()
    for request in attempt.glob("*.request.json"):
        data = json.loads(request.read_text())
        length = data.get("sequence_length")
        output = data.get("output_tokens")
        if type(length) is not int or type(output) is not int or length <= output:
            continue
        points.add((length - output, output))
    return points


def repair_c2(
    repo: Path,
    maxc_performance: Path,
    models_root: str,
    *,
    dry_run: bool,
) -> int:
    """Replace duplicated c2 entries with the receipt that served their points.

    Fails a cell rather than overwrite it when the receipt it would install
    cannot be shown to serve the cell's published c2 points, or when the
    entry already present is not the c1 duplicate this repair exists for.
    """
    failures = 0
    for cell_rel, attempt_rel in REPAIRS_C2.items():
        cell = repo / "perf" / cell_rel
        env_path = cell / "environment.json"
        if not env_path.is_file():
            print(f"SKIP {cell_rel}: no environment.json")
            failures += 1
            continue
        environment = json.loads(env_path.read_text())
        launches = environment.get("server_launch") or {}
        one, two = launches.get("1"), launches.get("2")
        if one is None:
            print(f"SKIP {cell_rel}: no concurrency-1 entry to compare against")
            failures += 1
            continue
        if two is not None and two != one:
            print(f"KEEP {cell_rel}: c2 entry already distinct from c1")
            continue

        attempt = maxc_performance / attempt_rel
        receipt = attempt / "launch.json"
        if not receipt.is_file():
            print(f"FAIL {cell_rel}: missing {receipt}")
            failures += 1
            continue
        launch = json.loads(receipt.read_text())
        argvs = launch.get("argv") or launch["shell"].split()

        points = set()
        for conc in cell.glob("isl*/c2"):
            match = POINT_RE.fullmatch(conc.parent.name)
            if match:
                points.add((int(match.group(1)), int(match.group(2))))
        served = attempt_points(attempt)
        if points != served:
            print(
                f"FAIL {cell_rel}: published c2 points {sorted(points)} do not "
                f"match attempt shapes {sorted(served)}"
            )
            failures += 1
            continue
        def value(flag: str) -> str | None:
            return argvs[argvs.index(flag) + 1] if flag in argvs else None

        if value("--max-num-seqs") != "2":
            print(
                f"FAIL {cell_rel}: receipt admits --max-num-seqs "
                f"{value('--max-num-seqs')}, the points ran at concurrency 2"
            )
            failures += 1
            continue
        if server_config(argvs) == server_config(launches["1"]["argv"]):
            print(f"FAIL {cell_rel}: receipt is indistinguishable from the c1 entry")
            failures += 1
            continue

        entry = build_entry(argvs, models_root)
        if "<models-root>/" not in " ".join(entry["argv"]):
            print(f"FAIL {cell_rel}: checkpoint path not rewritten to the placeholder")
            failures += 1
            continue

        verb = "would replace" if dry_run else "replaced"
        print(
            f"OK   {cell_rel}: {verb} c2 launch from {attempt_rel} "
            f"({len(points)} points, --max-num-seqs 2)"
        )
        if not dry_run:
            launches["2"] = entry
            environment["server_launch"] = {
                key: launches[key] for key in sorted(launches, key=int)
            }
            env_path.write_text(json.dumps(environment, indent=2) + "\n")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument(
        "--models-root", default="/mnt/weka/shrd/k2m/micah.villmow/models"
    )
    parser.add_argument(
        "--maxc-performance",
        type=Path,
        default=Path("/mnt/weka/shrd/k2m/micah.villmow/models/.chapter81-runs/maxc-phase1/performance"),
        help="maxc-phase1 performance root, scanned only for the c2 repair map",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    failures = 0
    for cell_rel, config in CONFIGS.items():
        cell = args.repo / "perf" / cell_rel
        env_path = cell / "environment.json"
        if not env_path.is_file():
            print(f"SKIP {cell_rel}: no environment.json")
            failures += 1
            continue
        environment = json.loads(env_path.read_text())
        launches = environment.get("server_launch") or {}
        if "1" in launches:
            print(f"KEEP {cell_rel}: already records concurrency 1")
            continue

        points = published_points(cell)
        if not points:
            print(f"SKIP {cell_rel}: ships no concurrency-1 points")
            continue

        config_dir = args.incoming / config
        covering = attempts_covering(config_dir, points)
        if not covering:
            print(f"FAIL {cell_rel}: no attempt of {config} covers any c1 point")
            failures += 1
            continue

        configs, entries, missing = {}, {}, []
        covered = set()
        for attempt in covering:
            receipt = attempt / "launch.json"
            if not receipt.is_file():
                missing.append(attempt.name)
                continue
            launch = json.loads(receipt.read_text())
            argvs = launch.get("argv") or []
            if launch.get("shell"):
                argvs = launch["shell"].split()
            entries.setdefault(server_config(argvs), []).append(attempt.name)
            configs.setdefault(server_config(argvs), argvs)
            for name in os.listdir(attempt):
                match = REQUEST_SEQ_RE.search(name)
                if match:
                    covered.add((int(match.group(1)), int(match.group(2))))

        if missing:
            print(f"FAIL {cell_rel}: attempts without launch.json: {missing}")
            failures += 1
            continue
        if len(entries) != 1:
            print(f"FAIL {cell_rel}: {len(entries)} distinct server configs:")
            for signature, attempts in entries.items():
                print(f"      {signature} <- {attempts[:4]}")
            failures += 1
            continue
        uncovered = points - covered
        if uncovered:
            print(f"FAIL {cell_rel}: no attempt for {sorted(uncovered)}")
            failures += 1
            continue

        signature, attempts = next(iter(entries.items()))
        if signature[1] != "1":
            print(
                f"FAIL {cell_rel}: archived --max-num-seqs is {signature[1]}, "
                "not the concurrency"
            )
            failures += 1
            continue

        launch = json.loads((config_dir / attempts[0] / "launch.json").read_text())
        raw = launch.get("argv") or launch["shell"].split()
        entry = build_entry(raw, args.models_root)
        if "<models-root>/" not in " ".join(entry["argv"]):
            print(f"FAIL {cell_rel}: checkpoint path not rewritten to the placeholder")
            failures += 1
            continue

        launches["1"] = entry
        environment["server_launch"] = {
            key: launches[key] for key in sorted(launches, key=int)
        }
        verb = "would add" if args.dry_run else "added"
        print(
            f"OK   {cell_rel}: {verb} concurrency-1 launch from {len(attempts)} "
            f"attempt(s), {len(points)} points, port "
            f"{entry['argv'][entry['argv'].index('--port') + 1] if '--port' in entry['argv'] else '?'}"
        )
        if not args.dry_run:
            env_path.write_text(json.dumps(environment, indent=2) + "\n")

    failures += repair_c2(
        args.repo,
        args.maxc_performance,
        args.models_root,
        dry_run=args.dry_run,
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())