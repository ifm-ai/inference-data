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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument(
        "--models-root", default="/mnt/weka/shrd/k2m/micah.villmow/models"
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

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())