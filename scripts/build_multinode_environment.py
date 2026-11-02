#!/usr/bin/env python3
"""Build the environment.json for a multi-node K2-Horizon cell.

The published single-node cells carry `cell.gpus` and `system.device_count`
equal to the device count on their one node. A multi-node cell cannot be
described that way: its placement is several nodes that each expose 8 GPUs,
and the total is a property of the placement rather than of any one host.

This derives a `<nodes>x<gpus-per-node>` cell from the published sibling in the
same model, runtime, and dtype so the schema, key order, and component pins
stay identical to what is already shipped, then fills the placement fields and
the launch receipt from the run that produced the cell.

Usage:
    build_multinode_environment.py --sibling <published environment.json>
        --archive-point <archive cell dir> --nodes <n> --gpus-per-node <n>
        --output <environment.json>

`--archive-point` is the directory the archiver wrote for the cell, for example
<archive>/<model>/2x8/sglang/bf16. Its `server_launch` and device inventory are
read from the attempt receipts so the published launch line is the one that
actually ran.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_inventory(path: Path) -> list[dict]:
    """Read a per-GPU inventory, tolerating the spacing nvidia-smi emits.

    `nvidia-smi --format=csv` writes a space after each comma and leaves the
    units in the value, so headers arrive as " uuid" and the memory column as
    "memory.total [MiB]". Normalize both when reading.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        header = [(name or "").strip() for name in (reader.fieldnames or [])]
        rows = []
        for raw in reader:
            row = {
                (key or "").strip(): (value or "").strip()
                for key, value in raw.items()
            }
            # One srun task per node, each writing its own header into the same
            # redirected file, so repeated header lines arrive as data.
            if [row.get(name, "") for name in header] == header:
                continue
            if row.get("uuid"):
                rows.append(row)
        return rows


def derive(
    sibling: dict,
    *,
    nodes: int,
    gpus_per_node: int,
    devices: list[str],
    launch: dict,
) -> dict:
    """Return the multi-node environment.json derived from a published sibling."""
    total = nodes * gpus_per_node
    if len(devices) != total:
        raise SystemExit(
            f"inventory lists {len(devices)} GPUs, placement needs {total}"
        )
    out = json.loads(json.dumps(sibling))  # preserve key order and shape
    out["cell"] = {
        **out["cell"],
        "topology": f"{nodes}x{gpus_per_node}",
        "gpus": total,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
    }
    system = out["system"]
    system["devices"] = devices
    system["device_count"] = total
    system["nodes"] = nodes
    # The launch receipt is the only truthful source for a two-node command;
    # the sibling's single-node argv would claim a placement that never ran.
    argv = list(launch["argv"])
    out["server_launch"] = {
        "1": {
            "argv": argv,
            "shell": launch.get("shell") or " ".join(argv),
        }
    }
    distributed = launch.get("distributed") or {}
    if distributed:
        out["server_launch"]["1"]["distributed"] = {
            "nodes": distributed.get("nodes"),
            "gpus_per_node": distributed.get("gpus_per_node"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sibling", type=Path, required=True)
    parser.add_argument("--archive-point", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    sparse = list(args.archive_point.glob("*/c*/run*/result.json"))
    if not sparse:
        raise SystemExit(f"no archived runs under {args.archive_point}")
    # Every run of a point shares one launch; take the first that has a receipt.
    launch = None
    inventory = None
    for probe in sorted(args.archive_point.glob("*/c*")):
        receipt = probe / "launch.json"
        if receipt.is_file():
            launch = json.loads(receipt.read_text())
            break
    for candidate in sorted(args.archive_point.rglob("gpu-inventory.csv")):
        inventory = candidate
        break
    if launch is None:
        raise SystemExit("no launch.json beside the archived runs")
    if inventory is None:
        raise SystemExit("no gpu-inventory.csv for the archived runs")

    devices = [row["name"] for row in load_inventory(inventory)]
    sibling = json.loads(args.sibling.read_text())
    result = derive(
        sibling,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        devices=devices,
        launch=launch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} ({args.nodes}x{args.gpus_per_node})")
    return 0


if __name__ == "__main__":
    sys.exit(main())