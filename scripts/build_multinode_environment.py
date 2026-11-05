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
    ranks: list[dict],
    request: dict | None = None,
) -> dict:
    """Return the multi-node environment.json derived from a published sibling.

    ``ranks`` holds one entry per node with the ``node_rank`` and
    ``dist_init_addr`` that node's server process reported. The base argv in
    the receipt is not runnable on its own: the wrapper appends those two flags
    per node, so publishing the base alone would not reproduce the placement.
    """
    total = nodes * gpus_per_node
    if len(devices) != total:
        raise SystemExit(
            f"inventory lists {len(devices)} GPUs, placement needs {total}"
        )
    if len(ranks) != nodes:
        raise SystemExit(f"{len(ranks)} rank receipts for a {nodes}-node placement")
    if sorted(rank["node_rank"] for rank in ranks) != list(range(nodes)):
        raise SystemExit(f"rank receipts do not cover 0..{nodes - 1}: {ranks}")
    addresses = {rank["dist_init_addr"] for rank in ranks}
    if len(addresses) != 1:
        raise SystemExit(f"rank receipts disagree on dist-init-addr: {addresses}")

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

    # One command per node, in rank order, each carrying the flags the wrapper
    # added. Launch them concurrently on their own hosts.
    base = list(launch["argv"])
    ordered = sorted(ranks, key=lambda rank: rank["node_rank"])
    out["server_launch"] = {
        "1": {
            "argv": base,
            "nodes": [
                {
                    "node_rank": rank["node_rank"],
                    "argv": base
                    + ["--dist-init-addr", rank["dist_init_addr"],
                       "--node-rank", str(rank["node_rank"])],
                }
                for rank in ordered
            ],
        }
    }
    out["server_launch"]["1"]["shell"] = (
        "run concurrently, one per node:\n"
        + "\n".join(
            f"[rank {entry['node_rank']}] " + " ".join(entry["argv"])
            for entry in out["server_launch"]["1"]["nodes"]
        )
    )
    # The sibling describes a different placement with a different served name,
    # port and token counts; inheriting it would publish a request that this
    # cell's runs never sent.
    if request is not None:
        prompts = request.get("prompts") or []
        out["client_request_example"] = {
            "base_url": request["base_url"],
            "model": request["model"],
            "concurrency": request["concurrency"],
            "seed": request["seed"],
            "sequence_length": request["sequence_length"],
            "output_tokens": request["output_tokens"],
            "prompts": {
                "count": len(prompts) or 1,
                "tokens_per_prompt": [
                    len(prompts[0]) if prompts else len(request.get("prompt_ids") or [])
                ],
                "note": (
                    "exact token-id lists omitted for size; fixed token ids drawn "
                    "once from the model tokenizer, identical across the three runs "
                    "of a point"
                ),
            },
        }
    else:
        out.pop("client_request_example", None)
    return out


def first_request(archive_point: Path) -> dict | None:
    """The first measured request the attempt recorded, for the example block."""
    for candidate in [*sorted(archive_point.glob("*.request.json")),
                      *sorted(archive_point.glob("*/*.request.json"))]:
        try:
            data = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        if "sequence_length" in data and "output_tokens" in data:
            return data
    return None


def rank_receipts(control: Path, nodes: int) -> list[dict]:
    """``node_rank`` and ``dist_init_addr`` each server process reported.

    The wrapper appends those flags per node, so they exist only in the
    processes' own startup logs, not in the base launch receipt.
    """
    import re

    logs = sorted(control.glob("server.log")) + sorted(control.glob("worker.log"))
    ranks = []
    for log in logs:
        text = log.read_text(errors="ignore")
        rank = re.search(r"'node_rank': (\d+)", text)
        address = re.search(r"'dist_init_addr': '([^']+)'", text)
        nnodes = re.search(r"'nnodes': (\d+)", text)
        if not rank or not address:
            continue
        if nnodes and int(nnodes.group(1)) != nodes:
            raise SystemExit(
                f"{log} reports nnodes={nnodes.group(1)}, placement needs {nodes}"
            )
        ranks.append(
            {"node_rank": int(rank.group(1)), "dist_init_addr": address.group(1)}
        )
    return ranks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sibling", type=Path, required=True)
    parser.add_argument("--archive-point", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--control", type=Path, required=True,
                        help="multinode-control/<job> holding server.log and worker.log")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    # Results are flat beside the receipts (<attempt>/prefill-1.json,
    # decode-1.json); older archives nest them under run dirs.
    sparse = (
        list(args.archive_point.glob("*.json"))
        or list(args.archive_point.glob("*/result.json"))
        or list(args.archive_point.glob("*/run*/result.json"))
        or list(args.archive_point.glob("*/c*/run*/result.json"))
    )
    measured = [
        path
        for path in sparse
        if path.name.startswith(("prefill-", "decode-"))
        and not path.name.endswith((".request.json", ".started.json"))
    ]
    if not measured:
        raise SystemExit(f"no archived runs under {args.archive_point}")
    # Every run of a point shares one launch; take the first that has a receipt.
    # A measure attempt keeps it flat; older archives nest it under c<N>/.
    launch = None
    for candidate in [
        args.archive_point / "launch.json",
        *sorted(args.archive_point.glob("*/c*/launch.json")),
        *sorted(args.archive_point.glob("*/launch.json")),
    ]:
        if candidate.is_file():
            launch = json.loads(candidate.read_text())
            break
    inventory = None
    # The control dir holds the inventory the launcher collected across every
    # node; one inside a point would describe a single host only.
    for candidate in [
        args.control / "gpu-inventory.csv",
        *sorted(args.archive_point.rglob("gpu-inventory.csv")),
    ]:
        if candidate.is_file():
            inventory = candidate
            break
    if launch is None:
        raise SystemExit("no launch.json beside the archived runs")
    if inventory is None:
        raise SystemExit("no gpu-inventory.csv for the archived runs")

    devices = [row["name"] for row in load_inventory(inventory)]
    ranks = rank_receipts(args.control, args.nodes)
    if not ranks:
        raise SystemExit(f"no server startup logs under {args.control}")
    sibling = json.loads(args.sibling.read_text())
    result = derive(
        sibling,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        devices=devices,
        launch=launch,
        ranks=ranks,
        request=first_request(args.archive_point),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} ({args.nodes}x{args.gpus_per_node})")
    return 0


if __name__ == "__main__":
    sys.exit(main())