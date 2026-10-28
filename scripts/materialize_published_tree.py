#!/usr/bin/env python3
"""Materialize the published Chapter 8.1 dataset into this repository.

Reads the uncompressed campaign archive tree (schema
``perfhc.inference_perf_data.run.v1``) and writes, under ``perf/``
(override with ``--out``), one gzipped ``run<N>_results.json.gz`` per leaf
in the point layout, mapped onto the published directory spellings
(``K2-Horizon-<X>`` -> ``k2-horizon-<x>`` with ``MoVA`` preserved).

Models whose mapped output directory does not already exist under ``--out``
in the repo are skipped with a warning (publish scope follows what the
repository already carries); override with ``--models``. Each shipped cell
must already carry its ``environment.json`` in the repo (preserved from the
existing publication; ``scripts/reproduce.py`` consumes it) — a missing one
fails a gate.

Records are transformed by ``scripts/publish_run.py``: the full record
including the ``upstream.itls_us`` per-block timing arrays, plus the added
``upstream.itl_us_stats``. Each ``.json.gz`` is an independently
decompressible gzip stream (~7× smaller than the raw record), tracked
through Git LFS (``.gitattributes``: ``*.gz``).

The lineage manifests and ``audit.json`` are NOT shipped in the repo;
pass ``--meta-out <dir>`` to write them somewhere for inspection (e.g.
``/tmp``). Every write is validated:

* each ``c<N>`` point ships exactly ``run1_results.json.gz``,
  ``run2_results.json.gz``, ``run3_results.json.gz`` — ``run4+`` is always
  fatal; partial points (<3 runs, e.g. runs still in flight) are fatal too
  unless ``--allow-partial`` is passed, in which case they are skipped and
  recorded in the audit;
* every shipped cell retains its ``environment.json`` in the repo
  (``scripts/reproduce.py`` consumes it);
* each source leaf carries ``upstream.itls_us`` (per-block timing); relax
  with ``--allow-missing-timing`` if ever needed;
* no ``_slim`` marker field appears in any written record;
* no ``.chapter81`` or absolute ``/mnt/`` reference survives in any written
  file (``_archive.source`` and manifest ``source`` fields are rewritten to
  their stage-relative form when they arrive unscrubbed);
* every written ``.json.gz`` is re-opened and reparsed as JSON.

Outputs are deterministic (gzip mtime=0): identical archive input yields
byte-identical files. Exits non-zero on the first violated gate.
Idempotent: re-running rewrites the tree from the current archive state.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from publish_run import publish_record  # noqa: E402

STAGE_PREFIX = "/mnt/weka/shrd/k2m/micah.villmow/models/.chapter81-runs/"
FORBIDDEN = (".chapter81", "/mnt/")
RUNS = ("run1", "run2", "run3")
MANIFESTS = {
    ".chapter81-archive-manifest-consolidated.json": "manifest-consolidated.json",
    ".chapter81-archive-manifest-capacity.json": "manifest-capacity.json",
}
LEAF_RE = re.compile(r"/(run\d+)/result\.json$")
SUFFIX = "_results.json.gz"


def model_dir(archive_model: str) -> str:
    """Published directory spelling of an archive model dir.

    ``K2-Horizon-7B`` -> ``k2-horizon-7b``; ``K2-Horizon-MoVA-36B-A4B`` ->
    ``k2-horizon-MoVA-36b-a4b`` (the published tree keeps the ``MoVA`` camel).
    """
    suffix = archive_model.removeprefix("K2-Horizon-")
    return "k2-horizon-" + suffix.lower().replace("mova", "MoVA")


def scrub(value: str) -> str:
    return value[len(STAGE_PREFIX):] if value.startswith(STAGE_PREFIX) else value


def scrub_obj(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                node[key] = scrub(value)
            else:
                scrub_obj(value)
    elif isinstance(node, list):
        for item in node:
            scrub_obj(item)


def rel_point(point_dir: Path, root: Path) -> list[str]:
    return point_dir.relative_to(root).parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="uncompressed archive root (contains K2-Horizon-*/ )")
    parser.add_argument("--repo", required=True, help="inference-perf-data worktree root")
    parser.add_argument("--out", default="perf",
                        help="repo-relative output root for the published tree")
    parser.add_argument("--meta-out",
                        help="optional directory (outside the repo) for the lineage manifests "
                             "and audit.json; omitted: none are written")
    parser.add_argument("--models",
                        help="comma-separated archive model names to publish (default: all whose "
                             "mapped directory already exists under --out)")
    parser.add_argument("--allow-missing-timing", action="store_true",
                        help="permit leaves that lack upstream.itls_us (default: refuse)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="skip points holding fewer than run1..run3 (default: refuse; "
                             "run4+ is always refused)")
    args = parser.parse_args()

    source = Path(args.source)
    repo_root = Path(args.repo)
    out_rel = args.out.strip("/")
    out_root = repo_root / out_rel
    meta_dir = Path(args.meta_out) if args.meta_out else None
    allow = set(args.models.split(",")) if args.models else None

    points = 0
    leaves = 0
    cells: dict[tuple[str, ...], int] = {}
    bytes_raw = 0
    bytes_gz = 0
    skipped: dict[str, str] = {}
    partial: list[str] = []
    env_checked: set[tuple[str, ...]] = set()

    for point_dir in sorted(source.glob("K2-Horizon-*/*/*/*/*/c*")):
        rel = rel_point(point_dir, source)
        # K2-Horizon-<model>/<topology>/<runtime>/<dtype>/isl<I>-osl<O>/c<N>
        if len(rel) != 6 or not rel[-1].startswith("c"):
            continue
        archive_model = rel[0]
        mapped = model_dir(archive_model)
        if allow is not None:
            if archive_model not in allow:
                skipped[archive_model] = "not in --models"
                continue
        elif not (out_root / mapped).is_dir():
            skipped[archive_model] = f"{out_rel}/{mapped} not present in repo"
            continue
        rel = (mapped, *rel[1:])
        actual = sorted(
            child.name
            for child in point_dir.iterdir()
            if child.is_dir() and (child / "result.json").is_file()
        )
        if actual and actual != list(RUNS):
            if len(actual) < len(RUNS) and args.allow_partial:
                partial.append("/".join(rel))
                continue
            raise SystemExit(f"gate: {point_dir} has runs {actual}, expected {list(RUNS)}")
        if not actual:
            continue
        cell_rel = tuple(rel[:4])
        if cell_rel not in env_checked:
            env_checked.add(cell_rel)
            env_file = out_root.joinpath(*cell_rel, "environment.json")
            if not env_file.is_file():
                raise SystemExit(
                    f"gate: {env_file} missing — shipped cells must retain their "
                    "environment.json (scripts/reproduce.py consumes it)")
        for run in RUNS:
            leaf = point_dir / run / "result.json"
            record = json.loads(leaf.read_text())
            if "itls_us" not in (record.get("upstream") or {}) and not args.allow_missing_timing:
                raise SystemExit(
                    f"gate: {leaf} lacks upstream.itls_us per-block timing; "
                    "pass --allow-missing-timing to waive"
                )
            scrub_obj(record)
            published = publish_record(record)
            if "_slim" in published:
                raise SystemExit(f"gate: _slim marker present in published {leaf}")
            data = (json.dumps(published, indent=2) + "\n").encode()
            for token in FORBIDDEN:
                if token.encode() in data:
                    raise SystemExit(f"gate: forbidden reference {token!r} in published {leaf}")
            dest = out_root.joinpath(*rel, f"{run}{SUFFIX}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                with gzip.GzipFile(fileobj=fh, mode="wb", compresslevel=6, mtime=0) as gz:
                    gz.write(data)
            with gzip.open(dest, "rt") as fh:  # re-open and reparse what shipped
                if not isinstance(json.load(fh), dict):
                    raise SystemExit(f"gate: shipped {dest} did not reparse to a JSON object")
            json.loads(data)  # the pre-compression record likewise parses
            bytes_raw += len(data)
            bytes_gz += dest.stat().st_size
            leaves += 1
        points += 1
        cells[(rel[0], rel[1], rel[2], rel[3])] = cells.get((rel[0], rel[1], rel[2], rel[3]), 0) + 1

    if meta_dir is not None:
        # Lineage manifests: rewrite to repo-relative shipped paths, keep only
        # records whose leaf shipped. Inspection only — not shipped in the repo.
        shipped = {str(p.relative_to(out_root)) for p in out_root.rglob(f"run*{SUFFIX}")}
        for src_name, dst_name in MANIFESTS.items():
            src_file = source / src_name
            if not src_file.is_file():
                continue
            records = json.loads(src_file.read_text())
            scrub_obj(records)
            kept = []
            for record in records:
                dest_abs = record.get("destination", "")
                marker = "/models/K2-Horizon-"
                if marker in dest_abs:
                    dest_rel = dest_abs.split(marker, 1)[1]
                    dest_rel = "K2-Horizon-" + dest_rel
                else:
                    dest_rel = dest_abs
                if dest_rel.startswith("K2-Horizon-"):
                    head, _, tail = dest_rel.partition("/")
                    dest_rel = f"{model_dir(head)}/{tail}"
                dest_rel = LEAF_RE.sub(rf"/\1{SUFFIX}", dest_rel)
                record["destination"] = f"{out_rel}/{dest_rel}"
                if dest_rel in shipped:
                    kept.append(record)
            out_file = meta_dir / dst_name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(kept, indent=2) + "\n"
            for token in FORBIDDEN:
                if token in text:
                    raise SystemExit(f"gate: forbidden reference {token!r} in manifest {src_name}")
            out_file.write_text(text)
            print(f"manifest {dst_name}: {len(records)} lineage records -> {len(kept)} shipped")

        audit = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "source": scrub(str(source)) if not scrub(str(source)).startswith("/") else "<local campaign archive root>",
            "points": points,
            "leaves": leaves,
            "bytes_raw": bytes_raw,
            "bytes_gz": bytes_gz,
            "cells": {"/".join(k): v for k, v in sorted(cells.items())},
            "skipped_models": skipped,
            "partial_points_skipped": partial,
            "gates": {
                "run4_refused": True,
                "partial_points_are_fatal_unless_waived": not args.allow_partial,
                "timing_arrays_present": not args.allow_missing_timing,
                "no_slim_marker": True,
                "no_chapter81_refs": True,
                "no_absolute_paths": True,
                "env_json_present": True,
                "shipped_files_reparsed": True,
            },
        }
        audit_path = meta_dir / "audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        print(f"audit -> {audit_path}")
    print(f"points={points} leaves={leaves} raw={bytes_raw/1e6:.1f} MB gz={bytes_gz/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
