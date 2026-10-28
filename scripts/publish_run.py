#!/usr/bin/env python3
"""Prepare one archived Chapter 8.1 run record for publication.

The campaign archiver (`chapter81_archive_layout.py`, schema
``perfhc.inference_perf_data.run.v1``) keeps ``upstream.itls_us``, one
integer-microsecond inter-token-latency (ITL) array per request — the
per-block timing data. The published form keeps every field, including
those raw per-token timing arrays, and adds per-request ITL statistics
alongside them.

Transform, applied to one leaf ``result.json``:

* keep the run record itself — ``concurrency``, ``observed_running``,
  ``scope``, ``token_ids_requested``, ``qualification``, ``streams``, and
  all ``upstream`` aggregates and arrays: ``duration``, token totals,
  throughputs, per-request ``ttfts``, ``input_lens``, ``output_lens``,
  errors, and ``itls_us`` (per-block timing);
* drop the packaging provenance blocks ``_archive`` and ``_encoding``
  (they reference campaign stage paths and harness internals; per-run
  lineage is carried by the published manifests instead);
* add ``upstream.itl_us_stats``: ``[{count, mean, min, p50, p90, p95, p99,
  max}]`` aligned with the per-request arrays (``ttfts``, ``output_lens``),
  percentiles computed nearest-rank (``ceil(n * f)``, 1-based) — the same
  convention as ``perfhc_bench.inferencex_catalog._quantile``.

No transform marker field (``_slim`` or otherwise) is added.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


def _nearest_rank(values: list[int | float], fraction: float) -> int | float | None:
    """Nearest-rank percentile, mirroring perfhc_bench _quantile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[min(rank - 1, len(ordered) - 1)]


def request_stats(itls_us: list[Any]) -> list[dict[str, Any] | None]:
    """One stats object per request, aligned with upstream.itls_us."""
    stats: list[dict[str, Any] | None] = []
    for per_request in itls_us:
        if not isinstance(per_request, list) or not per_request:
            stats.append(None)
            continue
        if not all(isinstance(v, (int, float)) for v in per_request):
            raise ValueError("upstream.itls_us holds a non-numeric entry")
        stats.append(
            {
                "count": len(per_request),
                "mean": sum(per_request) / len(per_request),
                "min": min(per_request),
                "p50": _nearest_rank(per_request, 0.50),
                "p90": _nearest_rank(per_request, 0.90),
                "p95": _nearest_rank(per_request, 0.95),
                "p99": _nearest_rank(per_request, 0.99),
                "max": max(per_request),
            }
        )
    return stats


def publish_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out.pop("_archive", None)
    out.pop("_encoding", None)
    upstream = dict(record.get("upstream") or {})
    itls_us = upstream.get("itls_us")
    if itls_us is not None:
        if not isinstance(itls_us, list):
            raise ValueError("upstream.itls_us is not a list")
        stats = request_stats(itls_us)
        if any(entry is not None for entry in stats):
            upstream["itl_us_stats"] = stats
    out["upstream"] = upstream
    return out


def publish_text(raw: str) -> str:
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("run record is not a JSON object")
    return json.dumps(publish_record(record), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", help="default: stdout")
    args = parser.parse_args()
    text = open(args.input).read()
    published = publish_text(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(published)
    else:
        sys.stdout.write(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
