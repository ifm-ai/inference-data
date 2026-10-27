# Chapter 8.1 — K2-Horizon inference-performance archive

Published benchmark archive backing the **Chapter 8.1 technical report,
"Inference Optimization (Techniques and Performance)"**:
[`LLM360/PerfHC` — `chapter_8_1/chapter_8_1_draft.md`](https://github.com/LLM360/PerfHC/blob/main/chapter_8_1/chapter_8_1_draft.md)
(manuscript on the default branch; supporting evidence under
[`chapter_8_1/`](https://github.com/LLM360/PerfHC/tree/main/chapter_8_1)).

## What this dataset is

Systematic single-node endpoint measurements of the K2-Horizon model family on
H200 GPUs, swept over input-sequence length (ISL), output-sequence length
(OSL), and concurrency (C), for both vLLM and SGLang runtimes at BF16 and
(where available) FP8 precision. Each measured point holds exactly three
independent repetitions (`run1`, `run2`, `run3`) with identical requested
token counts and concurrency.

Model cells: `K2-Horizon-3.7B`, `-7B`, `-7B-Uno-Linear`, `-7B-Uno-Tree`,
`-900M`, `-900M-Uno-Linear`, `-900M-Uno-Tree`, `-32B`, `-MoVA-36B-A4B`,
`-375B-A23B`.

## Layout

One gzipped JSON per leaf — each `run<N>_results.json.gz` is an
independently decompressible record (`zcat … | jq .`; ~7× smaller than the
raw record):

```
k2-horizon/
  K2-Horizon-<model>/
    1x<gpus>/                    # single-node GPU count (1, 2, 4, 8)
      <runtime>/                 # vllm | sglang
        <dtype>/                 # bf16 | fp8
          isl<I>-osl<O>/         # requested token counts
            c<N>/                # concurrency
              run{1,2,3}_results.json.gz
```

`manifest-consolidated.json` and `manifest-capacity.json` give per-run
lineage: stage source directory, packaged destination, model, runtime,
topology, dtype, parallel mode, ISL/OSL/C, run index.

## Record schema

Leaves follow `perfhc.inference_perf_data.run.v1` (see `_encoding` in each
record), published **in full — per-block timing retained**:

* kept: identity/provenance (`_archive`, `_encoding`), `concurrency`,
  `observed_running`, `scope`, `token_ids_requested`, `qualification`
  (pass flag, aggregate prefill/decode TPS), `streams` pass metadata, and
  all `upstream` aggregates **and arrays** — duration, token totals,
  throughputs, per-request `ttfts`, `input_lens`, `output_lens`, errors,
  and `upstream.itls_us` (per-token inter-token-latency arrays in integer
  microseconds, one array per request — the per-block timing data);
* added: `upstream.itl_us_stats` — per-request nearest-rank
  (`ceil(n·f)`) ITL statistics `{count, mean, min, p50, p90, p95, p99, max}`
  in integer microseconds, aligned with the per-request arrays.

No `_slim` or other transform marker field is added. Each record ships as
its own gzipped file (raw per-block timing is the bulk of record bytes;
per-file compression recovers ~7×).

Regenerate with `tools/materialize_published_tree.py` (transform in
`tools/publish_run.py`). `audit.json` records the packaging-time counts.

## Publication rules

* A point ships **iff** `run1`, `run2`, `run3` are all present; `run4+`
  are dropped; partial points are removed whole.
* Points at concurrency above a cell's verified Max-C (search-descent
  probes) are excluded: they cannot hold three full runs.
* UNO-linear cells ship as-is (bf16 only); two UNO-linear searches are
  permanently excluded (upstream SGLang IMA defect, documented in the
  campaign repo).
* Requested-token exactness gate: `usage == request`, enforced by the
  campaign's ingestion tooling; round-3 repair manifests used the
  exactly-servable counts for SGLang (`requested - 2` at the servable
  boundary).

## Provenance

* Campaign: `chapter-8-updated-m2-20260924`, Slurm `lowprio`, seed 3,
  image sha256 `135beadc…752474`, vLLM `0.29.1rc1.dev452+g3df4ae153`.
* Consolidated stage: `20260924-m2` (sweep + anchor workloads).
  Capacity stage: `maxc-phase1` (Max-C search/measure descent).
* PR sequence: this PR is cut only after the campaign's `post-repair3` and
  `final` refresh phases land, so the packaged tree is the complete archive.
