# inference-data

Data store of tech report information, information for future blog posts, and open-source inference investigations.

## What this dataset is

Systematic endpoint measurements of the K2-Horizon checkpoint family on
NVIDIA H200 GPUs, sweeping input-sequence length (ISL), output-sequence
length (OSL), and concurrency (C) across serving runtimes (vLLM and
SGLang today; more may appear). Every measured point holds exactly three
independent repetitions (`run1`, `run2`, `run3`) with identical requested
token counts and concurrency.

Model datasets published so far (each with its own README listing its
cells): [`k2-horizon-3.7b`](perf/k2-horizon-3.7b/),
[`k2-horizon-7b`](perf/k2-horizon-7b/), [`k2-horizon-32b`](perf/k2-horizon-32b/),
[`k2-horizon-MoVA-36b-a4b`](perf/k2-horizon-MoVA-36b-a4b/),
[`k2-horizon-375b-a23b`](perf/k2-horizon-375b-a23b/),
[`k2-horizon-900m`](perf/k2-horizon-900m/).

## Components

Every measurement was taken with these component versions (per-cell
receipts in each cell's `environment.json`):

| component | version | repository | commit | image |
|---|---|---|---|---|
| InferenceX | — | https://github.com/SemiAnalysisAI/InferenceX | 4ab85c1e33b66d6bd5a3087b3de5ba3e86cbbe80 | — |
| SGLang | 0.5.20 | https://github.com/sgl-project/sglang | 94602c9c2b7cbdb8efd5c52802dac6a1c180089e | lmsysorg/sglang:v0.5.20 |
| vLLM | 0.29.1rc1.dev452+g3df4ae153 | https://github.com/vllm-project/vllm | 3df4ae153eb385e27b52f26c81f8edb9e20b9984 | vllm/vllm-openai:nightly-3df4ae153eb385e27b52f26c81f8edb9e20b9984 |

## Layout

```
perf/k2-horizon-<model>/
  <nodes>x<gpus>/              # <nodes> nodes × <gpus> GPUs each — 1xG today,
                               # multi-node data lands as e.g. 2x8
    <runtime>/                 # vllm | sglang | …
      <dtype>/                 # bf16 | fp8 | … (more precisions going forward)
        environment.json       # component versions + exact launch commands
        isl<I>-osl<O>/         # requested token counts
          c<N>/                # concurrency
            run{1,2,3}_results.json.gz
```

Vocabulary used throughout: a **cell** is one
`<nodes>x<gpus>/<runtime>/<dtype>` directory (it carries the
`environment.json` receipt); a **point** is one `(placement, runtime,
dtype, ISL, OSL, concurrency)` combination — an `isl<I>-osl<O>/c<N>`
directory; the **leaves** of a point are its three run records.

## Record schema

Each `run<N>_results.json.gz` is one independently decompressible gzip
stream holding the full InferenceX client output for one repetition —
`zcat run1_results.json.gz | jq .`. Records keep everything the benchmark
emitted: identity fields (`concurrency`, `observed_running`, `scope`,
`token_ids_requested`), `qualification` (pass flag, aggregate
prefill/decode TPS), `streams` pass metadata, and all `upstream`
aggregates **and arrays** — duration, token totals, throughputs,
per-request `ttfts`, `input_lens`, `output_lens`, errors, and
`upstream.itls_us` (per-token inter-token-latency arrays in integer
microseconds, one array per request — the per-block timing data). The only
addition over the raw record is `upstream.itl_us_stats`: per-request
nearest-rank (`ceil(n·f)`) ITL statistics `{count, mean, min, p50, p90,
p95, p99, max}` in integer microseconds, aligned with the per-request
arrays.

A cell measured through the single-request capacity path keeps that path's
leaner record instead: it carries `prompt_token_ids` and the raw SSE body
rather than `token_ids_requested` and `streams`, and has no
`concurrency`/`observed_running` identity fields. Its `upstream` block is
identical to every other cell's, so the timing data is unaffected;
`validate_dataset.py` reports the difference as `RECORD_LEAN_SHAPE`.

## Launch metadata

Each cell's `environment.json` carries everything needed to reproduce its
measurements without any harness-internal tooling:

* `components`: name, repository, commit and container image of the serving
  runtime, and the repository/commit of the InferenceX benchmark client;
* `server_launch`: the exact server launch command (`argv` and shell form)
  per shipped concurrency, with the local checkpoint directory shown as
  `<models-root>/`;
* `client_request_example`: one verbatim client request payload (prompt
  token-id lists summarized for size).

### Placement

Single-node cells describe their placement implicitly: `cell.gpus`,
`system.device_count` and the length of `system.devices` all equal the
device count on their one host. A multi-node cell cannot be read that way,
so it carries the placement explicitly. Every cell may declare:

* `cell.nodes` — number of hosts the server spanned;
* `cell.gpus_per_node` — GPUs on each host (8 on this cluster);
* `system.nodes` — the same host count, recorded with the device inventory.

`cell.gpus`, `system.device_count` and `len(system.devices)` remain the
*placement total* (`nodes × gpus_per_node`), so a reader that only knows
the single-node fields still sees the full width. Such a cell's
`server_launch` entry carries `nodes`: one command per host in rank order,
each the base `server_launch.argv` plus the `--dist-init-addr` and
`--node-rank` that host ran. The wrapper appends those flags per node, so
the base argv alone does not reproduce the placement — launch the `nodes`
commands concurrently on their own hosts. `system.devices` lists every GPU
across the allocation, so entries repeat. A directory such as
`2x8/sglang/bf16` therefore means two nodes of eight GPUs each, sixteen
devices in total — the multi-node analogue of `1x8`.

All client requests share: seed 3, exact-token input (`input_lens` equals
ISL), exact-token output (`output_lens` equals OSL), and the
OpenAI-compatible endpoint of the server named in the launch command.
Point directories encode the request shape: `isl<I>-osl<O>` are the
requested input/output token counts and `c<N>` the concurrency. Prompt
token ids were drawn once from the model tokenizer and are identical
across the three runs of a point.

## Reproducing a point

`scripts/reproduce.py` walks the dataset and prints, for every measured
point, the exact two commands that produce it — the serving-runtime launch
(verbatim from the cell's `environment.json`, with the `<models-root>/`
placeholder resolved from `--models-root`) and the InferenceX client
replay (seed 3, exact ISL/OSL, `--max-concurrency C`). No job scheduler,
no harness tooling:

```
python3 scripts/reproduce.py [--cell <nodes>x<gpus>/<runtime>/<dtype>]
                             [--point isl<I>-osl<O>/c<N>]
                             [--models-root <path>] [--execute]
```

## Regenerating the dataset

The tree is rewritten from the campaign archive by gated tooling
(idempotent; deterministic gzip bytes — mtime=0):

```
scripts/publish_run.py                   # record transform: full record + upstream.itl_us_stats
scripts/materialize_published_tree.py    # archive -> perf/ writer
scripts/backfill_launch_receipts.py      # add a missing launch receipt from the archive
scripts/build_multinode_environment.py   # environment.json for a multi-node cell

python3 scripts/materialize_published_tree.py --source <archive-root> --repo . [--meta-out <dir>]
```

## Required point grid

Every concurrency of a cell holds the same ISL sweep. The sweep is capped at
the server's native context, and the maximum-sequence test is the largest
input whose input plus output reaches that context:

```
64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
131072, 196608, 262144, 327680, 393216, 458752   (+ the context test)
```

That is **18 ISL points for a 512K-context model, 14 for 256K and 12 for
128K**. A 1M-context model adds 768K and 1M on top of the sweep, for **20**.
Each cell also carries one `isl1` decode anchor. A cell with `c2` carries the
full sweep at `c2` as well, so coverage is counted per concurrency.

### Output length

The grid is measured at **osl 32**. Only a prefix of the response is needed to
resolve prefill behaviour across input length, and a short output keeps a
full sweep inside a sane run time.

The initial campaign measured the same sweep at **osl 1024**, long enough for
a response to reach a steady decode rate, so each sweep point also served as a
short decode-throughput sample. Those points stay in the tree and are not
reported as missing grid coverage (`LEGACY_SWEEP_OSL`). Points whose ISL falls
outside the grid at either output length are reported as
`GRID_UNEXPECTED_POINT`.

## Validating the dataset

`scripts/validate_dataset.py` reads the shipped tree back and reports every
inconsistency it finds. Nothing else does: the publisher only writes and
`reproduce.py` only re-emits commands.

```
python3 scripts/validate_dataset.py [--dataset-root perf] [--json]
                                    [--max-report 40] [--allow CODE[,CODE]]
```

It checks that every point holds exactly `run1`, `run2` and `run3` and no
other files; that each leaf decompresses and keeps the record shape; that
`input_lens` equals the requested ISL at the requested concurrency; that
`itl_us_stats` describes the `itls_us` arrays present; that each point's
`c<N>` directory is served by a launch that admits `N`; that a cell's
`environment.json` agrees with its directory name and device totals; that
each model README states the totals and per-cell counts the tree actually
holds; and that a placement measured in one dtype covers the same point
shapes in its sibling dtype. Repetitions of one point must agree on their
output length to within `REPETITION_TOLERANCE` (1) token; a wider spread is
`RUNS_DISAGREE_OUTPUT`.

A server may return up to `SHORTFALL_TOLERANCE` (3) fewer output tokens than
requested; both runtimes do this on maximum-length decodes. A shortfall within
that tolerance is `OUTPUT_SHORTFALL_WITHIN_TOLERANCE`, a longer one is
`OUTPUT_SHORTFALL`, and more tokens than requested is
`OUTPUT_OVER_REQUESTED`.

Dtype coverage is compared per placement and runtime. A cell measured in one
dtype only is `DTYPE_PAIR_INCOMPLETE`; a point shape present in one dtype and
absent in the other is `DTYPE_MISSING_COVERAGE`, which names each missing
shape so it can be queued.

Exit status is 0 when clean, 1 when any ERROR is reported, and 2 when only
WARNINGs are reported. Warnings cover conditions that are real but explained:
`ITLS_SHORT` and `RUNS_DISAGREE_ITL` (some vLLM runs report fewer
inter-token intervals than the tokens they produced), `OUTPUT_SHORTFALL`
(a server stopped short of the requested OSL), and `RECORD_LEAN_SHAPE`
(a cell produced through the single-request capacity path). Accept a known
condition with `--allow <CODE>` rather than dropping the check.
