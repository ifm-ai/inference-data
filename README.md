# inference-data

Data store of tech report information, information for future blog posts, and open-source inference investigations.

## What this dataset is

Systematic endpoint measurements of the K2-Horizon checkpoint family on
NVIDIA H200 GPUs, sweeping input-sequence length (ISL), output-sequence
length (OSL), and concurrency (C) across serving runtimes (vLLM and
SGLang today; more may appear). Every measured point holds exactly three
independent repetitions (`run1`, `run2`, `run3`) with identical requested
token counts and concurrency.

All measurements published so far are **single-node**; multi-node
measurements land under the same layout going forward (the layout encodes
the node count — see below). Likewise, precisions beyond `bf16`/`fp8`
will appear as new dtype directories as they are measured.

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

The `.gz` files are tracked with **Git LFS** (`.gitattributes`: `*.gz`).
Install `git-lfs` before cloning/pulling: without it you get pointer stubs
instead of the records.

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

python3 scripts/materialize_published_tree.py --source <archive-root> --repo . [--meta-out <dir>]
```

Gates (first violation is fatal): exactly run1..run3 per point — run4+ is
always refused, a partial point is refused unless `--allow-partial` skips
it; `upstream.itls_us` present in every leaf; each shipped cell retains
its `environment.json`; no harness-internal or absolute-path strings in
shipped records; every `.gz` re-decompressed and reparsed after writing.
Per-run lineage manifests and `audit.json` are written only with
`--meta-out` — they are not shipped in the repo.
