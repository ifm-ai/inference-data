# inference-data

Data store of tech report information, information for future blog posts, and open-source inference investigations.

## Layout

`perf/k2-horizon-<model>/` holds the K2-Horizon benchmark campaign results, one directory per model checkpoint — each carries its own README with the cell table and sizes:

```
perf/k2-horizon-<model>/
  1x<gpus>/                  # single-node GPU count (1, 2, 4, 8)
    <runtime>/               # vllm | sglang
      <dtype>/               # bf16 | fp8
        environment.json     # component versions + exact launch commands
        isl<I>-osl<O>/       # requested token counts
          c<N>/              # concurrency
            run{1,2,3}_results.json.gz
```

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

## Reproducing a point

`scripts/reproduce.py` walks the dataset and prints, for every measured
point, the exact two commands that produce it — the serving-runtime launch
(verbatim from the cell's `environment.json`, with the `<models-root>/`
placeholder resolved from `--models-root`) and the InferenceX client
replay (seed 3, exact ISL/OSL, `--max-concurrency C`). No job scheduler,
no harness tooling:

```
python3 scripts/reproduce.py [--cell <topology>/<runtime>/<dtype>]
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
