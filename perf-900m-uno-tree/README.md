# K2-Horizon-900M-Uno-Tree — inference-performance dataset

## What this dataset is

Systematic single-node endpoint measurements of **K2-Horizon-900M-Uno-Tree** on NVIDIA H200
GPUs, sweeping input-sequence length (ISL), output-sequence length (OSL),
and concurrency (C) across the vLLM and SGLang serving runtimes. Every
measured point holds exactly three independent repetitions (`run1`, `run2`,
`run3`) with identical requested token counts and concurrency. Sister models
of the K2-Horizon family are published on sibling `perf-data-*` branches of
this repository.

## Components

Every measurement of this dataset was taken with these component versions
(per-cell receipts in each cell's `environment.json`):

| component | version | repository | commit | image |
|---|---|---|---|---|
| InferenceX | — | https://github.com/SemiAnalysisAI/InferenceX | 4ab85c1e33b66d6bd5a3087b3de5ba3e86cbbe80 | — |
| SGLang | 0.5.20 | https://github.com/sgl-project/sglang | 94602c9c2b7cbdb8efd5c52802dac6a1c180089e | lmsysorg/sglang:v0.5.20 |

## Layout

```
<branch root>/
  scripts/
    reproduce.py               # shared across models: emits/reruns the exact commands
  perf-900m-uno-tree/
    1x<gpus>/                  # single-node GPU count (1, 2, 4, 8)
      <runtime>/               # vllm | sglang
        <dtype>/               # bf16 | fp8
          environment.json     # component versions + exact launch commands
          isl<I>-osl<O>/       # requested token counts
            c<N>/              # concurrency
              run{1,2,3}/result.json
```

## Cells in this dataset

Points = one `(topology, runtime, dtype, ISL, OSL, concurrency)`; leaves =
individual `run{1,2,3}` records. **17 points /
51 leaves / 2.5 MB.**

| cell (topology/runtime/dtype) | points | leaves |
|---|---:|---:|
| `1x1/sglang/bf16` | 17 | 51 |

## Record schema

Leaves carry the InferenceX benchmark client output for one repetition,
slimmed for publication (`_slim` records the transform version and drop
list):

* kept: `concurrency`, `observed_running`, `scope`, `token_ids_requested`,
  `qualification` (pass flag, aggregate prefill/decode TPS), `streams` pass
  metadata, and all `upstream` aggregates (duration, token totals,
  throughputs, per-request `ttfts`, `input_lens`, `output_lens`, errors);
* dropped: `upstream.itls_us` (raw per-token inter-token-latency arrays —
  99.9% of record bytes);
* added: `upstream.itl_us_stats` — per-request nearest-rank (`ceil(n·f)`)
  ITL statistics `{count, mean, min, p50, p90, p95, p99, max}` in integer
  microseconds, aligned with the per-request arrays.

## Reproducing this dataset

`scripts/reproduce.py` at the root of this branch (identical across model
branches) walks the dataset and emits, for every measured point, the exact
two commands that produce it — no job scheduler, no harness tooling:

1. the serving-runtime launch command, verbatim from the cell's
   `environment.json` (`server_launch`), with the `<models-root>/`
   placeholder resolved from `--models-root`;
2. the InferenceX benchmark client command
   (`python3 -m infx.bench_serving.benchmark_serving`) for that point's
   workload: `--seed 3 --dataset-name random --random-range-ratio 1.0`
   (exact ISL/OSL), `--num-prompts C --max-concurrency C`, `/v1/completions`,
   once per `run1..run3`, saving results under
   `<output-root>/` with the same directory layout as this dataset.

```bash
python3 scripts/reproduce.py --models-root /path/to/models --inferencex /path/to/InferenceX
python3 scripts/reproduce.py --models-root /path/to/models --inferencex /path/to/InferenceX --execute
```

Requirements for `--execute`: the serving runtime installed at the commit
recorded in the cell's `environment.json` (or its recorded container image),
an InferenceX checkout at the recorded commit (`components.inferencex`),
GPUs matching `cell`, and the model weights under `--models-root`. Without
`--execute` the commands are only printed.

Fidelity note: the published runs drove the InferenceX benchmark loop with
pre-generated exact prompt token IDs (uniform sample from the tokenizer
vocabulary, `random.Random(3)`, no decode/encode round trip). The CLI form
above regenerates an equivalent workload via InferenceX's own
random-dataset generator with the same seed and exact lengths; per-request
token IDs may differ from the published ones, while ISL/OSL/concurrency,
endpoint, and sampling semantics are identical.

## Launch metadata

Each cell's `environment.json` carries everything needed to reproduce its
measurements without any harness-internal tooling:

* `components`: name, repository, commit and container image of the serving
  runtime (vLLM or SGLang), and the repository/commit of the InferenceX
  benchmark client;
* `server_launch`: the exact server launch command (`argv` and shell form)
  per shipped concurrency, with the local checkpoint directory shown as
  `<models-root>/`;
* `client_request_example`: one verbatim client request payload (prompt
  token-id lists summarized for size).

All client requests share: seed 3, exact-token input (`input_lens` equals
ISL), exact-token output (`output_lens` equals OSL), and the
OpenAI-compatible endpoint of the server named in the launch command. Point
directories encode the request shape: `isl<I>-osl<O>` are the requested
input/output token counts and `c<N>` the concurrency. Prompt token ids were
drawn once from the model tokenizer and are identical across the three runs
of a point.
