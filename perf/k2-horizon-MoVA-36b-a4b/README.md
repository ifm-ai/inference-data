# K2-Horizon-MoVA-36B-A4B — inference-performance dataset

## What this dataset is

Systematic single-node endpoint measurements of **K2-Horizon-MoVA-36B-A4B** on NVIDIA H200
GPUs, sweeping input-sequence length (ISL), output-sequence length (OSL),
and concurrency (C) across the vLLM and SGLang serving runtimes. Every
measured point holds exactly three independent repetitions (`run1`, `run2`,
`run3`) with identical requested token counts and concurrency. Sister models
of the K2-Horizon family are published on sibling `perf/*` branches of
this repository.

## Components

Every measurement of this dataset was taken with these component versions
(per-cell receipts in each cell's `environment.json`):

| component | version | repository | commit | image |
|---|---|---|---|---|
| InferenceX | — | https://github.com/SemiAnalysisAI/InferenceX | 4ab85c1e33b66d6bd5a3087b3de5ba3e86cbbe80 | — |
| SGLang | 0.5.20 | https://github.com/sgl-project/sglang | 94602c9c2b7cbdb8efd5c52802dac6a1c180089e | lmsysorg/sglang:v0.5.20 |
| vLLM | 0.29.1rc1.dev452+g3df4ae153 | https://github.com/vllm-project/vllm | 3df4ae153eb385e27b52f26c81f8edb9e20b9984 | vllm/vllm-openai:nightly-3df4ae153eb385e27b52f26c81f8edb9e20b9984 |

## Layout

```
<branch root>/
  scripts/
    reproduce.py               # shared across models: emits/reruns the exact commands
  perf/k2-horizon-mova-36b-a4b/
    1x<gpus>/                  # single-node GPU count (1, 2, 4, 8)
      <runtime>/               # vllm | sglang
        <dtype>/               # bf16 | fp8
          environment.json     # component versions + exact launch commands
          isl<I>-osl<O>/       # requested token counts
            c<N>/              # concurrency
              run{1,2,3}_results.json.gz   # one gzipped record per repetition
```

## Cells in this dataset

Points = one `(topology, runtime, dtype, ISL, OSL, concurrency)`; leaves =
individual `run{1,2,3}` records. **141 points / 423 leaves / 111.3 MB.**

| cell (topology/runtime/dtype) | points | leaves |
|---|---:|---:|
| `1x2/sglang/bf16` | 22 | 66 |
| `1x2/vllm/bf16` | 22 | 66 |
| `1x2/vllm/fp8` | 23 | 69 |
| `1x4/sglang/bf16` | 22 | 66 |
| `1x4/vllm/bf16` | 25 | 75 |
| `1x4/vllm/fp8` | 23 | 69 |
| `1x8/vllm/bf16` | 2 | 6 |
| `1x8/vllm/fp8` | 2 | 6 |

## Record schema

Leaves carry the InferenceX benchmark client output for one repetition,
**in full — per-block timing retained** — each as its own gzipped file
(`zcat run1_results.json.gz | jq .`; ~7× smaller than the raw record). No
`_slim` or other transform marker field is added:

* kept: `concurrency`, `observed_running`, `scope`, `token_ids_requested`,
  `qualification` (pass flag, aggregate prefill/decode TPS), `streams` pass
  metadata, and all `upstream` aggregates **and arrays** — duration, token
  totals, throughputs, per-request `ttfts`, `input_lens`, `output_lens`,
  errors, and `upstream.itls_us` (per-token inter-token-latency arrays in
  integer microseconds, one array per request — the per-block timing data);
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
