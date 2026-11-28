# K2-Horizon inference performance dataset

Data store of technical report information, information for future blog posts, and open-source inference investigations.

[![Papers with Code: SOTA on BBH](https://paperswithcode.co/api/v1/papers/2611.21003/leaderboard-badge.svg?eval=49274&live=1)](https://paperswithcode.co/api/v1/papers/2611.21003/leaderboard-badge-link?eval=49274)
[![Papers with Code: #2 on MBPP](https://paperswithcode.co/api/v1/papers/2611.21003/leaderboard-badge.svg?eval=49280&live=1)](https://paperswithcode.co/api/v1/papers/2611.21003/leaderboard-badge-link?eval=49280)
[![Papers with Code: #3 on GSM8K](https://paperswithcode.co/api/v1/papers/2611.21003/leaderboard-badge.svg?eval=49278&live=1)](https://paperswithcode.co/api/v1/papers/2611.21003/leaderboard-badge-link?eval=49278)

This repository contains inference performance measurements for the
K2-Horizon model family on NVIDIA H200 GPUs. The data covers input length,
output length, concurrency, serving runtime, and precision. Each point has
three runs.

## Models

| Model | Context |
|---|---|
| [K2-Horizon-900M](perf/k2-horizon-900m/README.md) | 128K |
| [K2-Horizon-3.7B](perf/k2-horizon-3.7b/README.md) | 512K |
| [K2-Horizon-7B](perf/k2-horizon-7b/README.md) | 512K |
| [K2-Horizon-32B](perf/k2-horizon-32b/README.md) | 512K |
| [K2-Horizon-MoVA-36B-A4B](perf/k2-horizon-MoVA-36b-a4b/README.md) | 512K |
| [K2-Horizon-375B-A23B](perf/k2-horizon-375b-a23b/README.md) | 512K |

## Repository layout

```text
perf/
  k2-horizon-<model>/
    <nodes>x<gpus>/
      <runtime>/
        <dtype>/
          environment.json
          isl<I>-osl<O>/
            c<N>/
              run1_results.json.gz
              run2_results.json.gz
              run3_results.json.gz
```

- A **cell** is one placement, one runtime, and one precision. For example,
  `2x8/sglang/bf16` is two nodes with eight GPUs per node, SGLang, and bf16.
- A **point** is one cell with one input length, one output length, and one
  concurrency. `isl8192-osl32/c2` is 8192 input tokens, 32 output tokens, and
  concurrency 2.
- A **run** is one measurement of a point. Each point has three runs.

## Result records

Each `run*_results.json.gz` file is one result from the InferenceX benchmark
client. A record contains:

- the request: input tokens, output tokens, and concurrency;
- the timings: time to first token, inter-token latency, and total duration;
- the throughput: input, output, and total tokens per second;
- the outcome: completed tokens, request errors, and the qualification result.

The full field set is in each record. The file is self-contained.

## Runtime versions

| Component | Version |
|---|---|
| InferenceX benchmark client | current campaign build |
| SGLang | 0.5.20 |
| vLLM | 0.29.1 |

Each cell has an `environment.json` file. That file records the exact
component builds and the launch commands for that cell.

## Reproduce one point

Use `scripts/reproduce.py` to print the server and client commands for one
point:

```bash
python3 scripts/reproduce.py \
  --cell perf/k2-horizon-375b-a23b/2x8/sglang/bf16 \
  --point isl4096-osl32
```

The script prints the commands. It does not start a scheduler or reserve
GPUs. Add `--execute` to run the commands. Use `--models-root` to set the
local model directory.

## Validate the dataset

Run the validator:

```bash
python3 scripts/validate_dataset.py
```

The validator checks:

- each point has three runs;
- each run has the declared input and output lengths;
- the timings and token counts agree;
- each cell has an `environment.json` file;
- the model README totals match the tree;
- both runtimes have the same coverage when both can run.

The exit status is 0 for a clean dataset, 1 for an error, and 2 for warnings
only. Use `--json` for a machine-readable report.

## Measurement grid

Each cell has a full input-length sweep at concurrency 2. Each cell also has
a decode anchor at input length 1.

The input lengths are 1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
32768, 65536, 131072, 196608, 262144, 327680, 393216, and 458752. Each cell
also has a context test at its largest supported prompt.

The main sweep uses 32 output tokens. Legacy points at 1024 output tokens
remain in the tree.

A cell limited by GPU memory stops at the highest point it can host. A
missing point above that limit is intentional. The validator reports these
limits, and the model README lists the points for each cell.

## Known limits

- The 900M and 3.7B models have bf16 data only.
- The 32B and MoVA 36B models have no vLLM 1x1 cell. The vLLM engine fails
  at startup on one GPU.
- The MoVA 36B fp8 model has no SGLang cell. The SGLang loader rejects the
  checkpoint.
- The 375B bf16 1x4 cell is absent because the model does not fit in GPU
  memory. Use the fp8 cell instead.
- The 2x8 placement is available for the 375B model only.
- SGLang rejects prompts above the context length minus 6 tokens. vLLM
  accepts prompts up to the context length minus 1 token. The data records
  the largest prompt that each runtime accepts. The validator compares these
  two boundary points as equal.

## Regenerate the published tree

The archive is the source of truth for the published tree. Use the
materializer to rebuild `perf/` from an archive:

```bash
python3 scripts/materialize_published_tree.py \
  --source /path/to/archive \
  --repo .
```

Use `--meta-out` to keep the audit files. After a data change, run:

```bash
python3 scripts/refresh_model_readmes.py
```

This command updates the totals and cell table in each model README.