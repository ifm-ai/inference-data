# Methodology

Everything in [`../SUMMARY.md`](../SUMMARY.md) and the per-model reports was
produced by this pipeline. It is reproducible end to end from the scripts here.

## Measurement class

These are **`cpu_quality_eval`** datasets: accuracy on 8 lm-eval tasks plus
single-stream CPU throughput, measured on a CPU-only host. They are **not** GPU
throughput sweeps and are **not comparable** to the `perf-data-*` series in this
repository, which measure ISL/OSL/concurrency sweeps on GPU with vLLM/SGLang.
The two live side by side under distinct prefixes (`eval-data-*` vs
`perf-data-*`) for exactly that reason.

## Hardware

| | |
|---|---|
| CPU | Intel Xeon E5, Sandy Bridge-E, 8 threads, no AVX2 |
| Memory | 62 GiB DDR3 |
| GPU | GeForce GT 430 (1 GB, nouveau) — **not used for inference**; no CUDA, no usable Vulkan device, so everything runs CPU-only through llvmpipe/software paths |
| Storage | two ~458 GiB volumes, models split across them |
| OS | Linux |

Inference was CPU-only by necessity, not by choice. The host has no GPU capable
of running these models, which is why every throughput number here is a CPU
number and should not be compared against GPU figures from other reports.

## Software

| Component | Version |
|---|---|
| Lemonade | 11.9.0, built from source |
| llama.cpp | `MBZUAI-IFM/llama.cpp@model/K2Horizon`, built with `-DGGML_VULKAN=ON` |
| lm-eval | 0.4.13 (EleutherAI) |
| Inference endpoint | Lemonade `/v1/completions` (OpenAI-compatible) |

## Task suite

EleutherAI lm-eval, 8 tasks, **12 samples per task** (5 for segmentation):

| Task | Metric | Discriminating |
|---|---|---|
| HumanEval | `pass@1,create_test` | yes |
| MBPP | `pass_at_1,none` | yes |
| GSM8K | `exact_match,flexible-extract` | yes |
| CoQA | `f1,none` | yes |
| SQuAD (completion) | `contains,none` | yes |
| IFEval | `prompt_level_strict_acc,none` | yes |
| Tool-call (custom, 12-case) | `exact_match,extract-json` | yes |
| BIG-bench paragraph segmentation | `exact_match,none` | **no** |

Two exclusions worth knowing about:

- **BIG-bench paragraph segmentation scored 0.00 for every model**, including the
  FP8 baseline. Non-discriminating at this scale; excluded from all means but
  still reported per model.
- **`Q2_K_S` is corrupt upstream** (zeroed file header, no GGUF magic; SHA matches
  what Hugging Face serves). Never evaluated. `Q2_0` produced degenerate output
  and is treated as failing.

## Generations

```
temperature = 0 (greedy)
max_gen_toks = 256
n-shot = 0 (except gsm8k, which the harness configures at 5-shot)
batch_size = 1
```

Greedy decoding is what makes the per-sample logs comparable across models: with
temp 0 the only source of divergence between two quants is quantization noise.

## Tokenizers

Each model family uses its own tokenizer, which matters:

| Family | Tokenizer | Vocab |
|---|---|---|
| 36B MoE, 3.7B, 7B | `IFM/K2-Horizon-7B` | 250,000 |
| 0.9B | `IFM/K2-Horizon-0.9B` | 64,000 |

Using the 7B tokenizer for the 0.9B would score it on the wrong token
boundaries. The campaign scripts select the tokenizer per model.

## Throughput probe

One fixed prompt ("Explain why the sky is blue in one short paragraph."),
temperature 0, 64 generated tokens, 8 threads, one model loaded at a time, no
other process competing. Both `timings.predicted_per_second` (decode) and
`timings.prompt_per_second` (prefill) are recorded. Throughput is the weakest
part of this dataset: a single 64-token sample per model, no repeats, so
differences under ~10% should not be treated as real.

## Statistical limits — read before quoting any number

n=12 per task gives a per-task standard error of roughly **±0.08–0.15**, so the
7-task mean carries about **±0.03–0.05**. Consequences:

- Everything from IQ4_XS up to Q8_0 is a statistical tie. The study cannot rank
  them, and no claim in [`../SUMMARY.md`](../SUMMARY.md) does.
- CoQA F1 with n=12 cannot separate most models; ties are common.
- GSM8K's 1.00 scores are 12/12 and should be read as "top of the observed
  range", not as a real gap over a 0.92.
- The dense 0.9B family's six means cluster in a band narrower than its own
  standard error; only the gap up to the 3.7B is meaningful.

Raw per-sample outputs are in each model's `data/samples.jsonl` so any of these
claims can be re-checked or recomputed at higher n.

## Known measurement issues

- **`lemond` wedges under sustained load** (accepts connections, returns empty).
  Recovered with `systemctl restart lemond`; the backend `llama-server` stayed
  healthy. One model run was lost this way and re-run.
- **Parallel `lemonade pull` races `user_models.json`** (last-writer-wins), so
  downloads registered nothing. Re-pulled sequentially.
- **New files in `extra_models_dir` are invisible until restart.**
- **The full BF16 MoVA (69.8 GiB) does not fit 62 GiB RAM** at a usable context.
  It loads at ctx 512 and generates at 0.09 tok/s, so it is reported as a
  data point, not benchmarked.

## Layout

Each model gets its own top-level directory and its own PR:

```
eval-data-<slug>/
  README.md                 per-test report: score, stderr, n, context
  model-scores.png          per-task scores vs 7B baseline and study best
  data/results.json         consolidated lm-eval results + provenance
  data/samples.jsonl        every evaluated item: prompt, output, target, score
  data/throughput.json      fixed-prompt decode/prefill probe
  manifest.json             lineage + per-task metric record
  audit.json                packaging gate record
```

Repository-level, added by the summary PR:

```
SUMMARY.md                  cross-model report
eval-data-index.json        machine-readable index of all models
audit-summary.json          gate roll-up across all models
methodology/                this file, scripts, task definitions
assets/                     cross-model charts
```

`<slug>` is a stable lowercase identifier (`mova36b-a4b-iq4-xs`,
`k2horizon-3.7b-q5-k-m`) rather than an HF revision, so links survive a refresh.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/build-dataset.py` | consolidate raw lm-eval output into per-model `data/` |
| `scripts/build-reports.py` | write per-model reports and the cross-model report |
| `scripts/model-charts.py` | per-model score charts |
| `scripts/scaling-charts.py` | cross-model scaling / Pareto / speed charts |
| `scripts/package-eval-data.py` | reshape into `eval-data-<slug>/`, write `manifest.json` + `audit.json`, run the gates |
| `scripts/summary-links.py` | fix relative links for the published layout, rebuild `SUMMARY.md` and the indexes |
| `scripts/file-eval-pr.sh` | file one PR per model (branch `eval/<slug>`) |
| `scripts/run-campaign.sh` | per-model load → eval → unload harness campaign (MoVA) |
| `scripts/run-dense-campaign.sh` | same, with per-family tokenizer selection |
| `scripts/tps-probe.sh` | fixed-prompt throughput probe |
| `tasks/k2_toolcall.yaml`, `tasks/k2_toolcall.jsonl` | the custom tool-calling task |

Regenerate everything in place:

```bash
python3 scripts/build-dataset.py      # raw harness output -> per-model data/
python3 scripts/model-charts.py
python3 scripts/scaling-charts.py
python3 scripts/build-reports.py
python3 scripts/package-eval-data.py  # -> eval-data-<slug>/ + audit + manifest
python3 scripts/summary-links.py      # -> SUMMARY.md + indexes
```

## Gates

`audit.json` records a packaging-time gate per model, all of which must pass
before a PR is opened:

| Gate | Meaning |
|---|---|
| `results_reparsed` | every reported score reparses as a number |
| `samples_wellformed` | every `samples.jsonl` row carries `task` and `doc_id` |
| `no_absolute_paths` | no host paths (`/mnt/…`, `/home/…`) in the payload |
| `readme_present` | the per-test report exists |
| `chart_present` | the per-model chart exists |
| `throughput_present` | the throughput probe exists |

`audit-summary.json` rolls these up across all 43 models; the current snapshot
passes every gate.
