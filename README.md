# inference-perf-data

Data store of tech report information, information for future blog posts, and
open-source inference investigations.

## Dataset series

Two kinds of measurement live here under distinct prefixes, because they are
**not comparable to each other**:

| Prefix | Measurement class | What it captures |
|---|---|---|
| `perf-data-*` | GPU inference throughput | ISL/OSL/concurrency sweeps on H200 with vLLM/SGLang, 3 reps per point |
| `eval-data-*` | CPU quality evaluation | Accuracy on 8 lm-eval tasks + single-stream CPU throughput, CPU-only host |

---

## `eval-data-*` — K2-Horizon CPU quality evaluation

**[`SUMMARY.md`](SUMMARY.md)** — cross-model report. **43 models**, 8 EleutherAI
lm-eval tasks, 12 samples per task, raw per-sample outputs for every item.

Covers every GGUF build of the **36B MoE (4B active)** that fits on disk, plus
the dense **3.7B**, **0.9B** and **7B** siblings, on a CPU-only host using the
IFM K2-Horizon llama.cpp backend.

Headline findings:

- The **36B MoE is the quality ceiling** — every quant above ~15 GiB matches or
  beats the dense siblings.
- **Quality is flat across the upper ladder**: IQ4_XS (18.7 GiB) through Q8_0
  (37.1 GiB) sit inside a ~0.05 band, within the noise floor.
- The **dense 3.7B beats the dense 7B** with 44% fewer parameters, so the family
  does not scale monotonically in dense parameters.
- **Quantization is free at 3.7B**: all six builds land within 0.016 of each
  other.
- **Tool-calling is a model-family property, not a quantization artifact** — the
  3.7B scores 0.00 at every precision *including BF16*, while the 7B manages 0.25.
- **MoE breaks the speed-vs-size line**: 36B weights but ~4B active decodes ~2×
  faster than the dense 7B.

Each model ships as its own PR and its own `eval-data-<slug>/` directory with the
raw dataset, a per-test report, a chart, a manifest and a gate record. The
machine-readable index is [`eval-data-index.json`](eval-data-index.json); the
gate roll-up is [`audit-summary.json`](audit-summary.json).

> **Read [`methodology/README.md`](methodology/README.md) before quoting any
> single number.** At n=12 per task the standard error is ±0.08–0.15, so the top
> of the leaderboard is a statistical tie and this study deliberately does not
> rank within it.

## Layout

```
SUMMARY.md                 cross-model report for the eval-data-* series
eval-data-index.json       machine-readable index of all evaluated models
audit-summary.json         packaging gate roll-up across all models
methodology/               harness config, hardware, statistical limits, scripts
assets/                    cross-model charts
eval-data-<slug>/          per-model dataset (own PR)
  README.md                per-test report
  model-scores.png         per-task scores vs baseline and study best
  data/results.json        consolidated lm-eval results + provenance
  data/samples.jsonl       every evaluated item: prompt, output, target, score
  data/throughput.json     fixed-prompt decode/prefill probe
  manifest.json            lineage + per-task metric record
  audit.json               packaging gate record
```

Model directories use a stable slug (`mova36b-a4b-iq4-xs`,
`k2horizon-3.7b-q5-k-m`) rather than an HF revision, so links survive a refresh.
