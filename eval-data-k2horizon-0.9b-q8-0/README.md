# 0.9B Q8_0

`user.K2-0.9B-Q8_0` · 0.9B dense · 1.07 GiB

**21.61 tok/s** decode · 52.0 tok/s prefill · **mean .423** over 7 tasks

## Per-test results

| Test | Metric | Score | ± stderr | n | in mean |
|---|---|---|---|---|---|
| HumanEval | `pass@1,create_test` | .417 | .149 | 12 | yes |
| MBPP | `pass_at_1,none` | .417 | .149 | 12 | yes |
| GSM8K | `exact_match,flexible-extract` | .583 | .149 | 12 | yes |
| CoQA | `f1,none` | .542 | .120 | 12 | yes |
| SQuAD | `contains,none` | .667 | N/A | 12 | yes |
| IFEval | `prompt_level_strict_acc,none` | .167 | .112 | 12 | yes |
| Tool-call (custom) | `exact_match,extract-json` | .167 | .112 | 12 | yes |
| BIG-bench paragraph segmentation | `exact_match,none` | .000 | .000 | 5 | no |

![scores](model-scores.png)

> **Note.** BIG-bench paragraph segmentation scored `0.00` for every model in this study, including the FP8 baseline. The metric is non-discriminating at this scale and is excluded from the mean.

Mean over the 7 discriminating tasks is .423, below the 7B dense baseline's .466 (-0.043).

## Files

| Path | Contents |
|---|---|
| `data/results.json` | consolidated lm-eval results, provenance, raw per-filter metrics |
| `data/samples.jsonl` | every evaluated item: prompt, model output, target, per-item score |
| `data/throughput.json` | fixed-prompt decode/prefill probe |

See [`SUMMARY.md`](../SUMMARY.md) for the cross-model report and [`methodology/`](../methodology/) for how these numbers were produced.