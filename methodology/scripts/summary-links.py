#!/usr/bin/env python3
"""Fix relative links for the eval-data-<slug>/ top-level layout and rebuild SUMMARY.md.

model READMEs were written assuming models/<slug>/ (two levels down). At
eval-data-<slug>/ they are one level down, so ../../SUMMARY.md -> ../SUMMARY.md.
"""
import json
import os
import re

STAGING = "/mnt/Projects/k2-perf-staging"
EVALS = sorted(d for d in os.listdir(STAGING)
               if d.startswith("eval-data-") and
               os.path.isdir(os.path.join(STAGING, d)))

fixed = 0
for d in EVALS:
    p = os.path.join(STAGING, d, "README.md")
    if not os.path.isfile(p):
        continue
    s = open(p).read()
    n = s.replace("](../../SUMMARY.md)", "](../SUMMARY.md)")
    n = n.replace("](../../methodology/)", "](../methodology/)")
    if n != s:
        open(p, "w").write(n)
        fixed += 1
print(f"relinked {fixed}/{len(EVALS)} model READMEs")

# ------------------------------------------------------------------ SUMMARY
TASK_ORDER = ["humaneval", "mbpp", "gsm8k", "coqa", "squad_completion",
              "ifeval", "k2_toolcall",
              "bigbench_paragraph_segmentation_generate_until"]
FAMILY_ORDER = ["0.9B dense", "3.7B dense", "7B dense", "36B MoE (4B active)"]
BASE7B = 0.466


def load(slug):
    return json.load(open(os.path.join(STAGING, f"eval-data-{slug}",
                                       "data", "results.json")))


def audit(slug):
    return json.load(open(os.path.join(STAGING, f"eval-data-{slug}", "audit.json")))


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.3f}".lstrip("0")
    except (TypeError, ValueError):
        return str(v)


slugs = [d.replace("eval-data-", "") for d in EVALS]
rows = []
for s in slugs:
    p = load(s)
    if p.get("mean_over_7") is None:
        continue
    rows.append((p["mean_over_7"], s, p))
rows.sort(key=lambda r: -r[0])

L = ["# K2-Horizon CPU quality evaluation — cross-model summary", ""]
L.append("Quality and CPU throughput for every GGUF build of the K2-Horizon family "
         "evaluated on a CPU-only host with the IFM K2-Horizon llama.cpp backend: "
         "**36B MoE (4B active)**, plus the dense **3.7B**, **0.9B** and **7B** "
         "siblings.")
L.append("")
L.append(f"- **{len(rows)} models** with scores · 8 lm-eval tasks · 12 samples/task · "
         "1 fixed throughput probe each")
L.append("- Host: 8-thread Intel Xeon E5 (Sandy Bridge-E), 62 GiB DDR3, **no GPU**")
L.append("- Harness: EleutherAI `lm-eval` 0.4.13 over Lemonade's OpenAI-compatible endpoint")
L.append("")
L.append("> **Read this first.** n=12 per task gives a per-task standard error of "
         "roughly ±0.08–0.15, so the 7-task mean carries about ±0.03–0.05. "
         "Differences smaller than that are **not resolved by this study**. The top of "
         "the table is a statistical tie, and nothing here claims otherwise.")
L.append("")
L.append("This is a `cpu_quality_eval` dataset — accuracy plus single-stream CPU "
         "throughput. It is **not** a GPU throughput sweep and is not comparable to "
         "the `perf-data-*` series in this repository.")
L.append("")
L.append("## Ranking by mean score")
L.append("")
L.append("| # | Model | Family | Size | mean (7 tasks) | decode tok/s | vs 7B |")
L.append("|---:|---|---|---|---|---|---|")
for i, (mean, s, p) in enumerate(rows, 1):
    tp = (p.get("throughput") or {}).get("decode_tok_per_s")
    d = mean - BASE7B
    mark = "—" if p["model"] == "K2-Horizon-7B-GGUF" else f"{'+' if d >= 0 else ''}{d:.3f}"
    sz = "—" if p["size_gib"] is None else f"{p['size_gib']:.2f} GiB"
    L.append(f"| {i} | [{p['label']}](eval-data-{s}/README.md) | {p['family']} | "
             f"{sz} | **{fmt(mean)}** | {tp if tp else '—'} | {mark} |")
L.append("")
L.append("## Full score matrix")
L.append("")
hdr = ["human", "mbpp", "gsm8k", "coqa", "squad", "ifeval", "toolcall"]
L.append("| Model | " + " | ".join(hdr) + " | seg | mean |")
L.append("|---|" + "---|" * (len(hdr) + 2))
for mean, s, p in rows:
    cells = [fmt(p["tasks"].get(t, {}).get("score")) for t in
             ["humaneval", "mbpp", "gsm8k", "coqa", "squad_completion",
              "ifeval", "k2_toolcall"]]
    seg = p["tasks"].get("bigbench_paragraph_segmentation_generate_until", {})
    L.append(f"| [{p['label']}](eval-data-{s}/README.md) | " + " | ".join(cells) +
             f" | {fmt(seg.get('score'))} | **{fmt(mean)}** |")
L.append("")
L.append("`seg` (BIG-bench paragraph segmentation) scored 0.00 for every model "
         "including the FP8 baseline and is excluded from the mean.")
L.append("")
L.append("## Throughput")
L.append("")
L.append("![speed vs size](assets/size_vs_speed.png)")
L.append("")
L.append("![pareto frontier](assets/pareto_frontier.png)")
L.append("")
L.append("Fixed prompt, temperature 0, 64 generated tokens, 8 threads, one model "
         "loaded at a time.")
L.append("")
L.append("| Model | Size | decode tok/s | prefill tok/s |")
L.append("|---|---|---|---|")
tp_rows = []
for mean, s, p in rows:
    tp = p.get("throughput") or {}
    if tp.get("decode_tok_per_s"):
        tp_rows.append((tp["decode_tok_per_s"], s, p, tp))
for _, s, p, tp in sorted(tp_rows):
    sz = "—" if p["size_gib"] is None else f"{p['size_gib']:.2f} GiB"
    L.append(f"| [{p['label']}](eval-data-{s}/README.md) | {sz} | "
             f"{tp['decode_tok_per_s']} | {tp.get('prefill_tok_per_s', '—')} |")
L.append("")
L.append("Throughput is the weakest part of this dataset: a single 64-token sample "
         "per model with no repeats, so differences under ~10% are noise.")
L.append("")
L.append("## Findings")
L.append("")
L.append("1. **The 36B MoE is the quality ceiling.** Every MoVA quant from ~15 GiB up "
         "matches or beats every dense sibling; the best dense model is the 3.7B.")
L.append("2. **Quality is flat across the upper ladder.** IQ4_XS (18.7 GiB) through "
         "Q8_0 (37.1 GiB) sit inside a ~0.05 band — within the noise floor. Larger "
         "quants buy disk, not accuracy.")
L.append("3. **The dense 3.7B beats the dense 7B** (mean .518 vs .466) with 44% fewer "
         "parameters, so the family does not scale monotonically in dense parameters. "
         "At n=12 that gap is ~1σ, so treat it as suggestive rather than settled.")
L.append("4. **Quantization is free at 3.7B.** All six builds land within 0.016 of "
         "each other — take Q5_K_M at 3.4 GiB rather than BF16 at 9.4 GiB.")
L.append("5. **Tool-calling is a model-family property, not a quantization artifact.** "
         "Several MoVA quants score exactly 0.00, and the 3.7B scores 0.00 at *every* "
         "precision including BF16, while the 7B manages 0.25.")
L.append("6. **MoE breaks the speed-vs-size line.** Dense speed tracks total "
         "parameters (0.9B → 27.8 tok/s, 7B → 1.4 tok/s); the 36B MoE carries 10× the "
         "parameters of the 3.7B and still decodes ~2× faster than the 7B.")
L.append("")
L.append("## Per-model datasets")
L.append("")
L.append("Each model ships its own PR and directory containing the raw dataset, the "
         "per-test report, a chart, a manifest and a gate record. Machine-readable "
         "index: [`eval-data-index.json`](eval-data-index.json).")
L.append("")
L.append("| Slug | Model | PR |")
L.append("|---|---|---|")
for mean, s, p in rows:
    L.append(f"| `{s}` | {p['label']} | see repo PR list, head `eval/{s}` |")
L.append("")
L.append("## Method")
L.append("")
L.append("Full harness configuration, hardware, generation settings, tokenizer "
         "selection per family, statistical limits and known measurement issues: "
         "[`methodology/README.md`](methodology/README.md). Regeneration scripts are "
         "in [`methodology/scripts/`](methodology/scripts/).")
open(os.path.join(STAGING, "SUMMARY.md"), "w").write("\n".join(L))

# machine-readable index + gate roll-up
idx, fails = {}, []
for s in slugs:
    a = audit(s)
    p = load(s)
    idx[s] = {"model": p["model"], "label": p["label"], "family": p["family"],
              "size_gib": p["size_gib"], "mean_over_7": p["mean_over_7"],
              "throughput": p.get("throughput"),
              "scores": {t: p["tasks"].get(t, {}).get("score") for t in TASK_ORDER},
              "audit": a["gates"], "sampled_items": a["sampled_items"]}
    for k, v in a["gates"].items():
        if v is not True:
            fails.append((s, k))
json.dump(idx, open(os.path.join(STAGING, "eval-data-index.json"), "w"), indent=2)
json.dump({"models": len(idx), "gate_failures": fails,
           "all_gates_pass": not fails,
           "gates_checked": ["results_reparsed", "samples_wellformed",
                             "no_absolute_paths", "readme_present",
                             "chart_present", "throughput_present"]},
          open(os.path.join(STAGING, "audit-summary.json"), "w"), indent=2)
print(f"SUMMARY.md + index written for {len(idx)} models; gate failures: {fails or 'none'}")
