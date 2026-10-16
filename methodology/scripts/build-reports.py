#!/usr/bin/env python3
"""Write models/<slug>/README.md (per-test report) and SUMMARY.md."""
import json
import os

OUT = "/mnt/Projects/inference-perf-data"
MODELS = os.path.join(OUT, "models")
DATA = "/home/mvillmow/Random/k2-research/data.json"

TASK_ORDER = ["humaneval", "mbpp", "gsm8k", "coqa", "squad_completion",
              "ifeval", "k2_toolcall",
              "bigbench_paragraph_segmentation_generate_until"]
FAMILY_ORDER = ["0.9B dense", "3.7B dense", "7B dense", "36B MoE (4B active)"]
BASE7B = 0.466


def load(slug_):
    return json.load(open(os.path.join(MODELS, slug_, "data", "results.json")))


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.3f}".lstrip("0")
    except (TypeError, ValueError):
        return str(v)


def size_str(g):
    return "—" if g is None else f"{g:.2f} GiB"


# ---------------------------------------------------------------- per-model
def write_model_readme(index):
    for slug_, entry in sorted(index.items()):
        p = load(slug_)
        tp = p.get("throughput") or {}
        L = [f"# {p['label']}", ""]
        L.append(f"`{p['model']}` · {p['family']} · {size_str(p['size_gib'])}")
        L.append("")
        bits = []
        if tp.get("decode_tok_per_s"):
            bits.append(f"**{tp['decode_tok_per_s']} tok/s** decode")
        if tp.get("prefill_tok_per_s"):
            bits.append(f"{tp['prefill_tok_per_s']} tok/s prefill")
        if p.get("mean_over_7") is not None:
            bits.append(f"**mean {fmt(p['mean_over_7'])}** over 7 tasks")
        L.append(" · ".join(bits))
        L.append("")
        L.append("## Per-test results")
        L.append("")
        L.append("| Test | Metric | Score | ± stderr | n | in mean |")
        L.append("|---|---|---|---|---|---|")
        for t in TASK_ORDER:
            d = p["tasks"].get(t)
            if not d:
                continue
            L.append(f"| {d['display']} | `{d['metric']}` | {fmt(d['score'])} | "
                     f"{fmt(d['stderr'])} | {d.get('n_effective') or '—'} | "
                     f"{'yes' if d['in_mean'] else 'no'} |")
        L.append("")
        L.append("![scores](model-scores.png)")
        L.append("")
        seg = p["tasks"].get("bigbench_paragraph_segmentation_generate_until")
        if seg and (seg.get("score") or 0) == 0:
            L.append("> **Note.** BIG-bench paragraph segmentation scored `0.00` for "
                     "every model in this study, including the FP8 baseline. The metric "
                     "is non-discriminating at this scale and is excluded from the mean.")
            L.append("")
        verdict = p.get("mean_over_7")
        if verdict is not None:
            delta = verdict - BASE7B
            rel = ("above" if delta > 0 else "below") if abs(delta) > 1e-9 else "level with"
            L.append(f"Mean over the 7 discriminating tasks is {fmt(verdict)}, "
                     f"{rel} the 7B dense baseline's {fmt(BASE7B)} "
                     f"({'+' if delta >= 0 else ''}{delta:.3f}).")
            L.append("")
        L.append("## Files")
        L.append("")
        L.append("| Path | Contents |")
        L.append("|---|---|")
        L.append("| `data/results.json` | consolidated lm-eval results, provenance, "
                 "raw per-filter metrics |")
        L.append("| `data/samples.jsonl` | every evaluated item: prompt, model "
                 "output, target, per-item score |")
        L.append("| `data/throughput.json` | fixed-prompt decode/prefill probe |")
        L.append("")
        L.append("See [`SUMMARY.md`](../../SUMMARY.md) for the cross-model report and "
                 "[`methodology/`](../../methodology/) for how these numbers were produced.")
        open(os.path.join(MODELS, slug_, "README.md"), "w").write("\n".join(L))


# ---------------------------------------------------------------- summary
def write_summary(index):
    by = {}
    for slug_, e in index.items():
        p = load(slug_)
        if p.get("mean_over_7") is None:
            continue
        by.setdefault(p["family"], []).append((p["mean_over_7"], slug_, p))
    L = ["# K2-Horizon quantization study — CPU inference performance & quality", ""]
    L.append("Full results for every GGUF quant of the K2-Horizon family evaluated on a "
             "CPU-only host with the IFM K2-Horizon llama.cpp backend: **36B MoE "
             "(4B active)**, plus the dense **3.7B**, **0.9B** and **7B** siblings.")
    L.append("")
    L.append(f"- **{len(index)} models** evaluated · 8 lm-eval tasks · 12 samples/task "
             "· 1 fixed throughput probe each")
    L.append("- Host: 8-thread Intel Xeon E5 (Sandy Bridge-E), 62 GiB DDR3, **no GPU**")
    L.append("- Harness: EleutherAI `lm-eval` over Lemonade's OpenAI-compatible endpoint")
    L.append("")
    L.append("> **Read this first.** n=12 per task gives a per-task standard error of "
             "roughly ±0.08–0.15, so the 7-task mean carries about ±0.03–0.05. "
             "Differences smaller than that are not resolved by this study. Anything "
             "at the top of the table is a statistical tie with anything else near it.")
    L.append("")
    L.append("## Headline")
    L.append("")
    rows = []
    for fam in FAMILY_ORDER:
        rows += by.get(fam, [])
    rows.sort(key=lambda r: -r[0])
    L.append("| Rank | Model | Family | Size | mean (7 tasks) | decode tok/s | vs 7B |")
    L.append("|---:|---|---|---|---|---|---|")
    for i, (mean, slug_, p) in enumerate(rows, 1):
        tp = (p.get("throughput") or {}).get("decode_tok_per_s")
        d = mean - BASE7B
        mark = "—" if slug_ == "k2horizon-7b-bf16" else \
            f"{'+' if d >= 0 else ''}{d:.3f}"
        L.append(f"| {i} | [{p['label']}](models/{slug_}/README.md) | "
                 f"{p['family']} | {size_str(p['size_gib'])} | **{fmt(mean)}** | "
                 f"{tp if tp else '—'} | {mark} |")
    L.append("")
    L.append("## Full matrix")
    L.append("")
    hdr = ["human", "mbpp", "gsm8k", "coqa", "squad", "ifeval", "toolcall"]
    L.append("| Model | " + " | ".join(hdr) + " | seg | mean |")
    L.append("|---|" + "---|" * (len(hdr) + 2))
    for mean, slug_, p in rows:
        cells = [fmt(p["tasks"].get(t, {}).get("score")) for t in
                 ["humaneval", "mbpp", "gsm8k", "coqa", "squad_completion",
                  "ifeval", "k2_toolcall"]]
        seg = p["tasks"].get("bigbench_paragraph_segmentation_generate_until", {})
        L.append(f"| [{p['label']}](models/{slug_}/README.md) | " + " | ".join(cells) +
                 f" | {fmt(seg.get('score'))} | **{fmt(mean)}** |")
    L.append("")
    L.append("## Throughput")
    L.append("")
    L.append("![speed vs size](assets/size_vs_speed.png)")
    L.append("")
    L.append("![pareto](assets/pareto_frontier.png)")
    L.append("")
    L.append("Decode throughput on 8 CPU threads, fixed prompt, 64 generated tokens, "
             "temp 0.")
    L.append("")
    L.append("| Model | Size | decode tok/s | prefill tok/s |")
    L.append("|---|---|---|---|")
    tp_rows = []
    for slug_, e in index.items():
        p = load(slug_)
        tp = p.get("throughput") or {}
        if tp.get("decode_tok_per_s"):
            tp_rows.append((tp["decode_tok_per_s"], slug_, p, tp))
    for _, slug_, p, tp in sorted(tp_rows):
        L.append(f"| [{p['label']}](models/{slug_}/README.md) | "
                 f"{size_str(p['size_gib'])} | {tp['decode_tok_per_s']} | "
                 f"{tp.get('prefill_tok_per_s', '—')} |")
    L.append("")
    L.append("## Findings")
    L.append("")
    L.append("1. **The 36B MoE is the quality ceiling and nothing else is close.** "
             "Every MoVA quant from ~15 GB up matches or beats the dense siblings; "
             "the best dense model is the 3.7B.")
    L.append("2. **Quality is flat across the upper quant ladder.** IQ4_XS (18.7 GiB) "
             "through Q8_0 (37.1 GiB) sit in a ~0.05 band — inside the noise floor. "
             "Larger quants buy disk, not accuracy.")
    L.append("3. **The dense 3.7B beats the dense 7B** (mean and 44% fewer params), so "
             "the family does not scale monotonically in dense parameters.")
    L.append("4. **Quantization is free at 3.7B**: all six builds land within 0.016 of "
             "each other. Take Q5_K_M at 3.4 GiB, not BF16 at 9.4 GiB.")
    L.append("5. **Tool-calling is the cliff, and it is a model-family property.** "
             "Several MoVA quants score exactly 0.00; the 3.7B scores 0.00 at *every* "
             "precision including BF16. The 7B manages 0.25, so this is not "
             "quantization damage.")
    L.append("6. **MoE breaks the speed-vs-size line.** Dense speed tracks total "
             "parameters (0.9B → 27.8 tok/s, 7B → 1.4 tok/s); the 36B MoE carries 10× "
             "the parameters of the 3.7B and still decodes ~2× faster than the 7B.")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("See [`methodology/`](methodology/) — harness configuration, hardware, "
             "and the exact scripts used to produce every number here.")
    open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    index = json.load(open(os.path.join(MODELS, "index.json")))
    write_model_readme(index)
    write_summary(index)
    print(f"wrote {len(index)} model reports + SUMMARY.md")
