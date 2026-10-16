#!/usr/bin/env python3
"""Per-model chart: this model's per-task scores vs the 7B baseline and the
best score any model achieved on that task."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/Projects/inference-perf-data"
MODELS = os.path.join(OUT, "models")
ASSETS = os.path.join(OUT, "assets")
os.makedirs(ASSETS, exist_ok=True)

TASKS = [("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k", "GSM8K"),
         ("coqa", "CoQA"), ("squad_completion", "SQuAD"), ("ifeval", "IFEval"),
         ("k2_toolcall", "Tool-call")]
BASE = "k2horizon-7b-bf16"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load(s):
    return json.load(open(os.path.join(MODELS, s, "data", "results.json")))


index = json.load(open(os.path.join(MODELS, "index.json")))
best = {t: max((load(s)["tasks"].get(t, {}).get("score") or 0)
               for s in index) for t, _ in TASKS}
b7 = load(BASE)

for slug_, _e in sorted(index.items()):
    p = load(slug_)
    vals = [p["tasks"].get(t, {}).get("score") or 0 for t, _ in TASKS]
    ref = [b7["tasks"].get(t, {}).get("score") or 0 for t, _ in TASKS]
    top = [best[t] for t, _ in TASKS]

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = range(len(TASKS))
    w = 0.27
    ax.bar([i - w for i in x], vals, w, label="this model", color="#2980b9")
    ax.bar(list(x), ref, w, label="7B dense baseline", color="#bdc3c7")
    ax.bar([i + w for i in x], top, w, label="best in study", color="#f5b041",
           alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n for _, n in TASKS], rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"{p['label']} — per-task scores vs baseline and study best",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS, slug_, "model-scores.png"))
    plt.close(fig)

print(f"wrote {len(index)} model charts")
