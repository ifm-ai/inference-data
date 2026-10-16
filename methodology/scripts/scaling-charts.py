#!/usr/bin/env python3
"""Scaling + frontier charts across the four K2-Horizon families.

Reads data.json (scores) and tps.jsonl (throughput); real sizes from k2-sizes.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/mnt/Projects/lemonade-server/k2-report/assets"
DATA = "/home/mvillmow/Random/k2-research/data.json"
TPS = "/home/mvillmow/Random/k2-research/tps.jsonl"
SIZES = json.load(open("/mnt/Projects/lemonade-server/k2-sizes.json"))

TASKS = ["humaneval", "mbpp", "ifeval", "k2_toolcall", "coqa", "gsm8k",
         "squad_completion"]

# display label, family, ordering weight
DISPLAY = {
    "K2-Horizon-7B-GGUF": ("7B BF16", "7B dense", 0),
    "feffd71999eb06bfa2fbb8ad220059b694077457": ("FP8-Q8_0", "36B MoE (4B act)", 2),
    "K2-Horizon-0.9B-GGUF": ("0.9B BF16", "0.9B dense", -3),
    "K2-Horizon-3.7B-GGUF": ("3.7B BF16", "3.7B dense", -2),
}
PREFIX = {"user.K2-0.9B-": ("0.9B ", "0.9B dense", -3),
          "user.K2-3.7B-": ("3.7B ", "3.7B dense", -2),
          "user.K2-MoVA-": ("", "36B MoE (4B act)", 2),
          "K2-Horizon-MoVA-36B-A4B-": ("", "36B MoE (4B act)", 2)}
FAMILY_COLOR = {"0.9B dense": "#e67e22", "3.7B dense": "#8e44ad",
                "7B dense": "#7f8c8d", "36B MoE (4B act)": "#2980b9"}


def label(m):
    if m in DISPLAY:
        return DISPLAY[m][0], DISPLAY[m][1], DISPLAY[m][2]
    if m.startswith("user.K2-0.9B-"):
        return PREFIX["user.K2-0.9B-"][0] + m.rsplit("-", 1)[1], "0.9B dense", -3
    if m.startswith("user.K2-3.7B-"):
        return PREFIX["user.K2-3.7B-"][0] + m.rsplit("-", 1)[1], "3.7B dense", -2
    if m.startswith("user.K2-MoVA-"):
        return m.replace("user.K2-MoVA-", ""), "36B MoE (4B act)", 2
    return m.replace("K2-Horizon-MoVA-36B-A4B-", ""), "36B MoE (4B act)", 2


def size_gb(m):
    if m == "K2-Horizon-7B-GGUF":
        return 16.8
    if m == "K2-Horizon-0.9B-GGUF":
        return 2.01
    if m == "K2-Horizon-3.7B-GGUF":
        return 9.43
    if m.startswith("feffd"):
        n = SIZES.get("K2-Horizon-MoVA-36B-A4B-FP8-Q8_0.gguf")
    elif m.startswith("user.K2-0.9B-"):
        n = SIZES.get(f"K2-Horizon-1B-{m.rsplit('-', 1)[1]}.gguf")
    elif m.startswith("user.K2-3.7B-"):
        n = SIZES.get(f"K2-Horizon-4B-{m.rsplit('-', 1)[1]}.gguf")
    else:
        n = SIZES.get(f"K2-Horizon-MoVA-36B-A4B-{m.rsplit('-', 1)[1]}.gguf")
    return round(n / 2 ** 30, 2) if n else None


tps = {}
for line in open(TPS):
    line = line.strip()
    if line:
        d = json.loads(line)
        tps[d["model"]] = (d.get("predicted_per_second"), d.get("prompt_per_second"))

rows = []
for r in json.load(open(DATA)):
    lab, fam, w = label(r["model"])
    sc = [r[t] for t in TASKS]
    have = [x for x in sc if x is not None]
    t, pt = tps.get(r["model"], (None, None))
    rows.append({"model": r["model"], "label": lab, "fam": fam, "w": w,
                 "size": size_gb(r["model"]), "tps": t, "scores": sc,
                 "mean": sum(have) / len(have) if have else None})

ok = [r for r in rows if r["mean"] is not None and r["tps"] and r["size"]]
print(f"{len(ok)}/{len(rows)} models have score+tps+size")
ok.sort(key=lambda r: (r["w"], r["size"]))

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# 1. scaling: mean score vs decode throughput, one colour per family
fig, ax = plt.subplots(figsize=(10, 6))
for fam in ("0.9B dense", "3.7B dense", "7B dense", "36B MoE (4B act)"):
    pts = [r for r in ok if r["fam"] == fam]
    if not pts:
        continue
    ax.scatter([p["tps"] for p in pts], [p["mean"] for p in pts],
               s=90, c=FAMILY_COLOR[fam], alpha=0.75, edgecolors="k",
               linewidths=0.5, label=fam)
    for p in pts:
        ax.annotate(p["label"], (p["tps"], p["mean"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("Generation throughput (tok/s, log scale) - 8-core CPU")
ax.set_ylabel("Mean score over 7 harness tasks")
ax.set_title("Quality vs speed across the four K2-Horizon families")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/scaling_quality_speed.png")

# 2. Pareto frontier: a model is optimal if nothing else is both >= quality
#    and >= speed, with at least one strict.
pareto = []
for r in ok:
    dominated = any(
        (p["mean"] >= r["mean"] and p["tps"] >= r["tps"]) and
        (p["mean"] > r["mean"] or p["tps"] > r["tps"])
        for p in ok)
    if not dominated:
        pareto.append(r)
pareto.sort(key=lambda r: r["tps"])
fig, ax = plt.subplots(figsize=(10, 6))
for r in ok:
    front = r in pareto
    ax.scatter(r["tps"], r["mean"], s=110 if front else 55,
               c=FAMILY_COLOR[r["fam"]], alpha=0.95 if front else 0.35,
               edgecolors="#c0392b" if front else "none",
               linewidths=1.6 if front else 0, zorder=3 if front else 2)
for r in pareto:
    ax.annotate(f"{r['label']}\n{r['mean']:.2f} @ {r['tps']:.1f} t/s",
                (r["tps"], r["mean"]), xytext=(8, -2),
                textcoords="offset points", fontsize=7.5, color="#c0392b")
ax.set_xscale("log")
ax.set_xlabel("Generation throughput (tok/s, log scale)")
ax.set_ylabel("Mean score over 7 harness tasks")
ax.set_title("Pareto frontier: no model here is both faster and better "
             "than a red-outlined one")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pareto_frontier.png")

# 3. size vs speed: MoE vs dense lines
fig, ax = plt.subplots(figsize=(9, 5.5))
for fam, mk in (("0.9B dense", "o"), ("3.7B dense", "s"),
                ("7B dense", "^"), ("36B MoE (4B act)", "D")):
    pts = sorted([r for r in ok if r["fam"] == fam], key=lambda r: r["size"])
    if not pts:
        continue
    ax.plot([p["size"] for p in pts], [p["tps"] for p in pts], marker=mk,
            color=FAMILY_COLOR[fam], label=fam, alpha=0.8, markersize=6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Model size (GiB, log)")
ax.set_ylabel("Decode throughput (tok/s, log)")
ax.set_title("Speed vs size: MoE sits off the dense line (only ~4B active)")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/size_vs_speed.png")

# 4. family summary bars
fams = ["0.9B dense", "3.7B dense", "7B dense", "36B MoE (4B act)"]
best = {}
for f in fams:
    pts = [r for r in ok if r["fam"] == f]
    if pts:
        best[f] = max(pts, key=lambda r: r["mean"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
names = [f.replace(" (4B act)", "") for f in fams if f in best]
a1.bar(names, [best[f]["mean"] for f in fams if f in best],
       color=[FAMILY_COLOR[f] for f in fams if f in best])
for i, f in enumerate([f for f in fams if f in best]):
    a1.text(i, best[f]["mean"] + 0.005, f"{best[f]['mean']:.3f}",
            ha="center", fontsize=8)
a1.set_ylabel("Best mean score in family")
a1.set_title("Best of each family")
a1.set_ylim(0, 0.7)
a2.bar(names, [best[f]["tps"] for f in fams if f in best],
       color=[FAMILY_COLOR[f] for f in fams if f in best])
for i, f in enumerate([f for f in fams if f in best]):
    a2.text(i, best[f]["tps"] + 0.6, f"{best[f]['tps']:.1f}", ha="center",
            fontsize=8)
a2.set_ylabel("tok/s of that model")
a2.set_title("Throughput of each family's best model")
a2.set_ylim(0, 32)
fig.tight_layout()
fig.savefig(f"{OUT}/family_summary.png")

print("Pareto set:")
for r in pareto:
    print(f"  {r['label']:<18} {r['mean']:.3f}  {r['tps']:>6.2f} t/s  "
          f"{r['size']:>6.2f} GiB  ({r['fam']})")
print("charts done")
