#!/usr/bin/env python3
"""Build the per-model dataset for ifm-ai/inference-perf-data.

Emits, for every evaluated model:
  models/<name>/data/results.json     consolidated harness results (all tasks)
  models/<name>/data/samples.jsonl    raw per-sample prompt / output / score
  models/<name>/data/throughput.json  decode + prefill probe
  models/<name>/README.md             per-test report
  models/index.json                   machine-readable index
  SUMMARY.md                          full cross-model report
"""
import glob
import json
import os
import re

RES = "/mnt/Projects/lemonade-server/k2-eval-results"
SIZES = json.load(open("/mnt/Projects/lemonade-server/k2-sizes.json"))
OUT = "/mnt/Projects/inference-perf-data"
DATA = "/home/mvillmow/Random/k2-research/data.json"
TPS = "/home/mvillmow/Random/k2-research/tps.jsonl"

# task -> (display name, metric key, whether it counts toward the mean)
TASKS = {
    "humaneval": ("HumanEval", "pass@1,create_test", True),
    "mbpp": ("MBPP", "pass_at_1,none", True),
    "gsm8k": ("GSM8K", "exact_match,flexible-extract", True),
    "coqa": ("CoQA", "f1,none", True),
    "squad_completion": ("SQuAD", "contains,none", True),
    "ifeval": ("IFEval", "prompt_level_strict_acc,none", True),
    "k2_toolcall": ("Tool-call (custom)", "exact_match,extract-json", True),
    "bigbench_paragraph_segmentation_generate_until":
        ("BIG-bench paragraph segmentation", "exact_match,none", False),
}
TASK_ORDER = ["humaneval", "mbpp", "gsm8k", "coqa", "squad_completion",
              "ifeval", "k2_toolcall",
              "bigbench_paragraph_segmentation_generate_until"]
FAMILY = {"36B MoE (4B active)": "K2-Horizon-MoVA-36B-A4B", "3.7B dense": "K2-Horizon-3.7B",
          "0.9B dense": "K2-Horizon-0.9B", "7B dense": "K2-Horizon-7B"}

DISPLAY = {
    "K2-Horizon-7B-GGUF": "7B BF16 (baseline)", "K2-Horizon-0.9B-GGUF": "0.9B BF16",
    "K2-Horizon-3.7B-GGUF": "3.7B BF16",
    "feffd71999eb06bfa2fbb8ad220059b694077457": "FP8-Q8_0 (IFM-direct baseline)",
    "user.K2-MoVA-Q1_0": "Q1_0", "user.K2-MoVA-Q2_0": "Q2_0",
    "user.K2-0.9B-Q8_0": "0.9B Q8_0", "user.K2-0.9B-Q6_K": "0.9B Q6_K",
    "user.K2-0.9B-Q5_K_M": "0.9B Q5_K_M", "user.K2-0.9B-Q5_0": "0.9B Q5_0",
    "user.K2-0.9B-Q4_K_M": "0.9B Q4_K_M",
    "user.K2-3.7B-Q8_0": "3.7B Q8_0", "user.K2-3.7B-Q6_K": "3.7B Q6_K",
    "user.K2-3.7B-Q5_K_M": "3.7B Q5_K_M", "user.K2-3.7B-Q5_0": "3.7B Q5_0",
    "user.K2-3.7B-Q4_K_M": "3.7B Q4_K_M",
}
ALIAS = {
    "user.K2-MoVA-Q3_K_S": "K2-Horizon-MoVA-36B-A4B-Q3_K_S",
    "user.K2-MoVA-Q4_0": "K2-Horizon-MoVA-36B-A4B-Q4_0",
    "user.K2-MoVA-Q4_1": "K2-Horizon-MoVA-36B-A4B-Q4_1",
    "user.K2-MoVA-Q4_K_M": "K2-Horizon-MoVA-36B-A4B-Q4_K_M",
    "user.K2-MoVA-Q4_K_S": "K2-Horizon-MoVA-36B-A4B-Q4_K_S",
    "user.K2-MoVA-Q8_0": "K2-Horizon-MoVA-36B-A4B-Q8_0",
}


def slug(model):
    """Filesystem-safe, stable identifier."""
    s = ALIAS.get(model, model)
    if s.startswith("feffd"):
        return "mova36b-a4b-fp8-q8-0"
    if s == "K2-Horizon-7B-GGUF":
        return "k2horizon-7b-bf16"
    if s == "K2-Horizon-0.9B-GGUF":
        return "k2horizon-0.9b-bf16"
    if s == "K2-Horizon-3.7B-GGUF":
        return "k2horizon-3.7b-bf16"
    if s.startswith("user.K2-0.9B-"):
        s = "K2-Horizon-0.9B-" + s.rsplit("-", 1)[1]
    elif s.startswith("user.K2-3.7B-"):
        s = "K2-Horizon-3.7B-" + s.rsplit("-", 1)[1]
    s = s.replace("K2-Horizon-MoVA-36B-A4B-", "mova36b-a4b-")
    s = s.replace("K2-MoVA-", "mova36b-a4b-").replace("user.", "")
    s = s.replace("K2-Horizon-", "k2horizon-")
    return s.lower().replace("_", "-")


def label(model):
    if model in DISPLAY:
        return DISPLAY[model]
    m = ALIAS.get(model, model)
    if m.startswith("K2-Horizon-MoVA-36B-A4B-"):
        return m.replace("K2-Horizon-MoVA-36B-A4B-", "")
    return m


def family(model):
    m = ALIAS.get(model, model)
    if "0.9B" in m:
        return "0.9B dense"
    if "3.7B" in m:
        return "3.7B dense"
    if m == "K2-Horizon-7B-GGUF":
        return "7B dense"
    return "36B MoE (4B active)"


def size_gb(model):
    m = ALIAS.get(model, model)
    fixed = {"K2-Horizon-7B-GGUF": 16.8, "K2-Horizon-0.9B-GGUF": 2.01,
             "K2-Horizon-3.7B-GGUF": 9.43,
             "K2-Horizon-MoVA-36B-A4B-IQ1_S": 7.3,
             "K2-Horizon-MoVA-36B-A4B-IQ1_M": 8.1,
             "user.K2-MoVA-Q1_0": 5.4, "user.K2-MoVA-Q2_0": 10.4}
    if m in fixed:
        return fixed[m]
    if m.startswith("user.K2-0.9B-"):
        fn = f"K2-Horizon-1B-{m.rsplit('-', 1)[1]}.gguf"
    elif m.startswith("user.K2-3.7B-"):
        fn = f"K2-Horizon-4B-{m.rsplit('-', 1)[1]}.gguf"
    elif m.startswith("feffd"):
        fn = "K2-Horizon-MoVA-36B-A4B-FP8-Q8_0.gguf"
    else:
        fn = f"K2-Horizon-MoVA-36B-A4B-{m.rsplit('-', 1)[1]}.gguf"
    n = SIZES.get(fn)
    return round(n / 2 ** 30, 2) if n else None


def newest_results(model_dir):
    """Latest results file covering each task, merged into one dict.

    Each run writes two files (7-task set, then segmentation); n-samples and the
    provenance block are merged across both rather than overwritten by the last.
    """
    merged, meta, nsamples = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(model_dir, "results_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for t, v in (d.get("results") or {}).items():
            merged[t] = v
        for k in ("date", "lm_eval_version", "transformers_version",
                  "total_evaluation_time_seconds", "config", "versions",
                  "higher_is_better", "git_hash", "model_source"):
            if d.get(k) is not None:
                meta[k] = d[k]
        nsamples.update(d.get("n-samples") or {})
        meta["_files"] = meta.get("_files", []) + [os.path.basename(f)]
    meta["n-samples"] = nsamples
    return merged, meta


def collect_samples(model_dir):
    """Latest samples file per task -> {task: [row, ...]}"""
    files = {}
    for f in glob.glob(os.path.join(model_dir, "samples_*")):
        m = re.match(r"samples_(.+)_20\d{2}-", os.path.basename(f))
        if not m:
            continue
        t = m.group(1)
        if t not in files or os.path.getmtime(f) > os.path.getmtime(files[t]):
            files[t] = f
    out = {}
    for t, f in files.items():
        rows = []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        out[t] = rows
    return out


def score_of(task, block):
    """Pull the single headline metric for a task out of an lm-eval result block."""
    if not block:
        return None, None
    _, key, _ = TASKS[task]
    if key in block:
        return block[key], block.get(key.replace(",", "_stderr,").replace(
            "pass@1,", "pass@1_stderr,").replace("pass_at_1,", "pass_at_1_stderr,"))
    return None, None


def main():
    scores = {r["model"]: r for r in json.load(open(DATA))}
    tps = {}
    for line in open(TPS):
        if line.strip():
            d = json.loads(line)
            tps[ALIAS.get(d["model"], d["model"])] = {
                "decode_tok_per_s": d.get("predicted_per_second"),
                "prefill_tok_per_s": d.get("prompt_per_second"),
                "generated_tokens": d.get("predicted_n")}

    index = {}
    for model_dir in sorted(os.path.basename(d.rstrip("/"))
                            for d in glob.glob(f"{RES}/*/")):
        d = os.path.join(RES, model_dir)
        results, meta = newest_results(d)
        if not results:
            continue
        samples = collect_samples(d)
        model = ALIAS.get(model_dir, model_dir)
        key = model if model in scores else model_dir
        sc = scores.get(key, {})
        mean_vals = [sc[t] for t in TASKS if TASKS[t][2] and sc.get(t) is not None]
        mean = round(sum(mean_vals) / len(mean_vals), 4) if mean_vals else None

        slug_ = slug(key)
        root = os.path.join(OUT, "models", slug_)
        os.makedirs(os.path.join(root, "data"), exist_ok=True)

        payload = {"model": key, "label": label(key), "family": family(key),
                   "size_gib": size_gb(key), "mean_over_7": mean,
                   "throughput": tps.get(key),
                   "harness": meta, "tasks": {}}
        for t in TASK_ORDER:
            if t not in results and t not in samples:
                continue
            v = results.get(t, {})
            val, err = score_of(t, v)
            payload["tasks"][t] = {
                "display": TASKS[t][0], "metric": TASKS[t][1],
                "score": sc.get(t, val),
                "stderr": err, "in_mean": TASKS[t][2],
                "n_effective": (meta.get("n-samples") or {}).get(t, {}).get("effective"),
                "raw": {k: x for k, x in v.items()
                        if isinstance(x, (int, float)) and not k.endswith("_stderr")},
            }
        json.dump(payload, open(os.path.join(root, "data", "results.json"), "w"),
                  indent=2, default=str)

        with open(os.path.join(root, "data", "samples.jsonl"), "w") as fh:
            for t in TASK_ORDER:
                for r in samples.get(t, []):
                    fh.write(json.dumps({"task": t, **r}) + "\n")
        json.dump(tps.get(key) or {"note": "no throughput probe for this model"},
                  open(os.path.join(root, "data", "throughput.json"), "w"), indent=2)

        index[slug_] = {k: payload[k] for k in
                        ("model", "label", "family", "size_gib", "mean_over_7")}
        index[slug_]["throughput"] = tps.get(key)
        index[slug_]["scores"] = {t: sc.get(t) for t in TASKS}

    json.dump(index, open(os.path.join(OUT, "models", "index.json"), "w"), indent=2)
    print(f"built {len(index)} model dirs")
    return index


if __name__ == "__main__":
    main()
