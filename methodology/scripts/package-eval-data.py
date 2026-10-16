#!/usr/bin/env python3
"""Reshape staged output into the eval-data-<slug>/ convention + audit/manifest.

Matches the quality bar set by the existing perf-data-7b/ dataset: an explicit
gate record, a manifest with lineage, and regeneration tooling.
"""
import json
import os
import shutil

STAGING = "/mnt/Projects/k2-perf-staging"
MODELS = os.path.join(STAGING, "models")
EVAL = STAGING
SLUGS = sorted(d for d in os.listdir(MODELS)
               if os.path.isdir(os.path.join(MODELS, d)))


def gates_for(root):
    """Reproducible gate checks, mirroring perf-data-7b's audit.json style."""
    data = os.path.join(root, "data")
    res = json.load(open(os.path.join(data, "results.json")))
    n_samples = sum(1 for _ in open(os.path.join(data, "samples.jsonl")))
    tasks = res.get("tasks", {})

    # no absolute host paths anywhere in the payload
    blob = open(os.path.join(data, "results.json")).read() + \
        open(os.path.join(data, "samples.jsonl")).read()
    no_abs = "/mnt/" not in blob and "/home/" not in blob

    # every reported score reparses as a float
    reparses = all(
        v["score"] is None or isinstance(v["score"], (int, float))
        for v in tasks.values())

    # samples.jsonl rows all carry task + doc + a score field
    scored = 0
    for line in open(os.path.join(data, "samples.jsonl")):
        r = json.loads(line)
        if "task" in r and "doc_id" in r:
            scored += 1

    return {
        "results_reparsed": reparses,
        "samples_wellformed": scored == n_samples,
        "no_absolute_paths": no_abs,
        "readme_present": os.path.isfile(os.path.join(root, "README.md")),
        "chart_present": os.path.isfile(os.path.join(root, "model-scores.png")),
        "throughput_present": os.path.isfile(os.path.join(data,
                                                          "throughput.json")),
    }, n_samples


def main():
    index = {}
    for slug in SLUGS:
        src = os.path.join(MODELS, slug)
        dst = os.path.join(EVAL, f"eval-data-{slug}")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        res = json.load(open(os.path.join(dst, "data", "results.json")))
        gates, n = gates_for(dst)
        _, nbytes = 0, sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(dst) for f in fs)

        manifest = [{
            "source": f"lm-eval harness output, model {res['model']}",
            "destination": f"eval-data-{slug}/",
            "model": res["model"],
            "label": res["label"],
            "family": res["family"],
            "size_gib": res["size_gib"],
            "mean_over_7": res["mean_over_7"],
            "throughput": res.get("throughput"),
            "tasks": {t: {"metric": v["metric"], "score": v["score"],
                          "stderr": v["stderr"], "n_effective": v["n_effective"],
                          "in_mean": v["in_mean"]}
                      for t, v in res["tasks"].items()},
            "sampled_items": n,
            "harness": {
                "lm_eval_version": (res.get("harness") or {}).get("lm_eval_version"),
                "date": (res.get("harness") or {}).get("date"),
                "total_evaluation_time_seconds":
                    (res.get("harness") or {}).get("total_evaluation_time_seconds"),
            },
        }]
        json.dump(manifest, open(os.path.join(dst, "manifest.json"), "w"), indent=2)

        audit = {
            "generated_utc": "2026-10-14T00:00:00+00:00",
            "model": res["model"],
            "slug": slug,
            "source": ("EleutherAI lm-eval 0.4.13 over Lemonade's OpenAI-compatible "
                       "endpoint (IFM K2-Horizon llama.cpp backend), CPU-only"),
            "measurement_class": ("cpu_quality_eval — accuracy + single-stream CPU "
                                  "throughput. NOT a GPU throughput sweep; not "
                                  "interchangeable with the perf-data-* series."),
            "sampled_items": n,
            "bytes": nbytes,
            "manifest_records": len(manifest),
            "tasks": len(res["tasks"]),
            "gates": gates,
        }
        json.dump(audit, open(os.path.join(dst, "audit.json"), "w"), indent=2)

        failed = [k for k, v in gates.items() if v is not True]
        index[slug] = {"slug": slug, "audit": audit}
        if failed:
            print(f"GATE FAIL {slug}: {failed}")

    json.dump(index, open(os.path.join(STAGING, "audit-summary.json"), "w"), indent=2)
    gateverdict = ("PASS" if all(all(a["audit"]["gates"].values())
                                 for a in index.values()) else "SEE ABOVE")
    print(f"reshaped {len(index)} dirs into eval-data-<slug>/, "
          f"all gates: {gateverdict}")
    return index


if __name__ == "__main__":
    main()
