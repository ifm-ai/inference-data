#!/usr/bin/env python3
"""Emit (or execute) the exact commands that reproduce a dataset.

This script is identical across the repository and lives at
``scripts/reproduce.py`` at the root of every ``perf-data-<model>`` branch.
It scans ``--dataset-root`` (default: the repository root, i.e. two levels
up from this script) for cell ``environment.json`` files and, for every
measured ``point`` (one ``<topology>/<runtime>/<dtype>/isl<I>-osl<O>/c<N>``
directory holding ``run1..run3`` records), it generates:

1. **One server launch command**, taken verbatim from the cell's
   ``environment.json`` (``server_launch[<concurrency>].argv``). This is the
   exact command line the measurement used, with the ``<models-root>/``
   placeholder resolved from ``--models-root``.
2. **One InferenceX client command per repetition** (``run1``..``run3``):
   ``python3 -m infx.bench_serving.benchmark_serving`` with the point's
   workload — exact ISL/OSL (``--random-range-ratio 1.0``), ``C`` prompts at
   ``--max-concurrency C``, ``--seed 3``, ``/v1/completions`` — saving each
   repetition's result under ``<output-root>/<same layout as the dataset>/
   run<k>/result.inferencex.json``.

Default mode only prints the commands (pipe into a shell to run manually).
``--execute`` instead runs them sequentially per point: start the server,
wait for ``GET /health`` to answer 200, run the three client repetitions,
stop the server.

``--execute`` requirements: the serving runtime installed at the commit
recorded in the cell's ``environment.json`` (or its recorded container
image providing the same executables), an InferenceX checkout at the
recorded commit (``components.inferencex`` — verified against
``git rev-parse HEAD``), GPUs matching the cell's topology, and the model
weights under ``--models-root``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset layout constants
# ---------------------------------------------------------------------------
#
# A *cell* is one <topology>/<runtime>/<dtype> directory carrying an
# environment.json; a *point* is one isl<I>-osl<O>/c<N> directory below it;
# each point holds RUNS repetitions in run<k>/ subdirectories. The dataset
# root is --dataset-root, defaulting to the repository root when script and
# dataset are checked out together.

CELL_GLOB = "*x*/*/*/environment.json"   # <topo>/<runtime>/<dtype>/environment.json
RUNTIMES = ("vllm", "sglang")            # sane filter for the glob above
POINT_RE = re.compile(r"isl(\d+)-osl(\d+)")   # point directory naming scheme
CONC_RE = re.compile(r"c(\d+)$")              # concurrency directory naming scheme
RUNS = (1, 2, 3)                         # repetitions published per point


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def parse_flag(argv: list[str], flag: str) -> str | None:
    """Value that follows ``flag`` in a command argv, or None if absent."""
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


# ---------------------------------------------------------------------------
# Client command construction — one command per run repetition
# ---------------------------------------------------------------------------
#
# Generated command shape (mirrors the recorded client_request_example in
# each cell's environment.json):
#
#   python3 -m infx.bench_serving.benchmark_serving \
#     --backend openai --base-url http://127.0.0.1:<port> --model <served-name> \
#     --tokenizer <checkpoint-path> \
#     --dataset-name random --random-input-len <ISL> --random-output-len <OSL> \
#     --random-range-ratio 1.0 \
#     --num-prompts <C> --max-concurrency <C> --request-rate inf \
#     --seed 3 --ignore-eos --endpoint /v1/completions \
#     --save-result --result-dir <out>/run<k> \
#     --result-filename result.inferencex.json --metadata run=<k>
#
# --random-range-ratio 1.0 makes InferenceX generate exactly ISL input and
# OSL output tokens per request; --ignore-eos forces the server to emit all
# OSL tokens; C prompts are all admitted at once at concurrency C.


def server_model_path(server_argv: list[str]) -> str:
    """Checkpoint path carried by a server argv.

    SGLang launchers take ``--model-path <path>``; vLLM takes the path as the
    positional argument right after ``serve``. Both forms appear in the
    recorded environment.json server_launch entries.
    """
    path = parse_flag(server_argv, "--model-path")
    if path:
        return path
    if "serve" in server_argv:
        return server_argv[server_argv.index("serve") + 1]
    raise SystemExit("gate: cannot locate checkpoint path in server argv")


def client_argv(server_argv: list[str], isl: int, osl: int, conc: int,
                run: int, out_dir: Path) -> list[str]:
    """InferenceX client argv for one repetition of one point.

    Connection details (port, served model name, checkpoint path) are read
    back from the server argv so the emitted pair of commands always matches.
    The tokenizer flag is required: InferenceX tokenizes the random dataset
    locally, and the served model name is only a routing alias, not a
    loadable tokenizer reference.
    """
    port = parse_flag(server_argv, "--port") or "8000"
    model = parse_flag(server_argv, "--served-model-name") or "model"
    return [
        "python3", "-m", "infx.bench_serving.benchmark_serving",
        "--backend", "openai",
        "--base-url", f"http://127.0.0.1:{port}",
        "--model", model,
        "--tokenizer", server_model_path(server_argv),
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--random-range-ratio", "1.0",
        "--num-prompts", str(conc),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--seed", "3",
        "--ignore-eos",
        "--endpoint", "/v1/completions",
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", "result.inferencex.json",
        "--metadata", f"run={run}",
    ]


# ---------------------------------------------------------------------------
# Discovery: cells -> points -> concurrencies present in the dataset
# ---------------------------------------------------------------------------


def iter_cells(dataset_root: Path) -> list[Path]:
    """environment.json paths of every cell found below the dataset root.

    Search depth is not fixed, so the script works from a branch root (cells
    under ``perf-<model>/...``) as well as from inside a dataset
    directory (cells one level down). Paths are kept relative to
    ``dataset_root`` so emitted output mirrors the checked-out layout.
    """
    return sorted(
        path for path in dataset_root.rglob("environment.json")
        if path.parent.parent.name in RUNTIMES
        and any(path.parent.glob("isl*-osl*"))
    )


def iter_points(cell_env: Path):
    """Yield ``(isl, osl, conc, conc_dir)`` for each published point of a cell.

    A directory only counts as a point if it actually holds run records —
    mirroring what the dataset ships.
    """
    for isl_dir in sorted(cell_env.parent.glob("isl*-osl*")):
        point_match = POINT_RE.fullmatch(isl_dir.name)
        if not point_match:
            continue
        isl, osl = int(point_match.group(1)), int(point_match.group(2))
        for conc_dir in sorted(isl_dir.glob("c*")):
            conc_match = CONC_RE.fullmatch(conc_dir.name)
            if not conc_match or not any(conc_dir.glob("run*/result.json")):
                continue
            yield isl, osl, int(conc_match.group(1)), conc_dir


# ---------------------------------------------------------------------------
# Execution engine (--execute only)
# ---------------------------------------------------------------------------


def wait_health(url: str, timeout_s: float) -> None:
    """Block until ``GET <url>/health`` answers 200 (both runtimes expose it)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise SystemExit(f"gate: server at {url} did not become healthy in {timeout_s}s")


def resolve_checkout_head(inferencex: Path) -> str:
    """HEAD commit of a git checkout without requiring the git binary.

    Handles plain checkouts (``.git/`` directory) and linked worktrees
    (``.git`` pointer file), including refs stored only in ``packed-refs``.
    """
    gitdir = inferencex / ".git"
    if gitdir.is_file():                      # linked worktree pointer
        text = gitdir.read_text().strip()
        gitdir = Path(text.removeprefix("gitdir: ").strip())
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head                           # detached HEAD: raw commit id
    ref = head.removeprefix("ref:").strip()
    ref_file = gitdir / ref
    if ref_file.is_file():
        return ref_file.read_text().strip()
    packed = gitdir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0].strip()
    raise SystemExit(f"gate: cannot resolve {ref} in {inferencex}")


def check_inferencex_commit(inferencex: Path, pinned: str) -> None:
    """Refuse to execute against an InferenceX checkout at the wrong commit."""
    head = resolve_checkout_head(inferencex)
    if head != pinned:
        raise SystemExit(
            f"gate: InferenceX checkout is at {head}, dataset requires {pinned}")


def execute_point(server: list[str], environment: dict, isl: int, osl: int,
                  conc: int, point_out: Path, inferencex: Path,
                  health_timeout: float) -> None:
    """Run one point end-to-end: server up -> health gate -> run1..3 -> down.

    Server stdout/stderr goes to ``<point_out>/server.log``; each repetition
    writes ``<point_out>/run<k>/result.inferencex.json`` via the client's own
    ``--save-result``.
    """
    point_out.mkdir(parents=True, exist_ok=True)
    log_file = open(point_out / "server.log", "w")
    server_proc = subprocess.Popen(
        server, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        port = parse_flag(server, "--port") or "8000"
        wait_health(f"http://127.0.0.1:{port}", health_timeout)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(inferencex)   # makes `import infx` resolvable
        for run in RUNS:
            client = client_argv(server, isl, osl, conc, run,
                                 point_out / f"run{run}")
            (point_out / f"run{run}").mkdir(parents=True, exist_ok=True)
            subprocess.run(client, cwd=str(inferencex), env=env, check=True)
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log_file.close()


# ---------------------------------------------------------------------------
# Entry point: parse args -> walk the dataset -> emit or execute per point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root",
                        help="dataset checkout to walk (default: repository "
                             "root when script and dataset coexist)")
    parser.add_argument("--cell",
                        help="limit to one cell, e.g. 1x2/vllm/bf16")
    parser.add_argument("--point",
                        help="limit to one point directory, e.g. isl1024-osl1024")
    parser.add_argument("--models-root", default="<models-root>",
                        help="directory holding the model checkpoint dirs")
    parser.add_argument("--output-root",
                        help="where results are written (default: <dataset-root>/output)")
    parser.add_argument("--inferencex", default="./InferenceX",
                        help="InferenceX checkout (required for --execute)")
    parser.add_argument("--execute", action="store_true",
                        help="run the commands instead of printing them")
    parser.add_argument("--health-timeout", type=float, default=3600.0,
                        help="seconds to wait for the server /health endpoint")
    args = parser.parse_args()

    dataset_root = (Path(args.dataset_root).resolve() if args.dataset_root
                    else Path(__file__).resolve().parent.parent)
    out_root = (Path(args.output_root) if args.output_root
                else dataset_root / "output")
    inferencex = Path(args.inferencex).resolve()
    checked_commit = False

    # One pass over the dataset: cells -> points -> per-point command blocks.
    for env_path in iter_cells(dataset_root):
        cell_rel = env_path.parent.relative_to(dataset_root)
        if args.cell and str(cell_rel) != args.cell:
            continue
        environment = json.loads(env_path.read_text())
        launches = environment.get("server_launch") or {}

        if args.execute and not checked_commit:
            check_inferencex_commit(
                inferencex, environment["components"]["inferencex"]["commit"])
            checked_commit = True

        # For every point: resolve the server command for its concurrency,
        # then emit (or run) it together with one client command per run.
        for isl, osl, conc, conc_dir in iter_points(env_path):
            if args.point and isl_dir_name(conc_dir) != args.point:
                continue
            launch = launches.get(str(conc))
            if launch is None:
                raise SystemExit(
                    f"gate: {env_path} has no server_launch for concurrency {conc}")
            server = [arg.replace("<models-root>", args.models_root)
                      for arg in launch["argv"]]
            point_rel = cell_rel / conc_dir.parent.name / conc_dir.name
            point_out = out_root / point_rel

            # Command block header: point path (+ synthesis note if the
            # launch entry carries one).
            print(f"# === {point_rel} ===")
            if launch.get("note"):
                print(f"  # note: {launch['note']}")
            print(shlex.join(server))
            for run in RUNS:
                print(shlex.join(client_argv(server, isl, osl, conc, run,
                                             point_out / f"run{run}")))
            print()
            if args.execute:
                execute_point(server, environment, isl, osl, conc, point_out,
                              inferencex, args.health_timeout)
    return 0


def isl_dir_name(conc_dir: Path) -> str:
    """isl<I>-osl<O> name of a point's parent directory (for --point filter)."""
    return conc_dir.parent.name


if __name__ == "__main__":
    sys.exit(main())
