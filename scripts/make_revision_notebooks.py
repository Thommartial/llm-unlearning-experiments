"""Write the Kaggle notebooks of the revision campaign and their kernel metadata.

    python scripts/make_revision_notebooks.py

One folder per notebook under notebooks/revision/<nn>_<name>/ with the
.ipynb and kernel-metadata.json (T4 x2, internet on, private). The kernel
id is "${KAGGLE_USERNAME}/<slug>": scripts/kaggle_run.sh fills in the user
name and the REPO_COMMIT cell at push time, so neither is stored here.
The campaign notebooks are in NOTEBOOKS; DIAGNOSTICS holds short one-GPU
runs of a diagnostic script (no tests or smoke run).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "revision"
REPO_URL = "https://github.com/Thommartial/llm-unlearning-experiments.git"
# versions the code was tested with locally; torch stays Kaggle's CUDA build
PINS = ["transformers==5.9.0", "peft==0.19.1", "datasets==4.8.5", "accelerate==1.13.0"]
# run in a fresh interpreter after the installs: the remaining versions, and peft must not see torchao
VERSIONS = ("import importlib.metadata as m; from peft.import_utils import is_torchao_available; "
            "print({p: m.version(p) for p in ('torch', 'transformers', 'peft')}); "
            "print('torchao seen by peft:', is_torchao_available()); assert not is_torchao_available()")

NOTEBOOKS = {
    "00_pilot": {"campaign": "pilot", "gpu_seeds": {0: None},
                 "what": "the calibration of ANALYSIS_PLAN §7 on GPT-2 seed 0 (fine-tuning rate, FedAvg rounds, "
                         "unlearning radius), the TinyLlama LoRA rate and its first weight load, the LoRA GA "
                         "sanity check, and one central and one federated GPT-2 run timed alone"},
    "01_central": {"campaign": "central", "gpu_seeds": {0: [0, 2, 4], 1: [1, 3]}, "memory_probe": "gpt2-medium",
                   "what": "the central campaign: DistilGPT-2, GPT-2, GPT-2-medium x seeds 0-4"},
    "02_fed_iid": {"campaign": "fed_iid", "gpu_seeds": {0: [0, 2, 4], 1: [1, 3]}, "memory_probe": "gpt2-medium",
                   "what": "the federated campaign: DistilGPT-2, GPT-2, GPT-2-medium x seeds 0-4, K = 5, iid"},
    "03_fed_ext": {"campaign": "fed_ext", "gpu_seeds": {0: [0, 2], 1: [1]},
                   "what": "the federated extensions: GPT-2 x seeds 0-2; by author K = 5, iid K = 3 and K = 10"},
    "04_1b_lora": {"campaign": "lora", "gpu_seeds": {0: [0, 2, 4], 1: [1, 3]},
                   "what": "the LoRA campaign: TinyLlama v1.1 x seeds 0-4, central path"},
}
# diagnostics: one process on GPU 0, no tests or smoke run, results zipped like the campaigns
DIAGNOSTICS = {
    "05_lora_diag": {"script": "scripts/lora_diag.py", "args": ["--steps", "200"], "out": "results/revision/lora_diag",
                     "what": "the LoRA training diagnostic (`scripts/lora_diag.py`): TinyLlama, 200 steps per arm, "
                             "fp16 AMP against fp32 adapters and three one-change variants, every step logged"},
    "06_1b_diag": {"script": "scripts/lora_diag.py", "out": "results/revision/1b_diag",
                   "args": ["--arms", "tinyllama_fp32", "pythia_amp", "pythia_fp32", "--steps", "200", "--eval-check"],
                   "what": "the second LoRA diagnostic (`scripts/lora_diag.py`): an eval check of TinyLlama and "
                           "Pythia-1B under fp32 / fp16 weights, autocast, attention kernels and batch size, then "
                           "200 steps each of TinyLlama with an fp32 base and no autocast, Pythia-1B with fp16 AMP "
                           "and Pythia-1B in fp32"},
}
MEMORY_LIMIT_GIB = 14


def _code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + lines[-1:]}


def _md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines[:-1]] + lines[-1:]}


def _with_ids(cells: list[dict]) -> list[dict]:
    """nbformat 4.5 cell ids (required by newer nbformat versions)."""
    return [{"id": f"cell-{i}", **c} for i, c in enumerate(cells)]


def _setup_cells() -> list[dict]:
    """Clone at REPO_COMMIT, install, list the GPUs."""
    return [
        _code("""import os, subprocess, sys
assert REPO_COMMIT != "SET_BY_KAGGLE_RUN", "launch with scripts/kaggle_run.sh, which sets REPO_COMMIT"
# fewer fragmented reserves; set before torch starts CUDA here or in any subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO = "/tmp/repo"  # outside /kaggle/working: only results and logs are kept as output
subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
os.chdir(REPO)
subprocess.run(["git", "checkout", "-q", REPO_COMMIT], check=True)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
# sys.path.insert below reaches this kernel only: every subprocess gets src/ on PYTHONPATH as well
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [f"{REPO}/src", os.environ.get("PYTHONPATH")]))}"""),
        _code(f"""subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "peft",
                {", ".join(repr(p) for p in PINS)}], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)  # the package itself
# Kaggle ships torchao 0.10, which peft rejects when it builds LoRA layers; nothing here uses torchao
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
versions = subprocess.run([sys.executable, "-c", {VERSIONS!r}], capture_output=True, text=True, env=ENV)
print(versions.stdout, versions.stderr)
assert versions.returncode == 0, "the version check failed (is torchao still visible to peft?)" """),
        _code("""sys.path.insert(0, "src")
import torch, transformers, peft
n_gpu = torch.cuda.device_count()
print("torch", torch.__version__, "CUDA", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)
for i in range(n_gpu):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
assert n_gpu >= len(GPU_SEEDS), f"need {len(GPU_SEEDS)} GPU(s), found {n_gpu}" """),
    ]


def notebook(name: str, spec: dict) -> dict:
    camp = spec["campaign"]
    pilot = camp == "pilot"
    probe = spec.get("memory_probe")
    cells = [
        _md(f"""# Revision campaign: {name}

Runs {spec['what']}.

Launched with `scripts/kaggle_run.sh {name} <commit>`, which sets `REPO_COMMIT` below. Cells: clone the
repository at that commit, install, check the GPUs, run the tests (a skipped CUDA test fails the notebook),
run the smoke config on the GPU, {f"check that {probe} fits a T4 (memory probe), " if probe else ""}then {
"run the pilot in one process on GPU 0" if pilot else "run the campaign in two processes, one per T4, split by seed"
}. Success is judged by the result files.
Everything under `results/revision/` is zipped into `/kaggle/working/`; logs are in `/kaggle/working/logs/`."""),
        _code(f"""REPO_COMMIT = "SET_BY_KAGGLE_RUN"  # written by scripts/kaggle_run.sh
REPO_URL = "{REPO_URL}"
CAMPAIGN = "{camp}"
GPU_SEEDS = {spec['gpu_seeds']!r}  # GPU -> seeds (None: the pilot's own)"""),
        *_setup_cells(),
        _code("""# every test, CUDA-only ones included: a skipped CUDA test fails the notebook
tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], env={**ENV, "REQUIRE_CUDA_TESTS": "1"})
assert tests.returncode == 0, "tests failed (or a CUDA-only test was skipped)" """),
        _code("""# the smoke config on the GPU: every revision path and the pilot, tiny models
for args in (["--campaign", "smoke"], ["--campaign", "pilot", "--config", "configs/revision/smoke_cpu.yaml"]):
    smoke = subprocess.run([sys.executable, "scripts/run_revision.py", *args, "--device", "cuda",
                            "--out-dir", "/tmp/smoke"], env=ENV)
    assert smoke.returncode == 0, f"smoke run failed: {args}" """),
        *([_code(f"""# fail early if {probe} does not fit a T4: one step of each training phase, peaks printed
probe = subprocess.run([sys.executable, "scripts/run_revision.py", "--campaign", CAMPAIGN, "--memory-probe", "{probe}",
                        "--device", "cuda", "--max-reserved-gib", "{MEMORY_LIMIT_GIB}"],
                       env={{**ENV, "CUDA_VISIBLE_DEVICES": "0"}})
assert probe.returncode == 0, "memory probe failed: peak reserved memory over {MEMORY_LIMIT_GIB} GiB, see above" """)]
          if probe else []),
        _code("""import time
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs, logs = {}, {}
for gpu, seeds in GPU_SEEDS.items():
    logs[gpu] = f"/kaggle/working/logs/{CAMPAIGN}_gpu{gpu}.log"
    cmd = [sys.executable, "scripts/run_revision.py", "--campaign", CAMPAIGN]
    if seeds is not None:
        cmd += ["--seeds", *map(str, seeds), "--out", f"results/revision/{CAMPAIGN}_gpu{gpu}.csv"]
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "REPO_COMMIT": REPO_COMMIT}
    procs[gpu] = subprocess.Popen(cmd, stdout=open(logs[gpu], "w"), stderr=subprocess.STDOUT, env=env)
    print(f"GPU {gpu}: {' '.join(cmd)} -> {logs[gpu]}")
start = time.time()"""),
        _code("""# poll the logs until every process ends
def tail(path, n=3):
    with open(path) as f:
        return "".join(f.readlines()[-n:])

while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    print(f"--- {(time.time() - start) / 3600:.2f} h")
    for gpu in procs:
        state = "running" if procs[gpu].poll() is None else f"exit {procs[gpu].returncode}"
        print(f"GPU {gpu} ({state}):\\n{tail(logs[gpu])}")
print({gpu: p.returncode for gpu, p in procs.items()}, f"{(time.time() - start) / 3600:.2f} h")"""),
    ]
    if pilot:
        cells.append(_code("""# success is judged by the result files
import csv, json
report = json.load(open("results/revision/pilot_calibration.json"))
variants = sorted({r["variant"] for r in csv.DictReader(open("results/revision/pilot.csv"))})
print("selected:", json.dumps(report["selected"], indent=2))
print("flags:", report["flags"] or "none")
print("timing runs:", variants)
print(tail(logs[0], 12))
selected = {"finetune_lr", "rounds", "rho_factor", "lora_finetune_lr", "lora_learning_rate"}
ok = set(report["selected"]) == selected and variants == ["central", "iid_K5"]"""))
    else:
        cells.append(_code("""# success is judged by the result files, not by the exit codes
import glob
csvs = sorted(glob.glob(f"results/revision/{CAMPAIGN}_gpu*.csv"))
check = subprocess.run([sys.executable, "scripts/run_revision.py", "--campaign", CAMPAIGN, "--check", *csvs],
                       env=ENV)
if csvs:
    subprocess.run([sys.executable, "scripts/run_revision.py", "--merge", *csvs,
                    "--out", f"results/revision/{CAMPAIGN}.csv"], check=True, env=ENV)
ok = check.returncode == 0"""))
    cells.append(_code("""import shutil
archive = shutil.make_archive(f"/kaggle/working/revision_{CAMPAIGN}", "zip", "results/revision")
print("results ->", archive)
assert ok, "incomplete results: see the check above and the logs in /kaggle/working/logs/" """))
    return {"cells": _with_ids(cells), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def diag_notebook(name: str, spec: dict) -> dict:
    cells = [
        _md(f"""# Revision diagnostic: {name}

Runs {spec['what']}.

Launched with `scripts/kaggle_run.sh {name} <commit>`, which sets `REPO_COMMIT` below. Cells: clone the
repository at that commit, install, check the GPU, run `{spec['script']}` in one process on GPU 0, print its
log. Everything under `{spec['out']}/` is zipped into `/kaggle/working/`; the log is in `/kaggle/working/logs/`."""),
        _code(f"""REPO_COMMIT = "SET_BY_KAGGLE_RUN"  # written by scripts/kaggle_run.sh
REPO_URL = "{REPO_URL}"
GPU_SEEDS = {{0: None}}  # one process on GPU 0"""),
        *_setup_cells(),
        _code(f"""os.makedirs("/kaggle/working/logs", exist_ok=True)
log = "/kaggle/working/logs/{name}.log"
cmd = [sys.executable, "{spec['script']}", {", ".join(json.dumps(a) for a in spec['args'])}, "--device", "cuda",
       "--out", "{spec['out']}"]
with open(log, "w") as f:
    run = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                         env={{**ENV, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}})
print(open(log).read())"""),
        _code(f"""import shutil
archive = shutil.make_archive("/kaggle/working/{name}", "zip", "{spec['out']}")
print("results ->", archive)
assert run.returncode == 0, "the diagnostic failed: see the log above" """),
    ]
    return {"cells": _with_ids(cells), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def metadata(name: str) -> dict:
    slug = f"unlearning-revision-{name.replace('_', '-')}"
    return {"id": "${KAGGLE_USERNAME}/" + slug, "title": slug, "code_file": f"{name}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main() -> None:
    builds = [(n, notebook(n, s)) for n, s in NOTEBOOKS.items()] + [(n, diag_notebook(n, s))
                                                                    for n, s in DIAGNOSTICS.items()]
    for name, nb in builds:
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata(name), indent=2) + "\n")
        print(f"wrote {folder}")


if __name__ == "__main__":
    main()
