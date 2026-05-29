# LLM Unlearning Privacy Experiments

Reproducible experiments measuring **residual privacy leakage after machine
unlearning** in large language models. This repository accompanies the survey
*"Privacy Attacks on Machine Unlearning in Large Language Models: Attacks,
Defences, and Verification Challenges"* (CIS\*6590, University of Guelph, 2026)
and provides a controlled, config-driven study to complement the review.

> **Status:** infrastructure is complete and tested; the experiment logic
> (`data`, `unlearn`, `attacks`, `evaluate`) is scaffolded with documented
> stubs and is implemented in the experiment phase. The dry-run plan,
> configuration, seeding, tests, and CI all work today.

## Research question

Does approximate unlearning (gradient ascent, NPO) leave a *detectable,
exploitable* signal, and do standard checks understate it? Concretely, after
unlearning a forget set we run a membership-inference attack (MIA) and report
its AUC on forget vs. never-seen records, including an **outlier-vs-typical
slice** to test the minority-leakage effect.

## Repository structure

```
.
├── configs/            # YAML experiment configs (smoke + full)
├── src/unlearning/     # package: config, seed, data, unlearn, attacks, evaluate, plotting, cli
├── scripts/            # run_experiment.py, verify_gpu.py
├── tests/              # reproducibility smoke tests
├── results/            # run outputs (git-ignored)
├── data/               # datasets/caches (git-ignored)
├── environment.yml     # conda environment (PyTorch installed separately)
├── pyproject.toml      # package metadata, ruff + pytest config
└── .github/workflows/  # CI: lint + tests on every push
```

## Installation

```bash
# 1. Environment
conda env create -f environment.yml
conda activate unlearn

# 2. PyTorch with the right CUDA build (chosen for your hardware)
#    GPU (CUDA):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#    CPU only:    pip install torch --index-url https://download.pytorch.org/whl/cpu
make torch-gpu

# 3. The package (editable) + dev tools
make install
```

## Run on Colab (free T4)

No local GPU needed. Open the notebook and run top to bottom:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thommartial/llm-unlearning-experiments/blob/main/notebooks/colab_run.ipynb)

It clones the repo, installs deps (PyTorch is preinstalled on Colab), runs the
offline sanity check, then a real `gpt2-medium` + TOFU run, and shows the figure.

## Quickstart

```bash
make verify       # report PyTorch/CUDA and GPU memory
make smoke        # fast local dry-run plan (GPT-2, GTX-1050-friendly)
make test         # unit tests
make lint         # ruff
```

Run the full pipeline (once implemented) with:

```bash
python scripts/run_experiment.py --config configs/tofu_llama1b.yaml --run
```

## Reproducibility

- **Configs over flags.** Every run is fully described by a YAML file in
  `configs/`; the resolved config is saved to `results/<name>/config.resolved.json`.
- **Fixed seeds.** `unlearning.seed.set_seed` seeds Python, NumPy, and PyTorch
  and enables deterministic algorithms.
- **Pinned environment.** `environment.yml` + `requirements.txt` capture the
  software stack; PyTorch is installed explicitly to match your CUDA build.
- **CI.** GitHub Actions lints and runs the tests on every push.

## Hardware notes

| Setting | Model | Notes |
|---|---|---|
| Local (GTX 1050, 4 GB) | GPT-2 / Pythia-160M–410M | smoke tests, fp32/fp16; **no** bitsandbytes (Pascal) |
| Cloud T4 (16 GB, free) | Llama-3.2-1B / Phi-3-mini | full runs with LoRA/QLoRA |

## Citation

See `CITATION.cff`. Please cite both this software and the accompanying paper.

## License

MIT — see `LICENSE`.
