# Reproducible workflow. Uses '>' as the recipe prefix to avoid tab pitfalls.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help
SHELL := /bin/bash

help:  ## Show available targets
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
>   awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n",$$1,$$2}'

env:  ## Create the conda environment
> conda env create -f environment.yml

torch-gpu:  ## Install the CUDA build of PyTorch (run inside the env)
> pip install torch --index-url https://download.pytorch.org/whl/cu124

install:  ## Install the package (editable) + dev tools
> pip install -e ".[dev]"

verify:  ## Check the GPU / PyTorch install
> python scripts/verify_gpu.py

smoke:  ## Offline end-to-end run (tiny model + synthetic data; runs anywhere)
> python scripts/run_experiment.py --config configs/smoke_synthetic.yaml --run

smoke-local:  ## Real local run (GPT-2 + TOFU subset; GTX-1050-friendly)
> python scripts/run_experiment.py --config configs/smoke_gpt2.yaml --run

experiment:  ## Full run (Llama-3.2-1B + TOFU; intended for a cloud T4)
> python scripts/run_experiment.py --config configs/tofu_llama1b.yaml --run

sweep:  ## Multi-model/multi-seed sweep, two attacks (Colab/T4)
> python -m unlearning.sweep --config configs/sweep_tofu.yaml --models gpt2 gpt2-medium EleutherAI/pythia-410m --seeds 0 1 2 --out results/sweep

figures:  ## Regenerate figures from results
> python -m unlearning.plotting --results results

lint:  ## Lint
> ruff check src scripts tests

test:  ## Unit tests
> pytest -q

clean:  ## Remove caches and generated artifacts
> rm -rf .pytest_cache .ruff_cache **/__pycache__ results/* && touch results/.gitkeep

.PHONY: help env torch-gpu install verify smoke smoke-local experiment sweep figures lint test clean
