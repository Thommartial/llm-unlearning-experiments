# LLM Unlearning Privacy Experiments

Code for the experiments in *Privacy Vulnerabilities of Machine Unlearning in Large
Language Models* (CIS\*6590, University of Guelph). We fine-tune small language
models on a controlled benchmark, unlearn part of the training data, and
measure how much an attacker can still infer about the forgotten records.

## What it measures

After unlearning a forget set, we attack the model with membership inference
and report its AUC on forgotten vs. never-seen records. We split the forget
set into *typical* and *outlier* (most-memorised) records to check whether
unlearning protects them equally. In short: unlearning lowers the attack's
success but never removes it, the leak grows with model size, and outlier
records stay far more exposed than typical ones.

## Layout

```
configs/         experiment configs (YAML)
src/unlearning/  the package: config, seed, data, unlearn, attacks,
                 evaluate, plotting, cli, sweep
scripts/         run_experiment.py, make_paper_figures.py, verify_gpu.py
notebooks/       Colab/Kaggle notebooks for a single run and the model sweep
tests/           unit tests (no GPU needed)
figures/         generated figures
```

## Setup

```bash
conda env create -f environment.yml
conda activate unlearn
make torch-gpu      # CUDA build of PyTorch; swap whl/cu124 for whl/cpu if no GPU
make install
make verify         # check the GPU is visible
```

PyTorch is installed on its own so the CUDA build matches your machine. On
Colab/Kaggle it is already there, so just `pip install -r requirements.txt`
followed by `pip install -e .`.

## Running

```bash
make smoke          # offline end-to-end run (tiny model, synthetic data, seconds)
make smoke-local    # GPT-2 + TOFU subset on a local GPU
make experiment     # GPT-2-medium + TOFU (one model, a cloud T4)
make sweep          # DistilGPT-2, GPT-2, GPT-2-medium x 3 seeds x 2 attacks
```

Each single run writes `results/<name>/` with `metrics.json`, the per-record
scores, and a figure. The sweep writes `results/sweep/sweep_raw.csv` and is
resumable: re-running skips model/seed pairs that already finished. Rebuild the
paper figures from a sweep CSV with:

```bash
python scripts/make_paper_figures.py --csv results/sweep/sweep_raw.csv --out figures
```

## Federated experiment

A federated extension trains a global model by FedAvg over several clients
(client 0 holds the forget set, the rest shard the retain set), then unlearns
that client and attacks the result. It reports the same residual-leakage and
equity metrics plus an `update_signal_auc`: whether the unlearning update itself
reveals which records were removed (the second-order signal in a distributed
setting).

```bash
# validate offline in seconds (tiny model, synthetic data, CPU):
python scripts/run_federated.py --config configs/smoke_fed_synthetic.yaml --run
# real run on a GPU (edit model.name for distilgpt2 / gpt2 / gpt2-medium):
python scripts/run_federated.py --config configs/fed_tofu.yaml --run
```

Knobs live under `federated:` in the config (`num_clients`, `rounds`,
`local_epochs`, and `unlearn_method: approx | retrain`). `approx` reuses the
configured `unlearn.method` server-side; `retrain` re-runs FedAvg over the
retain clients as the gold reference. On Kaggle, run `notebooks/kaggle_fed_sweep.ipynb`,
which sweeps models and seeds and is resumable across sessions. GPT-2-medium
needs the 16 GB card; a 4 GB GTX 1050 handles the smoke test, DistilGPT-2, and
GPT-2.

Rebuild the federated paper figures (scaling, equity, and update-signal) from
the sweep CSV with `make fed-figures` (or `python scripts/make_fed_figures.py
--csv results/fed_sweep.csv --out figures`). It runs on CPU in seconds.

## Notes

- Runs are config-driven; the resolved config is saved next to the results, and
  `set_seed` fixes the Python, NumPy, and PyTorch RNGs.
- The models are small (82M–355M parameters) so the sweep fits a free Colab or
  Kaggle T4. A 4 GB GTX 1050 handles DistilGPT-2 and GPT-2; GPT-2-medium needs
  the 16 GB T4.
- Gradient ascent tends to diverge; NPO with retain regularisation is the
  stable default.

## Try it on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thommartial/llm-unlearning-experiments/blob/main/notebooks/colab_run.ipynb)

## Citation and licence

See `CITATION.cff`; the code is MIT-licensed (`LICENSE`).
