"""Cost instrumentation: wall time and peak GPU memory per phase, the run
environment, and analytic communication for the federated runs."""

from __future__ import annotations

import contextlib
import time

import torch

BYTES_PER_PARAM = 4  # fp32 weights are exchanged (fp16 AMP keeps fp32 master weights)
_active = []


def _cuda(device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


@contextlib.contextmanager
def phase(device):
    """Measure one phase; yields a dict filled on exit with `wall_s` and
    `peak_mem_mib`.

    On CUDA the device is synchronized before and after, so queued kernels
    count towards the phase they belong to, and the peak is
    `max_memory_allocated` after `reset_peak_memory_stats` (it includes what
    was already allocated at the start, e.g. the weights). On CPU the peak is
    None. Phases cannot be nested: the reset would hide the outer peak.
    """
    if _active:
        raise RuntimeError("cost phases cannot be nested")
    cuda = _cuda(device)
    out = {}
    _active.append(out)
    try:
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        yield out
        if cuda:
            torch.cuda.synchronize(device)
        out["wall_s"] = time.perf_counter() - start
        out["peak_mem_mib"] = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
    finally:
        _active.pop()


def combine(*phases: dict) -> dict:
    """Several phases run one after the other: summed wall time, largest peak."""
    peaks = [p["peak_mem_mib"] for p in phases if p.get("peak_mem_mib") is not None]
    return {"wall_s": sum(p["wall_s"] for p in phases), "peak_mem_mib": max(peaks) if peaks else None}


def columns(costs: dict) -> dict:
    """{phase: cost or None} -> flat row columns `time_s_<phase>` and
    `peak_mem_mib_<phase>` (None for a phase that did not run)."""
    out = {}
    for name, c in costs.items():
        out[f"time_s_{name}"] = None if c is None else c["wall_s"]
        out[f"peak_mem_mib_{name}"] = None if c is None else c["peak_mem_mib"]
    return out


def environment(device) -> dict:
    """GPU name (or "cpu"), torch version and the CUDA version torch was built with."""
    cuda = _cuda(device)
    return {"gpu_name": torch.cuda.get_device_name(device) if cuda else "cpu",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda}


def trainable_params(model) -> int:
    """Parameters that training changes, counted once (tied weights once);
    with LoRA only the adapter weights."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def communication(n_params: int, uploads: int, downloads: int) -> dict:
    """Analytic communication: (uploads + downloads) x parameters x 4 bytes."""
    return {"comm_uploads": uploads, "comm_downloads": downloads,
            "comm_bytes": (uploads + downloads) * n_params * BYTES_PER_PARAM}


def fedavg_transfers(n_clients: int, rounds: int) -> tuple[int, int]:
    """(uploads, downloads) of FedAvg: each round, every client downloads the
    global model and uploads its update."""
    return n_clients * rounds, n_clients * rounds
