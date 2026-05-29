#!/usr/bin/env python
"""Report PyTorch / CUDA availability and GPU memory."""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. See README 'Installation'.")
        return

    print(f"torch {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"device: {name} ({total:.1f} GiB VRAM)")


if __name__ == "__main__":
    main()
