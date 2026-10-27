"""With REQUIRE_CUDA_TESTS=1 (the Kaggle notebooks), a CUDA-only test that is
skipped counts as a failure, so a missing or unsupported GPU fails the run."""

from __future__ import annotations

import os

import pytest

CUDA_SKIP_REASON = "needs a GPU this PyTorch build supports"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("REQUIRE_CUDA_TESTS") == "1" and report.skipped and CUDA_SKIP_REASON in str(report.longrepr):
        report.outcome = "failed"
        report.longrepr = f"CUDA-only test skipped although REQUIRE_CUDA_TESTS=1: {CUDA_SKIP_REASON}"
