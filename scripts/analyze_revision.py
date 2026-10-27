"""Analyse the revision campaign: statistics, LaTeX tables and figures (CPU).

    python scripts/analyze_revision.py --results results/revision --figures figures/revision --tables tables/revision

Reads the merged campaign CSVs, pilot.csv, scores/*/*.npz and configs/*/*.json
under --results (see unlearning.analysis); writes CSV summaries and LaTeX
tables to --tables and figures (PDF + PNG) to --figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlearning.analysis import analyse  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/revision")
    ap.add_argument("--figures", default="figures/revision")
    ap.add_argument("--tables", default="tables/revision")
    args = ap.parse_args(argv)
    a = analyse(args.results, args.figures, args.tables)
    print(f"{len(a['rows'])} rows from {a['rows']['campaign'].nunique()} campaign(s)")
    for p in [*a["tables"], *a["figures"]]:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
