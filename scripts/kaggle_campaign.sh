#!/usr/bin/env bash
# Run the revision campaign notebooks 01 -> 04 on Kaggle in sequence with one commit,
# stopping at the first failure (run 00_pilot first, alone, and set the calibration values).
#
#   scripts/kaggle_campaign.sh [<commit>]
set -euo pipefail

commit=$(git rev-parse --verify "${1:-HEAD}^{commit}")
here=$(dirname "$0")
for nb in 01_central 02_fed_iid 03_fed_ext 04_1b_lora; do
    echo "== $nb at $commit"
    if ! "$here/kaggle_run.sh" "$nb" "$commit"; then
        echo "stopped: $nb did not complete" >&2
        exit 1
    fi
done
echo "all notebooks completed"
