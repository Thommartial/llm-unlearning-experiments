#!/usr/bin/env bash
# Push one revision notebook to Kaggle, wait for it to finish, download its output.
#
#   scripts/kaggle_run.sh <nn_name> [<commit>]      e.g. scripts/kaggle_run.sh 00_pilot
#
# <nn_name> is a folder under notebooks/revision/; <commit> defaults to HEAD and must be
# on GitHub (the notebook clones the repository at that commit). Needs the official
# Kaggle CLI (pip install kaggle) with its usual credentials (~/.kaggle/kaggle.json, or
# KAGGLE_USERNAME and KAGGLE_KEY) and KAGGLE_USERNAME set; no credential is stored in the
# repository. The commit hash and the user name are written into a temporary copy of the
# notebook folder, which is what gets pushed. Polls the status every KAGGLE_POLL_SECONDS
# (default 300); on completion downloads the output to results/revision/kaggle/<nn_name>/,
# prints the final status and the log path, and exits non-zero unless the kernel completed.
set -euo pipefail

name=${1:?usage: scripts/kaggle_run.sh <nn_name> [<commit>]}
: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME to your Kaggle user name}"
root=$(git rev-parse --show-toplevel)
src="$root/notebooks/revision/$name"
[[ -f "$src/$name.ipynb" && -f "$src/kernel-metadata.json" ]] || { echo "no notebook in $src" >&2; exit 2; }
commit=$(git rev-parse --verify "${2:-HEAD}^{commit}")
if [[ -z $(git branch -r --contains "$commit" 2>/dev/null) ]]; then
    echo "commit $commit is not on any remote branch: push it first (the notebook clones it)" >&2
    exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 - "$src" "$work" "$name" "$commit" <<'PY'
import json, os, re, sys
src, work, name, commit = sys.argv[1:]
nb = json.load(open(f"{src}/{name}.ipynb"))
hits = 0
for cell in nb["cells"]:
    for i, line in enumerate(cell["source"]):
        if line.startswith("REPO_COMMIT = "):
            cell["source"][i] = re.sub(r'"[^"]*"', f'"{commit}"', line, count=1)
            hits += 1
if hits != 1:
    sys.exit(f"expected one REPO_COMMIT line in {name}.ipynb, found {hits}")
json.dump(nb, open(f"{work}/{name}.ipynb", "w"), indent=1)
meta = json.load(open(f"{src}/kernel-metadata.json"))
meta["id"] = meta["id"].replace("${KAGGLE_USERNAME}", os.environ["KAGGLE_USERNAME"])
json.dump(meta, open(f"{work}/kernel-metadata.json", "w"), indent=2)
PY
id=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$work/kernel-metadata.json")

echo "pushing $name at $commit as $id"
kaggle kernels push -p "$work"

failures=0
while true; do
    sleep "${KAGGLE_POLL_SECONDS:-300}"
    line=$(kaggle kernels status "$id" 2>&1) || true
    status=$(sed -n 's/.*has status "\([^"]*\)".*/\1/p' <<<"$line" | tr '[:upper:]' '[:lower:]')
    echo "$(date '+%F %T') ${status:-?} ($line)"
    case "$status" in
        *complete*) final=complete; break ;;
        *error*|*cancel*) final=$status; break ;;
        *running*|*queued*) failures=0 ;;
        *) failures=$((failures + 1))   # the CLI or the API failed; try again
           if (( failures >= 6 )); then final="unknown (status unavailable)"; break; fi ;;
    esac
done

out="$root/results/revision/kaggle/$name"
mkdir -p "$out"
kaggle kernels output "$id" -p "$out" || echo "could not download the output of $id" >&2
echo "final status: $final"
echo "output: $out"
echo "log: $(ls "$out"/*.log 2>/dev/null | head -1 || true)"
[[ $final == complete ]]
